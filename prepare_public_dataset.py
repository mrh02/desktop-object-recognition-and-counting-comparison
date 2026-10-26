"""Build a bounded COCO-80 pilot subset from existing public box labels.

Downloads public data only. No image upload, model API, training or app changes.
Use --download explicitly; reruns reuse the fixed selection and checked cache.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import zipfile

import requests
from PIL import Image
import yaml

PROJECT = Path(__file__).resolve().parent
LABEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"
# The official dataset documentation uses this HTTP image endpoint. Do not disable TLS.
IMAGE_BASE = "http://images.cocodataset.org"
FOCUS = {41: "cup", 64: "mouse", 76: "scissors", 66: "keyboard", 45: "bowl", 73: "book"}
SEED = "lingjing-public-pilot-v1"
MIB = 1024 * 1024


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(MIB), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


class Budget:
    def __init__(self, limit):
        self.limit, self.used, self.lock = limit, 0, threading.Lock()

    def consume(self, size):
        with self.lock:
            if self.used + size > self.limit:
                raise RuntimeError("Public-data download byte budget exhausted; cached files retained")
            self.used += size


def download(url, path, max_bytes, budget):
    if path.exists():
        if not 0 < path.stat().st_size <= max_bytes:
            raise ValueError(f"Invalid cached file size: {path.name}")
        return {"cached": True, "bytes": path.stat().st_size, "sha256": sha256(path)}
    part = path.with_suffix(path.suffix + ".part")
    size = 0
    with requests.get(url, stream=True, timeout=(15, 40)) as response:
        response.raise_for_status()
        declared = int(response.headers.get("Content-Length", 0))
        if declared > max_bytes:
            raise ValueError("Remote file exceeds per-file byte cap")
        with part.open("wb") as f:
            for chunk in response.iter_content(64 * 1024):
                if not chunk:
                    continue
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError("Downloaded file exceeds per-file byte cap")
                budget.consume(len(chunk))
                f.write(chunk)
    if not size or (declared and size != declared):
        raise ValueError("Empty or truncated download")
    part.replace(path)
    return {"cached": False, "bytes": size, "sha256": sha256(path)}


def read_labels(text):
    rows, clipped = [], 0
    for line in text.splitlines():
        values = line.split()
        if not values:
            continue
        if len(values) != 5:
            raise ValueError("Expected five YOLO box-label columns")
        cls_float, x, y, w, h = map(float, values)
        if not all(math.isfinite(v) for v in (cls_float, x, y, w, h)):
            raise ValueError("Non-finite label value")
        cls = int(cls_float)
        if cls != cls_float or not 0 <= cls < 80 or w <= 0 or h <= 0:
            raise ValueError("Invalid COCO-80 box label")
        left, top, right, bottom = x-w/2, y-h/2, x+w/2, y+h/2
        bounded = [max(0., left), max(0., top), min(1., right), min(1., bottom)]
        if bounded[0] >= bounded[2] or bounded[1] >= bounded[3]:
            raise ValueError("Box lies outside image")
        if any(abs(a-b) > 1e-8 for a, b in zip((left, top, right, bottom), bounded)):
            clipped += 1
        left, top, right, bottom = bounded
        rows.append((cls, (left+right)/2, (top+bottom)/2, right-left, bottom-top))
    return rows, clipped


def choose(index, split, quota):
    candidates = sorted((v for v in index if v["source_split"] == split),
                        key=lambda v: hashlib.sha256((SEED+split+v["image_id"]).encode()).hexdigest())
    selected = {}
    # Scarce target classes first. Reuse co-occurring classes instead of duplicating images.
    for cls in sorted(FOCUS, key=lambda c: sum(c in v["classes"] for v in candidates)):
        for row in candidates:
            if sum(cls in v["classes"] for v in selected.values()) >= quota:
                break
            if cls in row["classes"]:
                selected[row["image_id"]] = row
    # Preserve broad category coverage and all labels within every selected image.
    for cls in range(80):
        if any(cls in v["classes"] for v in selected.values()):
            continue
        for row in candidates:
            if cls in row["classes"]:
                selected[row["image_id"]] = row
                break
    return [dict(v, split="train" if split == "train2017" else "val") for v in selected.values()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--train-per-focus", type=int, default=40)
    parser.add_argument("--val-per-focus", type=int, default=10)
    parser.add_argument("--max-download-mib", type=int, default=200)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    if not (1 <= args.train_per_focus <= 60 and 1 <= args.val_per_focus <= 15):
        parser.error("Pilot per-class caps are 60 train / 15 val images")
    if not 1 <= args.max_download_mib <= 200:
        parser.error("This pilot permits at most 200 MiB of network transfer per run")
    if not args.download:
        print(json.dumps(vars(args), indent=2))
        print("No download performed. Use --download to prepare the public pilot.")
        return
    root = PROJECT / "outputs" / "public_data" / "coco_desktop_pilot_v1"
    root.mkdir(parents=True, exist_ok=True)
    cached_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and p.suffix in (".zip", ".jpg", ".part"))
    total_cap = args.max_download_mib * MIB
    if cached_bytes > total_cap:
        raise ValueError("Existing download cache already exceeds this pilot's total byte cap")
    budget = Budget(total_cap - cached_bytes)
    archive = root / "coco2017labels.zip"
    archive_meta = download(LABEL_URL, archive, 64*MIB, budget)
    print(f"Public annotation archive: {archive_meta['bytes']} bytes", flush=True)
    write_json(root / "annotation_source.json", {"url": LABEL_URL, **archive_meta})
    plan_path = root / "selection.json"
    with zipfile.ZipFile(archive) as zf:
        if zf.testzip() is not None:
            raise ValueError("Annotation archive CRC check failed")
        if plan_path.exists():
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            if plan["archive_sha256"] != archive_meta["sha256"] or plan["quotas"] != [args.train_per_focus, args.val_per_focus]:
                raise ValueError("Existing selection differs; do not overwrite a previous experiment")
            selected = plan["images"]
        else:
            index = []
            for info in zf.infolist():
                match = re.fullmatch(r"coco/labels/(train2017|val2017)/(\d{12})\.txt", info.filename)
                if not match:
                    continue
                text = zf.read(info).decode("utf-8-sig")
                classes = sorted({int(line.split()[0]) for line in text.splitlines() if line.strip()})
                if any(c < 0 or c >= 80 for c in classes):
                    raise ValueError("Archive class ID is outside COCO-80")
                index.append({"source_split": match[1], "image_id": match[2], "entry": info.filename, "classes": classes})
            if not index:
                raise ValueError("No expected COCO labels found in archive")
            selected = choose(index, "train2017", args.train_per_focus) + choose(index, "val2017", args.val_per_focus)
            if len(selected) > 500:
                raise ValueError("Pilot selection exceeds 500 images")
            plan = {"seed": SEED, "archive_sha256": archive_meta["sha256"], "quotas": [args.train_per_focus, args.val_per_focus],
                    "focus_classes": FOCUS, "images": selected}
            write_json(plan_path, plan)
        for row in selected:
            labels, clipped = read_labels(zf.read(row["entry"]).decode("utf-8-sig"))
            row["label_rows"] = labels
            row["boundary_clipped_boxes"] = clipped
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    print(f"Fixed public selection: {Counter(v['split'] for v in selected)}", flush=True)

    def fetch(row):
        row = dict(row)
        filename = f"coco_{row['source_split']}_{row['image_id']}.jpg"
        destination = root / "images" / row["split"] / filename
        url = f"{IMAGE_BASE}/{row['source_split']}/{row['image_id']}.jpg"
        meta = download(url, destination, 3*MIB, budget)
        with Image.open(destination) as im:
            im.load()
            if im.getexif().get(274, 1) not in (0, 1):
                raise ValueError("Unexpected image orientation; box alignment needs inspection")
            rgb = im.convert("RGB")
            width, height = rgb.size
            pixel_hash = hashlib.sha256(f"{width}x{height}:RGB:".encode()+rgb.tobytes()).hexdigest()
        label_path = root / "labels" / row["split"] / Path(filename).with_suffix(".txt")
        text = "".join(f"{c} {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n" for c, x, y, w, h in row.pop("label_rows"))
        label_path.write_text(text, encoding="utf-8")
        return {**row, **meta, "image": str(destination), "labels": str(label_path), "url": url,
                "width": width, "height": height, "pixel_sha256": pixel_hash, "label_sha256": sha256(label_path)}

    records, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, row): row for row in selected}
        for future in as_completed(futures):
            row = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:
                failures.append({"source_split": row["source_split"], "image_id": row["image_id"], "error": f"{type(exc).__name__}: {str(exc)[:220]}"})
            if (len(records)+len(failures)) % 25 == 0:
                print(f"Prepared {len(records)}/{len(selected)}, errors {len(failures)}, transfer {budget.used/MIB:.1f} MiB", flush=True)
                write_json(root / "download_progress.json", {"completed": len(records), "failed": failures, "network_bytes_this_run": budget.used})
    records.sort(key=lambda r: (r["split"], r["image_id"]))
    # Exclude exact duplicated content from all later splits, preserving downloaded originals.
    unique, seen, duplicates = [], set(), []
    for row in records:
        if row["pixel_sha256"] in seen:
            duplicates.append({"image": row["image"], "reason": "duplicate_pixel_content_excluded_from_lists"})
        else:
            seen.add(row["pixel_sha256"])
            unique.append(row)
    records = unique
    names_file = PROJECT / ".venv" / "Lib" / "site-packages" / "ultralytics" / "cfg" / "datasets" / "coco.yaml"
    names = yaml.safe_load(names_file.read_text(encoding="utf-8"))["names"]
    if len(names) != 80 or any(names[c] != name for c, name in FOCUS.items()):
        raise ValueError("Installed COCO names do not match the original 80-class schema")
    stats = {}
    for split in ("train", "val"):
        subset = [r for r in records if r["split"] == split]
        counts = Counter()
        image_counts = Counter()
        for row in subset:
            labels, _ = read_labels(Path(row["labels"]).read_text(encoding="utf-8"))
            counts.update(r[0] for r in labels)
            image_counts.update(set(r[0] for r in labels))
        stats[split] = {"images": len(subset), "boxes": sum(counts.values()), "classes_present": len(counts),
                        "focus_instances": {name: counts[c] for c, name in FOCUS.items()},
                        "focus_images": {name: image_counts[c] for c, name in FOCUS.items()}}
        (root / f"{split}.txt").write_text("".join(Path(r["image"]).as_posix()+"\n" for r in subset), encoding="utf-8")
    ready = not failures and all(stats[s]["focus_images"][name] >= quota for s, quota in (("train", args.train_per_focus), ("val", args.val_per_focus)) for name in FOCUS.values())
    manifest = {"created_at": datetime.now().isoformat(), "status": "ready_public_pilot" if ready else "incomplete_do_not_train",
                "source": "COCO 2017 boxes redistributed by Ultralytics", "source_url": LABEL_URL,
                "source_documentation": "https://docs.ultralytics.com/datasets/detect/coco/", "class_id_scheme": "original_COCO80",
                "all_boxes_in_selected_images_retained": True, "stats": stats, "images": records, "failures": failures,
                "duplicates_excluded": duplicates, "network_bytes_this_run": budget.used, "network_budget_remaining_at_start": budget.limit,
                "total_download_cache_cap_bytes": total_cap, "existing_download_bytes_at_start": cached_bytes,
                "annotation_archive": archive_meta, "API_calls": 0, "API_tokens": 0, "training_started": False,
                "evaluation_note": "Disjoint official train2017/val2017 and exact pixel hashes. COCO was used in pretraining; this is development validation, not independent application acceptance. Small category coverage is not a guarantee against forgetting.",
                "private_photo_policy": "Personal and synthetic auto-label candidates kept separately; do not silently mix unverified partial labels into public ground truth."}
    write_json(root / "manifest.json", manifest)
    if ready:
        config = {"path": root.as_posix(), "train": "train.txt", "val": "val.txt", "names": names}
        (root / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    elif (root / "data.yaml").exists():
        raise RuntimeError("Dataset became incomplete; inspect manifest before using previously generated data.yaml")
    print(json.dumps({"status": manifest["status"], "stats": stats, "failures": failures, "manifest": str(root / "manifest.json"), "network_mib": round(budget.used/MIB, 2)}, ensure_ascii=False, indent=2), flush=True)
    if not ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
