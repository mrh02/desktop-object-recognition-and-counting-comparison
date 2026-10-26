"""Screen private/AI candidates locally and assemble an experimental training mix.

Uses cached Grounding DINO/cloud verdicts and local YOLO11s only. No API calls.
Public validation always excludes private pseudo-labels and exact image duplicates.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps
import yaml
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iou(a, b):
    inter = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union > 0 else 0.


def match_all(left, right):
    used_l, used_r, pairs = set(), set(), []
    candidates = sorted(((iou(a["box"], b["box"]), i, j) for i, a in enumerate(left) for j, b in enumerate(right)), reverse=True)
    for overlap, i, j in candidates:
        if overlap < 0.5:
            break
        if i in used_l or j in used_r:
            continue
        used_l.add(i)
        used_r.add(j)
        pairs.append({"left": i, "right": j, "iou": overlap, "class_agrees": left[i]["cls"] == right[j]["cls"]})
    return len(pairs) == len(left) == len(right) and all(p["class_agrees"] for p in pairs), pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-report', type=Path, required=True,
                        help='Explicit bidirectional v2 cached review; old one-way reviews are not accepted')
    args = parser.parse_args()
    review_path = args.review_report.resolve()
    review = json.loads(review_path.read_text(encoding='utf-8'))
    if review.get('review_policy') != 'bidirectional_geometric_and_class_agreement_v2':
        raise ValueError('Old one-way review cannot be reused: rerun assemble_training_candidates.py first')
    public = PROJECT / "outputs/public_data/coco_desktop_pilot_v1"
    manifest = json.loads((public / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "ready_public_pilot":
        raise ValueError("Public dataset is not ready")
    refined_path = PROJECT / "outputs/refined_auto_labels/20261023_120616_182604/results.json"
    refined = json.loads(refined_path.read_text(encoding="utf-8"))
    sources = json.loads((PROJECT / "photo_source_notes.json").read_text(encoding="utf-8"))
    run = PROJECT / "outputs/training_datasets" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    (run / "images/train").mkdir(parents=True)
    (run / "labels/train").mkdir(parents=True)
    records = []
    original_hashes = {str(p): digest(p) for folder in ("training_photos", "test_images") for p in (PROJECT/folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    known_pixels = {r["pixel_sha256"] for r in manifest["images"]}
    test_pixels = set()
    for p in (PROJECT / "test_images").iterdir():
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        with Image.open(p) as raw:
            im = ImageOps.exif_transpose(raw).convert("RGB")
            test_pixels.add(hashlib.sha256(f"{im.width}x{im.height}:RGB:".encode()+im.tobytes()).hexdigest())
    model = YOLO(str(PROJECT / "yolo11s.pt"))
    synthetic = [r for r in refined["images"] if sources.get(r["source_sha256"], {}).get("training_use_allowed_by_user")]

    def export(source, expected_sha, label_text, source_type, provenance):
        if source.parent.resolve() != (PROJECT / "training_photos").resolve() or digest(source) != expected_sha:
            raise ValueError("Candidate source moved or changed")
        with Image.open(source) as raw:
            im = ImageOps.exif_transpose(raw).convert("RGB")
        pixel_hash = hashlib.sha256(f"{im.width}x{im.height}:RGB:".encode()+im.tobytes()).hexdigest()
        if pixel_hash in known_pixels or pixel_hash in test_pixels:
            return {"image": str(source), "status": "excluded_duplicate", "source_type": source_type}
        known_pixels.add(pixel_hash)
        stem = "private_" + expected_sha[:16]
        image = run / "images/train" / (stem+".png")
        label = run / "labels/train" / (stem+".txt")
        im.save(image)
        label.write_text(label_text, encoding="utf-8")
        return {"image": str(source), "source_sha256": expected_sha, "pixel_sha256": pixel_hash, "source_type": source_type,
                "status": "provisional_training_only", "labels_verified": False, "output_image": str(image),
                "output_label": str(label), "box_count": len(label_text.splitlines()), "provenance": provenance}

    for row in review["images"]:
        if row["status"] != "cross_model_agreement_only":
            continue
        source = Path(row["image"])
        label = review_path.parent / "consistent_candidates_coco80" / (source.stem+".txt")
        records.append(export(source, row["source_sha256"], label.read_text(encoding="utf-8"), "personal_cached_cross_model_agreement", str(review_path)))
    for row in synthetic:
        source = Path(row["image"])
        if digest(source) != row["source_sha256"]:
            raise ValueError("Synthetic source changed since cached local labeling")
        with Image.open(source) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        if list(image.size) != [row["width"], row["height"]]:
            raise ValueError("Cached box coordinate frame changed")
        proposals = [d for d in row["detections"] if d["kept_after_nms"] and d["coco_class_id"] is not None]
        left = [{"cls": d["coco_class_id"], "box": d["box_xyxy"], "score": d["confidence"]} for d in proposals]
        unknown = [d for d in row["detections"] if d["kept_after_nms"] and d["name"] is None]
        result = model.predict(image, device="cpu", imgsz=1280, conf=0.25, iou=0.7, agnostic_nms=True, verbose=False)[0]
        right = [{"cls": int(b.cls.item()), "box": b.xyxy[0].tolist(), "score": float(b.conf.item())} for b in result.boxes]
        agrees, pairs = match_all(left, right)
        reasons = []
        if unknown:
            reasons.append("unresolved_local_foreground")
        if not left:
            reasons.append("empty_not_confirmed_background")
        if not agrees:
            reasons.append("local_teacher_boxes_or_classes_disagree")
        details = {"image": str(source), "source_sha256": row["source_sha256"], "source_type": "synthetic_AI_user_allowed",
                   "status": "quarantined" if reasons else "provisional_training_only", "reasons": reasons,
                   "dino": left, "yolo11s": right, "matches": pairs, "labels_verified": False}
        if not reasons:
            w, h = image.size
            lines = []
            for d in left:
                x1, y1, x2, y2 = d["box"]
                lines.append(f"{d['cls']} {(x1+x2)/(2*w):.8f} {(y1+y2)/(2*h):.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}\n")
            details.update(export(source, row["source_sha256"], "".join(lines), "synthetic_AI_user_allowed", str(refined_path)))
        records.append(details)
        print(f"AI {source.name}: {details['status']}; {reasons}", flush=True)
    included = [r for r in records if r["status"] == "provisional_training_only"]
    names = yaml.safe_load((public / "data.yaml").read_text(encoding="utf-8"))["names"]
    train_images = [r["image"] for r in manifest["images"] if r["split"] == "train"] + [r["output_image"] for r in included]
    val_images = [r["image"] for r in manifest["images"] if r["split"] == "val"]
    (run / "train.txt").write_text("".join(Path(p).as_posix()+"\n" for p in train_images), encoding="utf-8")
    (run / "val.txt").write_text("".join(Path(p).as_posix()+"\n" for p in val_images), encoding="utf-8")
    (run / "data.yaml").write_text(yaml.safe_dump({"path": run.as_posix(), "train": "train.txt", "val": "val.txt", "names": names}, sort_keys=False), encoding="utf-8")
    if any(digest(Path(path)) != value for path, value in original_hashes.items()):
        raise ValueError("An original photo changed")
    report = {"public_manifest": str(public / "manifest.json"), "public_manifest_sha256": digest(public / "manifest.json"),
              "private_review": str(review_path), "private_review_sha256": digest(review_path),
              "review_policy": review['review_policy'],
              "data_yaml": str(run / "data.yaml"), "public_only_data_yaml": str(public / "data.yaml"),
              "status": "ready_experimental_mix_with_weak_labels", "counts": {"train_images": len(train_images), "val_images": len(val_images),
              "private_weak_images": len(included), "AI_weak_images": sum(r['source_type']=='synthetic_AI_user_allowed' for r in included)},
              "private_records": records, "original_photo_hashes": original_hashes, "originals_unchanged": True,
              "new_API_calls": 0, "new_API_tokens": 0, "training_started": False,
              "note": "Cross-model agreement is a screening heuristic, not ground truth or annotation completeness. Private/AI labels are weak experimental training data only. Validation is public-only, disjoint from fine-tuning but not independent of COCO pretraining. Known development photos are not acceptance data. AI source alone never causes exclusion."}
    (run / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(run / "manifest.json"), "counts": report["counts"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
