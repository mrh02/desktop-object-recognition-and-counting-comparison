"""Read-only photo inventory; exact-pixel duplicate selection, no model or API."""
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import struct

from PIL import Image, ImageOps

PROJECT = Path(__file__).resolve().parent
SUPPORTED = {".jpg", ".jpeg", ".png"}


def main():
    run = PROJECT / "outputs/photo_import" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run.mkdir(parents=True)
    rows, duplicates = [], defaultdict(list)
    observation_path = PROJECT / "photo_source_notes.json"
    observations = json.loads(observation_path.read_text(encoding="utf-8")) if observation_path.exists() else {}
    for folder in ("training_photos", "test_images", "acceptance_photos"):
        source = PROJECT / folder
        if not source.exists():
            continue
        for path in sorted(p for p in source.rglob("*") if p.is_file()):
            row = {"image": str(path), "folder": folder, "relative_path": str(path.relative_to(source)),
                   "source_type": "unconfirmed", "status": "unsupported_extension"}
            if path.suffix.lower() not in SUPPORTED:
                rows.append(row)
                continue
            try:
                row["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                with Image.open(path) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                    image.load()
                    row.update({"width": image.width, "height": image.height, "format": opened.format,
                                "pixel_sha256": hashlib.sha256(struct.pack(">II", image.width, image.height) + image.tobytes()).hexdigest()})
                row["status"] = "readable"
                observation = observations.get(row["file_sha256"])
                if observation:
                    row["source_type"] = observation["source_type"]
                    row["source_observation"] = observation["observation"]
                duplicates[row["pixel_sha256"]].append(row)
            except Exception as error:
                row.update({"status": "unreadable", "error_type": type(error).__name__})
            rows.append(row)
    training = [r for r in rows if r["folder"] == "training_photos" and r["status"] == "readable"]
    by_pixels = defaultdict(list)
    for row in training:
        by_pixels[row["pixel_sha256"]].append(row)
    selected = [items[0] for items in by_pixels.values()]
    stems = defaultdict(list)
    for row in selected:
        stems[Path(row["relative_path"]).stem.casefold()].append(row["relative_path"])
    collisions = [items for items in stems.values() if len(items) > 1]
    duplicate_groups = [[r["image"] for r in group] for group in duplicates.values() if len(group) > 1]
    cross_folder = [group for group in duplicate_groups if len({Path(p).parent.name for p in group}) > 1]
    report = {"source_files_modified": False, "new_api_calls": 0, "training_started": False,
              "counts_by_folder": dict(Counter(r["folder"] for r in rows)), "training_readable": len(training),
              "training_unique_pixels": len(selected), "duplicate_groups": duplicate_groups,
              "cross_folder_duplicate_groups": cross_folder, "label_stem_collisions": collisions,
              "unsupported_files": [r["image"] for r in rows if r["status"] == "unsupported_extension"],
              "unreadable_files": [r["image"] for r in rows if r["status"] == "unreadable"],
              "selected_training_files": [r["relative_path"] for r in selected] if not collisions else [],
              "note": "Only exact normalized RGB-pixel duplicates are skipped. Similar or differently cropped images are not automatically removed. AI-marked images remain candidate-training material; no new independent real-photo acceptance set is established by importing them. Previous test images used for tuning remain development images.",
              "files": rows}
    (run / "inventory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, ensure_ascii=False, indent=2), flush=True)
    print(f"Inventory: {run / 'inventory.json'}", flush=True)


if __name__ == "__main__":
    main()
