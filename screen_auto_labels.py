"""Check pseudo-label consistency. Passing is not proof of semantic correctness."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import inspect
import json
import math
from pathlib import Path
import time

import auto_label_trial as trial


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0


def compare_views(original, flipped, minimum_iou=0.7):
    reasons, pairs = [], []
    for view_name, detections in (("original", original), ("hflip", flipped)):
        for item in detections:
            if item["name"] is None:
                reasons.append(f"{view_name}: ambiguous/unmapped phrase: {item['phrase']}")
    left = [item for item in original if item["coco_class_id"] is not None]
    right = [item for item in flipped if item["coco_class_id"] is not None]
    if not left or not right:
        reasons.append("No mapped positive objects in at least one view; no background label exported.")
    unused = set(range(len(right)))
    for item in sorted(left, key=lambda item: item["confidence"], reverse=True):
        index = max(unused, key=lambda j: box_iou(item["box_xyxy"], right[j]["box_xyxy"]), default=None)
        overlap = box_iou(item["box_xyxy"], right[index]["box_xyxy"]) if index is not None else 0
        if index is None or overlap < minimum_iou:
            reasons.append(f"Unstable/missing box: {item['name']}; best IoU={overlap:.3f}")
            continue
        unused.remove(index)
        other = right[index]
        if item["coco_class_id"] != other["coco_class_id"]:
            reasons.append(f"Class conflict: {item['name']} vs {other['name']}; IoU={overlap:.3f}")
        else:
            pairs.append({"name": item["name"], "iou": overlap,
                          "minimum_score": min(item["confidence"], other["confidence"])})
    for index in sorted(unused):
        reasons.append(f"Additional hflip object: {right[index]['name']}")
    # Quarantine the whole image: silently dropping a disputed positive would
    # turn that object into an incorrect background example during training.
    return not reasons, reasons, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", choices=("test_images", "training_photos"), default="test_images")
    parser.add_argument("--files", nargs="+")
    args = parser.parse_args()
    source = trial.PROJECT_DIR / args.source_dir
    if args.files:
        paths = [source / name for name in args.files]
    elif args.source_dir == "test_images":
        paths = [source / name for name in ("02_simple.jpg", "07_occlusion.jpg", "16.jpg")]
    else:
        paths = sorted(path for path in source.iterdir() if path.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not paths or any(not path.is_file() for path in paths):
        parser.error("Input images are missing.")

    import torch
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    from torchvision.ops import nms
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    model_id = "IDEA-Research/grounding-dino-tiny"
    print("Loading cached model locally; no new download.", flush=True)
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, local_files_only=True).to("cpu").eval()
    threshold_key = "threshold" if "threshold" in inspect.signature(processor.post_process_grounded_object_detection).parameters else "box_threshold"
    text = ". ".join(trial.PROMPTS) + "."
    run = trial.PROJECT_DIR / "outputs" / "auto_label_screen" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for name in ("previews", "consistent_candidates_coco80", "quarantine"):
        (run / name).mkdir(parents=True, exist_ok=False)
    report = {"model": model_id, "model_revision": getattr(model.config, "_commit_hash", None),
              "source_directory": str(source), "prompts": list(trial.PROMPTS),
              "views": ["original", "hflip"], "box_threshold": 0.35, "text_threshold": 0.25,
              "nms_iou": 0.7, "match_iou": 0.7, "labels_verified": False,
              "training_started": False, "note": "Same-model agreement can retain consistent mistakes. All outputs remain candidates; an entire inconsistent image is quarantined.",
              "images": []}
    print(f"Output: {run}", flush=True)

    def detect(image, mirror):
        view = ImageOps.mirror(image) if mirror else image
        inputs = processor(images=view, text=text, return_tensors="pt")
        with torch.inference_mode():
            outputs = model(**inputs)
        found = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, **{threshold_key: 0.35}, text_threshold=0.25,
            target_sizes=[(view.height, view.width)])[0]
        phrases = found["text_labels"] if "text_labels" in found else found["labels"]
        detections = []
        for raw_box, score, phrase in zip(found["boxes"], found["scores"], phrases):
            raw_box = raw_box.tolist()
            if not all(math.isfinite(v) for v in raw_box) or not math.isfinite(float(score)):
                raise ValueError("Non-finite model output")
            x1, y1, x2, y2 = [max(0.0, min(v, limit)) for v, limit in zip(raw_box, (image.width, image.height, image.width, image.height))]
            if x2 <= x1 or y2 <= y1:
                raise ValueError("Degenerate model box")
            if mirror:
                x1, x2 = image.width-x2, image.width-x1
            name, cid = trial.LABELS.get(trial.normalize_label(phrase), (None, None))
            detections.append({"name": name, "coco_class_id": cid, "phrase": str(phrase),
                               "confidence": float(score), "box_xyxy": [x1, y1, x2, y2]})
        if not detections:
            return []
        indices = nms(torch.tensor([item["box_xyxy"] for item in detections]),
                      torch.tensor([item["confidence"] for item in detections]), 0.7).tolist()
        return [detections[index] for index in indices]

    for path in paths:
        started = time.perf_counter()
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        original, flipped = detect(image, False), detect(image, True)
        passed, reasons, matches = compare_views(original, flipped)
        mapped = [item for item in original if item["coco_class_id"] is not None]
        record = {"image": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "width": image.width, "height": image.height,
                  "automatic_consistency_pass": passed, "reasons": reasons, "matches": matches,
                  "original": original, "hflip_mapped_back": flipped,
                  "original_counts": dict(Counter(item["name"] for item in mapped)),
                  "seconds": time.perf_counter()-started}
        if passed:
            rows = []
            for item in mapped:
                x1, y1, x2, y2 = item["box_xyxy"]
                xywh = ((x1+x2)/(2*image.width), (y1+y2)/(2*image.height), (x2-x1)/image.width, (y2-y1)/image.height)
                rows.append(str(item["coco_class_id"])+" "+" ".join(f"{v:.6f}" for v in xywh))
            (run / "consistent_candidates_coco80" / f"{path.stem}.txt").write_text("\n".join(rows)+"\n", encoding="utf-8")
        else:
            (run / "quarantine" / f"{path.stem}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        preview = image.copy()
        preview.thumbnail((1200, 1200))
        draw = ImageDraw.Draw(preview)
        font_path = Path("C:/Windows/Fonts/arial.ttf")
        font = ImageFont.truetype(str(font_path), 20) if font_path.exists() else ImageFont.load_default()
        color = "green" if passed else "red"
        for item in original:
            coords = [item["box_xyxy"][i]*(preview.width/image.width if i%2 == 0 else preview.height/image.height) for i in range(4)]
            draw.rectangle(coords, outline=color, width=3)
            draw.text((coords[0], max(0, coords[1]-23)), f"{item['name'] or item['phrase']} {item['confidence']:.3f}", font=font, fill=color, stroke_width=1, stroke_fill="white")
        preview.save(run / "previews" / f"{path.stem}.jpg")
        report["images"].append(record)
        report["summary"] = {"checked": len(report["images"]),
                             "consistency_pass": sum(item["automatic_consistency_pass"] for item in report["images"])}
        (run / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{path.name}: {'CONSISTENT CANDIDATE' if passed else 'QUARANTINED'}; {record['original_counts']}", flush=True)
        for reason in reasons:
            print(f"  {reason}", flush=True)
    print(f"Finished: {run}; summary={report['summary']}; no training started.", flush=True)


if __name__ == "__main__":
    main()
