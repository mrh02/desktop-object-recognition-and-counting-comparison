"""Offline class-span decoding with ambiguity rejection; never calls an API."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT = Path(__file__).resolve().parent


def resolve_class(scores, minimum_score=0.35, minimum_margin=0.10):
    # Group synonyms by the same COCO ID before measuring ambiguity. Distinct
    # non-COCO concepts remain separate. No filename or expected answer is used.
    grouped = {}
    for item in scores:
        score = item["max_token_score"]
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid class token score")
        cid, name = item["coco_class_id"], item["name"]
        if cid is not None and (type(cid) is not int or not 0 <= cid < 80):
            raise ValueError("Invalid original COCO-80 ID")
        key = ("coco", cid) if cid is not None else ("other", name or item["prompt"])
        if key not in grouped or score > grouped[key]["score"]:
            grouped[key] = {"name": name, "coco_class_id": cid, "score": float(score), "prompt": item["prompt"]}
    ranked = sorted(grouped.values(), key=lambda item: item["score"], reverse=True)
    if not ranked:
        return {"status": "no_class_scores", "name": None, "coco_class_id": None, "score": 0.0, "margin": 0.0}
    first = ranked[0]
    second = ranked[1]["score"] if len(ranked) > 1 else 0.0
    margin = first["score"] - second
    common = {"score": first["score"], "margin": margin, "ranked_alternatives": ranked[:4]}
    if first["score"] < minimum_score:
        return {**common, "status": "low_score", "name": None, "coco_class_id": None}
    if margin + 1e-9 < minimum_margin:
        return {**common, "status": "ambiguous_classes", "name": None, "coco_class_id": None}
    if first["coco_class_id"] is None:
        return {**common, "status": "ignored_cable" if first["name"] == "cable" else "outside_COCO80",
                "name": first["name"], "coco_class_id": None}
    return {**common, "status": "single_class_candidate", "name": first["name"], "coco_class_id": first["coco_class_id"]}


def refine_row(row, minimum_score, minimum_margin):
    revised = deepcopy(row)
    changed, rejected = 0, 0
    for item in revised["detections"]:
        if "prompt_class_scores" not in item:
            raise ValueError("Input lacks per-prompt scores; do not infer a class from a mixed phrase")
        old_name = item.get("phrase_name", item.get("name"))
        old_id = item.get("phrase_class_id", item.get("coco_class_id"))
        resolved = resolve_class(item["prompt_class_scores"], minimum_score, minimum_margin)
        item.update({"phrase_name": old_name, "phrase_class_id": old_id,
                     "name": resolved["name"], "coco_class_id": resolved["coco_class_id"],
                     "class_resolution": resolved, "detector_confidence": item.get("detector_confidence", item["confidence"]),
                     "confidence": resolved["score"]})
        if item["kept_after_nms"]:
            changed += (old_name, old_id) != (item["name"], item["coco_class_id"])
            rejected += resolved["status"] in {"ambiguous_classes", "low_score", "no_class_scores"}
    kept = [item for item in revised["detections"] if item["kept_after_nms"]]
    revised["counts"] = dict(Counter(item["name"] or "unresolved" for item in kept))
    revised["candidate_label_count"] = sum(item["coco_class_id"] is not None for item in kept)
    revised["unresolved_foreground_candidates"] = sum(item["class_resolution"]["status"] not in {"single_class_candidate", "ignored_cable"} for item in kept)
    revised["labels_verified"] = False
    revised["empty_is_confirmed_background"] = False
    revised["class_changes"] = int(changed)
    revised["ambiguity_or_low_score_rejections"] = int(rejected)
    return revised


def write_labels(row, path):
    lines = []
    width, height = row["width"], row["height"]
    for item in row["detections"]:
        if not item["kept_after_nms"] or item["coco_class_id"] is None:
            continue
        x1, y1, x2, y2 = item["box_xyxy"]
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Invalid box geometry")
        lines.append(f"{item['coco_class_id']} {(x1+x2)/(2*width):.6f} {(y1+y2)/(2*height):.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_preview(row, path):
    with Image.open(row["image"]) as opened:
        preview = ImageOps.exif_transpose(opened).convert("RGB")
    preview.thumbnail((1200, 1200))
    draw = ImageDraw.Draw(preview)
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/arial.ttf"
    font = ImageFont.truetype(str(font_path), 18) if font_path.is_file() else ImageFont.load_default()
    for item in row["detections"]:
        if not item["kept_after_nms"]:
            continue
        box = [v*(preview.width/row["width"] if i % 2 == 0 else preview.height/row["height"]) for i, v in enumerate(item["box_xyxy"])]
        resolved = item["class_resolution"]
        color = "#008000" if item["coco_class_id"] is not None else "#b85c00"
        title = f"{item['name'] or resolved['status']} s={resolved['score']:.2f} gap={resolved['margin']:.2f}"
        draw.rectangle(box, outline=color, width=2)
        origin = (box[0], max(0, box[1]-22))
        draw.rectangle(draw.textbbox(origin, title, font=font), fill="white")
        draw.text(origin, title, fill=color, font=font)
    preview.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--minimum-score", type=float, default=0.35)
    parser.add_argument("--minimum-margin", type=float, default=0.10)
    args = parser.parse_args()
    if not 0 <= args.minimum_score <= 1 or not 0 <= args.minimum_margin <= 1:
        parser.error("Thresholds must be in [0,1]")
    original = json.loads(args.source_report.read_text(encoding="utf-8"))
    # Validate the complete input before producing a partial label directory.
    stems = set()
    for row in original["images"]:
        path = Path(row["image"])
        if path.stem.casefold() in stems:
            raise ValueError("Output stems collide")
        stems.add(path.stem.casefold())
        if not row.get("source_sha256") or hashlib.sha256(path.read_bytes()).hexdigest() != row["source_sha256"]:
            raise ValueError("Missing or stale source-image hash")
        for item in row["detections"]:
            if "prompt_class_scores" not in item:
                raise ValueError("Per-class scores required")
    run = PROJECT / "outputs/refined_auto_labels" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    (run / "candidate_labels_coco80").mkdir(parents=True)
    (run / "previews").mkdir()
    report = {key: value for key, value in original.items() if key != "images"}
    report.update({"source_report": str(args.source_report.resolve()),
                   "source_report_sha256": hashlib.sha256(args.source_report.read_bytes()).hexdigest(),
                   "label_extraction": "class_token_span_max_with_distinct_class_margin",
                   "minimum_class_score": args.minimum_score, "minimum_class_margin": args.minimum_margin,
                   "score_note": "Maximum content-token similarity and the gap to a different canonical class guide provisional decoding. These scores are not calibrated probabilities or independent verification.",
                   "labels_verified": False, "training_started": False, "new_api_calls": 0,
                   "note": "Class-span decoding replaces stitched text labels. Ambiguous classes remain unresolved. Similarity is not calibrated probability; this does not recover missing boxes or prove class correctness/completeness. No trainable dataset is accepted automatically.",
                   "images": []})
    for row in original["images"]:
        revised = refine_row(row, args.minimum_score, args.minimum_margin)
        name = Path(row["image"]).stem
        write_labels(revised, run / "candidate_labels_coco80" / f"{name}.txt")
        write_preview(revised, run / "previews" / f"{name}.jpg")
        report["images"].append(revised)
        print(f"{name}: {revised['counts']}; changed={revised['class_changes']}; unresolved={revised['unresolved_foreground_candidates']}", flush=True)
    (run / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Refined report: {run / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
