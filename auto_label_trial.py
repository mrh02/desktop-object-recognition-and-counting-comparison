"""Local automatic-label diagnostic; generated labels are unverified candidates."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import time

PROJECT_DIR = Path(__file__).resolve().parent
DEPS_DIR = PROJECT_DIR / "outputs" / "auto_label_deps"
if DEPS_DIR.is_dir():
    sys.path.insert(0, str(DEPS_DIR))

# Keep downloads local to this experiment; do not load the app's API secrets.
os.environ["HF_HOME"] = str(PROJECT_DIR / "outputs" / "auto_label_models")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "15"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROMPTS = (
    "a paper cup", "a bowl", "a toilet", "a computer mouse", "scissors",
    "a book", "a remote control", "a mobile phone", "a wine glass",
    "a laptop", "a dining table", "a usb cable", "a spoon", "a fork",
)
# Preserve original COCO-80 IDs. A cable has no COCO-80 class.
LABELS = {
    "paper cup": ("cup", 41), "cup": ("cup", 41),
    "bowl": ("bowl", 45), "toilet": ("toilet", 61),
    "computer mouse": ("mouse", 64), "mouse": ("mouse", 64),
    "scissors": ("scissors", 76), "book": ("book", 73),
    "remote control": ("remote", 65), "remote": ("remote", 65),
    "mobile phone": ("cell phone", 67), "cell phone": ("cell phone", 67),
    "wine glass": ("wine glass", 40), "laptop": ("laptop", 63),
    "dining table": ("dining table", 60), "usb cable": ("cable", None),
    "spoon": ("spoon", 44), "fork": ("fork", 42),
    "keyboard": ("keyboard", 66), "potted plant": ("potted plant", 58),
    "clock": ("clock", 74), "bottle": ("bottle", 39),
    "headphones": ("headphones", None), "tablet": ("tablet", None),
    "saucer": ("saucer", None),
}
PROMPT_SETS = {
    "original": PROMPTS,
    "desktop_v2": tuple("a cup" if phrase == "a paper cup" else phrase for phrase in PROMPTS) + (
        "a keyboard", "a potted plant", "a clock", "a bottle", "headphones", "a tablet", "a saucer",
    ),
}


def normalize_label(text):
    text = re.sub(r"\s+", " ", str(text).lower()).strip(" .")
    text = re.sub(r"^(a|an|the) ", "", text)
    return re.sub(r" (a|an|the)$", "", text)


def build_prompt_spans(tokenizer, text, prompts):
    encoded = tokenizer(text, return_offsets_mapping=True)
    spans, cursor = [], 0
    for prompt in prompts:
        content = re.sub(r"^(a|an|the) ", "", prompt)
        start = cursor + len(prompt) - len(content)
        stop = cursor + len(prompt)
        indices = [i for i, (left, right) in enumerate(encoded["offset_mapping"])
                   if right > left and left >= start and right <= stop]
        if not indices:
            raise ValueError(f"No content tokens for prompt: {prompt}")
        name, cid = LABELS.get(normalize_label(prompt), (None, None))
        spans.append({"prompt": prompt, "name": name, "coco_class_id": cid, "token_indices": indices})
        cursor += len(prompt) + 2
    return encoded["input_ids"], spans


def token_class_scores(probabilities, spans):
    scores = []
    for span in spans:
        values = [probabilities[i] for i in span["token_indices"]]
        scores.append({key: span[key] for key in ("prompt", "name", "coco_class_id")} | {
            "mean_token_score": sum(values)/len(values), "max_token_score": max(values),
        })
    return sorted(scores, key=lambda item: item["mean_token_score"], reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", choices=("test_images", "training_photos"), default="test_images")
    parser.add_argument("--files", nargs="+")
    parser.add_argument("--prompt-set", choices=tuple(PROMPT_SETS), default="original")
    parser.add_argument("--label-decoder", choices=("token_span", "phrase"), default="token_span")
    parser.add_argument("--minimum-class-score", type=float, default=0.35)
    parser.add_argument("--minimum-class-margin", type=float, default=0.10)
    args = parser.parse_args()
    if not 0 <= args.minimum_class_score <= 1 or not 0 <= args.minimum_class_margin <= 1:
        parser.error("Class thresholds must be in [0,1]")
    source_dir = PROJECT_DIR / args.source_dir
    if args.files:
        image_paths = [source_dir / name for name in args.files]
    elif args.source_dir == "test_images":
        image_paths = [source_dir / name for name in ("02_simple.jpg", "07_occlusion.jpg")]
    else:
        image_paths = sorted(path for path in source_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not image_paths:
        parser.error(f"No images found: {source_dir}")
    for path in image_paths:
        if not path.is_file():
            parser.error(f"Image does not exist: {path}")
        if not path.resolve().is_relative_to(source_dir.resolve()):
            parser.error(f"Image must stay inside the selected source directory: {path}")
    stems = [path.stem.casefold() for path in image_paths]
    if len(stems) != len(set(stems)):
        parser.error("Image stems collide; refuse to overwrite candidate label files")
    prompts = PROMPT_SETS[args.prompt_set]

    import torch
    from torchvision.ops import nms
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from refine_label_candidates import resolve_class

    run_dir = PROJECT_DIR / "outputs" / "auto_label_trial" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    preview_dir = run_dir / "previews"
    label_dir = run_dir / "candidate_labels_coco80"
    preview_dir.mkdir(parents=True, exist_ok=False)
    label_dir.mkdir()
    print(f"Output: {run_dir}", flush=True)
    print("Loading cached Grounding DINO tiny on CPU; no model download.", flush=True)
    start = time.perf_counter()
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, local_files_only=True).to("cpu").eval()
    load_seconds = time.perf_counter() - start
    print(f"Model loaded in {load_seconds:.2f}s", flush=True)

    report = {
        "model": model_id, "device": "cpu", "prompts": list(prompts), "prompt_set": args.prompt_set,
        "box_threshold": 0.35, "text_threshold": 0.25,
        "class_agnostic_nms_iou": 0.7, "load_seconds": load_seconds,
        "label_extraction": args.label_decoder,
        "minimum_class_score": args.minimum_class_score,
        "minimum_class_margin": args.minimum_class_margin,
        "labels_verified": False,
        "purpose": "diagnostic_only_not_training_data" if args.source_dir == "test_images" else "training_candidates_not_accepted_dataset",
        "source_directory": str(source_dir),
        "class_id_scheme": "original_COCO80_not_three_class",
        "prompt_scope": "trial_vocabulary_not_all_possible_objects",
        "note": "Geometry checks and NMS do not verify semantic correctness. Token-span extraction rejects low scores or close competing classes. All exported labels are unverified candidates; no training is started.",
        "score_note": "Per-prompt token similarities are not calibrated probabilities. Token-span mode groups synonyms and requires a distinct-class margin; phrase mode reproduces the earlier decoder.",
        "images": [],
    }
    text = ". ".join(prompts) + "."
    expected_ids, prompt_spans = build_prompt_spans(processor.tokenizer, text, prompts)
    if len(expected_ids) > model.config.max_text_len:
        raise ValueError("Prompt vocabulary exceeds the local model token limit")
    report["prompt_token_count"] = len(expected_ids)
    method = processor.post_process_grounded_object_detection
    threshold_key = "threshold" if "threshold" in inspect.signature(method).parameters else "box_threshold"

    for path in image_paths:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        started = time.perf_counter()
        inputs = processor(images=image, text=text, return_tensors="pt")
        if inputs.input_ids[0].tolist() != expected_ids:
            raise ValueError("Processor text tokens differ from prompt offsets")
        with torch.inference_mode():
            outputs = model(**inputs)
        found = method(
            outputs, inputs.input_ids, **{threshold_key: 0.35},
            text_threshold=0.25, target_sizes=[(image.height, image.width)],
        )[0]
        phrases = found["text_labels"] if "text_labels" in found else found["labels"]
        probabilities = outputs.logits[0].sigmoid()
        query_indices = (probabilities.max(dim=-1).values > 0.35).nonzero(as_tuple=True)[0].tolist()
        if len(query_indices) != len(found["boxes"]):
            raise ValueError("Raw query order does not match postprocessed boxes")
        raw, valid_boxes, valid_scores, valid_indices = [], [], [], []
        for query_index, box, score, phrase in zip(query_indices, found["boxes"], found["scores"], phrases):
            x1, y1, x2, y2 = box.tolist()
            coords = [max(0.0, min(float(v), limit)) for v, limit in zip(
                (x1, y1, x2, y2), (image.width, image.height, image.width, image.height))]
            name, class_id = LABELS.get(normalize_label(phrase), (None, None))
            record = {"phrase": str(phrase), "name": name, "coco_class_id": class_id,
                      "confidence": float(score), "box_xyxy": coords, "kept_after_nms": False,
                      "query_index": query_index,
                      "prompt_class_scores": token_class_scores(probabilities[query_index].tolist(), prompt_spans)}
            if args.label_decoder == "token_span":
                resolved = resolve_class(record["prompt_class_scores"], args.minimum_class_score, args.minimum_class_margin)
                record.update({"phrase_name": name, "phrase_class_id": class_id,
                               "name": resolved["name"], "coco_class_id": resolved["coco_class_id"],
                               "detector_confidence": float(score), "confidence": resolved["score"],
                               "class_resolution": resolved})
            raw.append(record)
            if coords[2] > coords[0] and coords[3] > coords[1] and all(math.isfinite(v) for v in (*box.tolist(), float(score))):
                valid_indices.append(len(raw) - 1)
                valid_boxes.append(coords)
                valid_scores.append(float(score))
        if valid_boxes:
            indices = nms(torch.tensor(valid_boxes), torch.tensor(valid_scores), 0.7).tolist()
            for index in indices:
                raw[valid_indices[index]]["kept_after_nms"] = True
        kept = [item for item in raw if item["kept_after_nms"]]
        counts = dict(Counter(item["name"] or f"unmapped:{item['phrase']}" for item in kept))
        yolo_lines = []
        for item in kept:
            if item["coco_class_id"] is None:
                continue
            x1, y1, x2, y2 = item["box_xyxy"]
            xywh = ((x1+x2)/(2*image.width), (y1+y2)/(2*image.height),
                    (x2-x1)/image.width, (y2-y1)/image.height)
            yolo_lines.append(str(item["coco_class_id"]) + " " + " ".join(f"{v:.6f}" for v in xywh))
        (label_dir / f"{path.stem}.txt").write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

        # Preview only: labels above use original, EXIF-corrected image coordinates.
        preview = image.copy()
        preview.thumbnail((1400, 1400))
        draw = ImageDraw.Draw(preview)
        font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf"
        font = ImageFont.truetype(str(font_path), 22) if font_path.is_file() else ImageFont.load_default()
        for item in kept:
            coords = [item["box_xyxy"][i] * (preview.width/image.width if i % 2 == 0 else preview.height/image.height) for i in range(4)]
            color = "#00b050" if item["coco_class_id"] is not None else "#e06010"
            draw.rectangle(coords, outline=color, width=3)
            label = f"{item['name'] or item['phrase']} {item['confidence']:.3f}"
            origin = (coords[0], max(0, coords[1] - 26))
            draw.rectangle(draw.textbbox(origin, label, font=font), fill="white")
            draw.text(origin, label, font=font, fill=color)
        preview.save(preview_dir / f"{path.stem}.jpg")
        report["images"].append({
            "image": str(path), "width": image.width, "height": image.height,
            "counts": counts, "candidate_label_count": len(yolo_lines),
            "processing_seconds": time.perf_counter() - started,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "labels_verified": False,
            "unresolved_foreground_candidates": sum(item["coco_class_id"] is None and item["name"] != "cable" for item in kept),
            "empty_is_confirmed_background": False,
            "detections": raw,
        })
        (run_dir / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n{path.name}: {counts}; exported {len(yolo_lines)} UNVERIFIED candidate boxes", flush=True)
        for item in kept:
            print(f"  {item['phrase']}: {item['confidence']:.3f}, {item['box_xyxy']}", flush=True)
    print(f"\nFinished. Review previews and results: {run_dir}", flush=True)
    print("No original files changed; no training started; candidate labels still require independent review.", flush=True)


if __name__ == "__main__":
    main()
