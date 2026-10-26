"""Run a short, offline YOLO11s fine-tuning experiment; never replace app weights."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

PROJECT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def preflight(data_file):
    import yaml
    from ultralytics.data.utils import check_det_dataset, img2label_paths
    data = check_det_dataset(data_file, autodownload=False)
    if len(data["names"]) != 80 or data.get("download"):
        raise ValueError("Pilot requires a local, original COCO-80 dataset without a download script")
    records, seen_paths, seen_pixels = [], set(), set()
    from PIL import Image
    for split in ("train", "val"):
        paths = [Path(s.strip()).resolve() for s in Path(data[split]).read_text(encoding="utf-8").splitlines() if s.strip()]
        if not paths:
            raise ValueError("Empty dataset split")
        box_count = 0
        counts = Counter()
        for path in paths:
            if not path.is_relative_to(PROJECT / "outputs") or not path.is_file() or path in seen_paths:
                raise ValueError("Invalid or duplicated training/validation path")
            seen_paths.add(path)
            label = Path(img2label_paths([str(path)])[0])
            if not label.is_file():
                raise ValueError(f"Missing label: {label}")
            with Image.open(path) as raw:
                im = raw.convert("RGB")
                pixel = hashlib.sha256(f"{im.width}x{im.height}:RGB:".encode()+im.tobytes()).hexdigest()
            if pixel in seen_pixels:
                raise ValueError("Exact pixel duplicate within or across splits")
            seen_pixels.add(pixel)
            for line in label.read_text(encoding="utf-8").splitlines():
                values = list(map(float, line.split()))
                if len(values) != 5 or not all(math.isfinite(v) for v in values):
                    raise ValueError("Malformed label")
                cls, x, y, w, h = values
                if cls != int(cls) or not 0 <= cls < 80 or not 0 < w <= 1 or not 0 < h <= 1 or min(x-w/2, y-h/2) < -1e-6 or max(x+w/2,y+h/2) > 1+1e-6:
                    raise ValueError(f"Invalid normalized box in {label}")
                counts[int(cls)] += 1
                box_count += 1
            records.append({"split": split, "image": str(path), "image_sha256": sha(path), "label_sha256": sha(label)})
        print(f"Preflight {split}: {len(paths)} images, {box_count} boxes, {len(counts)} classes", flush=True)
    return data, records


def development_predictions(model, out):
    # Reuse existing user-confirmed count records; do not create training labels.
    expected = json.loads((PROJECT / "test_labels.json").read_text(encoding="utf-8"))
    rows = []
    for name, record in expected.items():
        if name == "04_multiple.jpg":
            continue  # Historical power-bank photo has incompatible/stale supported-category counts.
        image = PROJECT / "test_images" / name
        result = model.predict(str(image), device="cpu", imgsz=1280, conf=0.25, iou=0.7, agnostic_nms=True, verbose=False)[0]
        counts = Counter(result.names[int(b.cls.item())] for b in result.boxes)
        objects = [{"class": result.names[int(b.cls.item())], "confidence": float(b.conf.item()), "xyxy": b.xyxy[0].tolist()} for b in result.boxes]
        result.save(filename=str(out / (Path(name).stem+".jpg")))
        rows.append({"image": name, "source_sha256": sha(image), "expected_counts": record["counts"], "counts": dict(counts),
                     "exact_count_match": dict(counts) == record["counts"], "detections": objects})
    write_json(out / "predictions.json", {"purpose": "known_development_samples_not_independent_acceptance", "excluded": ["04_multiple.jpg"], "images": rows})
    return {"exact_count_matches": sum(r["exact_count_match"] for r in rows), "images": len(rows), "rows": rows}


def evaluate(model, data_path, run, name):
    start = time.perf_counter()
    metrics = model.val(data=str(data_path), device="cpu", imgsz=640, batch=4, workers=0, conf=0.001,
                        iou=0.7, agnostic_nms=True, plots=False, verbose=False, project=str(run), name=name, exist_ok=False)
    # Save results before later train()/val() calls mutate internal model state.
    focus = {}
    for cls in (41, 64, 76, 66, 45, 73):
        indices = list(metrics.box.ap_class_index)
        focus[model.names[cls]] = {"mAP50": float(metrics.box.ap50[indices.index(cls)]), "mAP50_95": float(metrics.box.ap[indices.index(cls)])} if cls in indices else None
    report = {"metrics": {str(k): float(v) for k, v in metrics.results_dict.items()}, "focus_metrics": focus,
              "seconds": time.perf_counter()-start, "imgsz": 640, "batch": 4, "device": "cpu",
              "purpose": "COCO_development_validation_already_seen_in_pretraining"}
    write_json(run / (name+".json"), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--preset", choices=("short_neck_head", "conservative_head"), default="short_neck_head")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(f"Prepared only. Add --execute for {args.epochs} offline pilot epochs; app is unchanged.")
        return
    data_path = args.data.resolve()
    if not data_path.is_relative_to(PROJECT / "outputs") or not data_path.is_file():
        raise ValueError("Use a prepared local dataset under project outputs")
    run = PROJECT / "outputs/training_pilot" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run.mkdir(parents=True)
    shutil.copyfile(Path(__file__), run / "runner_source.py")
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_CONFIG_DIR"] = str(run / "runtime")
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    # check_det_dataset may try to fetch a plotting font even with YOLO_OFFLINE.
    # Seed the task-local font cache from a local font instead of making a request.
    font_sources = [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/arial.ttf"]
    font_sources += sorted((PROJECT / "outputs/training_pilot").glob("*/runtime/Ultralytics/Arial.ttf"))
    source_font = next((p for p in font_sources if p.is_file()), None)
    if source_font is None:
        raise ValueError("No local font found for offline validation")
    font_dir = run / "runtime/Ultralytics"
    font_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_font, font_dir / "Arial.ttf")
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS
    SETTINGS.update({"sync": False, "wandb": False, "mlflow": False, "comet": False, "clearml": False, "dvc": False, "raytune": False, "tensorboard": False})
    torch.set_num_threads(8)
    original_paths = [PROJECT / p for p in ("yolo11s.pt", "yolo11n.pt", "app.py")]
    original_hashes = {str(p): sha(p) for p in original_paths}
    data, integrity = preflight(data_path)
    write_json(run / "dataset_integrity.json", integrity)
    run_report = {"status": "running", "data": str(data_path), "data_sha256": sha(data_path), "epochs": args.epochs,
                  "device": "cpu", "ultralytics": ultralytics.__version__, "torch": torch.__version__,
                  "base_weights": str(PROJECT / "yolo11s.pt"), "original_hashes": original_hashes,
                  "preset": args.preset, "runner_sha256": sha(run / "runner_source.py"),
                  "new_API_calls": 0, "new_API_tokens": 0, "app_changed": False}
    write_json(run / "summary.json", run_report)
    print(f"Pilot run: {run}", flush=True)
    baseline = YOLO(str(PROJECT / "yolo11s.pt"))
    baseline_val = evaluate(baseline, data_path, run, "baseline_val")
    dev_before = run / "development_before"
    dev_before.mkdir()
    baseline_dev = development_predictions(baseline, dev_before)
    model = YOLO(str(PROJECT / "yolo11s.pt"))
    started = time.perf_counter()
    logged_epochs = set()
    def epoch_note(trainer):
        epoch = trainer.epoch + 1
        if epoch <= args.epochs and epoch not in logged_epochs:
            logged_epochs.add(epoch)
            print(f"PILOT_EPOCH_COMPLETED {epoch}/{args.epochs}", flush=True)
    model.add_callback("on_fit_epoch_end", epoch_note)
    conservative = args.preset == "conservative_head"
    if conservative:
        def keep_bn_stats(trainer):
            for module in trainer.model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()
        model.add_callback("on_train_batch_start", keep_bn_stats)
    lr = 0.00005 if conservative else 0.0002
    model.train(data=str(data_path), epochs=args.epochs, imgsz=640, batch=4, device="cpu", workers=0,
                project=str(run), name="finetune", exist_ok=False, optimizer="AdamW", lr0=lr,
                lrf=0.1, cos_lr=True, freeze=23 if conservative else 10, warmup_epochs=0.5, warmup_bias_lr=lr,
                nbs=16, mosaic=0.0 if conservative else 0.25, close_mosaic=0 if conservative else 1,
                degrees=10.0 if conservative else 15.0, scale=0.1 if conservative else 0.25, hsv_s=0.2 if conservative else 0.3,
                seed=20261023, deterministic=True, amp=False, cache=False, plots=False, agnostic_nms=True,
                save=True, val=True, single_cls=False)
    training_seconds = time.perf_counter()-started
    best = run / "finetune/weights/best.pt"
    if not best.is_file():
        raise ValueError("Training did not produce best.pt")
    candidate = YOLO(str(best))
    candidate_val = evaluate(candidate, data_path, run, "candidate_val")
    dev_after = run / "development_after"
    dev_after.mkdir()
    candidate_dev = development_predictions(candidate, dev_after)
    unchanged = all(sha(Path(p)) == h for p, h in original_hashes.items())
    if not unchanged:
        raise ValueError("Original app/model unexpectedly changed")
    run_report.update({"status": "completed_experiment_not_promoted", "training_seconds": training_seconds,
                       "candidate_weights": str(best), "candidate_sha256": sha(best), "originals_unchanged": unchanged,
                       "baseline_public_validation": baseline_val, "candidate_public_validation": candidate_val,
                       "baseline_development": baseline_dev, "candidate_development": candidate_dev,
                       "decision": "Do not auto-promote. Inspect per-class regressions and known failures; independent application acceptance is still required."})
    write_json(run / "summary.json", run_report)
    print(json.dumps({"summary": str(run / "summary.json"), "baseline_mAP": baseline_val['metrics'], "candidate_mAP": candidate_val['metrics'],
                      "dev_before": baseline_dev['exact_count_matches'], "dev_after": candidate_dev['exact_count_matches'], "training_seconds": training_seconds}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
