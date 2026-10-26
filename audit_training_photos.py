"""Small, resumable whole-image pseudo-label audit. Never starts training."""
import argparse
import base64
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time

from PIL import Image, ImageDraw, ImageFont, ImageOps
import yaml

PROJECT = Path(__file__).resolve().parent
MODEL = "qwen3-vl-plus"
FILES = ("01.jpg", "05.jpg", "08.jpg", "09.jpg", "11.jpg", "14.jpg", "19.jpg", "20.jpg")
MAX_REQUESTS = 10
OBSERVED_TOKEN_STOP = 10000
NEXT_REQUEST_RESERVE = 1500
MAX_OUTPUT = 512
PRIOR_RECEIPT = "outputs/auto_label_audit/20261023_063332_574400/audit.json"


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def parse_objects(raw, names):
    text = raw.strip()
    lines = text.splitlines()
    if len(lines) >= 3 and lines[0].strip().lower() in ("```", "```json") and lines[-1].strip() == "```":
        text = "\n".join(lines[1:-1]).strip()
    data = json.loads(text)
    if not isinstance(data, dict) or type(data.get("review_needed")) is not bool or not isinstance(data.get("objects"), list):
        raise ValueError("Invalid JSON object schema")
    if len(data["objects"]) > 12:
        raise ValueError("Unexpected object count")
    valid_names = {value: int(key) for key, value in names.items()}
    objects = []
    for item in data["objects"]:
        if not isinstance(item, dict) or item.get("class_name") not in valid_names or item.get("confidence") not in {"high", "medium", "low"}:
            raise ValueError("Unknown class or invalid confidence")
        box = item.get("bbox")
        if not isinstance(box, list) or len(box) != 4 or any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1000 for v in box):
            raise ValueError("Invalid normalized box")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Nonpositive box")
        objects.append({"class_name": item["class_name"], "coco_class_id": valid_names[item["class_name"]],
                        "confidence": item["confidence"], "bbox_1000": box})
    return {"objects": objects, "review_needed": data["review_needed"],
            "counts": dict(Counter(item["class_name"] for item in objects))}


def save_preview(job, folder):
    with Image.open(job["image_sent"]) as opened:
        preview = opened.convert("RGB")
    draw = ImageDraw.Draw(preview)
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/arial.ttf"
    font = ImageFont.truetype(str(font_path), 16) if font_path.is_file() else ImageFont.load_default()
    for item in job["verdict"]["objects"]:
        box = [value * (preview.width if i % 2 == 0 else preview.height) / 1000 for i, value in enumerate(item["bbox_1000"])]
        draw.rectangle(box, outline="#00b050", width=2)
        position = (box[0], max(0, box[1] - 19))
        title = f"{item['class_name']} ({item['confidence']})"
        draw.rectangle(draw.textbbox(position, title, font=font), fill="white")
        draw.text(position, title, fill="black", font=font)
    preview.save(folder / f"{Path(job['source_image']).stem}.jpg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--send", action="store_true")
    args = parser.parse_args()
    names = yaml.safe_load((PROJECT / ".venv/Lib/site-packages/ultralytics/cfg/datasets/coco.yaml").read_text(encoding="utf-8"))["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    prompt = (
        "Independently inspect this photograph and list every recognizable physical object belonging to the COCO-80 classes below. "
        "Do not count shadows, floor, walls or cables. Include real background objects when recognizable. "
        "Count instances, not parts: multiple rims of one cup are not separate cups. "
        "For each object give a tight axis-aligned bounding box around its visible extent; do not invent hidden extent. "
        "Coordinates MUST be normalized from 0 to 1000 relative to the entire supplied image, in [left,top,right,bottom] order. "
        "Use review_needed=true for ambiguous objects, uncertain instance counts or incomplete visibility that prevents classification. "
        "Return only JSON with this schema: {\"objects\":[{\"class_name\":\"cup\",\"bbox\":[100,200,300,400],\"confidence\":\"high|medium|low\"}],\"review_needed\":false}. "
        "Use only these class names: " + ", ".join(names.values())
    )
    batch = PROJECT / "outputs/training_label_audit/batch_01"
    (batch / "images_sent").mkdir(parents=True, exist_ok=True)
    (batch / "previews").mkdir(exist_ok=True)
    report_path = batch / "audit.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["prompt"] != prompt or report["model"] != MODEL or report["selected_files"] != list(FILES):
            raise RuntimeError("Batch configuration changed; refuse to reset the budget")
    else:
        prior = json.loads((PROJECT / PRIOR_RECEIPT).read_text(encoding="utf-8"))
        if not prior["usage_known"]:
            raise RuntimeError("Prior usage unknown; refuse new requests")
        jobs = []
        for name in FILES:
            path = PROJECT / "training_photos" / name
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            original_size = list(image.size)
            image.thumbnail((768, 768))
            sent_path = batch / "images_sent" / name
            image.save(sent_path, format="JPEG", quality=85)
            jobs.append({"source_image": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "original_size": original_size, "image_sent": str(sent_path), "sent_size": list(image.size),
                         "sent_sha256": hashlib.sha256(sent_path.read_bytes()).hexdigest(), "status": "prepared"})
        report = {"created_at": datetime.now().isoformat(), "model": MODEL, "prompt": prompt,
                  "selected_files": list(FILES), "prior_receipt": PRIOR_RECEIPT,
                  "prior_requests": prior["requests_sent"], "prior_tokens": prior["actual_total_tokens_this_run"],
                  "new_requests": 0, "new_tokens": 0, "usage_known": True,
                  "max_combined_requests": MAX_REQUESTS, "observed_combined_token_stop": OBSERVED_TOKEN_STOP,
                  "next_request_reserve": NEXT_REQUEST_RESERVE, "max_output_tokens": MAX_OUTPUT,
                  "note": "Local observed-usage gate, not a server hard cap. No automatic retries; no training. Model predictions are not ground truth.",
                  "labels_verified": False, "training_started": False, "jobs": jobs}
        write_json(report_path, report)
    print(f"Batch: {report_path}", flush=True)
    if not args.send:
        print("Preparation only; no API requests.", flush=True)
        return
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(PROJECT / ".env")
    key, base_url = os.getenv("DASHSCOPE_API_KEY"), os.getenv("DASHSCOPE_BASE_URL")
    if not key or not base_url:
        raise RuntimeError("Missing configured API credentials")
    client = OpenAI(api_key=key, base_url=base_url, timeout=60, max_retries=0)
    try:
        for job in report["jobs"]:
            if hashlib.sha256(Path(job["source_image"]).read_bytes()).hexdigest() != job["source_sha256"] or hashlib.sha256(Path(job["image_sent"]).read_bytes()).hexdigest() != job["sent_sha256"]:
                raise RuntimeError("An image changed; refuse stale provenance")
            if job["status"] == "completed":
                continue
            if job["status"] == "received" and job.get("usage") and job.get("finish_reason") == "stop":
                job["verdict"] = parse_objects(job["raw_text"], names)
                job["status"] = "completed"
                save_preview(job, batch / "previews")
                write_json(report_path, report)
                continue
            if job["status"] != "prepared" or not report["usage_known"]:
                print("Stopped: unresolved prior request; no retry.", flush=True)
                break
            total_requests = report["prior_requests"] + report["new_requests"]
            total_tokens = report["prior_tokens"] + report["new_tokens"]
            if total_requests >= MAX_REQUESTS or OBSERVED_TOKEN_STOP - total_tokens < NEXT_REQUEST_RESERVE:
                print("Stopped at combined request/token gate.", flush=True)
                break
            encoded = base64.b64encode(Path(job["image_sent"]).read_bytes()).decode("ascii")
            job["status"] = "started"
            report["new_requests"] += 1
            report["usage_known"] = False
            write_json(report_path, report)
            started = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                        {"type": "text", "text": prompt},
                    ]}], max_tokens=MAX_OUTPUT, temperature=0,
                    response_format={"type": "json_object"},
                    extra_body={"enable_thinking": False, "vl_high_resolution_images": False},
                )
                job.update({"status": "received", "raw_text": response.choices[0].message.content or "",
                            "finish_reason": response.choices[0].finish_reason, "seconds": time.perf_counter() - started})
                write_json(report_path, report)
                usage = response.usage
                if usage is None or usage.total_tokens is None:
                    report["usage_known"] = False
                    write_json(report_path, report)
                    print("Stopped: provider usage missing.", flush=True)
                    break
                job["usage"] = {"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens, "total_tokens": usage.total_tokens}
                report["new_tokens"] += usage.total_tokens
                report["usage_known"] = True
                write_json(report_path, report)
                if job["finish_reason"] != "stop":
                    raise ValueError("Incomplete response")
                job["verdict"] = parse_objects(job["raw_text"], names)
                job["status"] = "completed"
                save_preview(job, batch / "previews")
                print(f"{Path(job['source_image']).name}: {job['verdict']['counts']}; review={job['verdict']['review_needed']}; tokens={usage.total_tokens}; cumulative={report['prior_tokens'] + report['new_tokens']}", flush=True)
            except Exception as error:
                if "usage" not in job:
                    report["usage_known"] = False
                job["error_type"] = type(error).__name__
                write_json(report_path, report)
                print(f"Stopped without retry: {type(error).__name__}", flush=True)
                break
            finally:
                write_json(report_path, report)
    finally:
        client.close()
        write_json(report_path, report)
    print(f"Combined requests={report['prior_requests'] + report['new_requests']}; combined tokens={report['prior_tokens'] + report['new_tokens']}; usage_known={report['usage_known']}", flush=True)


if __name__ == "__main__":
    main()
