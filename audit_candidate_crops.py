"""Two-crop API audit with resize, request limits, usage logging and caching."""
import argparse
import base64
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import os
import time

from PIL import Image, ImageOps
import yaml

PROJECT = Path(__file__).resolve().parent
MODEL = "qwen3-vl-plus"
MAX_REQUESTS = 2
TOKEN_STOP = 2000
RESERVE_PER_REQUEST = 1000  # conservative gate, not a server-side input-token cap


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_verdict(raw, id_by_name):
    text = raw.strip()
    lines = text.splitlines()
    if len(lines) >= 3 and lines[0].strip().lower() in ("```", "```json") and lines[-1].strip() == "```":
        text = "\n".join(lines[1:-1]).strip()
    data = json.loads(text)
    if not isinstance(data, dict) or data.get("class_name") not in set(id_by_name) | {"other", "unknown"} or data.get("confidence") not in {"high", "medium", "low"} or type(data.get("single_main_object")) is not bool:
        raise ValueError("Invalid audit response")
    return {**data, "coco_class_id": id_by_name.get(data["class_name"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--send", action="store_true", help="Actually send at most two cropped images to the configured API")
    args = parser.parse_args()
    source = PROJECT / "outputs/auto_label_trial/20261022_234121_501554/results.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    names = yaml.safe_load((PROJECT / ".venv/Lib/site-packages/ultralytics/cfg/datasets/coco.yaml").read_text(encoding="utf-8"))["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    id_by_name = {name: int(cid) for cid, name in names.items()}
    choices = ", ".join(id_by_name)
    prompt = (
        "Identify the one main physical object in the centre of this crop. "
        "Ignore shadows, background and attached cables. Do not guess if unclear. "
        "Return JSON only: {\"class_name\":\"English class\",\"confidence\":\"high|medium|low\",\"single_main_object\":true}. "
        "Choose a class from this COCO-80 list, or other/unknown: " + choices
    )
    # Deliberate diagnostic cases, not a batch training audit. Do not send the
    # detector's proposed class to the verifier (avoid anchoring its answer).
    selectors = (("16.jpg", "a remote control"), ("02_simple.jpg", "a paper cup a toilet"))
    run = PROJECT / "outputs/auto_label_audit" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run.mkdir(parents=True, exist_ok=False)
    cache = PROJECT / "outputs/auto_label_audit/cache"
    cache.mkdir(parents=True, exist_ok=True)
    jobs = []
    for filename, phrase in selectors:
        row = next(row for row in report["images"] if Path(row["image"]).name == filename)
        candidate = next(item for item in row["detections"] if item["kept_after_nms"] and item["phrase"] == phrase)
        source_path = Path(row["image"])
        with Image.open(source_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        x1, y1, x2, y2 = candidate["box_xyxy"]
        dx, dy = (x2-x1)*0.06, (y2-y1)*0.06
        crop_box = (max(0, int(x1-dx)), max(0, int(y1-dy)), min(image.width, int(x2+dx+1)), min(image.height, int(y2+dy+1)))
        crop = image.crop(crop_box)
        crop.thumbnail((512, 512))
        buffer = io.BytesIO()
        crop.save(buffer, format="JPEG", quality=85)
        jpeg = buffer.getvalue()
        crop_path = run / f"{source_path.stem}_crop.jpg"
        crop_path.write_bytes(jpeg)
        key = hashlib.sha256(jpeg + prompt.encode("utf-8") + MODEL.encode() + b"no_thinking_max96_v1").hexdigest()
        jobs.append({"source_image": str(source_path), "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                     "original_box": candidate["box_xyxy"], "detector_phrase": candidate["phrase"],
                     "crop": str(crop_path), "crop_size": list(crop.size), "crop_bytes": len(jpeg),
                     "cache_key": key, "status": "prepared"})
    audit = {"model": MODEL, "max_api_requests": MAX_REQUESTS, "max_output_tokens_per_request": 96,
             "max_crop_side": 512, "observed_token_stop": TOKEN_STOP,
             "reserve_gate_per_request": RESERVE_PER_REQUEST,
             "budget_note": "max_tokens limits output only; image/text input is metered by the provider. Resize and a two-call limit reduce input. No automatic retries.",
             "requests_sent": 0, "actual_total_tokens_this_run": 0, "usage_known": True,
             "training_started": False, "automatic_bulk_audit": False, "jobs": jobs}
    write_json(run / "audit.json", audit)
    print(f"Prepared exactly {len(jobs)} cropped images: {run}", flush=True)
    if not args.send:
        print("Preparation only: no API calls made.", flush=True)
        return

    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(PROJECT / ".env")
    api_key, base_url = os.getenv("DASHSCOPE_API_KEY"), os.getenv("DASHSCOPE_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("Missing local API configuration; no request sent.")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0, max_retries=0)
    try:
        for job in jobs:
            cached_path = cache / f"{job['cache_key']}.json"
            if cached_path.exists():
                cached = json.loads(cached_path.read_text(encoding="utf-8"))
                if cached.get("status") == "received" and cached.get("usage") and cached.get("finish_reason") == "stop":
                    try:
                        cached["verdict"] = parse_verdict(cached["raw_text"], id_by_name)
                        cached["status"] = "completed"
                        write_json(cached_path, cached)
                    except (KeyError, ValueError):
                        job["status"] = "stopped_cached_response_invalid_no_retry"
                        break
                if cached.get("status") != "completed":
                    job["status"] = "stopped_previous_request_unresolved"
                    break
                job.update({"status": "cached", "verdict": cached["verdict"], "historical_usage": cached["usage"]})
                continue
            if audit["requests_sent"] >= MAX_REQUESTS or TOKEN_STOP-audit["actual_total_tokens_this_run"] < RESERVE_PER_REQUEST:
                job["status"] = "stopped_by_budget_gate"
                break
            encoded = base64.b64encode(Path(job["crop"]).read_bytes()).decode("ascii")
            write_json(cached_path, {"status": "started", "model": MODEL, "source_image": job["source_image"]})
            audit["requests_sent"] += 1
            job["status"] = "request_started"
            write_json(run / "audit.json", audit)
            started = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                        {"type": "text", "text": prompt},
                    ]}], max_tokens=96, temperature=0,
                    response_format={"type": "json_object"},
                    extra_body={"enable_thinking": False, "vl_high_resolution_images": False},
                )
                usage = response.usage
                raw_text = response.choices[0].message.content or ""
                job.update({"raw_text": raw_text, "finish_reason": response.choices[0].finish_reason})
                receipt = {"status": "received", "model": MODEL, "raw_text": raw_text,
                           "finish_reason": response.choices[0].finish_reason, "usage": None}
                write_json(cached_path, receipt)
                if usage is None or usage.total_tokens is None:
                    audit["usage_known"] = False
                    job["status"] = "stopped_missing_usage"
                    break
                metering = {"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens, "total_tokens": usage.total_tokens}
                audit["actual_total_tokens_this_run"] += usage.total_tokens
                job.update({"usage": metering, "seconds": time.perf_counter()-started})
                receipt["usage"] = metering
                write_json(cached_path, receipt)
                if response.choices[0].finish_reason != "stop":
                    raise ValueError("Incomplete model response")
                verdict = parse_verdict(raw_text, id_by_name)
                job.update({"status": "completed", "verdict": verdict})
                write_json(cached_path, {**receipt, "status": "completed", "verdict": verdict})
                print(f"{Path(job['source_image']).name}: {verdict}; usage={metering}", flush=True)
            except Exception as error:
                job.update({"status": "stopped_error_no_retry", "error_type": type(error).__name__})
                audit["usage_known"] = bool("usage" in job)
                print(f"Stopped without retry: {type(error).__name__}", flush=True)
                break
            finally:
                write_json(run / "audit.json", audit)
    finally:
        client.close()
        write_json(run / "audit.json", audit)
    print(f"API calls sent: {audit['requests_sent']}; metered tokens: {audit['actual_total_tokens_this_run']}; usage_known={audit['usage_known']}", flush=True)
    print(f"Audit: {run / 'audit.json'}. No training and no bulk requests started.", flush=True)


if __name__ == "__main__":
    main()
