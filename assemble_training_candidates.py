"""Locally compare two label sources; export provisional labels, never train."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0


def compare(row, verdict, min_iou=0.6):
    local = [item for item in row["detections"] if item["kept_after_nms"] and item["name"] != "cable"]
    cloud = verdict["objects"]
    boxes = [[v*(row["width"] if i % 2 == 0 else row["height"])/1000 for i, v in enumerate(item["bbox_1000"])] for item in cloud]
    pairs = sorted(((iou(box, item["box_xyxy"]), ci, li) for ci, box in enumerate(boxes) for li, item in enumerate(local)), reverse=True)
    used_cloud, used_local, matches = set(), set(), []
    for overlap, ci, li in pairs:
        if overlap < min_iou:
            break
        if ci in used_cloud or li in used_local:
            continue
        used_cloud.add(ci)
        used_local.add(li)
        matches.append({"cloud_index": ci, "local_index": li, "iou": overlap,
                        "cloud_class": cloud[ci]["class_name"], "local_class": local[li]["name"],
                        "local_phrase": local[li]["phrase"], "class_agrees": cloud[ci]["coco_class_id"] == local[li]["coco_class_id"]})
    reasons = []
    if verdict["review_needed"]:
        reasons.append("cloud_requests_review")
    if not cloud:
        reasons.append("empty_object_list_not_accepted_as_background")
    if any(item["confidence"] != "high" for item in cloud):
        reasons.append("cloud_has_non_high_confidence_object")
    if len(used_cloud) != len(cloud):
        reasons.append("cloud_object_has_no_geometric_local_match")
    if any(not match["class_agrees"] for match in matches):
        reasons.append("class_disagreement_or_unmapped_local_phrase")
    # Never export only the agreeing subset while treating other proposed
    # foreground objects as background. A disagreement quarantines the image;
    # it does not decide which teacher is correct or delete nested objects.
    if len(used_local) != len(local):
        reasons.append("local_object_has_no_geometric_cloud_match")
    if any(item["coco_class_id"] is None for item in local):
        reasons.append("unresolved_or_unsupported_local_foreground")
    return {"status": "quarantined" if reasons else "cross_model_agreement_only", "reasons": reasons,
            "matches": matches, "unmatched_local_candidates": [item for i, item in enumerate(local) if i not in used_local]}


def label_text(verdict):
    lines = []
    for item in verdict["objects"]:
        x1, y1, x2, y2 = item["bbox_1000"]
        lines.append(f"{item['coco_class_id']} {(x1+x2)/2000:.6f} {(y1+y2)/2000:.6f} {(x2-x1)/1000:.6f} {(y2-y1)/1000:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-report", type=Path, default=PROJECT / "outputs/auto_label_trial/20261023_064214_395351/results.json")
    parser.add_argument("--audit-report", type=Path, default=PROJECT / "outputs/training_label_audit/batch_01/audit.json")
    args = parser.parse_args()
    local_report = json.loads(args.local_report.read_text(encoding="utf-8"))
    audit = json.loads(args.audit_report.read_text(encoding="utf-8"))
    by_path = {str(Path(row["image"]).resolve()).casefold(): row for row in local_report["images"]}
    run = PROJECT / "outputs/training_label_review" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    (run / "proposals_all").mkdir(parents=True)
    (run / "consistent_candidates_coco80").mkdir()
    rows = []
    for job in audit["jobs"]:
        if job["status"] != "completed":
            continue
        source = Path(job["source_image"])
        if source.parent.resolve() != (PROJECT / "training_photos").resolve():
            raise ValueError("Test or other images are forbidden in training-photo review")
        if hashlib.sha256(source.read_bytes()).hexdigest() != job["source_sha256"]:
            raise ValueError("Source hash changed")
        local = by_path[str(source.resolve()).casefold()]
        if local.get("source_sha256") != job["source_sha256"]:
            raise ValueError("Local and cloud results refer to different source content")
        if [local["width"], local["height"]] != job["original_size"]:
            raise ValueError("EXIF coordinate dimensions disagree")
        verdict = job["verdict"]
        comparison = compare(local, verdict)
        text = label_text(verdict)
        (run / "proposals_all" / f"{source.stem}.txt").write_text(text, encoding="utf-8")
        if comparison["status"] == "cross_model_agreement_only":
            (run / "consistent_candidates_coco80" / f"{source.stem}.txt").write_text(text, encoding="utf-8")
        rows.append({"image": str(source), "source_sha256": job["source_sha256"], "cloud_counts": verdict["counts"],
                     "cloud_object_count": len(verdict["objects"]), **comparison})
        print(f"{source.name}: {comparison['status']}; {comparison['reasons']}", flush=True)
    report = {"local_report": str(args.local_report), "audit_report": str(args.audit_report), "labels_verified": False,
              "training_started": False, "new_api_calls": 0, "class_id_scheme": "original_COCO80",
              "minimum_matching_iou": 0.6, "combined_requests": audit["prior_requests"] + audit["new_requests"],
              "combined_tokens": audit["prior_tokens"] + audit["new_tokens"],
              "status_counts": dict(Counter(row["status"] for row in rows)),
              "review_policy": "bidirectional_geometric_and_class_agreement_v2",
              "note": "Agreement is a heuristic, not correctness or label-completeness proof. Every non-cable local proposal and every cloud object must match geometrically and by class; unresolved or unsupported local foreground quarantines the image. Both teachers can still miss the same object. Quarantined images are not partially exported to consistent candidates. No dataset split, data.yaml or training.",
              "images": rows}
    (run / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Review: {run / 'review.json'}", flush=True)


if __name__ == "__main__":
    main()
