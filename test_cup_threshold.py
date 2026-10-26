import json
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

project_dir = Path(__file__).resolve().parent
image_path = project_dir / "test_images" / "07_occlusion.jpg"

model = YOLO(str(project_dir / "yolo11s.pt"))

run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = project_dir / "outputs" / "cup_threshold" / run_name
output_dir.mkdir(parents=True, exist_ok=True)

records = []

for threshold in (0.25, 0.10):
    result = model(
        str(image_path),
        device="cpu",
        imgsz=1280,
        conf=threshold,
        iou=0.7,
        agnostic_nms=True,
    )[0]

    counts = {}
    detections = []

    print(f"\n置信度阈值：{threshold}")

    for box in result.boxes:
        name = result.names[int(box.cls.item())]
        confidence = float(box.conf.item())

        counts[name] = counts.get(name, 0) + 1
        detections.append({
            "name": name,
            "confidence": confidence,
            "box": box.xyxy[0].tolist(),
        })

        print(f"{name}：{confidence:.3f}")

    print(f"计数：{counts}")

    result.save(
        filename=str(output_dir / f"conf_{threshold}.jpg")
    )

    records.append({
        "image": image_path.name,
        "confidence_threshold": threshold,
        "counts": counts,
        "detections": detections,
    })

(output_dir / "results.json").write_text(
    json.dumps(records, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"\n结果保存到：{output_dir}")