import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

project_dir = Path(__file__).resolve().parent
model = YOLO(str(project_dir / "yolov8s.pt"))

run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = project_dir / "outputs" / "yolov8s" / run_name
output_dir.mkdir(parents=True, exist_ok=True)

records = []

for filename in (
    "02_simple.jpg",
    "05_multiple.jpg",
    "06_clutter.jpg",
    "07_occlusion.jpg",
    "08_occlusion.jpg",
):
    result = model.predict(
        source=str(project_dir / "test_images" / filename),
        device="cpu",
        imgsz=1280,
        conf=0.25,
        iou=0.7,
        agnostic_nms=True,
    )[0]

    detections = []
    for box in result.boxes:
        detections.append({
            "name": result.names[int(box.cls.item())],
            "confidence": float(box.conf.item()),
            "box": box.xyxy[0].tolist(),
        })

    counts = dict(Counter(item["name"] for item in detections))

    print(f"\n图片：{filename}")
    for item in detections:
        print(f"{item['name']}：{item['confidence']:.3f}")
    print(f"计数：{counts}")

    result.save(filename=str(output_dir / filename))

    records.append({
        "image": filename,
        "counts": counts,
        "detections": detections,
        "speed_ms": result.speed,
    })

(output_dir / "results.json").write_text(
    json.dumps(records, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"\n结果保存到：{output_dir}")