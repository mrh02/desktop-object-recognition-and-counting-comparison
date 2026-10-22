import json
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

project_dir = Path(__file__).resolve().parent
image_path = project_dir / "test_images" / "08_occlusion.jpg"
model = YOLO(str(project_dir / "yolo11s.pt"))

run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = project_dir / "outputs" / "nms_experiment" / run_name
output_dir.mkdir(parents=True, exist_ok=True)

records = []

# 其他参数相同，只比较是否开启跨类别去重
for enabled in (False, True):
    result = model(
        str(image_path),
        device="cpu",
        imgsz=1280,
        conf=0.25,
        iou=0.7,
        agnostic_nms=enabled,
    )[0]

    counts = {}
    detections = []

    print(f"\n跨类别去重：{enabled}")

    for box in result.boxes:
        name = result.names[int(box.cls.item())]
        confidence = float(box.conf.item())

        counts[name] = counts.get(name, 0) + 1

        detections.append({
            "name": name,
            "confidence": confidence,
            "box": box.xyxy[0].tolist(),
        })

        print(f"{name}：置信度{confidence:.3f}")

    print(f"计数：{counts}")

    result.save(
        filename=str(output_dir / f"nms_{enabled}.jpg")
    )

    records.append({
        "image": image_path.name,
        "agnostic_nms": enabled,
        "counts": counts,
        "detections": detections,
    })

(output_dir / "results.json").write_text(
    json.dumps(records, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"\n实验结果保存到：{output_dir}")