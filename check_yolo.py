from pathlib import Path
from ultralytics import YOLO

project_dir = Path(__file__).resolve().parent
model = YOLO(str(project_dir / "yolo11n.pt"))

result = model(
    str(project_dir / "test_images" / "04_multiple.jpg"),
    device="cpu",
    imgsz=1280,
)[0]

for box in result.boxes:
    class_id = int(box.cls.item())
    confidence = float(box.conf.item())

    print(
        f"类别编号：{class_id}，"
        f"名称：{result.names[class_id]}，"
        f"置信度：{confidence:.2f}"
    )