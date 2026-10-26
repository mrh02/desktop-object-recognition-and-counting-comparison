from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

project_dir = Path(__file__).resolve().parent
model = YOLO(str(project_dir / "yolo26n.pt"))

run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = project_dir / "outputs" / "rotation" / run_name
output_dir.mkdir(parents=True, exist_ok=True)

with Image.open(project_dir / "test_images" / "02_simple.jpg") as image:
    original = image.convert("RGB")

for angle in (0, 90, 180, 270):
    rotated = original.rotate(angle, expand=True)

    result = model.predict(
        source=rotated,
        device="cpu",
        imgsz=1280,
        conf=0.25,
        iou=0.7,
        agnostic_nms=True,
    )[0]

    counts = Counter(
        result.names[int(box.cls.item())]
        for box in result.boxes
    )

    print(f"\n旋转 {angle} 度：{dict(counts)}")
    result.save(filename=str(output_dir / f"rotation_{angle}.jpg"))

print(f"\n结果保存到：{output_dir}")