import json
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

project_dir = Path(__file__).resolve().parent

# 先比较已经发现问题的图片
filenames = [
    "02_simple.jpg",
    "05_multiple.jpg",
    "06_clutter.jpg",
    "07_occlusion.jpg",
]

# 先检查文件，避免运行到一半才发现缺图
for filename in filenames:
    image_path = project_dir / "test_images" / filename
    if not image_path.is_file():
        raise FileNotFoundError(f"找不到图片：{image_path}")

# 每次实验单独保存，不覆盖以前的结果
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = project_dir / "outputs" / "model_comparison" / run_name
output_dir.mkdir(parents=True, exist_ok=True)

records = []

for model_name in ("yolo11n", "yolo11s"):
    # 首次使用YOLO11s时，会尝试自动下载权重
    model = YOLO(str(project_dir / f"{model_name}.pt"))

    for filename in filenames:
        image_path = project_dir / "test_images" / filename

        # 两个模型使用相同的推理设置
        result = model(
            str(image_path),
            device="cpu",
            imgsz=1280,
            conf=0.25,
            iou=0.7,
        )[0]

        counts = {}

        for box in result.boxes:
            class_id = int(box.cls.item())
            name = result.names[class_id]
            counts[name] = counts.get(name, 0) + 1

        save_name = f"{model_name}_{Path(filename).stem}.jpg"
        result.save(filename=str(output_dir / save_name))

        records.append({
            "model": model_name,
            "image": filename,
            "counts": counts,
            "inference_ms": result.speed["inference"],
        })

        print(f"\n{model_name} | {filename} | {counts}")

result_path = output_dir / "results.json"
result_path.write_text(
    json.dumps(records, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"\n实验结果已保存到：{output_dir}")