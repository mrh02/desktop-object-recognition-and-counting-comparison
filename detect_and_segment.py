from config import IMAGE_PATH
import json

from pathlib import Path
from ultralytics import YOLO, SAM

project_dir = Path(__file__).resolve().parent
image_path = IMAGE_PATH
if not image_path.is_file():
    raise FileNotFoundError(f"找不到照片：{image_path}")

output_dir = project_dir / "outputs"
output_dir.mkdir(exist_ok=True)

# 第一步：YOLO检测物品
detector = YOLO(str(project_dir / "yolo11n.pt"))

detection = detector(
    str(image_path),
    device="cpu",
    imgsz=1280
)[0]

detection.save(filename=str(output_dir / "detected.jpg"))

# 取出所有检测框，转成普通Python列表
boxes = detection.boxes.xyxy.cpu().tolist()

if not boxes:
    print("没有检测到物品，本次不进行分割。")

    # 保存本次的空计数，覆盖上一次的结果
    record = {
        "model": "yolo11n",
        "image_path": str(image_path),
        "counts": {},
    }

    result_path = output_dir / "yolo_result.json"
    result_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 删除旧分割图，避免将它误认为本次结果
    segmented_path = output_dir / "all_segmented.jpg"
    if segmented_path.exists():
        segmented_path.unlink()

    print(f"空计数结果已保存到：{result_path}")

else:
    # 显示YOLO识别到的类别和数量
    counts = {}

    for box in detection.boxes:
        class_id = int(box.cls.item())
        name = detection.names[class_id]
        counts[name] = counts.get(name, 0) + 1

    print("\nYOLO物品统计：")
    for name, number in counts.items():
        print(f"{name}：{number}个")
    # 保存YOLO计数结果
    record = {
        "model": "yolo11n",
        "image_path": str(image_path),
        "counts": counts,
    }

    result_path = output_dir / "yolo_result.json"
    result_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"YOLO计数已保存到：{result_path}")
    # 第二步：将检测框自动传给SAM
    segmenter = SAM(str(project_dir / "sam_b.pt"))

    segmentation = segmenter(
        detection.orig_img,
        bboxes=boxes,
        device="cpu"
    )[0]

    output_path = output_dir / "all_segmented.jpg"
    segmentation.save(filename=str(output_path))

    print(f"\n分割结果已保存到：{output_path}")