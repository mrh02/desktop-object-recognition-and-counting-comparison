from pathlib import Path
from ultralytics import YOLO

# 加载预训练模型，首次运行时会自动下载模型文件
model = YOLO("yolo11n.pt")

# 之前成功读取的照片路径
image_path = r"C:\Users\moruohan\Desktop\desk.jpg"

# 识别图片，先使用CPU跑通流程
results = model(image_path, device="cpu", imgsz=1280)

# 创建保存结果的文件夹
output_dir = Path(__file__).parent / "outputs"
output_dir.mkdir(exist_ok=True)

# 获取这张图片的识别结果
result = results[0]
result.save(filename=str(output_dir / "desk_detected_1280.jpg"))

# 保存每种物品的数量
counts = {}

for box in result.boxes:
    class_id = int(box.cls.item())
    name = result.names[class_id]
    confidence = float(box.conf.item())

    print(f"物品：{name}，置信度：{confidence:.2f}")

    if name in counts:
        counts[name] += 1
    else:
        counts[name] = 1

print("\n物品统计：")

if not counts:
    print("没有检测到物品")
else:
    for name, number in counts.items():
        print(f"{name}：{number}个")

print("标注图片已保存到 outputs/desk_detected_1280.jpg")