from pathlib import Path
from ultralytics import SAM

# 项目目录
project_dir = Path(__file__).resolve().parent

# 加载SAM模型，首次运行需要下载权重
model = SAM(str(project_dir / "sam_b.pt"))

# 使用原来的照片
image_path = r"C:\Users\moruohan\Desktop\desk.jpg"

# YOLO之前给出的鼠标位置：左、上、右、下
mouse_box = [1951, 1685, 2598, 2433]

# 告诉SAM：分割这个框中的物体
results = model(
    image_path,
    bboxes=[mouse_box],
    device="cpu"
)

# 保存带分割颜色的图片
output_dir = project_dir / "outputs"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "mouse_segmented.jpg"
results[0].save(filename=str(output_path))

print(f"分割结果已保存到：{output_path}")