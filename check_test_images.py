import json
from pathlib import Path
from PIL import Image

project_dir = Path(__file__).resolve().parent
image_dir = project_dir / "test_images"
labels_path = project_dir / "test_labels.json"

labels = json.loads(labels_path.read_text(encoding="utf-8-sig"))

all_ok = True

for filename, label in labels.items():
    image_path = image_dir / filename

    if not image_path.is_file():
        print(f"找不到图片：{filename}")
        all_ok = False
        continue

    try:
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size

        print(f"\n图片：{filename}")
        print(f"尺寸：{width}×{height}")
        print(f"人工答案：{label['counts']}")

    except OSError as error:
        print(f"图片读取失败：{filename}，{error}")
        all_ok = False

if all_ok:
    print("\n检查通过，所有标注图片均可读取。")
else:
    print("\n检查未通过，请先修正以上问题。")