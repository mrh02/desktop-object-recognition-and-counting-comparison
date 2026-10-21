import json
from pathlib import Path

project_dir = Path(__file__).resolve().parent
output_dir = project_dir / "outputs"

# 读取两种方案保存的结果
yolo = json.loads(
    (output_dir / "yolo_result.json").read_text(encoding="utf-8")
)
vlm = json.loads(
    (output_dir / "vlm_result.json").read_text(encoding="utf-8")
)

# 确认结果记录的是同一图片路径
if Path(yolo["image_path"]).resolve() != Path(vlm["image_path"]).resolve():
    raise ValueError("两种方案使用的图片路径不同，请重新检查。")

yolo_counts = yolo["counts"]
vlm_counts = vlm["counts"]

# 合并两边出现的类别，避免漏掉只被一方识别出的物品
categories = sorted(set(yolo_counts) | set(vlm_counts))

rows = []

print("类别 | YOLO数量 | 大模型数量 | 是否一致")
print("-" * 45)

for name in categories:
    yolo_number = yolo_counts.get(name, 0)
    vlm_number = vlm_counts.get(name, 0)
    same = yolo_number == vlm_number

    print(
        f"{name} | {yolo_number} | {vlm_number} | "
        f"{'一致' if same else '不一致'}"
    )

    rows.append({
        "name": name,
        "yolo_count": yolo_number,
        "vlm_count": vlm_number,
        "same": same,
    })

all_same = all(row["same"] for row in rows)

print(f"\n计数结果：{'全部一致' if all_same else '存在差异'}")
print("注意：两种方案一致，不代表一定正确，仍需人工核对。")

# 保存对比结果
comparison = {
    "image_path": yolo["image_path"],
    "items": rows,
    "all_same": all_same,
}

result_path = output_dir / "comparison.json"
result_path.write_text(
    json.dumps(comparison, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"对比结果已保存到：{result_path}")