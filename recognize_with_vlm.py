from config import IMAGE_PATH

import json
import time

import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# 从项目中的.env读取配置
project_dir = Path(__file__).resolve().parent
load_dotenv(project_dir / ".env")

api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = os.getenv("DASHSCOPE_BASE_URL")

if not api_key or not base_url:
    raise ValueError("请检查.env中的密钥和接口地址是否已经填写并保存。")

# 读取原照片，转换成接口支持的Base64格式
image_path = IMAGE_PATH
if not image_path.is_file():
    raise FileNotFoundError(f"找不到照片：{image_path}")
    
image_base64 = base64.b64encode(
    image_path.read_bytes()
).decode("utf-8")

# 创建调用客户端，关闭自动重试，便于首次排查
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=120.0,
    max_retries=0,
)

# 开始新任务前，清除上次结果，避免失败后误用旧数据
output_dir = project_dir / "outputs"
output_dir.mkdir(exist_ok=True)

for filename in ("vlm_result.json", "comparison.json"):
    old_result = output_dir / filename
    if old_result.exists():
        old_result.unlink()
        
print("正在发送图片，请等待模型回答……")
start_time = time.perf_counter()
response = client.chat.completions.create(
    model="qwen3-vl-plus",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "请清点图片中可见的独立物品，按类别统计数量。"
                        "不要统计地板、阴影，连接在物品上的线缆不要单独计数。"
                        "类别名称使用英文单数，杯子统一用cup，鼠标统一用mouse。"
                        "其他物品也按实际情况列出，不要只识别杯子和鼠标。"
                        "只输出合法JSON，不要输出Markdown代码块或其他解释。"
                        '格式为：{"items":[{"name":"英文类别","count":整数}]}。'
                        '没有物品时返回：{"items":[]}。'
                    ),
                },
            ],
        }
    ],
    max_tokens=500,
    extra_body={"enable_thinking": False},
)

elapsed = time.perf_counter() - start_time
answer = response.choices[0].message.content or ""

# 将JSON文本转换成Python字典
try:
    data = json.loads(answer)
except json.JSONDecodeError:
    print("模型没有返回合法JSON，原始回答为：")
    print(answer)
    raise SystemExit("本次解析失败，暂不保存统计结果。")

# 检查返回的数据结构
if not isinstance(data, dict) or not isinstance(data.get("items"), list):
    raise ValueError("返回格式不正确：items应当是列表。")

counts = {}

for item in data["items"]:
    if not isinstance(item, dict):
        raise ValueError("每个物品应当包含name和count。")

    name = item.get("name")
    count = item.get("count")

    if not isinstance(name, str) or not name.strip():
        raise ValueError("物品名称应当是非空字符串。")

    if type(count) is not int or count < 0:
        raise ValueError("物品数量应当是非负整数。")

    name = name.strip().lower()

    if name in counts:
        raise ValueError(f"模型重复返回了类别：{name}")

    counts[name] = count

print("\n大模型物品统计：")
for name, count in counts.items():
    print(f"{name}：{count}个")

print(f"总计：{sum(counts.values())}件")
print(f"接口调用耗时：{elapsed:.2f}秒")

# 保存结果，供后续自动对比使用
output_dir = project_dir / "outputs"
output_dir.mkdir(exist_ok=True)

record = {
    "model": "qwen3-vl-plus",
    "image_path": str(image_path),
    "counts": counts,
    "api_elapsed_seconds": elapsed,
    "total_tokens": (
        response.usage.total_tokens if response.usage else None
    ),
}

output_path = output_dir / "vlm_result.json"
output_path.write_text(
    json.dumps(record, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"结果已保存到：{output_path}")
if response.usage:
    print(f"本次总Token用量：{response.usage.total_tokens}")