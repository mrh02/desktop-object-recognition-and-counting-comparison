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
image_path = Path(r"C:\Users\moruohan\Desktop\desk.jpg")
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

print("正在发送图片，请等待模型回答……")

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
                        "请清点图片中可见的独立物品，"
                        "按物品类别列出数量。"
                        "不要统计地板、阴影，"
                        "连接在物品上的线缆不要单独计数。"
                        "无法确定时请说明，不要猜测。"
                        "请用中文简洁回答。"
                    ),
                },
            ],
        }
    ],
    max_tokens=500,
    extra_body={"enable_thinking": False},
)

print("\n大模型识别结果：")
print(response.choices[0].message.content)

if response.usage:
    print(f"\n本次总Token用量：{response.usage.total_tokens}")