import base64
import io
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def recognize_image(image):
    """接收网页上传的照片，返回物品统计文字。"""
    if image is None:
        return "请先上传一张照片。"

    project_dir = Path(__file__).resolve().parent
    load_dotenv(project_dir / ".env")

    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_BASE_URL")

    if not api_key or not base_url:
        return "缺少配置，请检查本地.env文件。"

    # 将上传的图片统一转换成JPEG
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=120.0,
        max_retries=0,
    )

    start = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model="qwen3-vl-plus",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded}"
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "清点图片中的独立物品，按类别统计数量。"
                                "不统计地板、阴影或连接在物品上的线缆。"
                                "类别用英文单数，杯子统一为cup，鼠标为mouse，"
                                "其他物品也按实际情况列出。"
                                "只返回合法JSON，不要Markdown或解释。"
                                '格式：{"items":[{"name":"cup","count":整数}]}。'
                                '没有物品时返回{"items":[]}。'
                            ),
                        },
                    ],
                }
            ],
            max_tokens=500,
            extra_body={"enable_thinking": False},
        )

        data = json.loads(response.choices[0].message.content or "")

        if not isinstance(data, dict) or not isinstance(data.get("items"), list):
            raise ValueError("返回格式错误")

        counts = {}

        for item in data["items"]:
            if not isinstance(item, dict):
                raise ValueError("物品格式错误")

            name = item.get("name")
            count = item.get("count")

            if not isinstance(name, str) or not name.strip():
                raise ValueError("类别名称错误")
            if type(count) is not int or count < 0:
                raise ValueError("数量格式错误")

            name = name.strip().lower()
            if name in counts:
                raise ValueError("重复类别")

            counts[name] = count

        lines = [
            f"{name}：{count}个"
            for name, count in counts.items()
        ]
        lines.append(f"总计：{sum(counts.values())}件")
        lines.append(f"接口调用耗时：{time.perf_counter() - start:.2f}秒")

        if response.usage:
            lines.append(f"Token用量：{response.usage.total_tokens}")

        return "\n".join(lines)

    except (json.JSONDecodeError, ValueError):
        return "模型返回格式不符合要求，本次未生成有效统计。"
    except Exception as error:
        # 只显示错误类型，避免把敏感配置带到网页上
        return (
            f"调用失败：{type(error).__name__}。"
            "请检查网络、模型权限和本地配置。"
        )
    finally:
        client.close()