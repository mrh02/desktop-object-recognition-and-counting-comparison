from pathlib import Path

import gradio as gr
import numpy as np
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent

# 启动时加载一次模型，后续点击按钮时重复使用
detector = YOLO(str(PROJECT_DIR / "yolo11n.pt"))


def detect_image(image):
    if image is None:
        return None, "请先上传一张照片。"

    # Gradio传入PIL图片，转换为RGB数组
    rgb_image = np.array(image.convert("RGB"))

    # YOLO接收的numpy图片使用BGR通道顺序
    bgr_image = rgb_image[:, :, ::-1].copy()

    result = detector(
        bgr_image,
        device="cpu",
        imgsz=1280,
    )[0]

    counts = {}

    for box in result.boxes:
        class_id = int(box.cls.item())
        name = result.names[class_id]
        counts[name] = counts.get(name, 0) + 1

    if counts:
        lines = [
            f"{name}：{number}个"
            for name, number in counts.items()
        ]
        lines.append(f"总计：{sum(counts.values())}件")
        summary = "\n".join(lines)
    else:
        summary = "没有检测到物品。"

    # YOLO绘图返回BGR，转回RGB供网页显示
    annotated_image = result.plot()[:, :, ::-1].copy()

    return annotated_image, summary


with gr.Blocks() as demo:
    gr.Markdown("# 桌面物品识别与清点对比系统")
    gr.Markdown("上传照片，点击按钮查看YOLO检测与计数结果。")

    with gr.Row():
        input_image = gr.Image(
            label="上传照片",
            type="pil",
            sources=["upload"],
        )

        output_image = gr.Image(
            label="YOLO检测结果",
            interactive=False,
        )

    detect_button = gr.Button("识别物品")
    result_text = gr.Textbox(
        label="物品统计",
        lines=5,
        interactive=False,
    )

    detect_button.click(
        fn=detect_image,
        inputs=input_image,
        outputs=[output_image, result_text],
    )

    # 换图时清除旧结果，等待重新识别
    input_image.change(
        fn=lambda: (None, ""),
        inputs=None,
        outputs=[output_image, result_text],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        inbrowser=True,
        share=False,
    )