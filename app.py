import os
import tempfile
from pathlib import Path
from vlm_service import recognize_image

# 必须在导入Gradio前设置，每次启动使用新的临时目录
os.environ["GRADIO_TEMP_DIR"] = tempfile.mkdtemp(
    prefix="lingjing-gradio-"
)

import gradio as gr

import numpy as np
from ultralytics import YOLO, SAM

PROJECT_DIR = Path(__file__).resolve().parent

# 启动时加载一次模型，后续点击按钮时重复使用
detector = YOLO(str(PROJECT_DIR / "yolo11n.pt"))
segmenter = SAM(str(PROJECT_DIR / "sam_b.pt"))

def detect_image(image):
    if image is None:
        return None, None, "请先上传一张照片。", None

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

    # 将YOLO的检测框传给SAM
    boxes = result.boxes.xyxy.cpu().tolist()

    if boxes:
        segmentation = segmenter(
            bgr_image,
            bboxes=boxes,
            device="cpu",
        )[0]

        segmented_image = segmentation.plot()[:, :, ::-1].copy()
    else:
        segmented_image = None
        summary += "\n未检测到物品，本次不进行分割。"

    return annotated_image, segmented_image, summary, counts

def compare_counts(yolo_counts, vlm_counts):
    if yolo_counts is None or vlm_counts is None:
        return [], "请先完成当前照片的本地检测和大模型识别。"

    categories = sorted(set(yolo_counts) | set(vlm_counts))
    rows = []

    for name in categories:
        yolo_number = yolo_counts.get(name, 0)
        vlm_number = vlm_counts.get(name, 0)

        rows.append([
            name,
            yolo_number,
            vlm_number,
            "一致" if yolo_number == vlm_number else "不一致",
        ])

    if not categories:
        message = "两种方案均返回0件物品，请人工核对。"
    elif yolo_counts == vlm_counts:
        message = "计数全部一致，但不代表一定正确，请人工核对。"
    else:
        message = "存在计数差异，请结合原图检查漏检、误检或类别名称差异。"

    return rows, message
with gr.Blocks() as demo:
    yolo_state = gr.State(value=None)
    vlm_state = gr.State(value=None)
    gr.Markdown("# 桌面物品识别与清点对比系统")
    gr.Markdown("上传照片，点击按钮查看YOLO检测、物品计数与SAM分割结果。")
    with gr.Row():
        input_image = gr.Image(
            label="上传照片",
            type="pil",
            sources=["upload"],
            format="png",
        )

        output_image = gr.Image(
            label="YOLO检测结果",
            interactive=False,
            format="png",
        )

    segmented_output = gr.Image(
        label="SAM分割结果",
        interactive=False,
        format="png",
    )

    detect_button = gr.Button("检测并分割")
    result_text = gr.Textbox(
        label="物品统计",
        lines=5,
        interactive=False,
    )

    detect_button.click(
        fn=detect_image,
        inputs=input_image,
        outputs=[
            output_image,
            segmented_output,
            result_text,
            yolo_state,
        ],
    )

    # 换图时清除上一张照片的结果
    input_image.change(
    fn=lambda: (None, None, "", None),
    inputs=None,
    outputs=[
            output_image,
            segmented_output,
            result_text,
            yolo_state,
        ],
    )

    gr.Markdown("## 多模态大模型识别")
    gr.Markdown("点击下方按钮会将当前照片发送至阿里云，并消耗模型额度。")

    vlm_button = gr.Button("大模型识别")
    vlm_text = gr.Textbox(
        label="大模型物品统计",
        lines=7,
        interactive=False,
    )

    vlm_button.click(
        fn=recognize_image,
        inputs=input_image,
        outputs=[vlm_text, vlm_state],
        concurrency_limit=1,
    )

    # 换图时清除旧的大模型回答
    input_image.change(
        fn=lambda: ("", None),
        inputs=None,
        outputs=[vlm_text, vlm_state],
    )
    
    gr.Markdown("## 两种方案计数对比")

    compare_button = gr.Button("比较当前结果")

    comparison_table = gr.Dataframe(
        headers=["类别", "YOLO数量", "大模型数量", "是否一致"],
        datatype=["str", "number", "number", "str"],
        value=[],
        interactive=False,
    )

    comparison_text = gr.Textbox(
        label="对比说明",
        interactive=False,
    )

    compare_button.click(
        fn=compare_counts,
        inputs=[yolo_state, vlm_state],
        outputs=[comparison_table, comparison_text],
    )

    # 换图或重新识别时，清除旧对比表
    def clear_comparison():
        return [], ""

    input_image.change(
        fn=clear_comparison,
        inputs=None,
        outputs=[comparison_table, comparison_text],
    )

    detect_button.click(
        fn=clear_comparison,
        inputs=None,
        outputs=[comparison_table, comparison_text],
    )

    vlm_button.click(
        fn=clear_comparison,
        inputs=None,
        outputs=[comparison_table, comparison_text],
    )
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        inbrowser=True,
        share=False,
    )