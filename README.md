# 桌面物品识别与清点

使用 Python、OpenCV 和 YOLO 预训练模型读取照片、检测物品、统计数量并保存标注图片。

## 当前功能

- `count_items.py`：使用模拟物品列表练习分类计数。
- `read_image.py`：读取照片、打印尺寸并显示等比例缩放的预览。
- `detect_items.py`：使用 YOLO11n 检测照片中的物品，输出类别、置信度与数量。

SAM 分割、多模态大模型识别及两种方案的对比尚未实现。

## 安装与运行

在本项目文件夹打开 PowerShell。首次配置时运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install opencv-python ultralytics
```

已有环境无需重复创建或安装。在 VS Code 中选择本项目 `.venv\Scripts\python.exe` 作为 Python 解释器。

`read_image.py` 和 `detect_items.py` 当前读取 `C:\Users\moruohan\Desktop\desk.jpg`；在其他电脑运行时，先将代码中的图片路径改为实际照片路径。

```powershell
.\.venv\Scripts\python.exe .\count_items.py
.\.venv\Scripts\python.exe .\read_image.py
.\.venv\Scripts\python.exe .\detect_items.py
```

点击照片预览窗口后按任意键即可关闭。首次加载 YOLO 时，如当前目录没有 `yolo11n.pt`，会尝试下载权重，需要网络连接。

当前检测使用 CPU 和 `imgsz=1280`，标注图保存在 `outputs/desk_detected_1280.jpg`。终端末尾的保存提示尚写为旧文件名，以实际保存路径为准。

## 初步实验

同一张包含两个杯子和一个鼠标的照片，单次运行结果如下：

| 推理尺寸设置 | 检测结果 | 模型推理耗时 |
| --- | --- | --- |
| 640 | 两个杯子，漏检鼠标 | 60.3 ms |
| 1280 | 两个杯子、一个鼠标 | 144.6 ms |

这些结果仅来自单张图片的单次运行，不代表整体准确率或稳定性能；推理耗时不包含模型加载和完整文件处理流程。

## 后续计划

学习检测框坐标，接入 SAM 分割；实现多模态大模型独立识别；在固定图片集上比较两种方案的识别、计数和耗时，整理文档与演示视频。
