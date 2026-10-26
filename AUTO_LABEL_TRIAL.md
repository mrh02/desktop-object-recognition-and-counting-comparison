# 自动标注试验

本试验使用 `IDEA-Research/grounding-dino-tiny` 在本地生成候选检测框，标注器用于准备数据，未替换网页检测器，也不会启动训练。

## 依赖和运行

在项目目录运行。依赖放入 `outputs/auto_label_deps`，由试验脚本单独加载。

```powershell
.\.venv\Scripts\python.exe -m pip install --no-cache-dir --target outputs/auto_label_deps -r requirements-autolabel.txt
.\.venv\Scripts\python.exe auto_label_trial.py --files 02_simple.jpg 07_occlusion.jpg 13.jpg 14.jpg 15.jpg 16.jpg 17.jpg
```

首次运行需要下载模型。模型缓存位于 `outputs/auto_label_models`，后续可复用。

## 输出

结果保存到 `outputs/auto_label_trial/时间戳/`：

- `previews/`：带类别、分数和检测框的预览图。
- `candidate_labels_coco80/`：候选 YOLO 检测标签，格式为类别 ID、归一化中心坐标、宽、高。
- `results.json`：原始候选、跨类别去重结果、提示词、阈值、计数、耗时及 `labels_verified: false` 标记。

当前提示词覆盖杯子、碗、马桶、鼠标、剪刀、书、遥控器、手机、酒杯、笔记本电脑、餐桌、线缆、勺子和叉子。这只是试验词表。保留竞争类别有助于暴露混淆；不把马桶标签强制改成杯子。

类别 ID 使用原 COCO-80 编号（如杯子 41、勺子 44、鼠标 64、书 73、剪刀 76）。线缆没有对应 COCO-80 类别；无法明确映射的词组只记入 JSON，不导出为训练标签。

## 使用限制

几何检查和 NMS 不能证明类别正确。生成的文件都是未验证候选，不能直接当作真实标签开始训练。脚本没有生成训练配置。

这里的已知失败图片和新增测试图片只用于诊断，不自动加入训练。若后续把某些照片用于训练，须另外保留独立照片做最终评估。

训练前还需解决标签可靠性、训练/验证划分和原有类别保留问题；只保留 80 类的编号不等于保留了这 80 类的识别能力。

参考：[Grounding DINO 模型](https://huggingface.co/IDEA-Research/grounding-dino-tiny)、[Transformers 文档](https://huggingface.co/docs/transformers/model_doc/grounding-dino)。

## 本次运行记录

结果目录：`outputs/auto_label_trial/20261022_234121_501554/`。

| 图片 | 导出候选框数 | 检查观察 |
| --- | ---: | --- |
| 02_simple.jpg | 0 | 找到杯子位置，但词组混合为 `a paper cup a toilet`；未强制映射成杯子，跳过导出。 |
| 07_occlusion.jpg | 4 | 检出杯子、鼠标、剪刀各 1 个，并额外检测到背景桌面。 |
| 13.jpg | 4 | 检出鼠标、剪刀、勺子和书；边缘区域有含糊的手机词组，未导出该词组。 |
| 14.jpg | 4 | 检出鼠标、剪刀、勺子和书；边缘区域有遥控器/手机混合词组，未导出该词组；线缆不导出。 |
| 15.jpg | 5 | 检出鼠标、剪刀、勺子和书，另有背景桌面；线缆不导出。 |
| 16.jpg | 5 | 检出了杯子，但把勺子标成遥控器；存在错误候选标签，不能直接用于训练。 |
| 17.jpg | 6 | 检出杯子、鼠标、剪刀、勺子和书，另有背景桌面；线缆不导出。 |

共导出 28 个未验证候选框。独立检查了每行的 5 列格式、COCO-80 编号范围、有限归一化坐标、正宽高，以及还原到像素坐标后的误差（小于 0.01 像素）。格式和坐标检查通过不代表类别全部正确。

对照 YOLO11s（CPU、1280、conf=0.25、iou=0.7、跨类别 NMS）在新增五张图片中多次把勺子识别成剪刀，16 的杯子识别成鼠标，17 的杯子则识别正确。线缆整理并未消除这些分类问题。

结论：自动生成候选标签的流程已运行，但语义可靠性仍不足，未启动训练，未将这些诊断图片加入训练数据，也未替换主程序模型。
