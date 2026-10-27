# 桌面物品识别与清点对比系统（Stage 2）

本项目用两条独立流程分析同一张照片：**本地 YOLO11s + SAM** 输出检测框、计数与分割图；**阿里云 qwen3-vl-plus** 输出物品类别和数量。Gradio 页面支持上传、对比和下载 JSON，最终页面只使用一个固定 YOLO 配置。

当前版本已完成可运行流程和开发样本验证，不要将程序能运行理解为识别准确率达标。完整结论和两种方案的体验比较见 [PROJECT_REPORT.md](PROJECT_REPORT.md)。

## 快速运行

Windows / Python 3.10，在项目目录打开 PowerShell：

```powershell
cd D:\lingjing.data\stage2-vision
.\.venv\Scripts\python.exe .\app.py
```

浏览器打开 `http://127.0.0.1:7861`。若旧版本仍在运行，先在原终端按 `Ctrl+C`，再启动本版；不要同时启动两个占用 7861 的进程。

新环境安装：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

项目根目录需要本地权重 `yolo11s.pt` 与 `sam_b.pt`。当前工作目录已具备两者；新环境请根据 [YOLO11 官方说明](https://docs.ultralytics.com/models/yolo11/)和 [SAM 官方说明](https://docs.ultralytics.com/models/sam/)准备权重。主网页缺少权重时返回处理失败，不静默下载大文件。

操作顺序：

1. 上传 JPG/PNG 等可读取图片，点击“检测并分割”。首次调用会加载模型，CPU 上 SAM 通常比仅 YOLO 检测慢。
2. 需要对比时，再点击“大模型识别”。此操作会上传当前图片并消耗额度；本地检测与对比按钮不调用云端。
3. 点击“比较当前结果”，查看各类别数量差异；按需下载三种 JSON。
4. 换图会清空当前结果；比较时还会核对图片内容哈希，拒绝把不同照片的结果拼在一起。

## 配置大模型

仅使用本地识别时不需要 API 密钥。需要云端识别时，在项目根目录创建 `.env`：

```dotenv
DASHSCOPE_API_KEY=填写自己的密钥
DASHSCOPE_BASE_URL=填写控制台提供的OpenAI兼容接口地址
```

接口地址与密钥的地域/业务空间需要匹配。请自行在百炼控制台查看当前剩余额度；程序不能读取账户余额。主界面使用 `qwen3-vl-plus`，输出上限 500 tokens、无自动重试，返回信息中显示服务端提供的 token 用量。图片按 JPEG 编码发送；输出上限不是整个请求的输入加输出 token 上限。


## 固定配置与输出

固定检测配置：`yolo11s.pt`、CPU、`imgsz=1280`、`conf=0.25`、`iou=0.7`、`agnostic_nms=True`。保留预训练模型完整的 80 类，不把类别限制为杯子、鼠标、剪刀；例如书和碗仍可识别，但支持类别不等于每张图都识别正确。

SAM 使用 YOLO 的框生成掩膜。**类别与数量来自 YOLO；SAM 不会纠正错误类别，也不会找回没有检测框的杯子。** 分割图上的数字是分割输出标识，物品名称应看 YOLO 图和统计。

每次成功处理单独保存在 `outputs/app_runs/<时间>_<local|cloud|comparison>/`：

| 文件 | 内容 |
|---|---|
| `result.json` | 方案、图片内容哈希、计数、处理耗时与相关记录；本地结果还包含框与置信度 |
| `input.png` | 本地流程处理的 RGB 图像副本 |
| `detected.png` | 本地检测图 |
| `segmented.png` | 有检测框时生成的 SAM 图 |

云端 JSON 保存返回摘要，其中包含接口耗时和服务端返回的 token 用量（如有）。对比 JSON 引用两份源结果。文件保存在本机，`.gitignore` 已排除 `outputs/`、`.env`、虚拟环境与权重。

页面本地计时包含预处理、YOLO、计数、SAM 和绘图，不含模型加载、队列等待、文件写入与浏览器渲染；首次模型加载单独显示。云端完整计时包括预处理、编码、客户端初始化、完整请求与解析，同样不含队列等待、写文件与浏览器渲染。首次推理初始化可能计入本地处理时间，比较速度需做多次热身后重复测试，不能拿 YOLO 单模型毫秒数对比完整 API 秒数。

## 结果与限制

2026-10-23 在 8 张已知开发图片上使用一致参数比较：

| 配置 | 整图计数完全一致 | 类别计数绝对误差总和 |
|---|---:|---:|
| YOLO11n，跨类别去重关闭 | 3/8 | 10 |
| YOLO11n，跨类别去重开启 | 3/8 | 10 |
| YOLO11s，跨类别去重开启 | 4/8 | 9 |

这些照片已经用于调试，不是新验收集；4/8 也不代表总体准确率。选择 11s 是这批样本上的开发折中：05、07、08 有改善，但 03 从正确变成错误；02 的杯子仍被判为 toilet，06 仍有杯子误判，07 仍漏杯子。

两组先前微调都没有超过训练前 YOLO11s，训练权重未接入。后续自动标注实验也未产出足够可靠的新标签，因此停止用错误标签追加训练。原照片、原权重和失败结果保留。

## 文件与实验入口

| 文件 | 用途 |
|---|---|
| `app.py` | 当前网页入口；上传、双方案结果、下载和比较 |
| `vision_pipeline.py` | 固定本地模型、YOLO+SAM、内容哈希、计时和结果保存 |
| `vlm_service.py` | 用户点击时独立调用 qwen3-vl-plus |
| `verify_delivery.py` | 临时 7862 页面上的真实本地接口测试，云端仅模拟 |
| `select_delivery_baseline.py` | 复现现有固定配置的已知开发图对照 |
| `PROJECT_REPORT.md` | 构建过程、方案体验、适用场景、失败与证据 |
| `TRAINING_PILOT.md` | 两轮训练与验证、未采用原因 |
| `LABEL_VIEW_DIAGNOSTIC.md` | 自动标签分歧、分块实验和双向审核修复 |
| `AUTO_LABEL_TRIAL.md`、`notes.md` | 保留历史探索过程 |

`detect_and_segment.py`、`recognize_with_vlm.py`、`compare_results.py` 等是早期命令行练习，可能使用不同模型或固定输出文件；正式演示使用 `app.py`，不要混用旧 CLI 结果与新网页记录。

重跑本地交付测试（不会请求云端）：

```powershell
.\.venv\Scripts\python.exe -B .\verify_delivery.py
```

测试涉及真实 YOLO+SAM 推理，会写入本地结果，结束后关闭临时 7862 服务，不关闭已运行的 7861 页面。

## 数据边界与后续验收

- `training_photos/` 是训练候选照片，含用户允许使用的 AI 图；没有因 AI 来源一概排除。
- `test_images/` 目前是已知开发样本；旧充电宝图 04 不进入本轮正式计数统计。
- 公开 COCO 训练/验证拆分虽互不重复，但 COCO 已参与预训练，不能称为完全独立新场景验证。
- 自动标注采用双向匹配，未知/多余/不一致候选暂缓使用；模型一致也不证明正确。
- 不要求用户重新手工画框。若以后继续训练，需要更贴近失败场景的可靠现成或自动验证标签，并保留其他类别能力。
- 独立验收仍需未参与调试的新实拍图。云端同条件重复测试和演示录屏尚未执行，不能编造成已完成。

## 参考

- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [Ultralytics SAM](https://docs.ultralytics.com/models/sam/)
- [COCO 数据说明](https://docs.ultralytics.com/datasets/detect/coco/)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [阿里云百炼视觉理解](https://help.aliyun.com/zh/model-studio/vision/)

本项目仅记录 Stage 2 的实现与实验，不混入 Stage 1 内容。
