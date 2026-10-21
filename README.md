# 桌面物品识别与清点对比系统

使用两种方案独立分析同一张图片，对比物品类别与数量：

- **视觉模型方案**：YOLO11n检测物品并计数，SAM根据检测框分割物体。
- **多模态大模型方案**：调用阿里云百炼的qwen3-vl-plus，直接分析原图并返回物品清单。

项目目前为命令行版本，正在开发交互界面与多图测试功能。

## 已实现功能

- 图片读取、尺寸查看和预览。
- YOLO物品检测、置信度输出与分类计数。
- 检测框坐标输出与单个物品裁剪。
- SAM框提示分割。
- YOLO检测框自动传入SAM，生成分割结果。
- 多模态大模型独立识别图片，并解析JSON回答。
- 保存模型计数结果、API调用耗时及Token用量。
- 按类别自动比较两种方案的数量。
- 使用统一配置文件设置输入图片。

## 项目文件

| 文件 | 用途 |
| --- | --- |
| config.py | 设置两种方案共用的图片路径 |
| detect_and_segment.py | YOLO检测、计数与SAM自动分割 |
| recognize_with_vlm.py | 大模型图片识别及结构化结果保存 |
| compare_results.py | 读取已有结果，比较物品数量 |
| count_items.py | 模拟物品列表的计数练习 |
| read_image.py | 读取并预览图片 |
| detect_items.py | YOLO检测、坐标输出及裁剪练习 |
| segment_item.py | 使用手动指定的框进行SAM分割 |
| notes.md | 学习过程与实验记录 |
| requirements.txt | Python依赖版本 |
| .gitignore | 排除虚拟环境、权重、结果与密钥 |

`.env`、模型权重和`outputs/`为本地文件，不上传到仓库。

## 环境与安装

当前开发环境为Windows、Python 3.10，示例使用CPU推理。

下载仓库后，在项目文件夹打开PowerShell，执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

在VS Code中，选择本项目的`.venv\Scripts\python.exe`作为Python解释器。

## 模型权重

视觉模型方案使用：

- `yolo11n.pt`
- `sam_b.pt`

将权重放在项目根目录。缺少权重时，Ultralytics会尝试自动下载，需要网络连接。

SAM下载较大，若自动下载失败，可从官方资源地址下载完整文件后放入项目目录：

- [YOLO11n](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt)
- [SAM](https://github.com/ultralytics/assets/releases/download/v8.4.0/sam_b.pt)

## 配置图片

修改`config.py`中的路径，指向自己的JPG照片：

```python
IMAGE_PATH = Path(r"C:\Users\你的用户名\Desktop\desk.jpg")
```

主要识别程序均读取这个配置。

早期练习脚本可能仍使用各自代码中的固定路径，运行前需检查。

当前大模型程序按JPEG格式发送图片，其他图片格式的支持待完善。

## 配置大模型API

在项目根目录新建`.env`文件：

```dotenv
DASHSCOPE_API_KEY=填写自己的API密钥
DASHSCOPE_BASE_URL=填写百炼控制台提供的OpenAI兼容地址
```

接口地址应与密钥所属地域、业务空间匹配，使用控制台提供的完整地址。

不要将真实密钥写入代码或提交到GitHub。仓库的`.gitignore`已忽略`.env`。

调用使用`qwen3-vl-plus`，图片会发送到阿里云服务。调用会消耗模型额度；使用前请核对价格、免费额度有效期及“免费额度用完即停”设置。

## 运行步骤

### 1. 运行YOLO检测与SAM分割

```powershell
.\.venv\Scripts\python.exe .\detect_and_segment.py
```

主要输出：

- `outputs/detected.jpg`：检测标注图。
- `outputs/all_segmented.jpg`：分割效果图。
- `outputs/yolo_result.json`：YOLO类别与数量。

SAM根据YOLO检测框生成掩膜，类别和计数来自YOLO。

### 2. 运行大模型识别

```powershell
.\.venv\Scripts\python.exe .\recognize_with_vlm.py
```

程序直接发送原图，不提供YOLO识别结果，独立生成物品清单。

输出`outputs/vlm_result.json`，包含计数、接口调用耗时及Token用量。

### 3. 对比两种方案

```powershell
.\.venv\Scripts\python.exe .\compare_results.py
```

程序读取两份JSON，按类别比较数量，生成`outputs/comparison.json`。

这一步不调用模型，不消耗API额度。

换图片后，应重新运行两种识别程序，再执行对比。

## 初步实验

当前仅对一张包含两个杯子和一个鼠标的照片进行了流程验证。

| 方案或设置 | 计数结果 | 单次耗时示例 |
| --- | --- | --- |
| YOLO，imgsz=640 | 两个杯子，漏检鼠标 | 模型推理60.3 ms |
| YOLO，imgsz=1280 | 两个杯子、一个鼠标 | 模型推理144.6 ms |
| SAM，接收三个检测框 | 生成三个物体主体的分割区域 | 模型推理约7.2秒 |
| qwen3-vl-plus | 两个杯子、一个鼠标 | 接口调用约2.99秒 |

一次结构化识别调用共消耗2629 Token，实际用量随输入和回答变化。

上述数据来自单次运行示例，不代表稳定平均性能。YOLO、SAM记录的是模型推理耗时；大模型记录的是包含网络传输和服务端处理的接口调用耗时，不能直接作为同一口径的速度比较。

两种方案计数一致不代表一定正确，仍需人工核对。

## 当前局限

- 尚未完成网页交互界面和多图片批量实验。
- YOLO仅识别其支持的类别，可能发生漏检或误检。
- SAM分割依赖检测框，不能自动找回YOLO漏检的物品。
- 大模型可能返回错误类别、数量或不符合要求的格式。
- 类别名称尚未建立完整的统一映射，近义词可能被视为不同类别。
- 图片一致性目前只检查路径，尚未检查图片内容。
- YOLO零检测或大模型调用失败时，可能保留上次结果；此时不要直接执行对比，应检查输出是否属于本次运行。
- 当前结果使用固定文件名，多次运行会覆盖旧结果。

## 后续计划

- 完善空结果、调用失败与旧结果处理。
- 增加图片内容校验。
- 开发上传图片及并排展示结果的交互页面。
- 准备多场景测试图片和人工标注。
- 比较计数表现、完整处理耗时及失败案例。
- 整理项目总结、个人感悟与运行视频。

## 参考资料

- [Ultralytics YOLO](https://docs.ultralytics.com/models/yolo11/)
- [Ultralytics SAM](https://docs.ultralytics.com/models/sam/)
- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)
- [阿里云百炼视觉理解](https://help.aliyun.com/zh/model-studio/vision/)

本项目参考相关文档学习模型调用与组合，并逐步实现计数、结果保存和方案对比。