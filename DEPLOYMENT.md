# Linux / Docker 部署（Stage 3）

第三阶段沿用第二阶段仓库和识别流程，新增 Linux 容器部署。浏览器上传一张图片后，可分别运行固定的 YOLO11s + SAM 和 qwen3-vl-plus，再比较计数、耗时并下载 JSON。模型比较、误判与训练历史仍见原有报告。本轮实际验证与未完成事项见 [DEPLOYMENT_VALIDATION.md](DEPLOYMENT_VALIDATION.md)。

## 1. 部署结构

```text
Windows 浏览器（本机验证）或 SSH 转发后的浏览器
  -> http://127.0.0.1:7863
  -> Docker 端口映射 7863:7861
  -> Gradio 网页及 Python 回调（容器内 0.0.0.0:7861）
       ├─ 本地按钮 -> YOLO11s -> SAM -> 检测图、分割图与 JSON
       └─ 大模型按钮 -> 阿里云兼容 API -> 物品统计 JSON
```

- 一个 `web` 服务；CPU 推理，不需要 CUDA。容器以普通用户运行。
- `yolo11s.pt` 和 `sam_b.pt` 从项目目录只读挂载，缺文件时 Compose 明确失败，不偷偷下载模型。
- `.env` 仅在容器启动时注入；可以不配置，此时本地流程仍可用。真实密钥不写入镜像或仓库。
- `app_results` 命名卷保存 `outputs/app_runs/`，普通重启和 `docker compose down` 后保留。不要使用 `down -v`，除非确实要删除这些结果。
- 网页监听和端口可配置；直接运行 Windows `app.py` 时仍默认 `127.0.0.1:7861` 并自动打开浏览器。
- 镜像只包含三个运行模块和依赖；训练照片、测试图片、历史结果、权重、Git 和 Windows 虚拟环境都不会进入构建上下文。

## 2. 当前本机环境

本轮在 Windows + WSL2 Ubuntu 22.04 中使用 Docker Engine，而不是 Docker Desktop。Docker 镜像基于 `python:3.10-slim-bookworm`，容器中的用户空间是 Debian Bookworm，不等同于外层 Ubuntu。

- 已安装：Docker Engine 29.9.2、Compose 5.6.1；本轮目标平台为 `linux/amd64`，其他架构未验证。
- WSL 可用内存约 7.6 GiB，另有 2 GiB swap；YOLO 和 SAM 均使用 CPU。
- Python 3.10；PyTorch 2.14.1 CPU / torchvision 0.29.1 CPU。
- YOLO 配置沿用 `vision_pipeline.py`，不更换类别、不改计数算法。
- Gradio 6.30.0、Ultralytics 8.4.183 等运行版本沿用原环境；`requirements.txt` 约束传递依赖，`requirements-docker.txt` 只选择网页所需依赖。

这是本机 Linux 容器验证。若考核要求独立云服务器或学校服务器，仍需将同一仓库部署到相应服务器并保存运行证据。

## 3. 准备项目

在 **Ubuntu 终端**进入已有项目：

```bash
cd /mnt/d/lingjing.data/stage2-vision
```

在新的 Linux 服务器上，则先安装 Docker Engine 和 Compose 插件，再获取仓库：

```bash
git clone https://github.com/mrh02/desktop-object-recognition-and-counting-comparison.git
cd desktop-object-recognition-and-counting-comparison
```

新服务器需要另外复制两份权重到仓库根目录：`yolo11s.pt`、`sam_b.pt`。权重不通过 Git 上传；可从已有项目安全传输，或按照 [YOLO11](https://docs.ultralytics.com/models/yolo11/) 与 [SAM](https://docs.ultralytics.com/models/sam/) 官方说明单独下载。本机已有权重，无需重复下载。Compose 的可选 `env_file` 写法要求 2.24.0 或更新版本。

需要云端识别时，把 `.env.example` 复制成 `.env` 并在编辑器填写自己的 `DASHSCOPE_API_KEY` 和 `DASHSCOPE_BASE_URL`。接口地域与密钥应匹配。不要把 `.env` 的内容贴到终端输出、录屏或仓库。

## 4. 构建与启动

网络可以直接访问 Docker Hub、Debian、PyPI 和 PyTorch 官方源时：

```bash
sudo docker compose build
sudo docker compose up -d --no-build --pull never
sudo docker compose ps
```

本机 Windows 代理为 `127.0.0.1:7897`，WSL 已启用镜像网络。Docker 后台已另行配置代理以便拉取镜像；这不等于构建步骤中的 apt/pip 自动有代理。本机首次构建使用：

```bash
sudo docker build --network=host --allow=network.host \
  --build-arg HTTP_PROXY=http://127.0.0.1:7897 \
  --build-arg HTTPS_PROXY=http://127.0.0.1:7897 \
  --build-arg NO_PROXY=localhost,127.0.0.1,::1 \
  -t desktop-vision:stage3 .
sudo docker compose up -d --no-build --pull never
sudo docker compose ps
```

这里主机网络仅用于构建时访问本机代理；运行服务仍用 Compose 默认桥接网络。代理参数不写入 Dockerfile 的 `ENV`，不会成为应用运行设置。其他机器不要照搬 `7897`，按它自己的网络情况选择普通构建或正确代理。

浏览器打开 **http://127.0.0.1:7863**。端口 7863 与原 Windows 页面 7861 分开，不会替换或关闭原页面。

第一次安装依赖需要下载 CPU 版 PyTorch 和其他软件，可能耗时较长；后续构建可利用 Docker 缓存。

## 5. 使用与验证

1. 上传图片，点击“检测并分割”，检查 YOLO 图、SAM 图、统计和可下载 JSON。
2. 需要实际云端对比时才点击“大模型识别”；这会上传当前照片并消耗 API 额度。
3. 两份结果都成功后，点击“比较当前结果”。类别与数量相同不代表识别一定正确。
4. 更换照片会清空之前的结果，防止比较不同照片。

查看服务及日志：

```bash
sudo docker compose ps
sudo docker compose logs --tail=50 web
```

健康检查只确认网页可以响应，不保证模型加载、检测准确率或外部 API 可用；这些需要分别验证。部署验证不会自动执行真实大模型调用。

项目带有 `verify_docker.py`，从浏览器同一地址通过 Gradio HTTP API 验证上传、YOLO+SAM、文件下载和空输入清理，不调用云端。已有 Windows 虚拟环境中可运行：

```powershell
.\.venv\Scripts\python.exe -B .\verify_docker.py
```

默认使用已有的 `test_images/05_multiple.jpg` 和 `11_empty.jpg`；也可在脚本后传入自己的图片路径。验证报告写入 `outputs/docker_checks/`。这验证软件流程，不等于新场景准确率验收。

将持久化结果复制回项目文件夹（运行中也可复制）：

```bash
mkdir -p outputs/docker_app_runs
sudo docker compose cp web:/app/outputs/app_runs/. ./outputs/docker_app_runs/
```

停止服务：

```bash
sudo docker compose down
```

修改 Python 代码后重新构建镜像，再执行 `up -d --no-build --pull never`。仅修改 `.env` 时无需构建，但要执行 `sudo docker compose up -d --force-recreate --no-build --pull never` 才会重新注入配置。

## 6. 远程服务器访问

默认只发布到服务器的 `127.0.0.1`。在自己电脑上可用 SSH 转发访问：

```bash
ssh -L 7863:127.0.0.1:7863 用户名@服务器地址
```

然后在自己电脑浏览器打开 `http://127.0.0.1:7863`。这样无需把含有云端调用按钮的演示页面直接公开。

若确需局域网/公网访问，可通过 `APP_BIND_ADDRESS` 和 `APP_PORT` 配置端口发布；公网部署前应在反向代理或应用上配置访问控制，防止别人使用自己的 API 额度。示例中不开放公网端口。

## 7. 提交材料

沿用当前 GitHub 仓库。提交部署文件、启动代码调整、本说明和验证说明；`.env`、权重、`.venv`、测试输出不提交。用户继续通过 VS Code 自己提交与同步。

项目说明应区分“本机 WSL 验证”和“独立服务器部署”；演示视频可展示 `docker compose ps`、网页上传、双方案结果和 JSON 下载。真实 API 演示需要自行点击并记录实际用量，不能用本地模拟测试冒充。

## 参考

- [Docker Engine 安装到 Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
- [Docker 后台代理](https://docs.docker.com/engine/daemon/proxy/)
- [Compose 环境变量](https://docs.docker.com/compose/how-tos/environment-variables/set-environment-variables/)
- [PyTorch CPU 安装](https://pytorch.org/get-started/locally/)
