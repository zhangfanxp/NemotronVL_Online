# Nemotron VL 图片与视频分析

一个在 Mac 本机运行的简单网页界面，通过 NVIDIA API 调用 `nvidia/nemotron-3.5-super-vl-120b-a12b`，支持用中文提问并分析图片或视频。

## 功能

- 上传图片并针对图片内容提问。
- 上传视频，按时间均匀抽取画面并提问。模型接收的是带时间标记的抽样帧，不是完整视频流。
- 调整视觉分析温度、最大生成长度和模型思考模式。
- 在浏览器中实时查看模型回复。

## 环境要求

- macOS，Python 3.11 或更高版本。
- NVIDIA Build API Key。请在 [Nemotron 模型页面](https://build.nvidia.com/nvidia/nemotron-3.5-super-vl-120b-a12b) 登录并创建 API Key。
- 推荐使用 [uv](https://docs.astral.sh/uv/) 管理 Python 环境和依赖。

## 安装和启动（推荐：uv）

在终端进入本项目目录：

```bash
cd /Users/zhang/Documents/ChatGPT/NemotronVL
```

编辑项目根目录的 `.env`，把 NVIDIA API Key 粘贴到 `NVIDIA_API_KEY=` 后面：

```dotenv
NVIDIA_API_KEY=nvapi-你的密钥
```

安装依赖并启动：

```bash
uv sync
uv run app.py
```

程序会自动打开浏览器，也可以访问 <http://127.0.0.1:7860>。停止服务时在终端按 `Control-C`。

## 不使用 uv

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

## 使用方法

1. 在“图片分析”或“视频分析”标签中上传文件。
2. 输入想让模型回答的问题。
3. 点击“分析图片”或“分析视频”。

视频抽帧数默认是 8 帧，可以在 2 到 24 帧之间调整。增加帧数可能帮助模型理解更多细节，同时会增加请求体积和处理时间。视频建议使用系统和 OpenCV 可读取的 MP4 或 MOV 编码。

## 配置

`.env` 中可以设置：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `NVIDIA_API_KEY` | 无 | NVIDIA API Key，必填 |
| `NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` | NVIDIA OpenAI 兼容 API 地址 |
| `NVIDIA_MODEL` | `nvidia/nemotron-3.5-super-vl-120b-a12b` | 模型标识 |
| `APP_PORT` | `7860` | 本地网页服务端口 |

图片最大支持 30 MB，视频最大支持 500 MB。项目的 `.gitignore` 已排除 `.env`，请勿把包含真实密钥的文件提交到代码仓库。

## 数据传输说明

网页和视频抽帧在本机处理。图片文件或视频抽取出的画面会通过 NVIDIA API 发送给 NVIDIA 进行模型推理；程序不会把完整视频文件作为视频流上传。请只处理你有权使用的图片和视频，并查看 NVIDIA API 服务条款和隐私说明。

## 依赖文件

- `pyproject.toml`：项目依赖声明。
- `uv.lock`：uv 解析出的依赖锁定版本。
- `requirements.txt`：pip 安装依赖清单。
- `.env.example`：环境变量示例；实际密钥填写在本地 `.env`。
