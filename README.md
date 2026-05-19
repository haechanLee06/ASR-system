# ASR-System 后端服务 (Backend)

本项目是 ASR（自动语音识别）系统的后端服务部分。后端采用 Python + Flask 构建，集成了基于 FunASR 的四川话语音识别、3D-Speaker 说话人分离、CAM++ 声纹识别，以及基于本地大语言模型（Ollama/Qwen）的心理状态分析与对话摘要功能。

## 🛠 技术栈

- **Web 框架**: Flask, Flask-RESTful, Flask-CORS
- **数据库 & ORM**: SQLite, Flask-SQLAlchemy
- **身份认证**: Flask-JWT-Extended
- **AI 模型集成**:
  - **语音识别 (ASR)**: FunASR (定制四川话离线模型)
  - **说话人分离 (Diarization)**: 3D-Speaker
  - **声纹识别 (VoicePrint)**: CAM++ (192维特征向量提取与匹配)
  - **大语言模型 (LLM)**: Ollama (本地部署 qwen-sichuan-psych 心理分析模型)
- **音频处理**: FFmpeg (通过 Python subprocess 及 WSL 调用)

## 📂 项目结构

```text
my_voice_project/
├── app/                            # Flask 应用核心模块
│   ├── __init__.py                 # 应用初始化与工厂函数 (create_app)
│   ├── models.py                   # 数据库表模型定义 (SQLAlchemy ORM)
│   ├── routes.py                   # 主业务路由 (音频上传、记录查询、工作台数据等)
│   ├── routes_auth.py              # 认证路由 (注册、登录)
│   ├── services/                   # AI 服务调度层
│   │   ├── ai_service.py           # 语音识别与说话人分离处理流
│   │   ├── audio_handler.py        # 音频上传预处理、FFmpeg 格式转换
│   │   ├── diarization.py          # 封装 3D-Speaker 接口
│   │   ├── llm_service.py          # 大语言模型接口封装 (Ollama)
│   │   └── voiceprint_service.py   # CAM++ 声纹特征提取独立服务
│   ├── static/                     # 静态资源存储
│   │   ├── uploads/                # 原始音频存放区
│   │   ├── separated/              # 按角色切割的分段音频存放区
│   │   └── voiceprints/            # 入库声纹特征矩阵文件 (.npy)
│   └── utils/                      # 工具类 (如 wsl_bridge 跨平台调用)
├── 3D-Speaker/                     # 3D-Speaker 官方库依赖包
├── models/                         # 本地离线 AI 模型权重存放目录 (未加入版本控制)
├── config.py                       # 全局配置文件
├── run.py                          # Flask 服务启动入口
├── setup_env.sh                    # 环境自动化配置脚本
├── migrate_db.py                   # 数据库结构升级/迁移脚本
├── requirements.txt                # Python 基础依赖表
└── voice_data.db                   # SQLite 本地业务数据库
```

## 🚀 快速启动

1. **环境准备**
   请确保系统已安装 [Anaconda](https://www.anaconda.com/) 或 Miniconda，以及 FFmpeg。

2. **自动配置环境**
   在项目根目录下运行环境搭建脚本，将自动创建 `asr_sys` 环境并安装所需的 PyTorch 及所有依赖：
   ```bash
   bash setup_env.sh
   ```

3. **激活环境**
   ```bash
   conda activate asr_sys
   ```

4. **启动服务**
   运行应用主入口：
   ```bash
   python run.py
   ```
   默认启动在 `http://0.0.0.0:5000`。

## 📖 核心业务流程

1. **音频接入**：通过 `POST /upload` 上传音频，保存后交由 FFmpeg 统一转码为 16kHz, 单声道 WAV 格式。
2. **AI 分析流**：
   - 使用 **3D-Speaker** 进行 Diarization，识别不同的发言人时间戳。
   - 使用 **FunASR** 配合定制模型，针对切分后的音频段进行高精度四川话识别。
   - 通过 **CAM++** 对提取的音频段进行声纹匹配，映射真实的注册用户身份。
3. **内容洞察**：将各分段对话汇总，通过本地部署的 **Qwen LLM** 生成深度心理学分析和内容摘要。
4. **数据持久化**：所有的中间状态、文字稿、声纹特征和结构化摘要，均持久化存入 SQLite，供前端通过 REST API 获取并展示。

## 📄 文档参考

如需查看具体的底层架构设计、数据库 E-R 详情、全量 API 接口规格说明，请参考本目录下的 `backend.md`。
