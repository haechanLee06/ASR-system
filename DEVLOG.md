# 后端开发日志（ASR-system）

## 项目概览
- 技术栈：Flask + SQLAlchemy、FFmpeg、WSL2、FunASR（本地模型）
- 目标：实现音频上传→转码→AI识别（时间戳/说话人/四川话）→句段切割→入库→返回前端 JSON
- 运行环境：Windows 主机为主，缺依赖时自动回退到 WSL Ubuntu 环境执行

## 目录与关键路径
- 应用根：`my_voice_project/`
- 路由与工厂：`app/__init__.py`、`app/routes.py`
- 服务模块：
  - 音频：`app/services/audio_handler.py`
  - AI：`app/services/ai_service.py`
- 数据模型：`app/models.py`
- 配置：`config.py`
- 静态资源：
  - 上传：`app/static/uploads/`
  - 切段：`app/static/separated/`
- 模型与输出：
  - 本地模型：`models/`
  - 识别输出：`outputs/`
- 备用路由文件：`app.py`（提供 `/audio/upload` 的独立实现）

## 时间线与重大改动
1. 初始骨架与数据库
   - 创建 Flask 工厂与蓝图，初始化 SQLite，建立 `AudioRecord` 模型
2. 音频上传与转码
   - 封装 FFmpeg，统一转 16kHz/mono WAV，支持 Windows/WSL 双路径
   - 处理空文件校验、子进程编码问题（GBK→UTF-8 容错）
3. FunASR 双模型管线
   - A 管线：通用模型用于句级时间戳+说话人识别（VAD/PUNC/SPK）
   - B 管线：四川话高精度识别（段内精修）
   - 强制本地加载 `models/`，禁用联网下载
4. 后端联通与切段入库
   - 在 `/upload` 路由中串联：转码→AI→切段→入库→返回 JSON
   - 新增 `Split` 表记录切段文件路径与时间范围
5. 调试与容错
   - 加入关键打印（上传、转码、AI 调用、WSL 回退、切段、入库）
   - 容忍 AI 脚本 stdout 混有调试输出，从中提取有效 JSON
6. 跨域支持
   - 安装并启用 `flask-cors`，支持 Vue 前端跨域联调

7. 前后端接口对齐
   - 统一上传路由为 `/upload`（同时兼容 `/audio/upload` 别名）
   - 修正返回 JSON 字段：将数据库 `DialogueSegment.content` 映射为响应 `text`
   - 验证静态资源服务：`/static/separated/<record_id>/<index>.wav` 可被前端直接访问

8. 系统化改造（多用户与认证）
   - 引入 JWT 认证（`flask-jwt-extended`），在 `create_app` 中初始化 `JWTManager`
   - 新增 `User` 模型（`username`、`password_hash`），`AudioRecord` 增加可空 `user_id` 外键
   - 认证接口：`POST /auth/register`、`POST /auth/login`，登录返回 `access_token`
   - 保护业务接口：`POST /upload`、`GET /history` 添加 `@jwt_required()`；上传关联当前用户，查询仅返回该用户记录
   - 兼容旧库：运行时检测表结构是否包含 `audio_record.user_id`，缺失时不强制写入并提示迁移

## 接口与流程
- `GET /`（首页）：列出上传记录（`app/routes.py:11`）
- `POST /upload`（上传主入口）：
  - 保存临时文件 → 转码为 16k WAV → 写 `AudioRecord`
  - 调用 AI（优先 Windows Python，失败则 WSL 回退）
  - 解析 JSON → `ffmpeg` 按 `start/end` 切段入 `static/separated/<record_id>/`
  - 写入 `DialogueSegment`/`Split` → 返回 JSON（含分段文件路径）
- `GET /detail/<record_id>`：查看记录详情（`app/routes.py:56`）
- 备用：`POST /audio/upload`（`app.py` 中的独立版本，用法同上）

## 数据模型
- `AudioRecord`
  - `original_filename`、`filename`（相对路径）、`duration`、`upload_time`、`status`
- `DialogueSegment`
  - `record_id`、`start_time`、`end_time`、`speaker`、`content`、`sentiment`
- `Split`
  - `record_id`、`segment_index`、`file_path`（相对路径）、`start_time`、`end_time`、`speaker`

## 服务实现
- 音频服务（`app/services/audio_handler.py`）
  - `save_upload_file(file)`：落地到 `app/static/uploads/temp/`
  - `convert_to_16k_wav(temp_path)`：FFmpeg 转码为 `app/static/uploads/<ts>_<uuid>.wav` 并返回时长
  - Windows 缺 FFmpeg 时自动使用 WSL（路径转换 `_win_to_wsl`）
- AI 服务（`app/services/ai_service.py`）
  - `AIServiceRunner`
    - `init_models()`：拼接 `models/` 本地路径（A/B/VAD/PUNC/SPK），禁用联网
    - `run(audio_path)`：A 管线获取句级时间戳+说话人，B 管线逐句识别文本，输出标准 JSON
    - 内存切片与重采样：`torchaudio` → numpy 数组输入 B 模型
  - 关键打印：加载路径、模型就绪、推理完成、段数统计、每段处理进度

## 配置与环境
- `config.py`
  - `FFMPEG_BIN`（默认 `ffmpeg`）
  - `FFPROBE_BIN`（可选）
  - `WSL_DISTRO`（默认 `Ubuntu-20.04`）
- 执行策略（`app/utils/wsl_bridge.py`）
  - Linux/WSL：本地直接执行 AI 脚本（使用当前解释器），不再通过 `wsl.exe`
  - Windows：桥接到 WSL，转换路径并调用 `.venv/python` 或 `python3`
- CORS（`app/__init__.py`）
  - `from flask_cors import CORS` → `CORS(app, resources={r"/*": {"origins": "*"}})`

## 调试与日志
- `/upload` 路由打印：
  - `[UPLOAD]`：请求入站、临时文件路径
  - `[CONVERT]`：转码开始与结果（相对路径与时长）
  - `[DB]`：记录入库 ID
  - `[AI]`：子进程命令、返回码与 stdout/stderr 头部
  - `[AI/WSL]`：回退执行的返回码与输出
  - `[CUT]`：每段切割目标路径与时间范围
  - `[DONE]`：事务提交段数
  - `[ERROR]/[CLEAN]`：错误与临时文件清理
- AI 脚本打印：模型路径、管线就绪、推理阶段、段数、完成标记

## 使用与部署
- 启动：
  ```bash
  # Windows（直接python）
  python run.py
  
  # WSL（指定端口并清理占用）
  cd /mnt/c/Users/asus/Desktop/毕业论文/ASR-system/my_voice_project
  source .venv/bin/activate
  fuser -k 5000/tcp || true
  PORT=5000 python run.py
  ```
- 上传：页面表单或前端 POST 到 `/upload`
- WSL 依赖准备（如需）：
  ```bash
  sudo apt update && sudo apt install -y python3 python3-venv ffmpeg
  cd /mnt/c/Users/asus/Desktop/毕业论文/ASR-system/my_voice_project
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```
- 本地模型放置：
  - `models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch`
  - `models/iic/speech_fsmn_vad_zh-cn-16k-common-vocab8404-pytorch`
  - `models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch`
  - `models/iic/speech_campplus_sv_zh-cn_16k-common`
  - `models/lukeewin01/paraformer-large-sichuan-offline`

## 注意事项
- 在 WSL 内直接运行后端时，`wsl_bridge.py` 走本地执行路径，避免重复 `/mnt` 前缀导致的路径错误
- Windows 缺 `funasr` 时自动回退到 WSL 执行；确保 WSL `.venv` 安装了所需依赖
- AI 脚本输出含调试时，后端会从 stdout 中提取最后一个合法 JSON；若解析失败，查看终端头部日志定位问题
- `.gitignore` 已忽略本地 venv、模型与输出目录，避免提交庞大文件

9. 详情接口
   - 新增 `GET /record/<int:record_id>`，需要认证
   - 校验记录归属当前用户并按开始时间返回分段详情
   - 返回 `info` 与 `segments`，段内 `path` 为相对路径供前端拼接

10. 识别体验优化
   - 引入独立标点恢复层：加载 `punc_ct-transformer_zh-cn-common-vocab272727-pytorch`，在四川话识别后与段合并后分别执行标点恢复
   - 调整 VAD 参数：`max_single_segment_time=60000`、`max_end_silence_chunk=800`，提升停顿容忍度，降低过度切割
   - 新增后处理合并：相邻同说话人且间隔 < 0.3s 的短片段合并；若上一段以 `？/！` 结尾则不合并；合并后再进行标点恢复
   - 文本清洗：正则去重堆叠标点，修复怪异组合，去除行首标点

11. 异步架构与数据管理
   - 架构升级：`POST /upload` 改为异步非阻塞模式
     - 主线程：保存文件 → 创建 `pending` 记录 → 启动后台线程 → 立即返回 `{ record_id, status: 'pending' }`
     - 后台线程：更新状态 `processing` → 执行 AI 识别 → 切割分段 → 更新状态 `success` 或 `failed`（记录 `error_message`）
     - 实现方式：使用 Python 原生 `threading` 模块，无需额外消息队列组件
   - 状态管理：
     - `AudioRecord` 新增 `status` 字段（pending/processing/success/failed）
     - `AudioRecord` 新增 `error_message` 字段用于记录失败原因
     - 包含数据库迁移脚本 `migrate_db.py` 以支持旧库升级
   - 接口增强：
     - 新增 `DELETE /record/<int:record_id>`：同时删除数据库记录、原始音频文件、分段文件夹
     - 更新 `GET /record/<int:record_id>`：返回体包含任务状态、错误信息与 `current_stage` 字段，前端可展示细粒度处理阶段（如“正在转码音频格式...”、“AI 模型正在推理 (首次运行需下载模型)...”、“处理完成”等）

12. 说话人分割与模型参数优化
   - 说话人分割升级：彻底切换为 3D-Speaker 原生 `infer_diarization` 接口，不再依赖 ModelScope pipeline，解决单说话人问题。
   - ASR 识别优化：针对四川话方言调整 VAD 参数（阈值 0.3，最大切分 60s，保留 200ms 上下文），提升微弱语气词识别率。
   - 接口完善：`GET /record/<id>` 返回 `audio_url` 字段（Web 相对路径），修复前端播放索引错位问题。
   - 日志增强：增加 WSL 桥接日志回显，支持在 Windows 控制台查看 AI 引擎的 stderr 输出。

13. 对话文本整合功能
   - 在 `AIServiceRunner` 中新增 `format_transcript(segments)` 方法
   - 支持将切片列表转换为三种视角的文本：
     - `full_transcript`: 按时间顺序拼接的完整对话（格式："spkX: 内容\n"）
     - `spk0_transcript`: 仅 spk0 的内容（纯文本）
     - `spk1_transcript`: 仅 spk1 的内容（纯文本）
   - 该方法作为独立功能模块，为接入 LLM 摘要/问答或前端纯文本展示做准备，不影响现有识别与切片流程。

14. 核查页与 LLM 数据标准化
   - 新增 API `GET /api/transcript_data/<file_id>`
   - 数据清洗与增强：
     - `seq_id`: 添加对话轮次序号
     - `role`: 强制映射 spk0->Role A, spk1->Role B
     - `side`: 根据角色分配 left/right 布局属性
   - 生成标准化 LLM 上下文：`[seq_id] 角色{role}: {text}`

15. 用户自纠功能接口
   - 新增 `PUT /record/<int:record_id>/segment/<int:segment_index>`，需要认证
   - 只更新 `DialogueSegment` 内的 `content` 字段。
   - 利用 `start_time` 升序获取正确索引，不触发音频重新切割。

16. 接入本地 Ollama 大模型进行对话总结
   - 新增 `POST /api/summary/<int:record_id>`，需要 `jwt` 认证，校验记录归属。
   - 请求本地的 `qwen-sichuan-psych` 模型进行结构化的总结分析。
   - 将模型按规定 Prompt 返回的分析 JSON 进行验证并持久化存储在 `AudioRecord.llm_summary` 字段供后续快速查询。

17. Dashboard 大盘状态接口
   - 新增 `GET /api/dashboard/ambient`：返回地理位置、天气、气温（通过高德地图免费天气 API）、以及服务持续运行时间（天数/小时）。
   - 新增 `GET /api/dashboard/health`：实时探测 Ollama（GET 127.0.0.1:11434）和 Paraformer 模型目录存在性，返回 `llm_online` 和 `asr_online` 标志供前端展示。
   - 服务启动时间记录在 `app.config['SERVER_START_TIME']`（`create_app` 内写入）。
   - 天气 API Key 改为免费方案：`ip-api.com`（IP 定位）+ `Open-Meteo`（天气），无需注册和密钥。

18. Dashboard 业务速览接口（多租户数据隔离重构）
   - 接口路径：`GET /api/dashboard/stats`，新增 `@jwt_required()` 强制鉴权。
   - `total_transcribed`：严格统计**当前用户**下状态为 `success` 的记录总数。
   - `total_summarized`：严格统计**当前用户**下已完成 `llm_summary` 的记录总数。
   - `uptime_hours`（累计护航时长）：改为统计**当前用户**所有处理成功记录的 `duration` 总和（单位：小时），实现业务层面的数据完全私有化。
