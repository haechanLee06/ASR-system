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
  - `models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`
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
