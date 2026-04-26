# ASR-System 后端底层架构与 API 深度报告

> **生成时间**：2026-04-26  
> **代码版本**：基于声纹识别持久化方案（2026-04 最新迭代）  
> **用途**：毕业论文数据库E-R图、系统时序图、算法流程图绘制参考  
> **技术栈**：Python 3.x · Flask · Flask-SQLAlchemy · Flask-JWT-Extended · Flask-CORS · SQLite · FunASR · 3D-Speaker · Ollama · CAM++(VoicePrint)

---

## 第一章：系统架构总览

### 1.1 项目目录结构

```
my_voice_project/
├── run.py                          # 程序入口（调用 create_app）
├── config.py                       # 全局配置类 Config
├── voice_data.db                   # SQLite 数据库文件（生产数据）
├── migrate_db.py                   # 数据库迁移脚本（手动执行）
├── DEVLOG.md                       # 开发日志（记录全部19个迭代阶段）
│
├── app/                            # Flask 应用包
│   ├── __init__.py                 # 应用工厂 create_app()
│   ├── models.py                   # SQLAlchemy ORM 模型（4个表）
│   ├── routes.py                   # 主蓝图 main（核心业务路由，858行）
│   ├── routes_auth.py              # 认证蓝图 auth（注册/登录）
│   │
│   ├── services/
│   │   ├── ai_service.py           # AI推理核心（AIServiceRunner类）
│   │   ├── audio_handler.py        # 音频处理（上传保存/FFmpeg转码）
│   │   ├── diarization.py          # 说话人分割（3D-Speaker封装）
│   │   └── llm_service.py          # 大模型服务（Ollama流式请求）
│   │
│   ├── utils/
│   │   └── wsl_bridge.py           # Windows/WSL跨平台执行桥接
│   │
│   ├── static/
│   │   ├── uploads/                # 上传音频存储目录
│   │   │   └── temp/               # 临时文件暂存区（处理后自动清除）
│   │   ├── separated/              # 切割分段音频存储（按记录ID分目录）
│   │   ├── voiceprints/            # 声纹库注册特征存储 (.npy)
│   │   └── record_speakers/        # 单次录音提取的各说话人特征 (.npy)
│   └── templates/                  # Jinja2 HTML 模板
│
├── models/                         # 本地AI模型目录（不提交Git）
│   ├── iic/
│   │   ├── speech_fsmn_vad_zh-cn-16k-common-pytorch          # VAD模型
│   │   └── punc_ct-transformer_zh-cn-common-vocab272727-pytorch # 标点恢复
│   └── lukeewin01/
│       └── paraformer-large-sichuan-offline                   # 四川话ASR模型
│
└── 3D-Speaker/                     # 3D-Speaker说话人分割库（子项目）
    └── speakerlab/bin/
        └── infer_diarization.py    # Diarization3Dspeaker 类入口
```

### 1.2 技术架构图（分层视图）

```
┌─────────────────────────────────────────────────────────────────────┐
│                         前端 (Vue 3 SPA)                             │
│           通过 Axios 发起 HTTP/REST 请求，Bearer JWT 鉴权             │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTP/JSON
┌──────────────────────────────▼──────────────────────────────────────┐
│                   Flask 应用层 (my_voice_project/)                   │
│  ┌─────────────┐  ┌──────────────────────────────────────────────┐  │
│  │ auth 蓝图    │  │                main 蓝图                     │  │
│  │/auth/register│  │/upload  /record/<id>  /history              │  │
│  │/auth/login   │  │/api/summary/<id>  /api/voiceprint/*         │  │
│  └─────────────┘  └─────────────────┬────────────────────────────┘  │
│                                     │                                │
│  ┌──────────────────────────────────▼─────────────────────────────┐ │
│  │                    Services 服务层                               │ │
│  │  audio_handler.py │ ai_service.py │ diarization.py │ llm_service │ │
│  └──────────────────────────────────┬─────────────────────────────┘ │
│                                     │                                │
│  ┌──────────────────────────────────▼─────────────────────────────┐ │
│  │              SQLAlchemy ORM + SQLite (voice_data.db)            │ │
│  │ User | AudioRecord | DialogueSegment | Split | VoicePrint | RS  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                     │                                │
│  ┌──────────────────────────────────▼─────────────────────────────┐ │
│  │              外部进程层（subprocess / threading）                │ │
│  │  FFmpeg(转码) │ FunASR(四川话ASR) │ 3D-Speaker(说话人分割)       │ │
│  │  Ollama:11434 (qwen-sichuan-psych LLM)                         │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 核心配置（config.py / Config 类）

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `SECRET_KEY` | `str` | `dev-secret-key` | Flask Session 签名密钥 |
| `SQLALCHEMY_DATABASE_URI` | `str` | `sqlite:///voice_data.db` | SQLite 数据库路径（绝对路径拼接） |
| `SQLALCHEMY_TRACK_MODIFICATIONS` | `bool` | `False` | 禁用变更追踪，节省内存 |
| `AI_SPEAKER_URL` | `str` | `http://127.0.0.1:3000/segment` | 历史遗留配置，当前已改用3D-Speaker本地调用 |
| `AI_BERT_URL` | `str` | `http://127.0.0.1:10086/textProcess` | 历史遗留配置 |
| `FFMPEG_BIN` | `str` | `ffmpeg` | FFmpeg 可执行文件路径 |
| `FFPROBE_BIN` | `str` | `ffprobe` | FFprobe 可执行文件路径 |
| `WSL_DISTRO` | `str` | `Ubuntu-20.04` | WSL 发行版名称 |
| `AMAP_KEY` | `str` | `""` | 高德地图API Key（当前已切换为免费方案，此项已弃用） |
| `JWT_SECRET_KEY` | `str` | `dev-secret-change-me` | JWT 签名密钥（在 `create_app` 中 `setdefault` 写入） |
| `SERVER_START_TIME` | `datetime` | 服务启动时刻 | （旧逻辑预留）当前 `/api/dashboard/ambient` 已改用数据库累计时长作为运行时间 |

---

## 第二章：数据库物理模型（E-R 详细说明）

### 2.1 SQLite 数据库基础信息

- **数据库引擎**：SQLite 3
- **ORM 框架**：Flask-SQLAlchemy
- **数据库文件路径**：`my_voice_project/voice_data.db`
- **创建方式**：`db.create_all()`（在 `create_app` 的 `app_context` 中自动执行）
- **迁移策略**：运行时检测 `audio_record` 表是否有 `user_id` 列，缺失时执行 `ALTER TABLE ... ADD COLUMN`（兼容旧库）

---

### 2.2 表 1：`user`（对应模型类 `User`）

**SQLAlchemy 定义文件**：`app/models.py:4-7`

| 字段名 | SQLAlchemy 类型 | 物理类型（SQLite） | Primary Key | Unique | Nullable | Default | 说明 |
|--------|-----------------|---------------------|-------------|--------|----------|---------|------|
| `id` | `db.Integer` | `INTEGER` | ✅ 主键 | — | NOT NULL | 自增 | 用户唯一ID |
| `username` | `db.String(80)` | `VARCHAR(80)` | ❌ | ✅ 唯一 | NOT NULL | — | 登录用户名 |
| `password_hash` | `db.String(256)` | `VARCHAR(256)` | ❌ | ❌ | NOT NULL | — | Werkzeug `generate_password_hash` 生成的哈希值 |

**外键关系**：无（被 `AudioRecord.user_id` 引用）

---

### 2.3 表 2：`audio_record`（对应模型类 `AudioRecord`）

**SQLAlchemy 定义文件**：`app/models.py:9-31`

| 字段名 | SQLAlchemy 类型 | 物理类型（SQLite） | Nullable | Default | 说明 |
|--------|-----------------|---------------------|----------|---------|------|
| `id` | `db.Integer` | `INTEGER` | NOT NULL | 自增 | 主键 |
| `user_id` | `db.Integer, ForeignKey("user.id")` | `INTEGER` | ✅ 可为空 | NULL | 归属用户ID（外键；nullable=True 兼容迁移前旧数据） |
| `original_filename` | `db.String(256)` | `VARCHAR(256)` | NOT NULL | — | 用户上传时的原始文件名，如 `test.mp3` |
| `filename` | `db.String(256)` | `VARCHAR(256)` | NOT NULL | — | 服务器存储的相对路径，如 `static/uploads/xxx.wav`；异步模式下初始值为 `"pending"` |
| `duration` | `db.Float` | `REAL` | ✅ 可为空 | `0.0` | 音频时长（秒），FFprobe 或 wave 模块读取 |
| `upload_time` | `db.DateTime` | `DATETIME` | ✅ 可为空 | `datetime.utcnow` | 记录创建时间（UTC） |
| `status` | `db.String(20)` | `VARCHAR(20)` | ✅ 可为空 | `"pending"` | 任务状态：`pending` / `processing` / `success` / `failed` |
| `error_message` | `db.Text` | `TEXT` | ✅ 可为空 | NULL | 失败时的错误信息 |
| `current_stage` | `db.String(100)` | `VARCHAR(100)` | ✅ 可为空 | `"等待处理"` | 细粒度阶段描述，前端轮询展示（如"正在转码音频格式..."） |
| `llm_summary` | `db.Text` | `TEXT` | ✅ 可为空 | NULL | LLM 生成的 JSON 格式心理分析报告（序列化字符串） |
| `voiceprint_status` | `db.Integer` | `INTEGER` | ✅ 可为空 | `0` | 声纹匹配状态：0-未匹配，1-已处理映射 |

**外键约束**：
- `audio_record.user_id → user.id`  
- 注意：无 `CASCADE DELETE`（SQLAlchemy 层面未声明），删除 `AudioRecord` 时需手动 `DELETE` 关联的 `DialogueSegment` 和 `Split`（见 `routes.py:309-310`）

---

### 2.4 表 3：`dialogue_segment`（对应模型类 `DialogueSegment`）

**SQLAlchemy 定义文件**：`app/models.py:33-40`

| 字段名 | SQLAlchemy 类型 | 物理类型（SQLite） | Nullable | Default | 说明 |
|--------|-----------------|---------------------|----------|---------|------|
| `id` | `db.Integer` | `INTEGER` | NOT NULL | 自增 | 主键 |
| `record_id` | `db.Integer, ForeignKey("audio_record.id")` | `INTEGER` | NOT NULL | — | 关联音频记录（外键） |
| `start_time` | `db.Float` | `REAL` | NOT NULL | — | 分段起始时间（秒） |
| `end_time` | `db.Float` | `REAL` | NOT NULL | — | 分段结束时间（秒） |
| `speaker` | `db.String(50)` | `VARCHAR(50)` | ✅ 可为空 | NULL | 说话人标签，归一化后为 `spk0` / `spk1` |
| `content` | `db.Text` | `TEXT` | ✅ 可为空 | NULL | 四川话ASR识别文本，支持用户在线修改 |
| `sentiment` | `db.String(50)` | `VARCHAR(50)` | ✅ 可为空 | NULL | 情感标签（当前版本写入 `None`，预留字段） |

**外键约束**：
- `dialogue_segment.record_id → audio_record.id`
- 无 `CASCADE DELETE`，需手动清理（`routes.py:309`：`DialogueSegment.query.filter_by(record_id=rec.id).delete()`）

---

### 2.5 表 4：`split`（对应模型类 `Split`）

**SQLAlchemy 定义文件**：`app/models.py:42-49`

| 字段名 | SQLAlchemy 类型 | 物理类型（SQLite） | Nullable | Default | 说明 |
|--------|-----------------|---------------------|----------|---------|------|
| `id` | `db.Integer` | `INTEGER` | NOT NULL | 自增 | 主键 |
| `record_id` | `db.Integer, ForeignKey("audio_record.id")` | `INTEGER` | NOT NULL | — | 关联音频记录（外键） |
| `segment_index` | `db.Integer` | `INTEGER` | NOT NULL | — | 分段序号（从 1 开始，对应 `idx+1`） |
| `file_path` | `db.String(256)` | `VARCHAR(256)` | NOT NULL | — | 分段音频文件相对路径，如 `static/separated/{id}/0000.wav` |
| `start_time` | `db.Float` | `REAL` | NOT NULL | — | 分段起始时间（秒） |
| `end_time` | `db.Float` | `REAL` | NOT NULL | — | 分段结束时间（秒） |
| `speaker` | `db.String(50)` | `VARCHAR(50)` | ✅ 可为空 | NULL | 说话人标签（与 `DialogueSegment.speaker` 一致） |

**外键约束**：
- `split.record_id → audio_record.id`
- 无 `CASCADE DELETE`，需手动清理（`routes.py:310`：`Split.query.filter_by(record_id=rec.id).delete()`）

---

### 2.6 表 5：`voice_print`（对应模型类 `VoicePrint`）

**SQLAlchemy 定义文件**：`app/models.py:69-89`

| 字段名 | SQLAlchemy 类型 | 物理类型 | Nullable | Default | 说明 |
|--------|-----------------|----------|----------|---------|------|
| `id` | `db.Integer` | `INTEGER` | NOT NULL | 自增 | 主键 |
| `user_id` | `db.Integer, ForeignKey("user.id")` | `INTEGER` | NOT NULL | — | 归属用户 ID (支持用户间声纹隔离) |
| `person_name` | `db.String(128)` | `VARCHAR` | NOT NULL | — | 说话人真实姓名 (认证身份) |
| `source_filename` | `db.String(256)` | `VARCHAR` | ✅ 可为空 | — | 注册时提供的原始音频名 |
| `embedding_path` | `db.String(512)` | `VARCHAR` | NOT NULL | — | CAM++ 提取的 .npy 特征文件存储路径 |
| `created_at` | `db.DateTime` | `DATETIME` | ✅ 可为空 | `utcnow` | 入库时间 |

---

### 2.7 表 6：`record_speaker`（对应模型类 `RecordSpeaker`）

**SQLAlchemy 定义文件**：`app/models.py:94-101`

| 字段名 | SQLAlchemy 类型 | 物理类型 | Nullable | Default | 说明 |
|--------|-----------------|----------|----------|---------|------|
| `id` | `db.Integer` | `INTEGER` | NOT NULL | 自增 | 主键 |
| `record_id` | `db.Integer, ForeignKey("audio_record.id")` | `INTEGER` | NOT NULL | — | 关联的音频记录 ID |
| `raw_spk` | `db.String(50)` | `VARCHAR` | NOT NULL | — | 原始标签 (如 `spk0`) |
| `embedding_path` | `db.String(512)` | `VARCHAR` | NOT NULL | — | 该录音中提取的该 SPK 特征路径 |

---

### 2.8 E-R 关系汇总

```
User (1) ──────────────────── (0..N) AudioRecord
             user_id (FK, nullable)

AudioRecord (1) ────────────── (0..N) DialogueSegment
                   record_id (FK, NOT NULL)

AudioRecord (1) ────────────── (0..N) Split
                   record_id (FK, NOT NULL)

User (1) ───────────────────── (0..N) SystemSession
                   user_id (FK, NOT NULL)

User (1) ───────────────────── (0..N) VoicePrint
                   user_id (FK, NOT NULL)

AudioRecord (1) ────────────── (0..N) RecordSpeaker
                   record_id (FK, NOT NULL)
```

> **关键设计决策**：
> - `user_id` 设为 `nullable=True` 是为了兼容系统改造前（第8步之前）已存入数据库的历史记录
> - `DialogueSegment` 和 `Split` 互为平行关系，各自独立存储文本和文件路径信息，查询时通常联合使用
> - 无 ORM 层面的 `cascade="all, delete-orphan"`，删除操作在路由层用 `filter_by().delete()` 手动级联

---

## 第三章：API 接口契约（完整规格）

### 3.1 认证蓝图 `/auth`（`routes_auth.py`）

#### `POST /auth/register` — 用户注册

| 属性 | 值 |
|------|----|
| HTTP 方法 | `POST` |
| URL | `/auth/register` |
| JWT 鉴权 | ❌ 不需要 |
| Content-Type | `application/json` |

**请求体（Request Body JSON）**：
```json
{
  "username": "alice",
  "password": "mypassword123"
}
```

**响应体（Response Body）**：

| 场景 | HTTP 状态码 | 响应 JSON |
|------|------------|-----------|
| 注册成功 | `200 OK` | `{"code": 200, "msg": "注册成功"}` |
| 用户名/密码为空 | `400 Bad Request` | `{"code": 400, "msg": "username/password 必填"}` |
| 用户名已存在 | `400 Bad Request` | `{"code": 400, "msg": "用户名已存在"}` |

---

#### `POST /auth/login` — 用户登录

| 属性 | 值 |
|------|----|
| HTTP 方法 | `POST` |
| URL | `/auth/login` |
| JWT 鉴权 | ❌ 不需要 |
| Content-Type | `application/json` |

**请求体**：
```json
{
  "username": "alice",
  "password": "mypassword123"
}
```

**响应体**：

| 场景 | HTTP 状态码 | 响应 JSON |
|------|------------|-----------|
| 登录成功 | `200 OK` | `{"code": 200, "data": {"access_token": "<JWT_TOKEN>"}}` |
| 用户名或密码错误 | `401 Unauthorized` | `{"code": 401, "msg": "用户名或密码错误"}` |

> **JWT 说明**：使用 `flask_jwt_extended.create_access_token(identity=str(u.id))` 生成，`identity` 为用户ID的字符串形式。所有后续需要鉴权的接口均需在请求头携带：`Authorization: Bearer <JWT_TOKEN>`

---

### 3.2 主蓝图（`routes.py`）

#### `POST /upload`（别名：`POST /audio/upload`）— 音频上传（异步）

| 属性 | 值 |
|------|----|
| HTTP 方法 | `POST` |
| URL | `/upload` 或 `/audio/upload` |
| JWT 鉴权 | ✅ `@jwt_required()` |
| Content-Type | `multipart/form-data` |

**请求参数（Form-Data）**：

| 字段名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `file` | `File` | ✅ | 音频文件，支持 mp3/wav 等 FFmpeg 可解码格式 |

**响应体**：

| 场景 | HTTP 状态码 | 响应 JSON |
|------|------------|-----------|
| 上传成功（异步启动） | `200 OK` | `{"code": 200, "msg": "上传成功，正在后台处理", "data": {"record_id": 42, "status": "pending"}}` |
| 未选择文件 | `400 Bad Request` | `{"code": 400, "msg": "未选择文件"}` |
| 服务器内部错误 | `500 Internal Server Error` | `{"code": 500, "msg": "上传失败: <错误详情>"}` |

> **关键行为**：主线程在 `routes.py:254` (`t.start()`) 后立即在 `routes.py:256-263` 返回 `pending` 响应，后台处理由 `threading.Thread` 异步完成

---

#### `GET /record/<int:record_id>` — 获取录音详情（含状态轮询）

| 属性 | 值 |
|------|----|
| HTTP 方法 | `GET` |
| URL | `/record/<int:record_id>` |
| JWT 鉴权 | ✅ `@jwt_required()` |

**请求参数**：URL Path 参数 `record_id`（整数）

**响应体（成功 200）**：
```json
{
  "code": 200,
  "data": {
    "info": {
      "id": 42,
      "filename": "static/uploads/20240101120000_abcd1234.wav",
      "upload_time": "2024-01-01T12:00:00",
      "status": "success",
      "error_message": null,
      "current_stage": "处理完成"
    },
    "segments": [
      {
        "id": 101,
        "spk": "spk0",
        "text": "你好，这边是客服。",
        "start": 0.5,
        "end": 2.3,
        "path": "static/separated/42/0000.wav",
        "audio_url": "/static/separated/42/0000.wav"
      }
    ]
  }
}
```

| 场景 | HTTP 状态码 | 响应 JSON |
|------|------------|-----------|
| 记录不存在 | `404 Not Found` | `{"code": 404, "msg": "记录不存在"}` |
| 无权访问 | `403 Forbidden` | `{"code": 403, "msg": "无权访问该记录"}` |

---

#### `DELETE /record/<int:record_id>` — 删除录音记录

| 属性 | 值 |
|------|----|
| HTTP 方法 | `DELETE` |
| URL | `/record/<int:record_id>` |
| JWT 鉴权 | ✅ `@jwt_required()` |

**删除级联操作（代码层面手动实现）**：
1. 删除原始 WAV 文件（`app/static/uploads/xxx.wav`）
2. 删除分段目录（`app/static/separated/<record_id>/`，`shutil.rmtree`）
3. `DELETE FROM dialogue_segment WHERE record_id = ?`
4. `DELETE FROM split WHERE record_id = ?`
5. `DELETE FROM audio_record WHERE id = ?`

| 场景 | HTTP 状态码 | 响应 JSON |
|------|------------|-----------|
| 删除成功 | `200 OK` | `{"code": 200, "msg": "删除成功"}` |
| 记录不存在 | `404 Not Found` | `{"code": 404, "msg": "记录不存在"}` |
| 无权删除 | `403 Forbidden` | `{"code": 403, "msg": "无权删除该记录"}` |

---

#### `GET /history` — 获取当前用户历史记录列表

| 属性 | 值 |
|------|----|
| HTTP 方法 | `GET` |
| URL | `/history` |
| JWT 鉴权 | ✅ `@jwt_required()` |

**响应体（成功 200）**：
```json
{
  "code": 200,
  "data": [
    {
      "id": 42,
      "filename": "test.mp3",
      "filepath": "static/uploads/xxx.wav",
      "original_filename": "test.mp3",
      "duration": "02:35",
      "upload_time": "2024-01-01 12:00:00",
      "status": "success",
      "current_stage": "处理完成"
    }
  ]
}
```

> **注意**：`duration` 字段在此接口中格式化为 `MM:SS` 字符串，而非秒数浮点

---

#### `PUT /record/<int:record_id>/segment/<int:segment_index>` — 用户修改分段文本

| 属性 | 值 |
|------|----|
| HTTP 方法 | `PUT` |
| URL | `/record/<int:record_id>/segment/<int:segment_index>` |
| JWT 鉴权 | ✅ `@jwt_required()` |
| Content-Type | `application/json` |

**请求体**：
```json
{
  "text": "修改后的文字内容"
}
```

**索引逻辑**：`segment_index` 为 0-based，通过 `start_time` 升序排列后取第 `segment_index` 个 `DialogueSegment`

| 场景 | HTTP 状态码 | 响应 JSON |
|------|------------|-----------|
| 修改成功 | `200 OK` | `{"code": 200, "message": "Segment updated successfully", "data": {"index": 0, "new_text": "..."}}` |
| 缺少 text 字段 | `400 Bad Request` | `{"code": 400, "message": "Missing 'text' field"}` |

---

#### `PUT /record/<int:record_id>/segment/<int:segment_index>/speaker` — 修改分段说话人

| 属性 | 值 |
|------|----|
| HTTP 方法 | `PUT` |
| URL | `/record/<int:record_id>/segment/<int:segment_index>/speaker` |
| JWT 鉴权 | ✅ `@jwt_required()` |

**请求体**：
```json
{ "speaker": "张三" }
```

---

#### `POST /record/<int:record_id>/segment/<int:segment_index>/split` — 执行段内智能切分

| 属性 | 值 |
|------|----|
| HTTP 方法 | `POST` |
| URL | `/record/<int:record_id>/segment/<int:segment_index>/split` |
| JWT 鉴权 | ✅ `@jwt_required()` |

**请求参数**：
- `split_offset`: (float) 相对于该分段起始点的切分时刻（秒）

**逻辑说明**：
1. 使用 FFmpeg 对原分段 WAV 进行物理切分。
2. 调用 Python ASR 引擎（`--asr_only` 模式）对产生的两个新分段进行秒级重转写。
3. 更新数据库 `DialogueSegment` 与 `Split` 记录，并对后续序号执行自增平移。
| 索引越界 | `404 Not Found` | `{"code": 404, "message": "片段索引越界"}` |
| 无权访问 | `403 Forbidden` | `{"code": 403, "message": "无权访问该记录"}` |

---

#### `POST /api/summary/<int:record_id>` — 触发 LLM 深度分析

| 属性 | 值 |
|------|----|
| HTTP 方法 | `POST` |
| URL | `/api/summary/<int:record_id>` |
| JWT 鉴权 | ✅ `@jwt_required()` |
| 请求体 | 无（无需 Body） |

**处理逻辑**：
1. 若 `AudioRecord.llm_summary` 非空，直接返回缓存
2. 否则从 `DialogueSegment` 组装对话文本，调用 `request_local_llm()`
3. 结果以 JSON 字符串写入 `AudioRecord.llm_summary` 持久化

**响应体（成功 200）**：
```json
{
  "code": 200,
  "data": {
    "summary": {
      "summary": {
        "overview": {
          "scene_type": "客户投诉场景",
          "detailed_summary": "客户因空调问题投诉，客服处理态度..."
        },
        "analysis_breakdown": {
          "role_0": {
            "role_label": "强势型投诉客户",
            "emotional_state": "愤怒→焦虑→部分满意",
            "hidden_intent": "要求经济赔偿",
            "VFA_analysis": {
              "viewpoint": "服务态度需改进",
              "facts": ["开空调需要五块钱", "上次有人免费开"],
              "deep_analysis": "..."
            }
          },
          "role_1": { "...": "..." }
        }
      }
    }
  }
}
```

---

#### `GET /api/transcript_data/<int:record_id>` — 获取标准化对话数据（LLM/核查页专用）

| 属性 | 值 |
|------|----|
| HTTP 方法 | `GET` |
| URL | `/api/transcript_data/<int:record_id>` |
| JWT 鉴权 | ✅ `@jwt_required()` |

**响应体（成功 200）**：
```json
{
  "success": true,
  "data": {
    "display_data": [
      {
        "seq_id": 1,
        "role": "A",
        "side": "left",
        "text": "你好，我想投诉...",
        "start_time": 0.5,
        "end_time": 3.2,
        "speaker": "spk0"
      }
    ],
    "llm_context": "[1] 角色A: 你好，我想投诉...\n[2] 角色B: 好的，请说..."
  }
}
```

**说话人角色映射**：`spk0 → {role: "A", side: "left"}` · `spk1 → {role: "B", side: "right"}`

---

### 3.3 声纹库管理蓝图 API (`app/routes.py`)

#### `POST /api/voiceprint/enroll` — 声纹注册/入库
- **功能**：由用户提供样本音频和姓名，后端调用 CAM++ 模型提取 192 维特征向量并保存为 `.npy` 文件。
- **存储路径**：`static/voiceprints/{user_id}/{uuid}.npy`

#### `GET /api/voiceprint/list` — 获取声纹列表
- **功能**：返回当前用户声纹库中所有已注册成员的姓名、入库时间等元数据。

#### `PUT /api/voiceprint/<int:vp_id>` — 修改声纹信息（含历史同步）
- **功能**：更新说话人姓名。
- **核心逻辑**：修改姓名后，系统会自动触发全局扫描，将该用户下所有历史记录（`DialogueSegment` 和 `Split`）中匹配旧姓名的标签同步更新。

#### `DELETE /api/voiceprint/<int:vp_id>` — 从声纹库删除记录
- **功能**：从数据库移除声纹，并物理删除对应的 `.npy` 特征文件。

#### `GET /api/voiceprint/<int:vp_id>/history` — 身份参与历史
- **功能**：查询该特定说话人在所有音频记录中的出现情况。
- **返回内容**：记录标题、上传时间、在该音频中的发言段落总数 (segment_count)。

#### `GET /api/voiceprint/match_suggestions/<int:record_id>` — 获取匹配建议
- **算法**：余弦相似度 (Cosine Similarity)。
- **逻辑**：将录音中提取的 `RecordSpeaker` 特征与用户声纹库逐一比对，相似度 > 0.65 时推荐匹配。

#### `POST /api/voiceprint/apply_mapping/<int:record_id>` — 执行身份映射转换
- **功能**：接收用户确认的映射表 (如 `{"spk0": "张三"}`), 执行批量 SQL `UPDATE` 替换标签。

---

### 3.4 Dashboard 接口（`routes.py:571-857`）

#### `GET /api/dashboard/health` — AI 引擎健康检测

| 属性 | 值 |
|------|----|
| HTTP 方法 | `GET` |
| JWT 鉴权 | ❌ 不需要 |

**检测逻辑**：
- `llm_online`：向 `http://127.0.0.1:11434/` 发送 GET，`status_code == 200` 则为 `true`，超时 3s
- `asr_online`：检查磁盘路径 `models/lukeewin01/paraformer-large-sichuan-offline` 目录是否存在

**响应体**：
```json
{"code": 200, "data": {"asr_online": true, "llm_online": true}}
```

---

#### `GET /api/dashboard/ambient` — 环境时钟与状态（系统累计护航量）

| 属性 | 值 |
|------|----|
| HTTP 方法 | `GET` |
| JWT 鉴权 | ❌ 不需要 |

**数据来源**（两步免费API调用）：
1. `ip-api.com/json/` → 获取城市名、纬度、经度（无需 Key，限速 45次/分钟）
2. `api.open-meteo.com/v1/forecast` → 获取 WMO 天气码、气温（完全开源免费）

**响应体**：
```json
{
  "code": 200,
  "data": {
    "location": "Chengdu",
    "weather": "多云",
    "temperature": 18,
    "uptime_days": 2,
    "uptime_hours": 5
  }
}
```

> **注**：`uptime_days/hours` 在此接口中表示**全系统的有效在位时长**（基于 `SystemSession` 心跳记录累计）。

---

#### `GET /api/dashboard/stats` — 当前用户业务速览

| 属性 | 值 |
|------|----|
| HTTP 方法 | `GET` |
| JWT 鉴权 | ✅ `@jwt_required()` |

**SQL 等效查询**：
```sql
-- total_transcribed
SELECT COUNT(*) FROM audio_record WHERE user_id = ? AND status = 'success';

-- total_summarized  
SELECT COUNT(*) FROM audio_record WHERE user_id = ? AND llm_summary IS NOT NULL AND llm_summary != '';

-- uptime_days/hours (个人累计在线时长)
SELECT SUM(julianday(end_time) - julianday(start_time)) * 86400 FROM system_session WHERE user_id = ?;
-- 将总秒数换算为 D天 H小时
```

**响应体**：
```json
{"code": 200, "data": {"total_transcribed": 15, "total_summarized": 8, "uptime_days": 0, "uptime_hours": 3}}
```

---

#### `GET /api/dashboard/keywords` — 词云高频词

| 属性 | 值 |
|------|----|
| HTTP 方法 | `GET` |
| JWT 鉴权 | ✅ `@jwt_required()` |

**提取逻辑**：读取当前用户最近50条含有 `llm_summary` 的记录，拼合 `detailed_summary`、`scene_type`、角色 `emotional_state`、`hidden_intent`、`VFA_analysis` 等字段文本，使用 `jieba.analyse.extract_tags(topK=20, withWeight=True)` 提取关键词权重（权重 × 100 取整）

**响应体**：
```json
{
  "code": 200,
  "data": [
    {"name": "投诉", "value": 87},
    {"name": "服务", "value": 65}
  ]
}
```

---

#### `GET /api/dashboard/recent_records` — 最近5条记录

| 属性 | 值 |
|------|----|
| HTTP 方法 | `GET` |
| JWT 鉴权 | ✅ `@jwt_required()` |

**状态映射表**：
```python
status_map = {
    "pending": ("等待中", "pending"),
    "processing": ("转写中", "transcribing"),
    "success": ("分析完成", "analyzed"),
    "failed": ("失败", "failed")
}
```

**响应体**：
```json
{
  "code": 200,
  "data": [
    {
      "id": 42,
      "title": "customer_service_call.mp3",
      "created_at": "10分钟前",
      "status": "analyzed",
      "status_label": "分析完成"
    }
  ]
}
```

---

## 第四章：核心业务与算法调度流

### 4.1 AI 服务双管线架构（`ai_service.py`）

当前版本已从 **双管线（A+B 管线）** 演进为 **单管线 + 独立说话人分割** 的架构：

```
【旧架构（已废弃）】
A管线（FunASR通用 VAD+SPK）→ B管线（四川话精修）

【当前架构】
3D-Speaker DiarizationService.separate() → AIServiceRunner（仅B管线：四川话ASR）
```

#### `AIServiceRunner` 类方法签名

| 方法名 | 入参 | 返回值 | 说明 |
|--------|------|--------|------|
| `__init__(self)` | 无 | `None` | 调用 `setup_env()` 导入 FunASR/torch/torchaudio，加载 `DiarizationService` 实例 |
| `init_models(self)` | 无 | `None`（副作用：初始化 `self.pipeline_b` 和 `self.punc_model`） | 加载四川话 Paraformer 模型（B管线）和标点恢复模型 |
| `_preprocess_audio(self, audio_path: str)` | `audio_path: str` | `str`（处理后的音频路径） | 检测能量；若均值 < 0.01（静音），放大至 0.05，保存为 `_amp.wav` 并返回新路径 |
| `_normalize_speaker_labels(self, results: list)` | `results: list[dict]` | `list[dict]` | 将随机 spk ID（如 `spk3`, `spk7`）重映射为顺序标签 `spk0`, `spk1`... |
| `run(self, audio_path: str)` | `audio_path: str` | `list[dict]`（标准化分段列表） | 主入口：调用 `diarizer.separate()` → 逐段 ASR → 标点恢复 → 文本清洗 → 说话人标签归一化 |

#### `run()` 方法内部详细流程

```python
# step 1: 调用 3D-Speaker DiarizationService.separate()
segs = self.diarizer.separate(audio_path, out_dir)
# segs 格式: [{"spk": "spk0", "start": 0.5, "end": 3.2, "file": "app/static/separated/name/0000.wav"}, ...]
# out_dir = app/static/separated/<audio_basename>/

# step 2: 逐段循环
for i, seg in enumerate(segs):
    # step 2.1: 解析相对路径 → 绝对路径
    in_path = resolve_to_abs(seg["file"])
    
    # step 2.2: 能量预处理（NumPy数组，torchaudio.load() 读取）
    proc_path = self._preprocess_audio(in_path)
    # 中间变量: wav (torch.Tensor), energy (float, NumPy均值)
    # 若静音放大 → 新文件 *_amp.wav
    
    # step 2.3: B管线 ASR（四川话 Paraformer）
    res_b = self.pipeline_b.generate(input=proc_path)
    text_content = res_b[0].get('text', '')
    
    # step 2.4: 段内标点恢复（第一次）
    punc_res = self.punc_model.generate(input=text_content)
    text_content = punc_res[0].get('text', text_content)
    
    # step 2.5: 正则文本清洗
    cleaned = re.sub(r'([。？！，、])\1+', r'\1', text_content)  # 去重复标点
    cleaned = cleaned.replace("，。", "。")...                    # 修复异常组合
    cleaned = cleaned.lstrip("。？！，、")                        # 去行首标点
    
    final_output.append({speaker, text:cleaned, start, end, path:rel_path})

# step 3: 全局标点恢复（第二次，对合并后整体）
for m in merged:
    punc_res2 = self.punc_model.generate(input=m["text"])
    m["text"] = 再次清洗(punc_res2)

# step 4: [新增] 提取并保存各说话人声纹特征 (embedding)
# 遍历 merged 中独特的 speaker 标签，从 separated 目录读取首个分段提取 192 维特征
# 将特征保存为 static/record_speakers/{rec_id}/{spk}.npy

# step 5: 说话人标签归一化
merged = self._normalize_speaker_labels(merged)

# step 5: 输出 JSON 到 stdout
print(json.dumps(merged, ensure_ascii=False))
```

**关键模型参数**（`init_models()` 中硬编码）：
```python
vad_kwargs = {
    "max_single_segment_time": 60000,  # 最大单段时长 60s（ms）
    "threshold": 0.3,                  # VAD激活阈值
    "min_speech_duration_ms": 250,     # 最小语音时长 250ms
    "speech_pad_ms": 200               # 前后上下文填充 200ms
}
```

---

### 4.2 说话人分割服务（`diarization.py`）

#### `DiarizationService` 类

```python
class DiarizationService:
    def __init__(self):
        # 从 3D-Speaker/speakerlab/bin/infer_diarization.py 导入 Diarization3Dspeaker
        self.diarizer = Diarization3Dspeaker(device="cuda"|"cpu")
    
    def separate(self, audio_path: str, out_dir: str) -> list[dict]:
        """
        入参: audio_path (16kHz WAV绝对路径), out_dir (输出分段目录)
        出参: [{"spk": "spk0", "start": 0.5, "end": 3.2, "file": "app/static/..."}]
        """
        wav, sr = torchaudio.load(audio_path)
        # 调用 3D-Speaker 原生推理
        segments = self.diarizer(audio_path)  # 返回 [[start, end, spk_id], ...]
        
        for i, (start, end, spk_id) in enumerate(segments):
            sub_wav = wav[:, int(start*sr):int(end*sr)]  # torch.Tensor 切片
            filename = f"{i:04d}.wav"  # 0000.wav, 0001.wav ...
            torchaudio.save(out_dir/filename, sub_wav, sr)
            rel_path = os.path.relpath(save_path, project_root)  # 相对项目根的路径
            results.append({"spk": f"spk{spk_id}", "start": float(start), "end": float(end), "file": rel_path})
        return results
```

**输出目录规则**：分段文件统一命名为 `{i:04d}.wav`（4位0填充），保存在 `out_dir` 下

---

### 4.3 音频处理服务（`audio_handler.py`）

#### 函数签名与逻辑

| 函数名 | 入参 | 返回值 | 说明 |
|--------|------|--------|------|
| `save_upload_file(file_storage)` | `FileStorage` | `str`（临时文件绝对路径） | 保留原扩展名，用 `uuid4().hex` 生成随机文件名，保存至 `app/static/uploads/temp/` |
| `convert_to_16k_wav(input_path: str)` | `str`（临时文件绝对路径） | `(str, float)`（相对路径, 时长秒） | FFmpeg 转码为 16kHz 单声道 PCM WAV；FFprobe 测时长；wave 模块兜底；验证采样率须为 16000 Hz |

**FFmpeg 转码参数**：
```
-nostdin -hide_banner -y -fflags +genpts
-i {input}
-map 0:a:0          # 仅取第一条音频流
-af aresample=async=1:first_pts=0  # 异步重采样，避免时间戳跳跃
-c:a pcm_s16le      # 16-bit 有符号小端 PCM
-ar 16000           # 采样率 16kHz
-ac 1               # 单声道
{output}
```

**输出文件命名规则**：`{YYYYMMDDHHmmss}_{uuid4().hex}.wav`（UTC时间）

**WSL 路径转换逻辑**（`_win_to_wsl` 本地函数）：
```python
def _win_to_wsl(p: str) -> str:
    p = os.path.abspath(p)
    drive = p[0].lower()        # e.g. 'c'
    rest = p[2:].replace('\\', '/')  # 去掉盘符冒号后的部分
    return f"/mnt/{drive}/{rest}"  # e.g. /mnt/c/Users/asus/...
```

---

## 第五章：跨平台执行桥接（`wsl_bridge.py`）

### 5.1 核心判断逻辑

**函数**：`run_in_wsl(script_rel_path: str, audio_path: str) -> list[dict]`

**平台判断**（`wsl_bridge.py:22-23`）：
```python
current_os = platform.system()
if current_os == "Linux":
    # WSL 或 Linux 原生环境 → 直接本地执行
else:
    # Windows 环境 → 桥接到 WSL
```

**判断依据**：`platform.system()` 返回值
- 返回 `"Linux"` → 在 WSL 或 Linux 原生环境内运行，走**本地路径**
- 返回 `"Windows"` → 在 Windows 主机运行，走 **WSL 桥接路径**

### 5.2 Linux 路径（本地直接执行）

```python
python_exe = sys.executable               # 当前 Python 解释器路径
script_abs_path = project_root / script_rel_path
cmd = [python_exe, script_abs_path, audio_path]
res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
```

### 5.3 Windows 路径（WSL 桥接执行）

**wsl.exe 查找函数** `_get_wsl_path()`：
```python
candidates = [
    r"C:\Windows\Sysnative\wsl.exe",   # 32位进程访问64位系统目录
    r"C:\Windows\System32\wsl.exe"     # 标准路径
]
return 第一个存在的路径，否则返回 "wsl"（依赖PATH）
```

**Windows → WSL 路径转换**（`wsl_bridge.py:46-47`）：
```python
drive, tail = os.path.splitdrive(abs_audio)      # e.g. drive="C:", tail="\Users\..."
wsl_audio_path = "/mnt/" + drive.lower().rstrip(':') + tail.replace('\\', '/')
# 结果: /mnt/c/Users/asus/Desktop/毕业论文/ASR-system/...
```

**WSL 执行命令构造**：
```bash
wsl.exe bash -lc "export LC_ALL=C.UTF-8 && cd '/mnt/c/Users/asus/Desktop/毕业论文/ASR-system/my_voice_project' && source .venv/bin/activate && python app/services/ai_service.py '/mnt/c/...path_to_audio.wav'"
```

### 5.4 stdout JSON 提取（鲁棒性处理）

AI 脚本 stdout 可能混有调试输出，桥接函数从后往前扫描每一行，找到第一个合法 JSON 数组：

```python
for line in reversed(res.stdout.splitlines()):
    line = line.strip()
    if line.startswith('[') and line.endswith(']'):
        try:
            valid_json = json.loads(line)
            break  # 找到有效 JSON 即停止
        except json.JSONDecodeError:
            continue
```

---

## 第六章：异步并发与状态机机制

### 6.1 状态机全貌

```
                    ┌─────────────┐
  POST /upload      │   pending   │   主线程写入（routes.py:236）
  ─────────────────►│             │
                    └──────┬──────┘
                           │ t.start()（routes.py:254）
                           │ 后台线程 process_audio_background()
                           ▼
                    ┌─────────────┐
                    │ processing  │   routes.py:86（线程内第一次写）
                    │             │
                    └──────┬──────┘
              ┌────────────┴────────────┐
              │ 处理成功                 │ 处理失败（Exception）
              ▼                         ▼
       ┌─────────────┐          ┌─────────────┐
       │   success   │          │   failed    │
       │ routes.py:  │          │ routes.py:  │
       │ 187-188     │          │ 198-199     │
       └─────────────┘          └─────────────┘
```

### 6.2 状态流转代码节点（精确定位）

| 状态变化 | 代码位置 | 触发条件 |
|----------|----------|----------|
| 创建 `pending` 记录 | `routes.py:231-248` | 主线程：文件保存成功后立即创建 |
| `pending → processing` | `routes.py:86-89`（后台线程） | 后台线程启动后，转码前 |
| `processing → success` | `routes.py:187-189`（后台线程） | 所有分段入库成功后 |
| `processing → failed` | `routes.py:198-200`（后台线程） | `except Exception as e` 捕获任意错误后 |
| `processing → failed`（特殊：文件丢失） | `routes.py:106-109` | 转码后发现源文件不存在时的提前终止 |

### 6.3 主线程挂起返回点（精确行）

```python
# routes.py:251-263
app = current_app._get_current_object()  # 获取真实 app 对象（代理）
t = threading.Thread(target=process_audio_background, args=(app, rec.id, temp_path))
t.start()   # ← 第 254 行：此处启动后台线程，主线程继续向下执行

return jsonify({   # ← 第 256-263 行：立即返回 pending 响应给前端
    "code": 200, 
    "msg": "上传成功，正在后台处理", 
    "data": {"record_id": rec.id, "status": "pending"}
})
```

### 6.4 后台线程函数

**函数名**：`process_audio_background(app, record_id, temp_path)` （`routes.py:74`）

**执行序列**：
```
1. with app.app_context():               # 建立 Flask 应用上下文（线程内必须）
2. rec.status = 'processing' + commit    # 更新状态
3. convert_to_16k_wav(temp_path)         # FFmpeg 转码
4. rec.filename = rel_path + commit      # 更新文件路径
5. run_in_wsl("app/services/ai_service.py", abs_in)  # 调用 AI 引擎
6. for each segment in result:           # 处理每个分段
   - 复制分段文件到 static/separated/{id}/
   - 创建 DialogueSegment 对象
   - 创建 Split 对象
7. rec.status = 'success' + commit       # 最终成功状态
8. [finally] 删除临时文件 temp_path      # 清理临时文件
```

### 6.5 current_stage 细粒度进度更新

| 阶段 | `current_stage` 值 | 代码位置 |
|------|------------------|----------|
| 任务启动 | `"初始化任务..."` | `routes.py:83` |
| 转码中 | `"正在转码音频格式..."` | `routes.py:87` |
| 准备AI | `"正在启动 AI 引擎 (WSL)..."` | `routes.py:102` |
| AI推理中 | `"AI 模型正在推理 (首次运行需下载模型)..."` | `routes.py:112` |
| 文件丢失 | `"任务被取消或源文件已删除"` | `routes.py:107` |
| 完成 | `"处理完成"` | `routes.py:188` |

---

## 第七章：LLM 服务层（`llm_service.py`）

### 7.1 函数清单

| 函数名 | 入参 | 返回值 | 说明 |
|--------|------|--------|------|
| `format_transcript(segments: list[dict])` | `[{"speaker": "spk0", "text": "..."}]` | `dict`（含 `full_transcript`、`spk0_transcript`、`spk1_transcript`） | 将分段列表拼合为完整对话文本和角色独立文本 |
| `_extract_json_str(raw: str)` | `str`（模型原始输出） | `str`（清洗后的 JSON 字符串） | 去除 Markdown 代码块包裹（````json ... ``` `）|
| `_ensure_structure(data: dict)` | `dict` | `dict`（结构完整的嵌套字典） | 确保输出符合前端期望的完整 JSON Schema，缺失字段补全空值 |
| `request_local_llm(transcript_text: str)` | `str`（完整对话文本） | `dict`（经过结构验证的分析 JSON） | 流式调用 Ollama API，逐 chunk 拼接，三层 JSON 解析恢复机制 |

### 7.2 Ollama API 调用规格

```
URL:    http://127.0.0.1:11434/api/chat
Method: POST
Model:  qwen-sichuan-psych
Mode:   stream=True（流式，避免阻塞）
Timeout: (connect=10s, read=120s)
```

### 7.3 JSON 三层容错解析

```
原始输出 content_str
  → Layer 1: _extract_json_str()        去 Markdown 包裹
  → Layer 2: json.loads(json_str)       标准解析
     ↳ 失败 → re.search(r'\{.*\}', ...) 正则提取再解析
  → Layer 3: _ensure_structure(parsed)  Schema 完整性校验与填充
```

### 7.4 LLM 期望输出 Schema

```json
{
  "summary": {
    "overview": {
      "scene_type": "string（≤10字）",
      "detailed_summary": "string"
    },
    "analysis_breakdown": {
      "role_0": {
        "role_label": "string",
        "emotional_state": "string",
        "hidden_intent": "string",
        "VFA_analysis": {
          "viewpoint": "string",
          "facts": ["string"],
          "deep_analysis": "string"
        }
      },
      "role_1": { "...": "..." }
    }
  }
}
```

---

## 第八章：系统完整时序图（文字描述）

### 8.1 上传与异步识别时序

```
前端                      Flask主线程                    后台线程                    AI引擎(WSL)
 │                            │                              │                           │
 │─POST /upload (multipart)──►│                              │                           │
 │                            │─save_upload_file()───────────────────────────────────────│
 │                            │  临时文件→TEMP_DIR           │                           │
 │                            │─创建 AudioRecord(pending)────────────────────────────────│
 │                            │  db.session.commit()         │                           │
 │                            │─t.start()───────────────────►│                           │
 │◄──{record_id, "pending"}───│                              │                           │
 │                            │  (主线程结束)                │─status=processing+commit   │
 │                            │                              │─convert_to_16k_wav()       │
 │                            │                              │  FFmpeg转码16kHz WAV       │
 │                            │                              │─run_in_wsl(ai_service.py)─►│
 │                            │                              │                           │─DiarizationService.separate()
 │                            │                              │                           │  3D-Speaker推理
 │                            │                              │                           │─逐段 Paraformer B管线 ASR
 │                            │                              │                           │─标点恢复 (punc_model)
 │                            │                              │                           │─[新增] 提取 SPK 192D Embedding
 │                            │                              │◄─stdout JSON数组────────  │
 │                            │                              │─copy分段文件              │
 │                            │                              │─存储 RecordSpeaker 表     │
 │                            │                              │─写 DialogueSegment/Split  │
 │                            │                              │─status=success+commit     │
 │                            │                              │─删除临时文件              │
 │                            │                              │ (线程结束)                │
 │                            │                              │                           │
 │─GET /record/{id}──────────►│                              │                           │
 │◄──{status, segments}───────│                              │                           │
```

### 8.2 LLM 摘要时序

```
前端                    Flask主线程                      Ollama (localhost:11434)
 │                          │                                    │
 │─POST /api/summary/{id}──►│                                    │
 │                          │─检查 llm_summary 缓存              │
 │                          │─format_transcript(segs)            │
 │                          │─POST /api/chat (stream=True)──────►│
 │                          │                            ◄─chunk─│
 │                          │◄─逐chunk拼接 content_str ──────────│
 │                          │─_extract_json_str()                │
 │                          │─json.loads()                       │
 │                          │─_ensure_structure()                │
 │                          │─AudioRecord.llm_summary=json+commit│
 │◄──{summary JSON}─────────│                                    │
```

---

## 附录：核心技术组件版本与依赖

| 组件 | 用途 | 版本/规格 |
|------|------|-----------|
| Flask | Web 框架 | 见 requirements.txt |
| Flask-SQLAlchemy | ORM | SQLite 3 后端 |
| Flask-JWT-Extended | JWT 认证 | HS256 算法 |
| Flask-CORS | 跨域 | `origins: *`（开发模式） |
| Werkzeug | 密码哈希 | `generate_password_hash` + `check_password_hash` |
| FunASR `AutoModel` | B管线四川话ASR | 本地 Paraformer（`disable_update=True`） |
| 3D-Speaker | 说话人分割 | `Diarization3Dspeaker`，自动选 GPU/CPU |
| torchaudio | 音频IO + 内存切片 | PyTorch 生态 |
| Ollama | 本地大模型推理 | `qwen-sichuan-psych`，流式 API |
| FFmpeg | 音频转码/切割 | 16kHz, mono, PCM S16LE |
| jieba | 关键词抽取 | `jieba.analyse.extract_tags`（可选，无则降级） |
| ip-api.com | IP 地理定位 | 免费，无需 API Key |
| Open-Meteo | 实时天气 | 免费开源气象 API |

---

*报告生成完毕。本文档可直接用于 E-R 图（第二章）、系统时序图（第八章）、算法流程图（第四章）的精确绘制，以及毕业论文系统设计章节的撰写。*
