from datetime import datetime
from . import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

class AudioRecord(db.Model):
    # 主键ID
    id = db.Column(db.Integer, primary_key=True)
    # 归属用户，可为空以兼容旧数据和未迁移库
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)
    # 用户上传时的原始文件名，例如 "test.mp3"
    original_filename = db.Column(db.String(256), nullable=False)
    # 服务器本地存储的相对路径，例如 "static/uploads/xxx.wav"
    filename = db.Column(db.String(256), nullable=False)
    # 音频时长（秒）
    duration = db.Column(db.Float, default=0.0)
    # 上传时间
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    # 状态：默认为 'pending'
    status = db.Column(db.String(20), default="pending")
    # 记录自定义名称
    title = db.Column(db.String(256), nullable=True)
    # 最后修改时间 (自动跟随 update 操作更新)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # 错误信息
    error_message = db.Column(db.Text, nullable=True)
    current_stage = db.Column(db.String(100), default="等待处理")
    # 用于持久化存储 AI（LLM）生成的报告
    llm_summary = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f"<AudioRecord {self.original_filename}>"

class DialogueSegment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    record_id = db.Column(db.Integer, db.ForeignKey("audio_record.id"), nullable=False)
    start_time = db.Column(db.Float, nullable=False)
    end_time = db.Column(db.Float, nullable=False)
    speaker = db.Column(db.String(50))
    content = db.Column(db.Text)
    sentiment = db.Column(db.String(50))

class Split(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    record_id = db.Column(db.Integer, db.ForeignKey("audio_record.id"), nullable=False)
    segment_index = db.Column(db.Integer, nullable=False)
    file_path = db.Column(db.String(256), nullable=False)
    start_time = db.Column(db.Float, nullable=False)
    end_time = db.Column(db.Float, nullable=False)
    speaker = db.Column(db.String(50))

class SystemSession(db.Model):
    # 系统运行/会话计时表
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    # 会话开始时间
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    # 会话最后活跃时间（通过心跳更新，以此作为结束时间的估值）
    end_time = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<SystemSession {self.user_id} {self.start_time} to {self.end_time}>"
