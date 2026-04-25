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
    # 声纹库匹配状态：0-未匹配，1-已处理匹配（弹窗过且已操作）
    voiceprint_status = db.Column(db.Integer, default=0)

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

class VoicePrint(db.Model):
    """声纹库：存储用户注册的声纹身份。
    
    每条记录对应一个"真实身份"，转录完成后系统会将 diarization
    输出的 spkX 标签与库中 embedding 做 cosine 匹配，匹配成功则
    自动替换为 person_name。
    """
    __tablename__ = "voice_print"

    id = db.Column(db.Integer, primary_key=True)
    # 归属用户（用户隔离：每个用户只能看到和匹配自己的声纹）
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    # 该声纹对应的真实人名，转录后用于替换 spkX
    person_name = db.Column(db.String(128), nullable=False)
    # 入库时上传的原始音频文件名，仅用于展示
    source_filename = db.Column(db.String(256), nullable=True)
    # CAM++ embedding（.npy）在服务器上的相对路径
    # 例如: static/voiceprints/1/abc123.npy
    embedding_path = db.Column(db.String(512), nullable=False)
    # 入库时间
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<VoicePrint uid={self.user_id} name={self.person_name}>"

class RecordSpeaker(db.Model):
    """存储某次转录记录中发现的原始说话人（SPKX）的特征向量。"""
    __tablename__ = "record_speaker"
    id = db.Column(db.Integer, primary_key=True)
    record_id = db.Column(db.Integer, db.ForeignKey("audio_record.id"), nullable=False)
    raw_spk = db.Column(db.String(50), nullable=False) # "spk0", "spk1"
    embedding_path = db.Column(db.String(512), nullable=False) # .npy 路径
