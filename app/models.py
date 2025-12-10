from datetime import datetime
from . import db

class AudioRecord(db.Model):
    # 主键ID
    id = db.Column(db.Integer, primary_key=True)
    # 用户上传时的原始文件名，例如 "test.mp3"
    original_filename = db.Column(db.String(256), nullable=False)
    # 服务器本地存储的相对路径，例如 "static/uploads/xxx.wav"
    filename = db.Column(db.String(256), nullable=False)
    # 音频时长（秒）
    duration = db.Column(db.Float, default=0.0)
    # 上传时间
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    # 状态：默认为 'uploaded'
    status = db.Column(db.String(20), default="uploaded")

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

