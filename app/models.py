from datetime import datetime
from . import db

class AudioRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256), nullable=False)
    original_filename = db.Column(db.String(256), nullable=False)
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    duration = db.Column(db.Float, default=0.0)
    status = db.Column(db.String(20), default="uploaded")
    segments = db.relationship("DialogueSegment", backref="record", cascade="all, delete-orphan")

class DialogueSegment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    record_id = db.Column(db.Integer, db.ForeignKey("audio_record.id"), nullable=False)
    start_time = db.Column(db.Float, nullable=False)
    end_time = db.Column(db.Float, nullable=False)
    speaker = db.Column(db.String(50))
    content = db.Column(db.Text)
    sentiment = db.Column(db.String(50))

