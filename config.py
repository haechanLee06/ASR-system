import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key")
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(BASE_DIR, "voice_data.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    AI_SPEAKER_URL = os.environ.get("AI_SPEAKER_URL", "http://127.0.0.1:3000/segment")
    AI_BERT_URL = os.environ.get("AI_BERT_URL", "http://127.0.0.1:10086/textProcess")
    FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
    FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")
    WSL_DISTRO = os.environ.get("WSL_DISTRO", "Ubuntu-20.04")
    # 高德地图天气 API Key（免费申请：https://lbs.amap.com/）
    # 配置后 /api/dashboard/ambient 可返回真实天气；留空则返回默认占位值
    AMAP_KEY = os.environ.get("AMAP_KEY", "")

