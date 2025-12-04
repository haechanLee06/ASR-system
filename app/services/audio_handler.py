import os
import subprocess
from uuid import uuid4
from werkzeug.utils import secure_filename
import wave

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOADS_DIR = os.path.join(STATIC_DIR, "uploads")
TEMP_DIR = os.path.join(UPLOADS_DIR, "temp")

def _ensure_dirs():
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

def save_upload_file(file_storage):
    _ensure_dirs()
    ext = os.path.splitext(file_storage.filename)[1].lower()
    name = uuid4().hex + ext
    tmp_path = os.path.join(TEMP_DIR, name)
    file_storage.save(tmp_path)
    return tmp_path

def convert_to_16k_wav(input_path, output_filename):
    _ensure_dirs()
    safe_name = secure_filename(output_filename)
    out_abs = os.path.join(UPLOADS_DIR, safe_name)
    print("正在开始转码...")
    cmd = [
        "ffmpeg",
        "-i",
        input_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-y",
        out_abs,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())

    duration = None
    try:
        p = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                out_abs,
            ],
            capture_output=True,
            text=True,
        )
        if p.returncode == 0:
            duration = float(p.stdout.strip())
    except Exception:
        duration = None

    if duration is None:
        with wave.open(out_abs, "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            duration = frames / float(rate)

    verify_sr = None
    vp = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            out_abs,
        ],
        capture_output=True,
        text=True,
    )
    if vp.returncode == 0:
        try:
            verify_sr = int(vp.stdout.strip())
        except Exception:
            verify_sr = None
    if verify_sr != 16000:
        raise ValueError("转码失败：采样率异常")

    rel_path = os.path.join("static", "uploads", safe_name).replace("\\", "/")
    print(f"[SUCCESS] 文件已转换为 16k: {safe_name}, Duration: {duration}s")
    return rel_path, duration
