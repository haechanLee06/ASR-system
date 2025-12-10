import os
import subprocess
from uuid import uuid4
import wave
import shutil
from flask import current_app
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOADS_DIR = os.path.join(STATIC_DIR, "uploads")
TEMP_DIR = os.path.join(UPLOADS_DIR, "temp")

def _ensure_dirs():
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

def _decode_output(data: bytes) -> str:
    if data is None:
        return ""
    try:
        s = data.decode("utf-8", errors="ignore")
        if "\x00" in s:
            raise UnicodeDecodeError("utf-8", data, 0, 1, "NULs present")
        return s
    except Exception:
        try:
            return data.decode("utf-16-le", errors="ignore")
        except Exception:
            return data.decode(errors="ignore")

def _run(cmd, use_wsl: bool):
    if use_wsl:
        env = os.environ.copy()
        for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
            env.pop(k, None)
        proc = subprocess.run(cmd, capture_output=True, env=env)
        stdout = _decode_output(proc.stdout)
        stderr = _decode_output(proc.stderr)
        return proc.returncode, stdout, stderr
    else:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        return proc.returncode, proc.stdout or "", proc.stderr or ""

def save_upload_file(file_storage):
    """保存用户上传的原始文件到临时目录，保留原扩展名。"""
    _ensure_dirs()
    ext = os.path.splitext(file_storage.filename)[1].lower()
    name = uuid4().hex + ext
    tmp_path = os.path.join(TEMP_DIR, name)
    file_storage.save(tmp_path)
    return tmp_path

def convert_to_16k_wav(input_path):
    """
    将输入音频转码为 16kHz 单声道 WAV，输出文件命名为 {timestamp}_{uuid}.wav。
    返回 (系统相对路径, 时长秒)。
    """
    _ensure_dirs()
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    safe_name = f"{ts}_{uuid4().hex}.wav"
    out_abs = os.path.join(UPLOADS_DIR, safe_name)
    print("正在开始转码...")
    ffmpeg_bin = current_app.config.get("FFMPEG_BIN", "ffmpeg")
    def _win_to_wsl(p):
        p = os.path.abspath(p)
        drive = p[0].lower()
        rest = p[2:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"

    use_wsl = False
    if shutil.which(ffmpeg_bin) is None:
        use_wsl = True

    if use_wsl:
        distro = current_app.config.get("WSL_DISTRO", "Ubuntu-20.04")
        wsl_in = _win_to_wsl(input_path)
        wsl_out = _win_to_wsl(out_abs)
        cmd = [
            "wsl",
            "-d",
            distro,
            "--",
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-y",
            "-fflags", "+genpts",
            "-i", wsl_in,
            "-map", "0:a:0",
            "-af", "aresample=async=1:first_pts=0",
            "-c:a", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            wsl_out,
        ]
    else:
        cmd = [
            ffmpeg_bin,
            "-nostdin",
            "-hide_banner",
            "-y",
            "-fflags", "+genpts",
            "-i", input_path,
            "-map", "0:a:0",
            "-af", "aresample=async=1:first_pts=0",
            "-c:a", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            out_abs,
        ]
    rc, out, err = _run(cmd, use_wsl)
    if rc != 0:
        raise RuntimeError(err.strip() or out.strip() or "ffmpeg 转码失败")

    duration = None
    try:
        ffprobe_bin = current_app.config.get("FFPROBE_BIN", "ffprobe")
        if not use_wsl and shutil.which(ffprobe_bin):
            rc_dur, dur_out, dur_err = _run([
                ffprobe_bin,
                "-v","error",
                "-show_entries","format=duration",
                "-of","default=noprint_wrappers=1:nokey=1",
                out_abs,
            ], use_wsl=False)
        else:
            distro = current_app.config.get("WSL_DISTRO", "Ubuntu-20.04")
            wsl_out = _win_to_wsl(out_abs)
            rc_dur, dur_out, dur_err = _run([
                "wsl","-d",distro,"--","ffprobe",
                "-v","error",
                "-show_entries","format=duration",
                "-of","default=noprint_wrappers=1:nokey=1",
                wsl_out,
            ], use_wsl=True)
        if rc_dur == 0:
            duration = float(dur_out.strip())
    except Exception:
        duration = None

    if duration is None:
        with wave.open(out_abs, "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            duration = frames / float(rate)

    verify_sr = None
    ffprobe_bin = current_app.config.get("FFPROBE_BIN", "ffprobe")
    if not use_wsl and shutil.which(ffprobe_bin):
        rc_sr, sr_out, sr_err = _run([
            ffprobe_bin,
            "-v","error",
            "-select_streams","a:0",
            "-show_entries","stream=sample_rate",
            "-of","default=noprint_wrappers=1:nokey=1",
            out_abs,
        ], use_wsl=False)
    else:
        distro = current_app.config.get("WSL_DISTRO", "Ubuntu-20.04")
        wsl_out = _win_to_wsl(out_abs)
        rc_sr, sr_out, sr_err = _run([
            "wsl","-d",distro,"--","ffprobe",
            "-v","error",
            "-select_streams","a:0",
            "-show_entries","stream=sample_rate",
            "-of","default=noprint_wrappers=1:nokey=1",
            wsl_out,
        ], use_wsl=True)
    if rc_sr == 0:
        try:
            verify_sr = int(sr_out.strip())
        except Exception:
            verify_sr = None
    if verify_sr != 16000:
        raise ValueError("转码失败：采样率异常")

    rel_path = os.path.join("static", "uploads", safe_name).replace("\\", "/")
    print(f"[SUCCESS] 文件已转换为 16k: {safe_name}, Duration: {duration}s")
    return rel_path, duration
