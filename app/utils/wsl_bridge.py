import os
import subprocess
import json
import logging
import platform
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _get_wsl_path():
    candidates = [
        r"C:\\Windows\\Sysnative\\wsl.exe",
        r"C:\\Windows\\System32\\wsl.exe"
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return "wsl"

def run_in_wsl(script_rel_path, audio_path):
    current_os = platform.system()

    if current_os == "Linux":
        logger.info("[Bridge] Detected Linux/WSL environment. Running locally.")
        python_exe = sys.executable
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        script_abs_path = os.path.join(project_root, script_rel_path)
        cmd = [python_exe, script_abs_path, audio_path]
        logger.info(f"[Local] Executing: {cmd}")
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
        except Exception as e:
            raise RuntimeError(f"Local execution failed: {e}")
    else:
        logger.info("[Bridge] Detected Windows environment. Bridging to WSL.")
        wsl_exe = _get_wsl_path()
        wsl_project_root = "/mnt/c/Users/asus/Desktop/毕业论文/ASR-system/my_voice_project"
        abs_audio = os.path.abspath(audio_path)
        drive, tail = os.path.splitdrive(abs_audio)
        wsl_audio_path = "/mnt/" + drive.lower().rstrip(':') + tail.replace('\\', '/')
        bash_cmd = f"export LC_ALL=C.UTF-8 && cd '{wsl_project_root}' && source .venv/bin/activate && python {script_rel_path} '{wsl_audio_path}'"
        cmd = [wsl_exe, "bash", "-lc", bash_cmd]
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
        except FileNotFoundError:
            raise RuntimeError(f"wsl.exe not found. Is WSL installed?")

    if res.returncode != 0:
        logger.error(f"[Exec Fail] Stderr: {res.stderr}")
        raise RuntimeError(f"AI Process Error: {res.stderr[:500]}")
    else:
        # Log stderr even on success, as it contains important [3D-Speaker] logs
        if res.stderr:
            logger.info(f"[WSL Log] {res.stderr}")

    valid_json = None
    for line in reversed(res.stdout.splitlines()):
        line = line.strip()
        if line.startswith('[') and line.endswith(']'):
            try:
                valid_json = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    if valid_json is not None:
        return valid_json

    logger.error(f"[Bridge] No JSON found. Output: {res.stdout}")
    raise ValueError("No JSON response from AI service.")
