import os
import sys
import json
import shutil
import subprocess
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, current_app
from . import db
from .models import AudioRecord, DialogueSegment, Split
from .services.audio_handler import save_upload_file, convert_to_16k_wav
import threading

main = Blueprint("main", __name__)

@main.route("/")
def index():
    # 首页展示所有录音记录，按上传时间倒序
    records = AudioRecord.query.order_by(AudioRecord.upload_time.desc()).all()
    return render_template("index.html", records=records)

@main.route("/audio/upload", methods=["POST"])
def upload():
    def _ensure_dir(p):
        os.makedirs(p, exist_ok=True)

    def _win_to_wsl(p):
        p = os.path.abspath(p)
        drive = p[0].lower()
        rest = p[2:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"

    def _run_ffmpeg_cut(in_path, out_path, start, end):
        ffmpeg_bin = current_app.config.get("FFMPEG_BIN", "ffmpeg")
        use_wsl = shutil.which(ffmpeg_bin) is None
        _ensure_dir(os.path.dirname(out_path))
        if use_wsl:
            distro = current_app.config.get("WSL_DISTRO", "Ubuntu-20.04")
            wsl_in = _win_to_wsl(in_path)
            wsl_out = _win_to_wsl(out_path)
            cmd = [
                "wsl", "-d", distro, "--", "ffmpeg",
                "-nostdin", "-hide_banner", "-y",
                "-ss", str(float(start)),
                "-to", str(float(end)),
                "-i", wsl_in,
                "-map", "0:a:0",
                "-af", "aresample=async=1:first_pts=0",
                "-c:a", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                wsl_out,
            ]
            env = os.environ.copy()
            for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
                env.pop(k, None)
            proc = subprocess.run(cmd, capture_output=True, env=env)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore").strip())
        else:
            cmd = [
                ffmpeg_bin,
                "-nostdin", "-hide_banner", "-y",
                "-ss", str(float(start)),
                "-to", str(float(end)),
                "-i", in_path,
                "-map", "0:a:0",
                "-af", "aresample=async=1:first_pts=0",
                "-c:a", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                out_path,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.strip())

    def _extract_json_from_output(s: str):
        try:
            return json.loads(s.strip())
        except Exception:
            pass
        if not s:
            return None
        start = s.find('[')
        end = s.rfind(']')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                pass
        start = s.find('{')
        end = s.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                pass
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        for ln in reversed(lines):
            try:
                return json.loads(ln)
            except Exception:
                continue
        return None

    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"code": 400, "msg": "未选择文件"}), 400
    try:
        temp_path = None
        print("[UPLOAD] request received")
        temp_path = save_upload_file(file)
        print(f"[UPLOAD] temp_path={temp_path}")
        try:
            size = os.path.getsize(temp_path)
            if size == 0:
                return jsonify({"code": 400, "msg": "音频文件为空或损坏，请重新录制或重新上传"}), 400
        except Exception:
            pass
        print("[CONVERT] start convert to 16k wav...")
        rel_path, duration = convert_to_16k_wav(temp_path)
        print(f"[CONVERT] rel_path={rel_path}, duration={duration}")
        rec = AudioRecord(
            original_filename=file.filename,
            filename=rel_path,
            duration=duration,
            upload_time=datetime.utcnow(),
            status="uploaded",
        )
        db.session.add(rec)
        db.session.commit()
        print(f"[DB] audio record committed id={rec.id}")

        abs_in = os.path.join(current_app.root_path, rel_path)
        py = sys.executable
        ai_script = os.path.join(current_app.root_path, "services", "ai_service.py")
        cmd = [py, ai_script, abs_in]
        print(f"[AI] invoking: {cmd}")
        env = os.environ.copy()
        for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
            env.pop(k, None)
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore", env=env)
        print(f"[AI] returncode={proc.returncode}")
        print(f"[AI] stdout_head={proc.stdout[:200]}")
        print(f"[AI] stderr_head={proc.stderr[:200]}")

        result = None
        if proc.returncode == 0:
            result = _extract_json_from_output(proc.stdout)
            if result is not None:
                print(f"[AI] parsed items={len(result)}")
            else:
                print(f"[AI] parse failed, stdout_head={proc.stdout[:200]}")
        
        if result is None:
            print("[AI] fallback to WSL Python")
            distro = current_app.config.get("WSL_DISTRO", "Ubuntu-20.04")
            base_root = os.path.dirname(current_app.root_path)
            wsl_root = _win_to_wsl(base_root)
            wsl_audio = _win_to_wsl(abs_in)
            cmd_wsl = [
                "wsl", "-d", distro, "--", "bash", "-lc",
                f"cd '{wsl_root}' && if [ -x .venv/bin/python ]; then .venv/bin/python app/services/ai_service.py '{wsl_audio}'; else python3 app/services/ai_service.py '{wsl_audio}'; fi"
            ]
            env2 = os.environ.copy()
            for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
                env2.pop(k, None)
            proc2 = subprocess.run(cmd_wsl, capture_output=True, env=env2)
            stdout2 = proc2.stdout.decode("utf-8", errors="ignore")
            stderr2 = proc2.stderr.decode("utf-8", errors="ignore")
            print(f"[AI/WSL] returncode={proc2.returncode}")
            print(f"[AI/WSL] stdout_head={stdout2[:200]}")
            print(f"[AI/WSL] stderr_head={stderr2[:200]}")
            if proc2.returncode != 0:
                return jsonify({"code": 500, "msg": f"AI处理失败(WSL): {stderr2.strip() or stdout2.strip()}"}), 500
            result = _extract_json_from_output(stdout2)
            if result is None:
                return jsonify({"code": 500, "msg": f"AI输出解析失败(WSL): {stdout2[:200]}"}), 500

        sep_root = os.path.join(current_app.root_path, "static", "separated", str(rec.id))
        _ensure_dir(sep_root)
        payload = []
        for idx, item in enumerate(result):
            print(f"[CUT] idx={idx} item={item}")
            speaker = str(item.get("speaker", "spk0"))
            text = str(item.get("text", ""))
            start = float(item.get("start", 0.0))
            end = float(item.get("end", start))
            out_name = f"{idx+1:04d}.wav"
            out_abs = os.path.join(sep_root, out_name)
            print(f"[CUT] ffmpeg cut to {out_abs} from {start} to {end}")
            _run_ffmpeg_cut(abs_in, out_abs, start, end)

            seg = DialogueSegment(
                record_id=rec.id,
                start_time=start,
                end_time=end,
                speaker=speaker,
                content=text,
                sentiment=None,
            )
            db.session.add(seg)

            split = Split(
                record_id=rec.id,
                segment_index=idx + 1,
                file_path=os.path.join("static", "separated", str(rec.id), out_name).replace("\\", "/"),
                start_time=start,
                end_time=end,
                speaker=speaker,
            )
            db.session.add(split)

            payload.append({
                "speaker": speaker,
                "text": text,
                "start": start,
                "end": end,
                "path": split.file_path,
            })

        db.session.commit()
        print(f"[DONE] commit segments={len(payload)} for record={rec.id}")
        return jsonify({"code": 200, "msg": "上传并处理成功", "data": {"record_id": rec.id, "segments": payload}})
    except Exception as e:
        current_app.logger.error(str(e))
        db.session.rollback()
        print(f"[ERROR] {str(e)}")
        return jsonify({"code": 500, "msg": f"上传失败: {str(e)}"}), 500
    finally:
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"[CLEAN] removed temp {temp_path}")
        except Exception:
            pass

@main.route("/detail/<int:record_id>")
def detail(record_id):
    record = AudioRecord.query.get_or_404(record_id)
    return render_template("detail.html", record=record)

