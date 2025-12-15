import os
import sys
import json
import shutil
import subprocess
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from sqlalchemy import inspect
from . import db
from .models import AudioRecord, DialogueSegment, Split
from .services.audio_handler import save_upload_file, convert_to_16k_wav
from .utils.wsl_bridge import run_in_wsl
import threading

main = Blueprint("main", __name__)

@main.route("/")
def index():
    # 首页展示所有录音记录，按上传时间倒序
    records = AudioRecord.query.order_by(AudioRecord.upload_time.desc()).all()
    return render_template("index.html", records=records)

@main.route("/audio/upload", methods=["POST"])
@main.route("/upload", methods=["POST"])
@jwt_required()
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
        current_user_id = int(get_jwt_identity())
        rec = AudioRecord(
            original_filename=file.filename,
            filename=rel_path,
            duration=duration,
            upload_time=datetime.utcnow(),
            status="uploaded",
        )
        try:
            inspector = inspect(db.engine)
            cols = [c['name'] for c in inspector.get_columns('audio_record')]
            if 'user_id' in cols:
                rec.user_id = current_user_id
        except Exception:
            pass
        db.session.add(rec)
        db.session.commit()
        print(f"[DB] audio record committed id={rec.id}")

        abs_in = os.path.join(current_app.root_path, rel_path)
        ai_script_rel = "app/services/ai_service.py"
        print("[AI] invoking via WSL/local...")
        result = run_in_wsl(ai_script_rel, abs_in)
        print(f"[AI] segments_count={len(result)}")

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
                "spk": speaker,
                "start": start,
                "end": end,
                "text": seg.content,
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

@main.route("/history", methods=["GET"])
@jwt_required()
def history():
    try:
        inspector = inspect(db.engine)
        cols = [c['name'] for c in inspector.get_columns('audio_record')]
        q = AudioRecord.query
        if 'user_id' in cols:
            uid = int(get_jwt_identity())
            q = q.filter_by(user_id=uid)
        records = q.order_by(AudioRecord.upload_time.desc()).all()
        data = []
        for r in records:
            data.append({
                "id": r.id,
                "filename": r.filename,
                "original_filename": r.original_filename,
                "duration": r.duration,
                "upload_time": r.upload_time.isoformat(),
                "status": r.status,
            })
        return jsonify({"code": 200, "data": data})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)})

@main.route("/record/<int:record_id>", methods=["GET"])
@jwt_required()
def record_detail(record_id):
    try:
        inspector = inspect(db.engine)
        cols = [c['name'] for c in inspector.get_columns('audio_record')]
        r = AudioRecord.query.get(record_id)
        if r is None:
            return jsonify({"code": 404, "msg": "记录不存在"}), 404
        if 'user_id' in cols:
            uid = int(get_jwt_identity())
            if r.user_id is not None and r.user_id != uid:
                return jsonify({"code": 403, "msg": "无权访问该记录"}), 403
        segs = DialogueSegment.query.filter_by(record_id=record_id).order_by(DialogueSegment.start_time.asc()).all()
        payload = []
        for idx, s in enumerate(segs):
            out_name = f"{idx+1:04d}.wav"
            path = os.path.join("static", "separated", str(record_id), out_name).replace("\\", "/")
            payload.append({
                "id": s.id,
                "spk": s.speaker,
                "text": s.content,
                "start": s.start_time,
                "end": s.end_time,
                "path": path,
            })
        data = {
            "info": {
                "id": r.id,
                "filename": r.filename,
                "upload_time": r.upload_time.isoformat(),
            },
            "segments": payload,
        }
        return jsonify({"code": 200, "data": data})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)})

