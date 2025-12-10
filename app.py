import os
import sys
import json
import shutil
import subprocess
from datetime import datetime
from flask import request, jsonify, current_app
from app import create_app, db
from app.models import AudioRecord, DialogueSegment, Split
from app.services.audio_handler import save_upload_file, convert_to_16k_wav

app = create_app()

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

@app.route("/audio/upload", methods=["POST"])
def audio_upload():
    print("[UPLOAD] request received")
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"code": 400, "msg": "未选择文件"}), 400
    temp_path = None
    try:
        print("[UPLOAD] saving temp file...")
        temp_path = save_upload_file(file)
        print(f"[UPLOAD] temp_path={temp_path}")
        if os.path.getsize(temp_path) == 0:
            return jsonify({"code": 400, "msg": "音频文件为空或损坏"}), 400

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
        if proc.returncode != 0:
            return jsonify({"code": 500, "msg": f"AI处理失败: {proc.stderr.strip() or proc.stdout.strip()}"}), 500

        try:
            result = json.loads(proc.stdout.strip())
            print(f"[AI] parsed items={len(result)}")
        except Exception:
            return jsonify({"code": 500, "msg": f"AI输出解析失败: {proc.stdout[:200]}"}), 500

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

if __name__ == "__main__":
    app.run(debug=True)
