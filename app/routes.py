import os
import sys
import json
import shutil
import subprocess
import threading
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from sqlalchemy import inspect, func
from . import db
from .models import User, AudioRecord, DialogueSegment, Split, SystemSession
from .services.audio_handler import save_upload_file, convert_to_16k_wav, is_video_file, probe_has_audio
from .services.ai_service import AIServiceRunner
from .services.llm_service import request_local_llm, format_transcript
from .utils.wsl_bridge import run_in_wsl

main = Blueprint("main", __name__)

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

def process_audio_background(app, record_id, temp_path):
    with app.app_context():
        app.logger.info(f"[Thread] Starting processing for record_id={record_id}")
        
        try:
            rec = AudioRecord.query.get(record_id)
            if not rec:
                app.logger.error(f"[Thread] Record {record_id} not found")
                return
            rec.current_stage = "初始化任务..."
            db.session.commit()

            rec.status = 'processing'
            if is_video_file(temp_path):
                rec.current_stage = "正在从视频中提取音频..."
            else:
                rec.current_stage = "正在转码音频格式..."
            rec.status = 'processing'
            db.session.commit()
            app.logger.info(f"[Thread] Status updated to processing for {record_id}")

            app.logger.info("[CONVERT] start convert to 16k wav...")
            rel_path, duration = convert_to_16k_wav(temp_path)
            app.logger.info(f"[CONVERT] rel_path={rel_path}, duration={duration}")

            rec.filename = rel_path
            rec.duration = duration
            db.session.commit()

            abs_in = os.path.join(current_app.root_path, rel_path)
            ai_script_rel = "app/services/ai_service.py"
            rec.current_stage = "正在启动 AI 引擎 (WSL)..."
            db.session.commit()
            if not (os.path.exists(temp_path) or os.path.exists(abs_in)):
                app.logger.warning(f"[Thread] Source file missing for record {record_id}, stopping task gracefully")
                rec.status = 'failed'
                rec.current_stage = "任务被取消或源文件已删除"
                rec.error_message = "源音频文件不存在，任务已终止"
                db.session.commit()
                return

            rec.current_stage = "AI 模型正在推理 (首次运行需下载模型)..."
            db.session.commit()
            app.logger.info("[AI] invoking via WSL/local...")
            
            result = run_in_wsl(ai_script_rel, abs_in)
            app.logger.info(f"[AI] segments_count={len(result)}")

            sep_root = os.path.join(current_app.root_path, "static", "separated", str(rec.id))
            _ensure_dir(sep_root)
            
            payload = []
            for idx, item in enumerate(result):
                app.logger.info(f"[CUT] idx={idx} item={item}")
                speaker = str(item.get("speaker", "spk0"))
                text = str(item.get("text", ""))
                start = float(item.get("start", 0.0))
                end = float(item.get("end", start))
                
                # [Fix] Do not re-cut. Use the file provided by AI service.
                source_rel_path = item.get("path")
                final_rel_path = ""
                
                if source_rel_path:
                    # source_rel_path is relative to project root (e.g. app/static/separated/name/0000.wav)
                    source_abs = os.path.abspath(source_rel_path)
                    if os.path.exists(source_abs):
                        # Copy to the record's separated folder
                        fname = os.path.basename(source_rel_path)
                        target_abs = os.path.join(sep_root, fname)
                        try:
                            shutil.copy2(source_abs, target_abs)
                            # final path relative to app root for frontend (static/separated/id/name)
                            final_rel_path = f"static/separated/{rec.id}/{fname}"
                        except Exception as e:
                            app.logger.error(f"[COPY] Failed to copy {source_abs} to {target_abs}: {e}")
                            final_rel_path = source_rel_path # Fallback
                    else:
                        app.logger.warning(f"[COPY] Source file not found: {source_abs}")
                
                if not final_rel_path:
                    # Fallback if no path provided (should not happen with new ai_service)
                    out_name = f"{idx+1:04d}.wav"
                    out_abs = os.path.join(sep_root, out_name)
                    # We strictly avoid ffmpeg cutting as per requirement, but if file is missing, we can't do much.
                    # Just mark it.
                    final_rel_path = f"static/separated/{rec.id}/{out_name}"

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
                    file_path=final_rel_path,
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

            rec.status = 'success'
            rec.current_stage = "处理完成"
            db.session.commit()
            app.logger.info(f"[Thread] Record {record_id} completed successfully.")

        except Exception as e:
            db.session.rollback()
            app.logger.error(f"[Thread] Failed processing record {record_id}: {e}")
            # Re-fetch record to avoid session issues
            rec = AudioRecord.query.get(record_id)
            if rec:
                rec.status = 'failed'
                rec.error_message = str(e)
                db.session.commit()
        finally:
            try:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
                    app.logger.info(f"[CLEAN] removed temp {temp_path}")
            except Exception:
                pass

@main.route("/")
def index():
    # 首页展示所有录音记录，按上传时间倒序
    records = AudioRecord.query.order_by(AudioRecord.upload_time.desc()).all()
    return render_template("index.html", records=records)

@main.route("/audio/upload", methods=["POST"])
@main.route("/upload", methods=["POST"])
@jwt_required()
def upload():
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"code": 400, "msg": "未选择文件"}), 400
    
    try:
        print("[UPLOAD] request received")
        temp_path = save_upload_file(file)
        print(f"[UPLOAD] temp_path={temp_path}")

        # 视频文件预检：探测是否包含音频流
        if is_video_file(file.filename):
            print(f"[UPLOAD] 检测到视频文件，正在探测音频流...")
            if not probe_has_audio(temp_path):
                import os as _os
                try:
                    _os.remove(temp_path)
                except Exception:
                    pass
                return jsonify({"code": 400, "msg": "该视频文件不包含音频轨道，无法转录"}), 400
            print(f"[UPLOAD] 视频音频流探测通过，将提取音频进行转录")
        
        current_user_id = int(get_jwt_identity())
        
        # 创建记录，初始状态为 pending
        rec = AudioRecord(
            original_filename=file.filename,
            filename="pending", # 占位，等待转码完成后更新
            duration=0.0,
            upload_time=datetime.utcnow(),
            status="pending",
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
        print(f"[DB] audio record created id={rec.id} status=pending")
        
        # 启动后台线程处理
        app = current_app._get_current_object()
        t = threading.Thread(target=process_audio_background, args=(app, rec.id, temp_path))
        t.start()
        
        return jsonify({
            "code": 200, 
            "msg": "上传成功，正在后台处理", 
            "data": {
                "record_id": rec.id, 
                "status": "pending"
            }
        })
        
    except Exception as e:
        current_app.logger.error(str(e))
        return jsonify({"code": 500, "msg": f"上传失败: {str(e)}"}), 500

@main.route("/detail/<int:record_id>")
def detail(record_id):
    record = AudioRecord.query.get_or_404(record_id)
    return render_template("detail.html", record=record)

@main.route("/record/<int:record_id>", methods=["DELETE"])
@jwt_required()
def delete_record(record_id):
    try:
        rec = AudioRecord.query.get(record_id)
        if not rec:
            return jsonify({"code": 404, "msg": "记录不存在"}), 404
            
        # Check permissions
        inspector = inspect(db.engine)
        cols = [c['name'] for c in inspector.get_columns('audio_record')]
        if 'user_id' in cols:
            uid = int(get_jwt_identity())
            if rec.user_id is not None and rec.user_id != uid:
                return jsonify({"code": 403, "msg": "无权删除该记录"}), 403
                
        # Delete files
        # 1. Main file (only if not pending placeholder)
        if rec.filename and rec.filename != "pending":
            file_abs = os.path.join(current_app.root_path, rec.filename)
            if os.path.exists(file_abs):
                try:
                    os.remove(file_abs)
                except Exception as e:
                    print(f"[DELETE] Failed to remove file {file_abs}: {e}")
                    
        # 2. Separated dir
        sep_dir = os.path.join(current_app.root_path, "static", "separated", str(rec.id))
        if os.path.exists(sep_dir):
            try:
                shutil.rmtree(sep_dir)
            except Exception as e:
                print(f"[DELETE] Failed to remove dir {sep_dir}: {e}")
                
        # Delete DB records
        DialogueSegment.query.filter_by(record_id=rec.id).delete()
        Split.query.filter_by(record_id=rec.id).delete()
        db.session.delete(rec)
        db.session.commit()
        
        return jsonify({"code": 200, "msg": "删除成功"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"code": 500, "msg": str(e)}), 500

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
            # Format duration as MM:SS
            dur = r.duration if r.duration else 0.0
            m = int(dur // 60)
            s = int(dur % 60)
            duration_str = f"{m:02d}:{s:02d}"
            
            # Format upload_time
            time_str = r.upload_time.strftime('%Y-%m-%d %H:%M:%S') if r.upload_time else ""
            
            # Determine filename display
            display_name = r.original_filename if r.original_filename else r.filename

            data.append({
                "id": r.id,
                "title": r.title,
                "filename": display_name,
                "filepath": r.filename,
                "original_filename": r.original_filename,
                "duration": duration_str,
                "upload_time": time_str,
                "status": r.status,
                "current_stage": getattr(r, 'current_stage', None)
            })
        return jsonify({"code": 200, "data": data})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)})

@main.route("/api/transcript_data/<int:record_id>", methods=["GET"])
@jwt_required()
def get_transcript_data(record_id):
    """
    专门服务于 LLM 预处理和前端核查页的标准化数据接口
    """
    try:
        # 1. 权限与存在性校验
        inspector = inspect(db.engine)
        cols = [c['name'] for c in inspector.get_columns('audio_record')]
        rec = AudioRecord.query.get(record_id)
        
        if not rec:
            return jsonify({"success": False, "msg": "记录不存在"}), 404
            
        if 'user_id' in cols:
            uid = int(get_jwt_identity())
            if rec.user_id is not None and rec.user_id != uid:
                return jsonify({"success": False, "msg": "无权访问该记录"}), 403
                
        # 2. 获取原始 segments
        segments = DialogueSegment.query.filter_by(record_id=record_id).order_by(DialogueSegment.start_time).all()
        
        # 3. 数据清洗与增强
        display_data = []
        llm_context_lines = []
        
        # 映射规则：spk0 -> A, spk1 -> B
        role_map = {
            "spk0": {"role": "A", "side": "left"},
            "spk1": {"role": "B", "side": "right"}
        }
        
        for idx, seg in enumerate(segments, 1):
            # 获取映射信息，默认 fallback 到 Unknown/left
            spk_info = role_map.get(seg.speaker, {"role": "Unknown", "side": "left"})
            
            # 构建前端展示数据
            item = {
                "seq_id": idx,
                "role": spk_info["role"],
                "side": spk_info["side"],
                "text": seg.content if seg.content else "",
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "speaker": seg.speaker # 保留原始标签备查
            }
            display_data.append(item)
            
            # 构建 LLM 上下文行
            # 格式：[seq_id] 角色{role}: {text}
            line = f"[{idx}] 角色{spk_info['role']}: {item['text']}"
            llm_context_lines.append(line)
            
        # 4. 返回结果
        return jsonify({
            "success": True,
            "data": {
                "display_data": display_data,
                "llm_context": "\n".join(llm_context_lines)
            }
        })
        
    except Exception as e:
        return jsonify({"success": False, "msg": str(e)}), 500


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
        splits = Split.query.filter_by(record_id=record_id).order_by(Split.start_time.asc()).all()
        payload = []
        for idx, s in enumerate(segs):
            if idx < len(splits) and splits[idx].file_path:
                path = splits[idx].file_path.replace("\\", "/")
            else:
                out_name = f"{idx:04d}.wav"
                path = os.path.join("static", "separated", str(record_id), out_name).replace("\\", "/")
            
            audio_url = f"/{path}" if not path.startswith("/") else path
            
            payload.append({
                "id": s.id,
                "spk": s.speaker,
                "text": s.content,
                "start": s.start_time,
                "end": s.end_time,
                "path": path,
                "audio_url": audio_url,
            })
        
        # Join with User table to get username
        user_name = "Unknown"
        if r.user_id:
            u = User.query.get(r.user_id)
            if u:
                user_name = u.username

        # Updated response structure
        data = {
            "info": {
                "id": r.id,
                "title": r.title,
                "original_filename": r.original_filename,
                "filename": r.filename,
                "upload_time": r.upload_time.isoformat(),
                "updated_at": r.updated_at.isoformat() if r.updated_at else r.upload_time.isoformat(),
                "user_name": user_name,
                "status": r.status,
                "error_message": r.error_message,
                "current_stage": r.current_stage
            },
            "segments": payload,
        }
        return jsonify({"code": 200, "data": data})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)})

@main.route("/record/<int:record_id>/segment/<int:segment_index>", methods=["PUT"])
@jwt_required()
def update_segment(record_id, segment_index):
    req_json = request.get_json()
    if not req_json or 'text' not in req_json:
        return jsonify({"code": 400, "message": "Missing 'text' field"}), 400
        
    new_text = req_json['text']
    
    try:
        inspector = inspect(db.engine)
        cols = [c['name'] for c in inspector.get_columns('audio_record')]
        rec = AudioRecord.query.get(record_id)
        if not rec:
            return jsonify({"code": 404, "message": "记录不存在"}), 404
            
        if 'user_id' in cols:
            uid = int(get_jwt_identity())
            if rec.user_id is not None and rec.user_id != uid:
                return jsonify({"code": 403, "message": "无权访问该记录"}), 403
                
        # 按照 start_time 升序获取 segments
        segments = DialogueSegment.query.filter_by(record_id=record_id).order_by(DialogueSegment.start_time.asc()).all()
        
        # 越界检查
        if segment_index < 0 or segment_index >= len(segments):
            return jsonify({"code": 404, "message": "片段索引越界"}), 404
            
        target_segment = segments[segment_index]
        target_segment.content = new_text
        
        # 手动触发表记录的时间戳更新 (因为只修改了子表记录)
        rec.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({
            "code": 200, 
            "message": "Segment updated successfully", 
            "data": {
                "index": segment_index, 
                "new_text": new_text
            }
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"code": 500, "message": str(e)}), 500

@main.route("/record/<int:record_id>/segment/<int:segment_index>/speaker", methods=["PUT"])
@jwt_required()
def update_segment_speaker(record_id, segment_index):
    req_json = request.get_json()
    if not req_json or 'speaker' not in req_json:
        return jsonify({"code": 400, "message": "Missing 'speaker' field"}), 400
        
    new_speaker = req_json['speaker']
    
    try:
        inspector = inspect(db.engine)
        cols = [c['name'] for c in inspector.get_columns('audio_record')]
        rec = AudioRecord.query.get(record_id)
        if not rec:
            return jsonify({"code": 404, "message": "记录不存在"}), 404
            
        if 'user_id' in cols:
            uid = int(get_jwt_identity())
            if rec.user_id is not None and rec.user_id != uid:
                return jsonify({"code": 403, "message": "无权访问该记录"}), 403
                
        segments = DialogueSegment.query.filter_by(record_id=record_id).order_by(DialogueSegment.start_time.asc()).all()
        if segment_index < 0 or segment_index >= len(segments):
            return jsonify({"code": 404, "message": "片段索引越界"}), 404
            
        target_segment = segments[segment_index]
        target_segment.speaker = new_speaker
        
        # 同时也需要更新对应的 Split 记录中的角色
        splits = Split.query.filter_by(record_id=record_id).order_by(Split.segment_index.asc()).all()
        if 0 <= segment_index < len(splits):
            splits[segment_index].speaker = new_speaker
            
        # 手动触发表记录的时间戳更新
        rec.updated_at = datetime.utcnow()
            
        db.session.commit()
        
        return jsonify({
            "code": 200, 
            "message": "Speaker updated successfully", 
            "data": {
                "index": segment_index, 
                "speaker": new_speaker
            }
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"code": 500, "message": str(e)}), 500

@main.route("/record/<int:record_id>/segment/<int:segment_index>/split", methods=["POST"])
@jwt_required()
def split_segment(record_id, segment_index):
    req_json = request.get_json()
    if not req_json or 'split_offset' not in req_json:
        return jsonify({"code": 400, "message": "Missing 'split_offset' field"}), 400
        
    split_offset = float(req_json['split_offset'])
    
    try:
        inspector = inspect(db.engine)
        cols = [c['name'] for c in inspector.get_columns('audio_record')]
        rec = AudioRecord.query.get(record_id)
        if not rec:
            return jsonify({"code": 404, "message": "记录不存在"}), 404
            
        if 'user_id' in cols:
            uid = int(get_jwt_identity())
            if rec.user_id is not None and rec.user_id != uid:
                return jsonify({"code": 403, "message": "无权访问该记录"}), 403
                
        segments = DialogueSegment.query.filter_by(record_id=record_id).order_by(DialogueSegment.start_time.asc()).all()
        if segment_index < 0 or segment_index >= len(segments):
            return jsonify({"code": 404, "message": "片段索引越界"}), 404
            
        target_seg = segments[segment_index]
        splits = Split.query.filter_by(record_id=record_id).order_by(Split.segment_index.asc()).all()
        target_split = splits[segment_index]
        
        original_start = target_seg.start_time
        original_end = target_seg.end_time
        split_point = original_start + split_offset
        
        if split_point <= original_start or split_point >= original_end:
            return jsonify({"code": 400, "message": "分割点超出当前片段范围"}), 400
            
        speaker = target_seg.speaker
        
        old_file_path = os.path.join(current_app.root_path, target_split.file_path)
        sep_root = os.path.join(current_app.root_path, "static", "separated", str(rec.id))
        
        import uuid
        part1_name = f"{uuid.uuid4().hex[:8]}.wav"
        part2_name = f"{uuid.uuid4().hex[:8]}.wav"
        part1_abs = os.path.join(sep_root, part1_name)
        part2_abs = os.path.join(sep_root, part2_name)
        
        duration = original_end - original_start
        _run_ffmpeg_cut(old_file_path, part1_abs, 0, split_offset)
        _run_ffmpeg_cut(old_file_path, part2_abs, split_offset, duration)
        
        part1_text = ""
        part2_text = ""
        ai_script_rel = "app/services/ai_service.py"
        
        # Translate part2_abs to WSL path format to pass as an extra arg
        drive2, tail2 = os.path.splitdrive(part2_abs)
        wsl_part2_abs = f"/mnt/{drive2.lower().rstrip(':')}{tail2.replace(os.sep, '/')}"
        
        try:
            # Batch process both parts in a single WSL call to cut total execution time by 50%
            batch_res = run_in_wsl(ai_script_rel, part1_abs, wsl_part2_abs, "--asr_only")
            if batch_res and len(batch_res) >= 2:
                part1_text = batch_res[0].get("text", "")
                part2_text = batch_res[1].get("text", "")
            elif batch_res and len(batch_res) == 1:
                # Fallback if somehow only 1 processed
                part1_text = batch_res[0].get("text", "")
        except Exception as e:
            current_app.logger.error(f"Batch retranscribe failed: {e}")
            
        db.session.delete(target_seg)
        db.session.delete(target_split)
        
        new_seg1 = DialogueSegment(
            record_id=rec.id, start_time=original_start, end_time=split_point, speaker=speaker, content=part1_text
        )
        new_seg2 = DialogueSegment(
            record_id=rec.id, start_time=split_point, end_time=original_end, speaker=speaker, content=part2_text
        )
        db.session.add(new_seg1)
        db.session.add(new_seg2)
        
        new_split1 = Split(
            record_id=rec.id, segment_index=target_split.segment_index, file_path=f"static/separated/{rec.id}/{part1_name}",
            start_time=original_start, end_time=split_point, speaker=speaker
        )
        new_split2 = Split(
            record_id=rec.id, segment_index=target_split.segment_index+1, file_path=f"static/separated/{rec.id}/{part2_name}",
            start_time=split_point, end_time=original_end, speaker=speaker
        )
        db.session.add(new_split1)
        db.session.add(new_split2)
        
        for s in splits[segment_index+1:]:
            s.segment_index += 1
            
        # 手动触发表记录的时间戳更新
        rec.updated_at = datetime.utcnow()
            
        db.session.commit()
        
        return jsonify({"code": 200, "message": "Split successful"})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"code": 500, "message": str(e)}), 500

@main.route("/api/summary/<int:record_id>", methods=["POST"])
@jwt_required()
def generate_summary(record_id):
    try:
        inspector = inspect(db.engine)
        cols = [c['name'] for c in inspector.get_columns('audio_record')]
        rec = AudioRecord.query.get(record_id)
        if not rec:
            return jsonify({"code": 404, "message": "记录不存在"}), 404
            
        if 'user_id' in cols:
            uid = int(get_jwt_identity())
            if rec.user_id is not None and rec.user_id != uid:
                return jsonify({"code": 403, "message": "无权访问该记录"}), 403

        # 检查是否已有缓存的摘要
        if getattr(rec, 'llm_summary', None):
            try:
                cached_obj = json.loads(rec.llm_summary)
                return jsonify({"code": 200, "data": {"summary": cached_obj}})
            except Exception:
                pass # 如果解析失败则回退到重新生成
                
        # 获取按照时间排序的分段
        segs = DialogueSegment.query.filter_by(record_id=record_id).order_by(DialogueSegment.start_time.asc()).all()
        if not segs:
            return jsonify({"code": 400, "message": "当前记录无对话片段"}), 400
            
        # 组装格式化所需的结构
        dict_segs = [{"speaker": s.speaker, "text": s.content} for s in segs]
        
        transcript_data = format_transcript(dict_segs)
        full_text = transcript_data.get("full_transcript", "")
        
        if not full_text:
            return jsonify({"code": 400, "message": "提取的对话文本为空"}), 400
            
        # 请求本地微调大模型
        summary_dict = request_local_llm(full_text)
        
        # 保存 AI 给出的 JSON 报告结构
        rec.llm_summary = json.dumps(summary_dict, ensure_ascii=False)
        db.session.commit()
        
        return jsonify({"code": 200, "data": {"summary": summary_dict}})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"code": 500, "message": str(e)}), 500


# ---------------------------------------------------------------------------
# Dashboard 接口
# ---------------------------------------------------------------------------

@main.route("/api/system/heartbeat", methods=["POST"])
@jwt_required()
def system_heartbeat():
    """
    系统活跃心跳接口：由前端定时调用。
    逻辑：若该用户最近一次会话的结束时间在 60 秒内，则更新结束时间；否则开启新会话。
    以此解决“无法确定结束时间”的问题，将会话持续时间累加进数据库。
    """
    uid = int(get_jwt_identity())
    now = datetime.utcnow()
    last_session = SystemSession.query.filter_by(user_id=uid).order_by(SystemSession.end_time.desc()).first()
    
    # 允许 60 秒的回旋余地（心跳频率通常为 30 秒一次）
    if last_session and (now - last_session.end_time).total_seconds() < 60:
        last_session.end_time = now
    else:
        new_session = SystemSession(user_id=uid, start_time=now, end_time=now)
        db.session.add(new_session)
    
    db.session.commit()
    return jsonify({"code": 200, "msg": "Heartbeat received"})

def _get_cumulative_usage_seconds(uid=None):
    """内部工具函数：计算累计系统使用时长（秒）"""
    query = db.session.query(
        func.sum((func.julianday(SystemSession.end_time) - func.julianday(SystemSession.start_time)) * 86400)
    )
    if uid is not None:
        query = query.filter(SystemSession.user_id == uid)
    res = query.scalar()
    return float(res) if res else 0.0

@main.route("/api/dashboard/health", methods=["GET"])
def dashboard_health():
    """
    实时检测核心 AI 引擎存活状态。
    - LLM：向 Ollama 发送轻量探活请求 (GET /)
    - ASR：检查 Paraformer 模型目录是否存在于磁盘
    """
    import requests as _req

    # --- LLM 检测 ---
    llm_online = False
    try:
        resp = _req.get("http://127.0.0.1:11434/", timeout=3)
        llm_online = resp.status_code == 200
    except Exception:
        llm_online = False

    # --- ASR 检测 ---
    # 检查 Paraformer 模型目录是否在磁盘上存在（模型已下载 = 可调用）
    asr_online = False
    try:
        project_dir = os.path.dirname(current_app.root_path)
        paraformer_path = os.path.join(
            project_dir, "models", "lukeewin01", "paraformer-large-sichuan-offline"
        )
        asr_online = os.path.isdir(paraformer_path)
    except Exception:
        asr_online = False

    return jsonify({
        "code": 200,
        "data": {
            "asr_online": asr_online,
            "llm_online": llm_online,
        }
    })


@main.route("/api/dashboard/ambient", methods=["GET"])
def dashboard_ambient():
    """
    返回地理位置、天气、气温和全系统累计运行时间（基于会话累计）。
    """
    import requests as _req

    # WMO 天气码 → 中文描述
    WMO_WEATHER = {
        0: "晴", 1: "晴间多云", 2: "多云", 3: "阴", 45: "雾", 48: "冻雾",
        51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨", 61: "小雨", 63: "中雨", 65: "大雨",
        71: "小雪", 73: "中雪", 75: "大雪", 77: "冰粒", 80: "阵雨", 81: "中阵雨", 82: "强阵雨",
        85: "小阵雪", 86: "强阵雪", 95: "雷暴", 96: "雷暴伴冰雹", 99: "强雷暴伴冰雹",
    }

    # --- Uptime 计算 (从 SystemSession 统计全系统总的使用时长) ---
    total_seconds = _get_cumulative_usage_seconds()
    uptime_days = int(total_seconds // 86400)
    uptime_hours = int((total_seconds % 86400) // 3600)

    location = "未知"
    weather = "无法获取"
    temperature = None

    try:
        geo_resp = _req.get("http://ip-api.com/json/?lang=zh-CN&fields=city,lat,lon,status", timeout=5)
        geo_data = geo_resp.json()
        if geo_data.get("status") == "success":
            location = geo_data.get("city", "未知")
            lat, lon = geo_data.get("lat"), geo_data.get("lon")
            if lat is not None and lon is not None:
                weather_resp = _req.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={"latitude": lat, "longitude": lon, "current": "temperature_2m,weathercode", "timezone": "Asia/Shanghai"},
                    timeout=5,
                )
                weather_data = weather_resp.json()
                current = weather_data.get("current", {})
                wmo_code = current.get("weathercode")
                temperature = round(current.get("temperature_2m")) if current.get("temperature_2m") is not None else None
                weather = WMO_WEATHER.get(wmo_code, f"天气码 {wmo_code}")
    except Exception as e:
        current_app.logger.warning(f"[Dashboard] Ambient fetch failed: {e}")

    return jsonify({
        "code": 200,
        "data": {
            "location": location,
            "weather": weather,
            "temperature": temperature,
            "uptime_days": uptime_days,
            "uptime_hours": uptime_hours,
        }
    })


@main.route("/api/dashboard/stats", methods=["GET"])
@jwt_required()
def dashboard_stats():
    """
    当前用户的业务速览数据：
    - uptime_hours/days: 统计该用户在 SystemSession 表中的累计活跃时长。
    """
    uid = int(get_jwt_identity())

    # 1. 累计转写数
    total_transcribed = AudioRecord.query.filter(AudioRecord.user_id == uid, AudioRecord.status == "success").count()

    # 2. 深度总结数
    total_summarized = AudioRecord.query.filter(AudioRecord.user_id == uid, AudioRecord.llm_summary.isnot(None), AudioRecord.llm_summary != "").count()

    # 3. 用户累计系统使用时长 (从会话表计算)
    total_seconds = _get_cumulative_usage_seconds(uid)
    uptime_days = int(total_seconds // 86400)
    uptime_hours = int((total_seconds % 86400) // 3600)

    return jsonify({
        "code": 200,
        "data": {
            "total_transcribed": total_transcribed,
            "total_summarized": total_summarized,
            "uptime_days": uptime_days,
            "uptime_hours": uptime_hours,
        }
    })

@main.route("/api/dashboard/keywords", methods=["GET"])
@jwt_required()
def dashboard_keywords():
    """
    高频词云数据：
    - 读取当前用户拥有 LLM 总结的文本
    - 提取高频词及权重（简化实现，结合结巴分词或直接字符串统计）
    """
    uid = int(get_jwt_identity())
    records = AudioRecord.query.filter(
        AudioRecord.user_id == uid,
        AudioRecord.llm_summary.isnot(None),
        AudioRecord.llm_summary != ""
    ).limit(50).all()  # 取最近的即可避免性能问题

    text_corpus = ""
    for r in records:
        try:
            summary = json.loads(r.llm_summary)
            # 拼合可能产生独立意图或总结的文本
            overview = summary.get("summary", {}).get("overview", {})
            text_corpus += overview.get("scene_type", "") + " "
            text_corpus += overview.get("detailed_summary", "") + " "
            
            breakdown = summary.get("summary", {}).get("analysis_breakdown", {})
            for role_k, role_v in breakdown.items():
                if isinstance(role_v, dict):
                    text_corpus += role_v.get("role_label", "") + " "
                    text_corpus += role_v.get("emotional_state", "") + " "
                    text_corpus += role_v.get("hidden_intent", "") + " "
                    vfa = role_v.get("VFA_analysis", {})
                    if isinstance(vfa, dict):
                        text_corpus += vfa.get("viewpoint", "") + " "
                        facts = vfa.get("facts", [])
                        if isinstance(facts, list):
                            text_corpus += " ".join(facts) + " "
                        text_corpus += vfa.get("deep_analysis", "") + " "
        except Exception:
            pass

    # 尝试使用 jieba 提取关键词
    word_counts = {}
    if text_corpus.strip():
        try:
            import jieba.analyse
            # 提取 top 20 关键词，并将权重放大适配前端 ECharts
            tags = jieba.analyse.extract_tags(text_corpus, topK=20, withWeight=True)
            for tag, weight in tags:
                word_counts[tag] = int(weight * 100)
        except ImportError:
            # Fallback：无 jieba 时，用正则对中文字符做词频统计
            import re
            from collections import Counter
            # 提取长度 >= 2 的连续中文词片段
            words = re.findall(r'[\u4e00-\u9fa5]{2,6}', text_corpus)
            # 简单停用词过滤
            stop = {"的", "了", "是", "在", "我", "你", "他", "她", "我们", "就是", "这个", "那个",
                    "一个", "没有", "可以", "这样", "那么", "因为", "所以"}
            counter = Counter(w for w in words if w not in stop)
            for word, cnt in counter.most_common(20):
                word_counts[word] = cnt

    data = [{"name": k, "value": v} for k, v in word_counts.items() if v > 0]
    
    # 按 value 降序
    data = sorted(data, key=lambda x: x["value"], reverse=True)

    return jsonify({
        "code": 200,
        "data": data
    })

@main.route("/api/dashboard/recent_records", methods=["GET"])
@jwt_required()
def dashboard_recent_records():
    """
    按时间倒序获取当前用户的最近 5 条记录。
    """
    uid = int(get_jwt_identity())
    records = AudioRecord.query.filter_by(user_id=uid).order_by(
        AudioRecord.upload_time.desc()
    ).limit(5).all()

    def format_time_ago(d):
        if not d:
            return "位置时间"
        now = datetime.utcnow()
        diff = now - d
        sec = diff.total_seconds()
        if sec < 60:
            return "刚刚"
        elif sec < 3600:
            return f"{int(sec // 60)}分钟前"
        elif sec < 86400:
            return f"{int(sec // 3600)}小时前"
        else:
            return f"{int(sec // 86400)}天前"
            
    # 状态展示映射
    status_map = {
        "pending": ("等待中", "pending"),
        "processing": ("转写中", "transcribing"),
        "success": ("分析完成", "analyzed"),
        "failed": ("失败", "failed")
    }

    data = []
    for r in records:
        lbl, st = status_map.get(r.status, ("未知", "unknown"))
        data.append({
            "id": r.id,
            "title": r.title,
            "original_filename": r.original_filename,
            "upload_time": r.upload_time.isoformat() if r.upload_time else None,
            "created_at": format_time_ago(r.upload_time),
            "status": st,
            "status_label": lbl
        })

    return jsonify({
        "code": 200,
        "data": data
    })

@main.route("/record/<int:record_id>/title", methods=["PUT"])
@jwt_required()
def update_title(record_id):
    req_json = request.get_json()
    if not req_json or 'title' not in req_json:
        return jsonify({"code": 400, "message": "Missing 'title' field"}), 400
        
    new_title = req_json['title']
    
    try:
        inspector = inspect(db.engine)
        cols = [c['name'] for c in inspector.get_columns('audio_record')]
        rec = AudioRecord.query.get(record_id)
        if not rec:
            return jsonify({"code": 404, "message": "记录不存在"}), 404
            
        if 'user_id' in cols:
            uid = int(get_jwt_identity())
            if rec.user_id is not None and rec.user_id != uid:
                return jsonify({"code": 403, "message": "无权访问该记录"}), 403
                
        rec.title = new_title
        db.session.commit()
        
        return jsonify({
            "code": 200, 
            "message": "Title updated successfully", 
            "data": {
                "id": record_id, 
                "title": new_title,
                "updated_at": rec.updated_at.isoformat() if rec.updated_at else datetime.utcnow().isoformat()
            }
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"code": 500, "message": str(e)}), 500
