import os
import sys
import json
import shutil
import subprocess
import threading
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from sqlalchemy import inspect
from . import db
from .models import AudioRecord, DialogueSegment, Split
from .services.audio_handler import save_upload_file, convert_to_16k_wav
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
        payload = []
        for idx, s in enumerate(segs):
            # [Fix] Files are 0-indexed (0000.wav), so use idx directly
            out_name = f"{idx:04d}.wav"
            path = os.path.join("static", "separated", str(record_id), out_name).replace("\\", "/")
            
            # [Requirement] Add audio_url with leading slash
            audio_url = f"/{path}"
            
            payload.append({
                "id": s.id,
                "spk": s.speaker,
                "text": s.content,
                "start": s.start_time,
                "end": s.end_time,
                "path": path,
                "audio_url": audio_url,
            })
        
        # Updated response structure
        data = {
            "info": {
                "id": r.id,
                "filename": r.filename,
                "upload_time": r.upload_time.isoformat(),
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

