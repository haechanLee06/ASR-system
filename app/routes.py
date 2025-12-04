import os
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, current_app
from werkzeug.utils import secure_filename
from . import db
from .models import AudioRecord
from .services.audio_handler import save_upload_file, convert_to_16k_wav

main = Blueprint("main", __name__)

@main.route("/")
def index():
    records = AudioRecord.query.order_by(AudioRecord.upload_time.desc()).all()
    return render_template("index.html", records=records)

@main.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"code": 400, "msg": "未选择文件"}), 400
    try:
        temp_path = save_upload_file(file)
        base = os.path.splitext(secure_filename(file.filename))[0]
        out_name = datetime.utcnow().strftime("%Y%m%d%H%M%S") + "_" + base + ".wav"
        rel_path, duration = convert_to_16k_wav(temp_path, out_name)
        rec = AudioRecord(
            filename=rel_path,
            original_filename=file.filename,
            upload_time=datetime.utcnow(),
            duration=duration,
            status="uploaded",
        )
        db.session.add(rec)
        db.session.commit()
        return jsonify({"code": 200, "msg": "上传成功", "data": {"id": rec.id}})
    except Exception as e:
        current_app.logger.error(str(e))
        return jsonify({"code": 500, "msg": f"上传失败: {str(e)}"}), 500
    finally:
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass

@main.route("/detail/<int:record_id>")
def detail(record_id):
    record = AudioRecord.query.get_or_404(record_id)
    return render_template("detail.html", record=record)

