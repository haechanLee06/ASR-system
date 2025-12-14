from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from sqlalchemy import inspect, text

db = SQLAlchemy()

def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object("config.Config")
    db.init_app(app)
    CORS(app, resources={r"/*": {"origins": "*"}})
    app.config.setdefault("JWT_SECRET_KEY", "dev-secret-change-me")
    JWTManager(app)

    from .routes import main
    app.register_blueprint(main)
    try:
        from .routes_auth import auth
        app.register_blueprint(auth, url_prefix="/auth")
    except Exception:
        pass

    with app.app_context():
        db.create_all()
        try:
            insp = inspect(db.engine)
            cols = [c['name'] for c in insp.get_columns('audio_record')]
            if 'user_id' not in cols:
                db.session.execute(text('ALTER TABLE audio_record ADD COLUMN user_id INTEGER'))
                db.session.commit()
        except Exception:
            pass

    return app

