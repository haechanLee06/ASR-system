from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

db = SQLAlchemy()

def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object("config.Config")
    db.init_app(app)
    CORS(app, resources={r"/*": {"origins": "*"}})

    from .routes import main
    app.register_blueprint(main)

    with app.app_context():
        db.create_all()

    return app

