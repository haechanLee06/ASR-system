from app import create_app
import os

app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    host = os.getenv("HOST", "0.0.0.0")
    app.run(debug=True, host=host, port=port)

