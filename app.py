from flask import Flask
from routes.chat_routes import chat_bp

app = Flask(__name__)
app.secret_key = "tu_secreto_aqui"
app.register_blueprint(chat_bp)  # sin url_prefix está bien para que la raíz sea /

if __name__ == "__main__":
    app.run(debug=True)
