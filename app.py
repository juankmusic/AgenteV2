#app.py

# Archivo principal de la aplicación Flask
from flask import Flask
from routes.chat_routes import chat_bp

# Configuración de la aplicación Flask
app = Flask(__name__)
app.secret_key = "tu_secreto_aqui"
app.register_blueprint(chat_bp)  

# Ejecutar la aplicación Flask
if __name__ == "__main__":
    app.run(debug=True)
