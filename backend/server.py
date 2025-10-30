from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from utils.tokenization_module import TokenizerModule
import os

# Rutas relativas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "../frontend")
MODEL_PATH = os.path.join(BASE_DIR, "Models/model_lstm_v1.keras")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer")

app = Flask(__name__, static_folder=os.path.join(FRONTEND_DIR), static_url_path="/")

#Cargar modelo
model = tf.keras.models.load_model(MODEL_PATH)

#Carga del tokenizador
tok = TokenizerModule()
tok.load_vectorizer(VECTORIZER_PATH)
encoder = tok.vectorizer


@app.route("/")
def home():
    return app.send_static_file("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    reviews = data.get("reviews", [])

    if not reviews:
        return jsonify({"error": "No se enviaron reseñas"}), 400

    X = encoder(reviews)
    preds = model.predict(X)

    preds = np.array(preds).flatten()
    results = preds.tolist()


    return jsonify({"predictions": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))