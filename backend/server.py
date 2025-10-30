from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from utils.tokenization_module import TokenizerModule

app = Flask(__name__, static_folder="../frontend", static_url_path="/")

# Cargar modelo al iniciar el servidor
model_path = "backend/Models/model_lstm_v1.keras"
model = tf.keras.models.load_model(model_path)

#Carga del tokenizador
tok = TokenizerModule()
tok.load_vectorizer("backend/vectorizer")
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

    preproccesed = [encoder([r]) for r in reviews]

    X = np.vstack(preproccesed)
    preds = model.predict(X)

    results = [float(p) for p in np.squeeze(preds)]


    return jsonify({"predictions": results})

if __name__ == "__main__":
    app.run(debug=True)