from tokenization_module import TokenizerModule
import numpy as np
import tensorflow as tf

model_path = "Models/model_lstm_v1.keras"
model = tf.keras.models.load_model(model_path)

string = input("Introduzca un texto para analizar sentimiento: ")
print(f"Input text: {string}")

# Vectorizar primero
tok = TokenizerModule()
tok.load_vectorizer("vectorizer")
encoder = tok.vectorizer
sample_seq = encoder([string])  # Tensor de enteros

# Predecir
prediction = model.predict(sample_seq)
pred_class = (prediction[0] >= 0.5).astype(int)  # 1 si >= 0.5, sino 0
print(f"Predicted class: {pred_class}, Probabilities: {prediction}")

label = "Positivo" if pred_class == 1 else "Negativo"
print(f"Predicted sentiment: {label} ({prediction[0][0]:.4f})")


print("Done")

