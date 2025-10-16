from tokenization_module import TokenizerModule
import numpy as np
import tensorflow as tf

model_path = "model/sentiment_model.keras"
model = tf.keras.models.load_model(model_path)

string = input("Introduzca un texto para analizar sentimiento: ")
print(f"Input text: {string}")

# Vectorizar primero
tok = TokenizerModule()
tok.load_vectorizer("vectorizer")
encoder = tok.vectorizer
sample_seq = encoder([string])  # Tensor de enteros

# Predecir
predictions = model.predict(sample_seq)
print(f"Predicted class: {np.argmax(predictions[0])}, Probabilities: {predictions[0]}")

if np.argmax(predictions[0]) == 0:
    print("Sentimiento negativo")
elif np.argmax(predictions[0]) == 1:
    print("Sentimiento neutral")
else:
    print("Sentimiento positivo")

print("Done")

