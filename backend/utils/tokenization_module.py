import tensorflow as tf
import pandas as pd
import json
import os

class TokenizerModule:

    def __init__(self, vocab_size=20000, sequence_length=500):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.vectorizer = None
        self.vocabulary = None
    
    def build_vectorizer(self):
        self.vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=self.vocab_size,
            output_mode='int',
            output_sequence_length=self.sequence_length,
            standardize='lower_and_strip_punctuation'
        )
    
    def fit_vectorizer(self, texts):
        if self.vectorizer is None:
            self.build_vectorizer()
        self.vectorizer.adapt(texts)
    
    def vectorize_texts(self, texts):
        if self.vectorizer is None:
            raise ValueError("Vectorizer has not been built or fitted.")
        
        return self.vectorizer(texts)

    def save_vectorizer(self, path = "vectorizer"):
        if self.vectorizer is None:
            raise ValueError("Vectorizer has not been built or fitted.")
        
        os.makedirs(path, exist_ok=True)
        vocab = self.vectorizer.get_vocabulary()


        with open(os.path.join(path, 'vocab.json'), 'w', encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

    def load_vectorizer(self, path = "vectorizer"):
        with open(os.path.join(path, 'vocab.json'), 'r', encoding="utf-8") as f:
            vocab = json.load(f)

        self.build_vectorizer()
        self.vectorizer.set_vocabulary(vocab)

if __name__ == "__main__":
    # Crear y guardar el vectorizador solo si se ejecuta directamente
    train = pd.read_csv("data/dataset_clean_v2/train_clean_v2.csv")
    train_texts = train['review_body'].astype(str)

    tok = TokenizerModule()
    tok.fit_vectorizer(train_texts)
    tok.save_vectorizer("backend/vectorizer")

    print("✅ Vectorizador creado y guardado en 'vectorizer/vocab.json'")
