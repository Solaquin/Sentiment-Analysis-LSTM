import pandas as pd
import tensorflow as tf
from tokenization_module import TokenizerModule

#Cargar datasets
train = pd.read_csv("dataset_clean/train_clean.csv")
val = pd.read_csv("dataset_clean/validation_clean.csv")
test = pd.read_csv("dataset_clean/test_clean.csv")

train_texts = train['review_body'].astype(str)

tok = TokenizerModule()
tok.fit_vectorizer(train_texts)
tok.save_vectorizer("vectorizer")

#Vectorizar los textos
X_train = tok.vectorize_texts(train['review_body'].astype(str))
X_val = tok.vectorize_texts(val['review_body'].astype(str))
X_test = tok.vectorize_texts(test['review_body'].astype(str))

#Tomar labels
Y_train = train["label"]
Y_val = val["label"]
Y_test = test["label"]




