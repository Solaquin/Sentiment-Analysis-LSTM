import pandas as pd
import tensorflow as tf
from tokenization_module import TokenizerModule

#Cargar datasets
train = pd.read_csv("dataset_clean/train_clean.csv")
val = pd.read_csv("dataset_clean/validation_clean.csv")
test = pd.read_csv("dataset_clean/test_clean.csv")

tok = TokenizerModule(vocab_size=20000, sequence_length=200)
tok.fit_vectorizer(train['review_body'].astype(str))