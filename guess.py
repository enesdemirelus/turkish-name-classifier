import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = tf.keras.models.load_model("turkish_name_model.keras")

while True:
    name = input("Name: ").strip()
    if name == "":
        continue
    seq = tokenizer.texts_to_sequences([name])
    pad = pad_sequences(seq, maxlen=30)
    pred = model.predict(pad)[0][0]
    if (pred > 0.5):
        print(f"{int(pred * 100)}% Turkish")
    else:
        print(f"{int(100 - (pred * 100))}% Non Turkish")
