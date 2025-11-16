import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

turkish_names = pd.read_csv("dataset/turkish_names.csv")
nonturkish_names = pd.read_csv("dataset/nonturkish_names.csv")

turkish_names["label"] = 1
nonturkish_names["label"] = 0

df = pd.concat([turkish_names, nonturkish_names]).reset_index(drop=True)
df = df.sample(frac=1).reset_index(drop=True)

X = df["name"].astype(str)
y = df["label"].astype(int)

tokenizer = Tokenizer(char_level=True) 
tokenizer.fit_on_texts(X)

X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=30) 


X_train, X_val, y_train, y_val = train_test_split(
    X_pad, y, test_size=0.2, random_state=42
)

vocab_size = len(tokenizer.word_index) + 1

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64, input_length=30),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=32
)


with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
    
model.save("turkish_name_model.keras")
