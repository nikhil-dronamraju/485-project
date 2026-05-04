import pandas as pd
import tensorflow as tf
from keras import layers
from model import model


train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/val.csv")

train_texts, train_labels = train_df["text"].astype(str).values, train_df["label"].values
val_texts, val_labels = val_df["text"].astype(str).values, val_df["label"].values

vectorizer = layers.TextVectorization(max_tokens=20_000, output_sequence_length=128)
vectorizer.adapt(train_texts)

X_train = vectorizer(train_texts)
X_val = vectorizer(val_texts)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("Training")
history = model.fit(
    X_train, train_labels,
    epochs=3,
    batch_size=32,
    validation_data=(X_val, val_labels),
    verbose=1
)

model.save("models/best_model.keras")