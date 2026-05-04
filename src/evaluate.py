import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.metrics import classification_report, confusion_matrix

test_df = pd.read_csv("data/processed/test.csv").sample(10000, random_state=42)
adversarial_df = pd.read_csv("data/processed/adversarial.csv").sample(5000, random_state=42)

test_texts, test_labels = test_df["text"].astype(str).values, test_df["label"].values
adv_texts, adv_labels = adversarial_df["text"].astype(str).values, adversarial_df["label"].values

train_df = pd.read_csv("data/processed/train.csv")
vectorizer = layers.TextVectorization(max_tokens=20_000, output_sequence_length=128)
vectorizer.adapt(train_df["text"].astype(str).values)

model = tf.keras.models.load_model("models/best_model.keras")

def evaluate(texts, labels):
    X = vectorizer(texts)
    probs = model.predict(X).flatten()
    preds = (probs >= 0.5).astype(int)
    print("Classification report:")
    print(classification_report(labels, preds, target_names=["Human", "LLM"]))
    print("Confusion matrix:")
    print(confusion_matrix(labels, preds))

evaluate(test_texts, test_labels)