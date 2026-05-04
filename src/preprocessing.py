# TODO: Data may need additional cleaning
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import numpy as np


df = pd.read_csv("data/raw/data.csv")
df["label"] = np.where(df["source"] == "Human", 0, 1)
adversarial = df[df["source"] == "GPT-3.5"].drop(columns=["source"])
df_main = df[df["source"] != "GPT-3.5"]

sampled_groups = []

for source_name, group_df in df_main.groupby("source"):
    n = min(len(group_df), 50_000)
    sampled = group_df.sample(n, random_state=42)
    sampled_groups.append(sampled)

df_balanced = pd.concat(sampled_groups).reset_index(drop=True)

train, temp = train_test_split(df_balanced, test_size=0.3, stratify=df_balanced["label"], random_state=42) 
val, test = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=42)

train.to_csv("data/processed/train.csv", index=False, quoting=csv.QUOTE_ALL)
val.to_csv("data/processed/val.csv", index=False, quoting=csv.QUOTE_ALL)
test.to_csv("data/processed/test.csv", index=False, quoting=csv.QUOTE_ALL)
adversarial.to_csv("data/processed/adversarial.csv", index=False, quoting=csv.QUOTE_ALL)
