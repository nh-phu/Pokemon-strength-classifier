import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

dataset = pd.read_csv("PokemonDatabase.csv")
dataset.head()

from sklearn.model_selection import train_test_split

features = ["Health Stat", "Attack Stat", "Defense Stat", "Special Attack Stat", "Special Defense Stat", "Speed Stat", "Base Stat Total"]

tier_map = {'G': 0,'F': 1, 'E': 2, 'D': 3, 'C': 4, 'B': 5, 'A': 6, 'S': 7}
dataset['Power Ranking'] = dataset['Power Ranking'].astype(str).str.replace('"', '').str.strip().map(tier_map)

X = dataset[features].values
y = dataset["Power Ranking"].values

X_train, X_test_tmp, y_train, y_test_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_dev, X_test, y_dev, y_test = train_test_split(X_test_tmp, y_test_tmp, test_size=0.50, random_state=42, stratify=y_test_tmp)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_dev = scaler.transform(X_dev)
X_test = scaler.transform(X_test)

model = MLPClassifier(
    hidden_layer_sizes=(10,),
    activation="relu",
    alpha=0.0001,
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_dev_pred = model.predict(X_dev)

train_accuracy = accuracy_score(y_train, y_train_pred)
print("Train accuracy:", train_accuracy)
dev_accuracy = accuracy_score(y_dev, y_dev_pred)
print("Dev accuracy:", dev_accuracy)

class_report = classification_report(y_dev, y_dev_pred, target_names=["G", "F", "E", "D", "C", "B", "A", "S"])
print("Classification Report:\n", class_report)


import os
import json
import pandas as pd
from groq import Groq
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

api_key = input("Enter your Groq API key: ").strip()
client = Groq(api_key=api_key)

prompt = f"""
I have a Pokemon tier classification model using MLPClassifier with these results:

Train accuracy: {train_accuracy:.4f}
Dev accuracy: {dev_accuracy:.4f}

Classification Report:
{class_report}

Requirements:
Suggest better MLPClassifier hyperparameters to improve dev accuracy.
Reply ONLY with a valid JSON object, no explanation, no markdown:
{{
  "hidden_layer_sizes": [128, 64],
  "activation": "relu",
  "alpha": 0.001,
  "learning_rate_init": 0.001,
  "max_iter": 1000
}}
"""

print("Sending data to Groq for analysis... Please wait.")
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}]
)

print("\n========== REPORT FROM GROQ ==========\n")
raw = response.choices[0].message.content
print(raw)

try:
    params = json.loads(raw)
    params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])

    improved = MLPClassifier(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        activation=params.get("activation", "relu"),
        alpha=params.get("alpha", 0.0001),
        learning_rate_init=params.get("learning_rate_init", 0.001),
        max_iter=params.get("max_iter", 1000),
        random_state=42
    )
    improved.fit(X_train, y_train)

    new_train_acc = accuracy_score(y_train, improved.predict(X_train))
    new_dev_acc   = accuracy_score(y_dev,   improved.predict(X_dev))

    print("\n========== IMPROVED MODEL RESULTS ==========\n")
    print(f"Train accuracy : {new_train_acc:.4f}  (was {train_accuracy:.4f})")
    print(f"Dev   accuracy : {new_dev_acc:.4f}  (was {dev_accuracy:.4f})")
    print("\nClassification Report:\n",
          classification_report(y_dev, improved.predict(X_dev),
                                target_names=["G","F","E","D","C","B","A","S"]))

    if new_dev_acc > dev_accuracy:
        print("✅ Groq's suggestion improved the model!")
    else:
        print("⚠️  No improvement — try running again.")

except json.JSONDecodeError:
    print("❌ Could not parse Groq's response as JSON.")