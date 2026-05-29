import os
import json
import pandas as pd
from groq import Groq
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("PokemonDatabase.csv")

features = ["Health Stat", "Attack Stat", "Defense Stat",
            "Special Attack Stat", "Special Defense Stat",
            "Speed Stat", "Base Stat Total"]

tier_map = {'G': 0, 'F': 1, 'E': 2, 'D': 3, 'C': 4, 'B': 5, 'A': 6, 'S': 7}
dataset['Power Ranking'] = (dataset['Power Ranking']
                            .astype(str).str.replace('"', '').str.strip()
                            .map(tier_map))

X = dataset[features].values
y = dataset["Power Ranking"].values

X_train, X_test_tmp, y_train, y_test_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_dev, X_test, y_dev, y_test = train_test_split(X_test_tmp, y_test_tmp, test_size=0.50, random_state=42, stratify=y_test_tmp)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_dev   = scaler.transform(X_dev)
X_test  = scaler.transform(X_test)

model = MLPClassifier(hidden_layer_sizes=(10,), activation="relu", alpha=0.0001, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

train_accuracy = accuracy_score(y_train, model.predict(X_train))
dev_accuracy   = accuracy_score(y_dev,   model.predict(X_dev))
class_report   = classification_report(y_dev, model.predict(X_dev), target_names=["G","F","E","D","C","B","A","S"])

print("Train accuracy:", train_accuracy)
print("Dev accuracy:", dev_accuracy)
print("Classification Report:\n", class_report)

api_key = input("\nEnter your Groq API key: ").strip()
client = Groq(api_key=api_key)

prompt = f"""
I have a Pokemon tier classification model using MLPClassifier with these results:

Train accuracy: {train_accuracy:.4f}
Dev accuracy: {dev_accuracy:.4f}

Classification Report:
{class_report}

Please:
1. Identify which tiers are performing poorly and why
2. Explain what is causing the model to struggle
3. Give specific and actionable suggestions to improve the model
"""

print("\nSending data to Groq for analysis... Please wait.")
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}]
)

print("\n========== GROQ SUGGESTIONS ==========\n")
print(response.choices[0].message.content)