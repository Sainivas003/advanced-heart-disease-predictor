
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

data = {
    'age': [63, 37, 41, 56, 57, 57],
    'sex': [1, 1, 0, 1, 0, 1],
    'cp': [3, 2, 1, 1, 0, 0],
    'trestbps': [145, 130, 130, 120, 120, 140],
    'chol': [233, 250, 204, 236, 354, 192],
    'fbs': [1, 0, 0, 0, 0, 0],
    'restecg': [0, 1, 0, 1, 1, 1],
    'thalach': [150, 187, 172, 178, 163, 148],
    'exang': [0, 0, 0, 0, 1, 0],
    'oldpeak': [2.3, 3.5, 1.4, 0.8, 0.6, 0.4],
    'slope': [0, 0, 2, 2, 2, 1],
    'ca': [0, 0, 0, 0, 0, 0],
    'thal': [1, 2, 2, 2, 2, 1],
    'target': [1, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)
X = df.drop("target", axis=1)
y = df["target"]

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ model.pkl created.")
