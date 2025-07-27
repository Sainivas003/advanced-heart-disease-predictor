import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/Heart%20Disease/heart-disease.csv'
df = pd.read_csv(url)

# Features and labels
X = df.drop('target', axis=1)
y = df['target']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
