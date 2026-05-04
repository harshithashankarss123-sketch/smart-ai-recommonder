import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv('../data/dataset.csv')

X = data[['hours_studied', 'attendance', 'sleep']]
y = data['performance']

model = RandomForestClassifier()
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained successfully!")