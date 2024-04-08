import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

#df = pd.read_csv("data/train.csv")
# Introduce deliberate error in data loading process
try:
    df = pd.read_csv("non_existent_file.csv")  # Change to a non-existent file path
except FileNotFoundError:
    raise Exception("Failed to load data: File not found")

X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()

# Introduce deliberate error in model training process
try:
    model = LogisticRegression()  # Change to an invalid model class or parameters
    model.fit(X, y)
except Exception as e:
    raise Exception("Failed to train model: " + str(e))


#model = LogisticRegression().fit(X, y)

with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)
