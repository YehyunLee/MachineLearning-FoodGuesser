import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../utils')
from preprocess import preprocess

DATASET_PATH = '../data/cleanedWithScript/manual_cleaned_data_universal.csv'

# Load data in full mode to retrieve bag-of-words and one-hot encoded Label
df = preprocess(DATASET_PATH, normalize_and_onehot=False, mode="full")

# Identify label columns (assumed to be the one-hot encoded ones; they have "Label" in their names)
label_cols = [col for col in df.columns if col.startswith("Label")]

# Extract features by dropping "id" and label cols
X = df.drop(["id"] + label_cols, axis=1).values

# Convert one-hot labels to integer indices
y = np.argmax(df[label_cols].values, axis=1)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Naive Bayes Test Accuracy: {accuracy:.2f}%")
