import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
import sys
import pickle
import os
sys.path.append('../utils')
from preprocess import preprocess

DATASET_PATH = '../data/cleanedWithScript/manual_cleaned_data_universal.csv'

# Load data in full mode to retrieve bag-of-words and one-hot encoded Label
df = preprocess(DATASET_PATH, normalize_and_onehot=False, mode="full")

# Identify label columns (assumed to be the one-hot encoded ones; they have "Label" in their names)
label_cols = [col for col in df.columns if col.startswith("Label")]

# Extract features by dropping "id" and label cols
X = df.drop(["id"] + label_cols, axis=1)
feature_names = X.columns.tolist()
X = X.values

# Convert one-hot labels to integer indices
y = np.argmax(df[label_cols].values, axis=1)

# Save the label mapping for prediction
label_mapping = {i: col.replace('Label_', '') for i, col in enumerate(label_cols)}

# Split into train and test sets
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Train Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate accuracy
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred) * 100
loss = log_loss(y_val, model.predict_proba(X_val))
print(f"Naive Bayes Validation Accuracy: {accuracy:.2f}%")
print(f"Naive Bayes Validation Loss: {loss}")

# Save model parameters for later use without sklearn
output_dir = '../model_params'
os.makedirs(output_dir, exist_ok=True)

params = {
    'feature_names': feature_names,
    'label_mapping': label_mapping,
    'class_log_prior': model.class_log_prior_,
    'feature_log_prob': model.feature_log_prob_,
    'classes': model.classes_,
}

with open(f'{output_dir}/naive_bayes_params.pkl', 'wb') as f:
    pickle.dump(params, f)

print(f"Model parameters saved to {output_dir}/naive_bayes_params.pkl")
