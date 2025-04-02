import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Naive Bayes Test Accuracy: {accuracy:.2f}%")

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

# --- New Code: 5-Fold Cross-Validation Function ---

from sklearn.model_selection import KFold
from sklearn.base import clone

def k_fold_cv(model, X, y, k=5):
    """
    Performs k-fold cross-validation on the given model using the data X and y.
    
    Parameters:
      model: An untrained instance of a scikit-learn estimator.
      X: Features array.
      y: Labels array.
      k: Number of folds for cross-validation (default is 5).
      
    Returns:
      scores: A list of accuracy scores for each fold.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
        X_train_cv, X_val_cv = X[train_index], X[val_index]
        y_train_cv, y_val_cv = y[train_index], y[val_index]
        
        # Clone the model to ensure each fold gets a fresh instance.
        model_cv = clone(model)
        model_cv.fit(X_train_cv, y_train_cv)
        y_pred_cv = model_cv.predict(X_val_cv)
        fold_accuracy = accuracy_score(y_val_cv, y_pred_cv)
        scores.append(fold_accuracy)
        print(f"Fold {fold} Accuracy: {fold_accuracy * 100:.2f}%")
    
    mean_accuracy = np.mean(scores)
    print(f"Mean 5-Fold Cross Validation Accuracy: {mean_accuracy * 100:.2f}%")
    return scores

# Execute k-fold cross-validation on the entire dataset
print("\nPerforming 5-Fold Cross Validation on the entire dataset:")
k_fold_cv(MultinomialNB(), X, y, k=5)
