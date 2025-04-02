import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import sys

sys.path.append('../utils')
from preprocess import preprocess

DATASET_PATH = "../data/cleanedWithScript/manual_cleaned_data_universal.csv"

def main():
    # Use the preprocess function from utils/preprocess (bag-of-words for Q5 & Q6)
    df = preprocess(DATASET_PATH, mode="full")
    
    # Identify label columns (assumed to start with "Label")
    label_cols = [col for col in df.columns if col.startswith("Label")]
    # Features are all columns except "id" and label columns
    feature_cols = [col for col in df.columns if col not in ["id"] + label_cols]
    
    X = df[feature_cols].to_numpy()
    y = df[label_cols].to_numpy()
    
    # Convert one-hot encoded labels to class indices
    y = np.argmax(y, axis=1)
    
    print(f"Final data matrix X shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Train-test split (80% training, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    model = QuadraticDiscriminantAnalysis()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy}")

def k_fold_cv(k=5):
    # Load and preprocess data (same as in main)
    df = preprocess(DATASET_PATH, mode="full")
    
    label_cols = [col for col in df.columns if col.startswith("Label")]
    feature_cols = [col for col in df.columns if col not in ["id"] + label_cols]
    
    X = df[feature_cols].to_numpy()
    y = df[label_cols].to_numpy()
    y = np.argmax(y, axis=1)
    
    print(f"\nPerforming 5-Fold Cross Validation:")
    print(f"Data matrix X shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=1)
    fold = 1
    scores = []
    for train_index, test_index in kf.split(X):
        X_train_cv, X_val_cv = X[train_index], X[test_index]
        y_train_cv, y_val_cv = y[train_index], y[test_index]
        
        model_cv = QuadraticDiscriminantAnalysis()
        model_cv.fit(X_train_cv, y_train_cv)
        score = model_cv.score(X_val_cv, y_val_cv)
        print(f"Fold {fold} Accuracy: {score}")
        scores.append(score)
        fold += 1
    mean_accuracy = np.mean(scores)
    print(f"Mean 5-Fold CV Accuracy: {mean_accuracy}")

if __name__ == '__main__':
    main()
    k_fold_cv(k=5)
