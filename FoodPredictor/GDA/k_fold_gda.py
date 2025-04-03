import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import sys
from sklearn.metrics import log_loss, accuracy_score

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
    
    # Train-test split (70% training, 15% validation, 15% test)
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=1)
    
    model = QuadraticDiscriminantAnalysis()
    model.fit(X_train, y_train)
    
    # Calculate metrics on validation set
    val_accuracy = model.score(X_val, y_val)
    val_proba = model.predict_proba(X_val)
    val_loss = log_loss(y_val, val_proba)
    print(f"Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}")
    
    # Calculate metrics on test set
    test_accuracy = model.score(X_test, y_test)
    test_proba = model.predict_proba(X_test)
    test_loss = log_loss(y_test, test_proba)
    print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

def k_fold_cv(k=5):
    # Load and preprocess data (same as in main)
    df = preprocess(DATASET_PATH, mode="full")
    
    label_cols = [col for col in df.columns if col.startswith("Label")]
    feature_cols = [col for col in df.columns if col not in ["id"] + label_cols]
    
    X = df[feature_cols].to_numpy()
    y = df[label_cols].to_numpy()
    y = np.argmax(y, axis=1)
    
    print(f"\nPerforming {k}-Fold Cross Validation:")
    print(f"Data matrix X shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    kf = KFold(n_splits=k, shuffle=True, random_state=1)
    fold = 1
    val_scores = []
    test_scores = []
    
    for train_index, test_index in kf.split(X):
        print(f"\nFold {fold}/{k}")
        print("-" * 30)
        
        # Get training data for this fold
        X_fold = X[train_index]
        y_fold = y[train_index]
        
        # Further split training data into training and validation (70/15 split within the fold's training data)
        # This creates a 70-15-15 split where test is already 15% from the KFold split
        X_train, X_val, y_train, y_val = train_test_split(X_fold, y_fold, test_size=0.1765, random_state=1)
        # 0.1765 = 15/85 (15% of the 85% training data)
        
        X_test = X[test_index]
        y_test = y[test_index]
        
        model = QuadraticDiscriminantAnalysis()
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_score = model.score(X_val, y_val)
        val_scores.append(val_score)
        print(f"Validation Accuracy (Fold {fold}): {val_score:.4f}")
        
        # Evaluate on test set
        test_score = model.score(X_test, y_test)
        test_scores.append(test_score)
        print(f"Test Accuracy (Fold {fold}): {test_score:.4f}")
        
        fold += 1
    
    # Print summary statistics
    print("\n" + "="*50)
    print("CROSS VALIDATION RESULTS")
    print("="*50)
    for i, (val, test) in enumerate(zip(val_scores, test_scores)):
        print(f"Fold {i+1}: Validation Accuracy: {val:.4f}, Test Accuracy: {test:.4f}")
    print("-"*30)
    
    mean_val = np.mean(val_scores)
    std_val = np.std(val_scores)
    mean_test = np.mean(test_scores)
    std_test = np.std(test_scores)
    
    print(f"Average Validation Accuracy: {mean_val:.4f} ± {std_val:.4f}")
    print(f"Average Test Accuracy: {mean_test:.4f} ± {std_test:.4f}")
    print("="*50)

if __name__ == '__main__':
    main()
    k_fold_cv(k=5)
