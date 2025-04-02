import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
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
    
    # Train-test split (70% training, 15% validation, 15% test)
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=1)
    
    model = QuadraticDiscriminantAnalysis()
    model.fit(X_train, y_train)
    accuracy = model.score(X_val, y_val)
    loss = log_loss(y_val, model.predict_proba(X_val))
    print(f"Accuracy: {accuracy}, Loss: {loss}")

if __name__ == '__main__':
    main()
