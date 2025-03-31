import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys

sys.path.append('../utils')
from preprocess import preprocess

DATASET_PATH = '../data/cleanedWithScript/manual_cleaned_data_universal.csv'
RANDOM_STATE = 42

def load_data(dataset_path):
    """
    Loads and preprocesses the dataset.
    Returns:
      X: Feature matrix (numpy array of type float32)
      y: Integer labels (numpy array)
    """
    df = preprocess(dataset_path, mode="full")
    
    # Identify label columns (starting with "Label") and feature columns
    label_cols = [col for col in df.columns if col.startswith("Label")]
    feature_cols = [col for col in df.columns if col not in (['id'] + label_cols)]
    
    # Extract features and one-hot encoded labels
    X = df[feature_cols].values.astype(np.float32)
    y_onehot = df[label_cols].values.astype(np.float32)
    
    # Convert one-hot labels to integer labels (assuming one hot per row)
    y = np.argmax(y_onehot, axis=1)
    
    return X, y

def split_data(X, y):
    """
    Splits the data into train (70%), validation (15%), and test (15%) sets using stratification.
    Returns:
      X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_decision_tree(X_train, y_train):
    """
    Trains a DecisionTreeClassifier.
    Returns:
      The trained decision tree model.
    """
    dt_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt_model.fit(X_train, y_train)
    return dt_model

def train_adaboost(X_train, y_train):
    """
    Trains an AdaBoostClassifier with a decision tree as the base estimator.
    Returns:
      The trained AdaBoost model.
    """
    ada_model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        n_estimators=50,
        random_state=RANDOM_STATE
    )
    ada_model.fit(X_train, y_train)
    return ada_model

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name="Model"):
    """
    Evaluates the given model on train, validation, and test sets.
    Prints the accuracy for each set.
    """
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred) * 100
    val_acc = accuracy_score(y_val, y_val_pred) * 100
    test_acc = accuracy_score(y_test, y_test_pred) * 100
    
    print(f"{model_name} Results:")
    print("Train Accuracy: {:.2f}%".format(train_acc))
    print("Validation Accuracy: {:.2f}%".format(val_acc))
    print("Test Accuracy: {:.2f}%".format(test_acc))
    print()

def main():
    # Load and split the dataset
    X, y = load_data(DATASET_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Train and evaluate the Decision Tree model
    dt_model = train_decision_tree(X_train, y_train)
    evaluate_model(dt_model, X_train, y_train, X_val, y_val, X_test, y_test, model_name="Decision Tree")
    
    # Train and evaluate the AdaBoost (boosted decision tree) model
    ada_model = train_adaboost(X_train, y_train)
    evaluate_model(ada_model, X_train, y_train, X_val, y_val, X_test, y_test, model_name="AdaBoost")

if __name__ == '__main__':
    main()
