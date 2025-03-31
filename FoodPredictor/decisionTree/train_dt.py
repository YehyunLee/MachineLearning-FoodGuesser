import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
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

def tune_decision_tree(X_train, y_train):
    """
    Tunes hyperparameters for a DecisionTreeClassifier using GridSearchCV.
    Returns:
      best_dt: The best decision tree model.
    """
    param_grid_dt = {
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'criterion': ['gini', 'entropy']
    }
    
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    grid_dt = GridSearchCV(dt, param_grid_dt, cv=5, n_jobs=-1)
    grid_dt.fit(X_train, y_train)
    
    print("Best Decision Tree Parameters:")
    print(grid_dt.best_params_)
    print("Best CV Score: {:.2f}%".format(grid_dt.best_score_ * 100))
    return grid_dt.best_estimator_

def tune_adaboost(X_train, y_train):
    """
    Tunes hyperparameters for an AdaBoostClassifier (with a decision tree base estimator) using GridSearchCV.
    Returns:
      best_ada: The best AdaBoost model.
    """
    # Note: To tune parameters of the base estimator (DecisionTreeClassifier), prefix with "estimator__"
    param_grid_ada = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.5, 1.0, 1.5],
        'estimator__max_depth': [1, 2, 3]
    }
    
    # Create AdaBoost with a decision tree as the base estimator.
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        random_state=RANDOM_STATE
    )
    grid_ada = GridSearchCV(ada, param_grid_ada, cv=5, n_jobs=-1)
    grid_ada.fit(X_train, y_train)
    
    print("Best AdaBoost Parameters:")
    print(grid_ada.best_params_)
    print("Best CV Score: {:.2f}%".format(grid_ada.best_score_ * 100))
    return grid_ada.best_estimator_

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
    
    print(f"\n{model_name} Performance:")
    print("Train Accuracy: {:.2f}%".format(train_acc))
    print("Validation Accuracy: {:.2f}%".format(val_acc))
    print("Test Accuracy: {:.2f}%".format(test_acc))

def main():
    # Load and split the dataset
    X, y = load_data(DATASET_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Tune and evaluate the Decision Tree model
    best_dt = tune_decision_tree(X_train, y_train)
    evaluate_model(best_dt, X_train, y_train, X_val, y_val, X_test, y_test, model_name="Decision Tree")
    
    # Tune and evaluate the AdaBoost (boosted decision tree) model
    best_ada = tune_adaboost(X_train, y_train)
    evaluate_model(best_ada, X_train, y_train, X_val, y_val, X_test, y_test, model_name="AdaBoost")

if __name__ == '__main__':
    main()
