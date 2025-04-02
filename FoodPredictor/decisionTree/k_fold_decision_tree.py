import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

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
    Returns a dictionary with the accuracies.
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
    
    return {"model_name": model_name, "train": train_acc, "validation": val_acc, "test": test_acc}

def plot_performance(performance_list):
    """
    Displays a grouped bar chart comparing train, validation, and test accuracies
    for each model in performance_list.
    performance_list: list of dictionaries returned by evaluate_model.
    """
    models = [perf["model_name"] for perf in performance_list]
    metrics = ["train", "validation", "test"]
    
    # Prepare the data in the same order for each model
    data = []
    for perf in performance_list:
        data.append([perf[m] for m in metrics])
    data = np.array(data)  # shape: (num_models, num_metrics)
    
    x = np.arange(len(metrics))
    width = 0.35  # width of each bar
    
    fig, ax = plt.subplots()
    
    for i, model_data in enumerate(data):
        ax.bar(x + i * width, model_data, width, label=models[i])
    
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([metric.capitalize() for metric in metrics])
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Load and split the dataset
    X, y = load_data(DATASET_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Tune and evaluate the Decision Tree model
    best_dt = tune_decision_tree(X_train, y_train)
    dt_performance = evaluate_model(best_dt, X_train, y_train, X_val, y_val, X_test, y_test, model_name="Decision Tree")
    
    # Tune and evaluate the AdaBoost (boosted decision tree) model
    best_ada = tune_adaboost(X_train, y_train)
    ada_performance = evaluate_model(best_ada, X_train, y_train, X_val, y_val, X_test, y_test, model_name="AdaBoost")
    
    # Plot and display performance comparison
    plot_performance([dt_performance, ada_performance])

if __name__ == '__main__':
    main()
    
    # --- New Code: 5-Fold Cross-Validation Using Tuned Models ---
    from sklearn.model_selection import KFold

    def k_fold_cv_tuned(tune_fn, X, y, k=5):
        """
        Performs k-fold cross-validation using a tuning function (e.g., tune_decision_tree or tune_adaboost)
        to obtain a tuned model for each fold.
        
        Parameters:
          tune_fn: A tuning function that takes (X_train, y_train) and returns a tuned model.
          X: Features array.
          y: Labels array.
          k: Number of folds for cross-validation (default is 5).
          
        Returns:
          scores: A list of accuracy scores for each fold.
        """
        kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        fold_num = 1
        for train_index, val_index in kf.split(X):
            X_train_cv, X_val_cv = X[train_index], X[val_index]
            y_train_cv, y_val_cv = y[train_index], y[val_index]
            
            # Use the existing tuning function to get the tuned model for the fold.
            tuned_model = tune_fn(X_train_cv, y_train_cv)
            y_pred_cv = tuned_model.predict(X_val_cv)
            fold_accuracy = accuracy_score(y_val_cv, y_pred_cv)
            scores.append(fold_accuracy)
            print(f"Fold {fold_num} Accuracy: {fold_accuracy * 100:.2f}%")
            fold_num += 1
        
        mean_accuracy = np.mean(scores)
        print(f"Mean 5-Fold CV Accuracy using tuned model: {mean_accuracy * 100:.2f}%")
        return scores

    # Load full dataset for cross validation
    X_all, y_all = load_data(DATASET_PATH)
    
    print("\nPerforming 5-Fold Cross Validation on Tuned Decision Tree:")
    k_fold_cv_tuned(tune_decision_tree, X_all, y_all, k=5)
    
    print("\nPerforming 5-Fold Cross Validation on Tuned AdaBoost:")
    k_fold_cv_tuned(tune_adaboost, X_all, y_all, k=5)
