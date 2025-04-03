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
    # First, split into 70% train and 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    
    # Then split temp into equal validation and test sets (each 15% of original)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )
    
    # Print sizes to confirm split
    print(f"Training set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation set size: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def tune_decision_tree(X_train, y_train, X_val, y_val):
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
    
    # Evaluate on validation set
    best_dt = grid_dt.best_estimator_
    val_accuracy = accuracy_score(y_val, best_dt.predict(X_val)) * 100
    print(f"Validation Accuracy with Tuned Decision Tree: {val_accuracy:.2f}%")
    
    return best_dt

def tune_adaboost(X_train, y_train, X_val, y_val):
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
    
    # Evaluate on validation set
    best_ada = grid_ada.best_estimator_
    val_accuracy = accuracy_score(y_val, best_ada.predict(X_val)) * 100
    print(f"Validation Accuracy with Tuned AdaBoost: {val_accuracy:.2f}%")
    
    return best_ada

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
    width = 0.35 if len(models) <= 2 else 0.25  # adjust width based on number of models
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, model_data in enumerate(data):
        offset = width * (i - (len(models) - 1) / 2)
        bars = ax.bar(x + offset, model_data, width, label=models[i])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([metric.capitalize() for metric in metrics])
    ax.set_ylim(0, 110)  # Set y-axis limit to accommodate annotations
    ax.legend()
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = '../model_params'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/dt_performance_comparison.png')
    print(f"Performance comparison saved to {output_dir}/dt_performance_comparison.png")
    
    plt.show()

def main():
    # Load and split the dataset
    X, y = load_data(DATASET_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Tune and evaluate the Decision Tree model
    best_dt = tune_decision_tree(X_train, y_train, X_val, y_val)
    dt_performance = evaluate_model(best_dt, X_train, y_train, X_val, y_val, X_test, y_test, model_name="Decision Tree")
    
    # Tune and evaluate the AdaBoost (boosted decision tree) model
    best_ada = tune_adaboost(X_train, y_train, X_val, y_val)
    ada_performance = evaluate_model(best_ada, X_train, y_train, X_val, y_val, X_test, y_test, model_name="AdaBoost")
    
    # Plot and display performance comparison
    plot_performance([dt_performance, ada_performance])
    
    # Print the final accuracies clearly
    print("\n==== FINAL RESULTS ====")
    print(f"Decision Tree - Train: {dt_performance['train']:.2f}%, Validation: {dt_performance['validation']:.2f}%, Test: {dt_performance['test']:.2f}%")
    print(f"AdaBoost     - Train: {ada_performance['train']:.2f}%, Validation: {ada_performance['validation']:.2f}%, Test: {ada_performance['test']:.2f}%")
    
    # Save the best model
    output_dir = '../model_params'
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine the better model based on validation accuracy
    if dt_performance['validation'] > ada_performance['validation']:
        best_model = best_dt
        best_name = "Decision Tree"
    else:
        best_model = best_ada
        best_name = "AdaBoost"
        
    import pickle
    with open(f'{output_dir}/best_dt_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\nBest model ({best_name}) saved to {output_dir}/best_dt_model.pkl")

if __name__ == '__main__':
    main()
