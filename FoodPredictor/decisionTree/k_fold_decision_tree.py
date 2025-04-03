import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.base import clone

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
    param_grid_ada = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.5, 1.0, 1.5],
        'estimator__max_depth': [1, 2, 3]
    }
    
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
    plt.savefig(f'{output_dir}/dt_kfold_performance_comparison.png')
    print(f"K-fold performance comparison saved to {output_dir}/dt_kfold_performance_comparison.png")
    
    plt.show()

def k_fold_cv(tune_fn, X, y, k=5):
    """
    Performs k-fold cross-validation using a tuning function to obtain a tuned model for each fold.
    
    Parameters:
      tune_fn: A tuning function that takes (X_train, y_train) and returns a tuned model.
      X: Features array.
      y: Labels array.
      k: Number of folds for cross-validation (default is 5).
      
    Returns:
      scores: A list of accuracy scores for each fold.
      best_model: The model with the highest validation accuracy.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    fold_models = []
    fold_num = 1
    
    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]
        
        # Tune the model on this fold's training data
        print(f"\n--- Fold {fold_num}/{k} ---")
        model = tune_fn(X_train_fold, y_train_fold)
        
        # Evaluate on validation data
        y_val_pred = model.predict(X_val_fold)
        acc = accuracy_score(y_val_fold, y_val_pred)
        scores.append(acc)
        fold_models.append((model, acc))
        
        print(f"Fold {fold_num} Validation Accuracy: {acc * 100:.2f}%")
        fold_num += 1
    
    # Calculate and print mean accuracy
    mean_accuracy = np.mean(scores) * 100
    std_accuracy = np.std(scores) * 100
    print(f"\nMean {k}-Fold CV Accuracy: {mean_accuracy:.2f}% (±{std_accuracy:.2f}%)")
    
    # Find the best model across all folds
    best_model, best_acc = max(fold_models, key=lambda x: x[1])
    print(f"Best Fold Model Accuracy: {best_acc * 100:.2f}%")
    
    return scores, best_model

def main():
    # Load dataset
    X, y = load_data(DATASET_PATH)
    
    print("\n==== STANDARD TRAIN/VALIDATION/TEST SPLIT ====")
    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Tune and evaluate Decision Tree
    best_dt = tune_decision_tree(X_train, y_train)
    dt_performance = evaluate_model(best_dt, X_train, y_train, X_val, y_val, X_test, y_test, model_name="Decision Tree")
    
    # Tune and evaluate AdaBoost
    best_ada = tune_adaboost(X_train, y_train)
    ada_performance = evaluate_model(best_ada, X_train, y_train, X_val, y_val, X_test, y_test, model_name="AdaBoost")
    
    # Plot performance comparison
    plot_performance([dt_performance, ada_performance])
    
    # Print final results from standard split
    print("\n==== FINAL RESULTS (STANDARD SPLIT) ====")
    print(f"Decision Tree - Train: {dt_performance['train']:.2f}%, Validation: {dt_performance['validation']:.2f}%, Test: {dt_performance['test']:.2f}%")
    print(f"AdaBoost     - Train: {ada_performance['train']:.2f}%, Validation: {ada_performance['validation']:.2f}%, Test: {ada_performance['test']:.2f}%")
    
    print("\n==== K-FOLD CROSS VALIDATION ====")
    # Perform 5-fold CV on Decision Tree
    print("\nPerforming 5-Fold CV on Decision Tree:")
    dt_scores, best_cv_dt = k_fold_cv(tune_decision_tree, X, y, k=5)
    
    # Perform 5-fold CV on AdaBoost
    print("\nPerforming 5-Fold CV on AdaBoost:")
    ada_scores, best_cv_ada = k_fold_cv(tune_adaboost, X, y, k=5)
    
    # Train on full dataset using the best model from CV
    print("\n==== TRAINING FINAL MODELS ON FULL DATASET ====")
    
    # For Decision Tree, use the parameters from the best CV model
    final_dt = clone(best_cv_dt)
    final_dt.fit(X, y)
    
    # For AdaBoost, use the parameters from the best CV model
    final_ada = clone(best_cv_ada)
    final_ada.fit(X, y)
    
    # Print the final model parameters
    print("\n==== FINAL MODEL PARAMETERS ====")
    print("Decision Tree Parameters:")
    dt_params = final_dt.get_params()
    for key in sorted(dt_params.keys()):
        if not key.startswith('_'):
            print(f"  {key}: {dt_params[key]}")
    
    print("\nAdaBoost Parameters:")
    ada_params = final_ada.get_params()
    for key in sorted(ada_params.keys()):
        if not key.startswith('_') and key != 'estimator':
            print(f"  {key}: {ada_params[key]}")
    
    # Final accuracies (training on full dataset and evaluating on the same data)
    # This is for reference only since we're evaluating on the training data
    dt_full_acc = accuracy_score(y, final_dt.predict(X)) * 100
    ada_full_acc = accuracy_score(y, final_ada.predict(X)) * 100
    
    print("\n==== FINAL RESULTS (K-FOLD CV) ====")
    print(f"Decision Tree - Mean CV Accuracy: {np.mean(dt_scores) * 100:.2f}% (±{np.std(dt_scores) * 100:.2f}%)")
    print(f"AdaBoost     - Mean CV Accuracy: {np.mean(ada_scores) * 100:.2f}% (±{np.std(ada_scores) * 100:.2f}%)")
    
    # Save both models
    output_dir = '../model_params'
    os.makedirs(output_dir, exist_ok=True)
    
    import pickle
    with open(f'{output_dir}/dt_model.pkl', 'wb') as f:
        pickle.dump(final_dt, f)
    with open(f'{output_dir}/adaboost_model.pkl', 'wb') as f:
        pickle.dump(final_ada, f)
    
    print(f"\nModels saved to {output_dir}/dt_model.pkl and {output_dir}/adaboost_model.pkl")

if __name__ == '__main__':
    main()
