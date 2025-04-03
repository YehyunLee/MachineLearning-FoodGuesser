import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, log_loss, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
import sys
import pickle
import os
import time
sys.path.append('../utils')
from preprocess import preprocess

# Path to dataset
DATASET_PATH = '../data/cleanedWithScript/manual_cleaned_data_universal.csv'

def k_fold_tune_naive_bayes(k=5, test_size=0.3):
    print(f"Running {k}-fold cross-validation with hyperparameter tuning...")
    print(f"Using train-test split: {100-test_size*100}% training, {test_size*100}% testing")
    
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
    
    # Split into train and test sets (70% train, 30% test)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    print(f"Training set size: {X_train_full.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    # Define hyperparameters to tune
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],  # Smoothing parameter
        'fit_prior': [True, False]  # Whether to learn class prior probabilities
    }

    # Initialize stratified k-fold cross-validation (on training set only)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    # Track fold results
    fold_accuracies = []
    fold_log_losses = []
    best_params_list = []
    fold_models = []
    
    # Track time
    start_time = time.time()

    # Perform k-fold cross-validation on the training data only
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
        print(f"\nFold {fold+1}/{k}")
        X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
        y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]
        
        # Create and tune the model with GridSearchCV
        nb = MultinomialNB()
        grid_search = GridSearchCV(
            estimator=nb,
            param_grid=param_grid,
            cv=5,  # Internal cross-validation
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Get best parameters and model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        
        # Store best parameters
        best_params_list.append(best_params)
        print(f"Best parameters for fold {fold+1}: {best_params}")
        
        # Evaluate on validation set
        y_pred = best_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred) * 100
        fold_accuracies.append(accuracy)
        
        # Calculate log loss (lower is better)
        # Get probability predictions for each class
        y_pred_proba = best_model.predict_proba(X_val)
        logloss = log_loss(y_val, y_pred_proba)
        fold_log_losses.append(logloss)
        
        print(f"Fold {fold+1} - Validation Accuracy: {accuracy:.2f}%, Log Loss: {logloss:.4f}")
        
        # Store the model
        fold_models.append(best_model)

    # Calculate average metrics and standard deviations
    avg_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    avg_log_loss = np.mean(fold_log_losses)
    std_log_loss = np.std(fold_log_losses)
    
    print(f"\nK-fold Cross Validation Results (k={k}):")
    for i, (acc, ll) in enumerate(zip(fold_accuracies, fold_log_losses)):
        print(f"Fold {i+1} - Accuracy: {acc:.2f}%, Log Loss: {ll:.4f}")
    print(f"Average accuracy: {avg_accuracy:.2f}% ± {std_accuracy:.2f}%")
    print(f"Average log loss: {avg_log_loss:.4f} ± {std_log_loss:.4f}")
    
    # Find most common best parameters
    alpha_counts = {}
    fit_prior_counts = {True: 0, False: 0}
    
    for params in best_params_list:
        alpha = params['alpha']
        fit_prior = params['fit_prior']
        
        alpha_counts[alpha] = alpha_counts.get(alpha, 0) + 1
        fit_prior_counts[fit_prior] += 1
    
    most_common_alpha = max(alpha_counts, key=alpha_counts.get)
    most_common_fit_prior = max(fit_prior_counts, key=fit_prior_counts.get)
    
    print(f"\nMost common best alpha: {most_common_alpha}")
    print(f"Most common fit_prior: {most_common_fit_prior}")
    
    # Train final model with most common parameters on full training set
    print("\nTraining final model with most common best parameters on full training set...")
    final_model = MultinomialNB(alpha=most_common_alpha, fit_prior=most_common_fit_prior)
    final_model.fit(X_train_full, y_train_full)
    
    # Evaluate on held-out test set
    print("\n--- Final Evaluation on Test Set ---")
    y_test_pred = final_model.predict(X_test)
    y_test_pred_proba = final_model.predict_proba(X_test)
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_test_pred) * 100
    test_logloss = log_loss(y_test, y_test_pred_proba)
    test_precision = precision_score(y_test, y_test_pred, average='weighted') * 100
    test_recall = recall_score(y_test, y_test_pred, average='weighted') * 100
    test_f1 = f1_score(y_test, y_test_pred, average='weighted') * 100
    
    print(f"Test Set Metrics:")
    print(f"Accuracy: {test_accuracy:.2f}%")
    print(f"Log Loss: {test_logloss:.4f}")
    print(f"Precision: {test_precision:.2f}%")
    print(f"Recall: {test_recall:.2f}%")
    print(f"F1 Score: {test_f1:.2f}%")
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    target_names = [label_mapping[i] for i in range(len(label_mapping))]
    print(classification_report(y_test, y_test_pred, target_names=target_names))
    
    # Total time
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f} seconds")

    # Save model parameters for later use
    output_dir = '../model_params'
    os.makedirs(output_dir, exist_ok=True)

    params = {
        'feature_names': feature_names,
        'label_mapping': label_mapping,
        'class_log_prior': final_model.class_log_prior_,
        'feature_log_prob': final_model.feature_log_prob_,
        'classes': final_model.classes_,
        'best_params': {
            'alpha': most_common_alpha,
            'fit_prior': most_common_fit_prior
        },
        'fold_accuracies': fold_accuracies,
        'fold_log_losses': fold_log_losses,
        'avg_accuracy': avg_accuracy,
        'std_accuracy': std_accuracy,
        'avg_log_loss': avg_log_loss,
        'std_log_loss': std_log_loss,
        'test_metrics': {
            'accuracy': test_accuracy,
            'log_loss': test_logloss,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1
        }
    }

    with open(f'{output_dir}/k_fold_naive_bayes_params.pkl', 'wb') as f:
        pickle.dump(params, f)

    print(f"K-fold model parameters saved to {output_dir}/k_fold_naive_bayes_params.pkl")
    
    # Also save the full model
    with open(f'{output_dir}/k_fold_naive_bayes_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    
    print(f"Full K-fold model saved to {output_dir}/k_fold_naive_bayes_model.pkl")

if __name__ == "__main__":
    k_fold_tune_naive_bayes(k=5, test_size=0.1)
