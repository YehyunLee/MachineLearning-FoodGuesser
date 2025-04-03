import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import sys
import pickle
import os
import time
sys.path.append('../utils')
from preprocess import preprocess

# Path to dataset
DATASET_PATH = '../data/cleanedWithScript/manual_cleaned_data_universal.csv'

def tune_naive_bayes():
    print("Loading and preprocessing data...")
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

    print("Starting hyperparameter tuning...")
    start_time = time.time()

    # Define hyperparameters to tune
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],  # Smoothing parameter
        'fit_prior': [True, False]  # Whether to learn class prior probabilities
    }

    # Create the model
    nb = MultinomialNB()
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=nb,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='accuracy',
        n_jobs=-1,  # Use all available cores
        verbose=1
    )

    # Perform grid search
    grid_search.fit(X_train, y_train)

    # Get best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Training time
    tuning_time = time.time() - start_time
    print(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds")
    print(f"Best parameters: {best_params}")

    # Evaluate the best model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"Tuned Naive Bayes Test Accuracy: {accuracy:.2f}%")
    
    # Print detailed classification report
    print("\nClassification Report:")
    target_names = [label_mapping[i] for i in range(len(label_mapping))]
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Compare with base model
    base_model = MultinomialNB()
    base_model.fit(X_train, y_train)
    base_y_pred = base_model.predict(X_test)
    base_accuracy = accuracy_score(y_test, base_y_pred) * 100
    print(f"Base Naive Bayes Test Accuracy: {base_accuracy:.2f}%")
    print(f"Improvement: {accuracy - base_accuracy:.2f}%")

    # Save model parameters for later use
    output_dir = '../model_params'
    os.makedirs(output_dir, exist_ok=True)

    params = {
        'feature_names': feature_names,
        'label_mapping': label_mapping,
        'class_log_prior': best_model.class_log_prior_,
        'feature_log_prob': best_model.feature_log_prob_,
        'classes': best_model.classes_,
        'best_params': best_params
    }

    with open(f'{output_dir}/tuned_naive_bayes_params.pkl', 'wb') as f:
        pickle.dump(params, f)

    print(f"Tuned model parameters saved to {output_dir}/tuned_naive_bayes_params.pkl")
    
    # Also save the full model
    with open(f'{output_dir}/tuned_naive_bayes_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    print(f"Full tuned model saved to {output_dir}/tuned_naive_bayes_model.pkl")

if __name__ == "__main__":
    tune_naive_bayes()
