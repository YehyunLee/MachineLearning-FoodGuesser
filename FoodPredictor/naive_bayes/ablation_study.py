"""
Feature ablation study for the Naive Bayes model.
This script tests various feature combinations to measure their contribution to accuracy.
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append('../utils')
from preprocess import preprocess

# Constants
DATASET_PATH = '../data/cleanedWithScript/manual_cleaned_data_universal.csv'
RANDOM_SEED = 42
TEST_SIZE = 0.3

def run_ablation_test(df, exclude_prefixes=None, include_only_types=None):
    """
    Run a Naive Bayes model with specified feature inclusion/exclusion.
    
    Args:
        df: The preprocessed DataFrame
        exclude_prefixes: List of question prefixes to exclude (e.g., ['Q5', 'Q6'])
        include_only_types: Dictionary mapping feature type to prefixes that identify them
            e.g., {"numerical": ["Q1:", "Q2", "Q4"], "categorical": ["Q3:", "Q7:", "Q8:"], "bow": []}
        
    Returns:
        Accuracy score
    """
    # Identify label columns
    label_cols = [col for col in df.columns if col.startswith("Label_")]
    
    # Define feature groups based on preprocessing structure
    feature_groups = {
        "numerical": ["Q1:", "Q2 ", "Q4 "],  # Note the space after Q2 and Q4
        "categorical": ["Q3:", "Q7:", "Q8:"]  # Categorical columns are prefixed with question number
    }
    
    # All columns not starting with Q1-Q8, Label_, or id are bag-of-words columns
    all_columns = df.columns.tolist()
    id_columns = ['id']
    label_columns = [col for col in all_columns if col.startswith("Label_")]
    q_columns = [col for col in all_columns if any(col.startswith(f"Q{i}") for i in range(1, 9))]
    
    # All remaining columns are bag-of-words features
    bow_columns = set(all_columns) - set(id_columns) - set(label_columns) - set(q_columns)
    
    # We assume the first ~2/3 of bag-of-words columns are from Q5 (since it has more features)
    # and the remaining ~1/3 are from Q6
    bow_list = sorted(list(bow_columns))
    q5_count = int(len(bow_list) * 2/3)
    
    feature_groups["bow_q5"] = bow_list[:q5_count]
    feature_groups["bow_q6"] = bow_list[q5_count:]
    
    print(f"Identified {len(feature_groups['bow_q5'])} Q5 bag-of-words features")
    print(f"Identified {len(feature_groups['bow_q6'])} Q6 bag-of-words features")
    
    # Extract features based on inclusion/exclusion criteria
    if include_only_types:
        # Include only specific feature types
        feature_cols = []
        for feature_type in include_only_types:
            if feature_type in ["numerical", "categorical"]:
                for prefix in feature_groups[feature_type]:
                    feature_cols.extend([col for col in all_columns if col.startswith(prefix)])
            elif feature_type == "bow_q5":
                feature_cols.extend(feature_groups["bow_q5"])
            elif feature_type == "bow_q6":
                feature_cols.extend(feature_groups["bow_q6"])
    else:
        # Start with all non-label, non-id columns
        feature_cols = [col for col in all_columns if col not in id_columns and col not in label_columns]
        
        # Exclude specified feature types
        if exclude_prefixes:
            for prefix in exclude_prefixes:
                if prefix == "Q5":
                    # Exclude Q5 bag-of-words
                    for col in feature_groups["bow_q5"]:
                        if col in feature_cols:
                            feature_cols.remove(col)
                elif prefix == "Q6":
                    # Exclude Q6 bag-of-words
                    for col in feature_groups["bow_q6"]:
                        if col in feature_cols:
                            feature_cols.remove(col)
                else:
                    # Exclude columns with specific prefix
                    feature_cols = [col for col in feature_cols if not col.startswith(prefix)]
    
    # Check if we have any features left
    if not feature_cols:
        print("No features to use after filtering!")
        return 0.0
        
    print(f"Using {len(feature_cols)} features")
    
    # Extract features
    X = df[feature_cols].values
    
    # Convert one-hot labels to integer indices
    y = np.argmax(df[label_cols].values, axis=1)
    
    # Split into train and test sets - ensure EXACT same split as train_naive_bayes.py
    np.random.seed(RANDOM_SEED)  # Set NumPy random seed explicitly
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, 
                                                        random_state=RANDOM_SEED, shuffle=True)
    
    # Ensure Naive Bayes parameters match those in train_naive_bayes.py
    model = MultinomialNB(alpha=1.0, fit_prior=True)  # Default parameters, but explicitly set
    model.fit(X_train, y_train)
    
    # Predict and evaluate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    
    return accuracy

def main():
    """Run a comprehensive ablation study and display results."""
    print("Loading and preprocessing data...")
    # Ensure all preprocessing parameters match train_naive_bayes.py
    np.random.seed(RANDOM_SEED)  # Set global NumPy seed for reproducibility
    df = preprocess(DATASET_PATH, normalize_and_onehot=False, mode="full")
    # Drop column with "id"
    df.drop(columns=['id'], inplace=True)
    # print(df.columns)
    
    print(f"Preprocessed data shape: {df.shape}")
    print(f"Sample of columns: {list(df.columns)[:10]}...")
    
    # Define ablation tests
    ablation_tests = [
        {"name": "All features", "exclude": None, "include_only": None},
        {"name": "Without Q5 (Movie)", "exclude": ["Q5"], "include_only": None},
        {"name": "Without Q6 (Drink)", "exclude": ["Q6"], "include_only": None},
        {"name": "Without Q5 & Q6", "exclude": ["Q5", "Q6"], "include_only": None},
        {"name": "Without Q1 (Complexity)", "exclude": ["Q1"], "include_only": None},
        {"name": "Only numerical (Q1, Q2, Q4)", "exclude": None, 
         "include_only": ["numerical"]},
        {"name": "Only categorical (Q3, Q7, Q8)", "exclude": None, 
         "include_only": ["categorical"]}
    ]
    
    # Run tests and collect results
    results = []
    
    for test in ablation_tests:
        print(f"\nRunning test: {test['name']}")
        accuracy = run_ablation_test(
            df, 
            exclude_prefixes=test["exclude"], 
            include_only_types=test["include_only"]
        )
        results.append({"Test": test["name"], "Accuracy": accuracy})
        print(f"{test['name']} accuracy: {accuracy:.2f}%")
    
    # Run the "All features" test multiple times to check consistency
    all_features_accuracies = []
    for i in range(3):
        print(f"\nRunning consistency check {i+1} for All features")
        accuracy = run_ablation_test(df, exclude_prefixes=None, include_only_types=None)
        all_features_accuracies.append(accuracy)
        print(f"All features (run {i+1}) accuracy: {accuracy:.2f}%")
    
    print(f"\nAll features accuracy consistency check: {all_features_accuracies}")
    print(f"Mean: {np.mean(all_features_accuracies):.2f}%, Std: {np.std(all_features_accuracies):.4f}%")
    
    # Display results table
    results_df = pd.DataFrame(results)
    print("\n===== ABLATION STUDY RESULTS =====")
    print(results_df.to_string(index=False))
    
    # Save results to CSV
    results_df.to_csv('../naive_bayes/ablation_results.csv', index=False)
    print("\nResults saved to ablation_results.csv")

if __name__ == "__main__":
    main()
