import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.base import clone
import sys
import pickle
import os
import time
sys.path.append('../utils')
from preprocess import preprocess

DATASET_PATH = '../data/cleanedWithScript/manual_cleaned_data_universal.csv'

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

# Split into train, validation, and test sets (70:15:15)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Dataset split - Training: {X_train.shape[0]} samples, Validation: {X_val.shape[0]} samples, Testing: {X_test.shape[0]} samples")

# --- Hyperparameter Tuning ---
def tune_naive_bayes():
    print("\n=== Hyperparameter Tuning for Naive Bayes ===")
    # Define the parameter grid
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0],  # Smoothing parameter
        'fit_prior': [True, False]  # Whether to learn class prior probabilities
    }
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        MultinomialNB(), 
        param_grid, 
        cv=5, 
        scoring='accuracy', 
        verbose=1
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    tuning_time = time.time() - start_time
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate on validation set
    best_model = grid_search.best_estimator_
    y_val_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation accuracy with tuned model: {val_accuracy:.4f}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy with tuned model: {test_accuracy:.4f}")
    print(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds")
    
    return best_model

# --- Ensemble Methods ---
def evaluate_ensemble_methods():
    print("\n=== Evaluating Ensemble Methods with Naive Bayes ===")
    results = {}
    
    # Basic MultinomialNB (baseline)
    start_time = time.time()
    base_nb = MultinomialNB()
    base_nb.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_val_pred = base_nb.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    # Evaluate on test set
    y_pred = base_nb.predict(X_test)
    base_accuracy = accuracy_score(y_test, y_pred)
    base_time = time.time() - start_time
    results['Base MultinomialNB'] = {'test_accuracy': base_accuracy, 'val_accuracy': val_accuracy, 'time': base_time}
    print(f"Base MultinomialNB - Val Accuracy: {val_accuracy:.4f}, Test Accuracy: {base_accuracy:.4f}, Time: {base_time:.2f}s")
    
    # 1. Bagging with MultinomialNB
    start_time = time.time()
    bagging = BaggingClassifier(
        estimator=MultinomialNB(),
        n_estimators=10,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True,
        bootstrap_features=False,
        random_state=42
    )
    bagging.fit(X_train, y_train)
    
    # Validation accuracy
    y_val_pred = bagging.predict(X_val)
    bagging_val_accuracy = accuracy_score(y_val, y_val_pred)
    
    # Test accuracy
    y_pred = bagging.predict(X_test)
    bagging_accuracy = accuracy_score(y_test, y_pred)
    bagging_time = time.time() - start_time
    results['Bagging with MultinomialNB'] = {'test_accuracy': bagging_accuracy, 'val_accuracy': bagging_val_accuracy, 'time': bagging_time}
    print(f"Bagging with MultinomialNB - Val Accuracy: {bagging_val_accuracy:.4f}, Test Accuracy: {bagging_accuracy:.4f}, Time: {bagging_time:.2f}s")
    
    # 2. AdaBoost with MultinomialNB (with reduced learning rate for stability)
    try:
        start_time = time.time()
        adaboost = AdaBoostClassifier(
            estimator=MultinomialNB(),
            n_estimators=50,
            learning_rate=0.1,
            random_state=42
        )
        adaboost.fit(X_train, y_train)
        
        # Validation accuracy
        y_val_pred = adaboost.predict(X_val)
        adaboost_val_accuracy = accuracy_score(y_val, y_val_pred)
        
        # Test accuracy
        y_pred = adaboost.predict(X_test)
        adaboost_accuracy = accuracy_score(y_test, y_pred)
        adaboost_time = time.time() - start_time
        results['AdaBoost with MultinomialNB'] = {'test_accuracy': adaboost_accuracy, 'val_accuracy': adaboost_val_accuracy, 'time': adaboost_time}
        print(f"AdaBoost with MultinomialNB - Val Accuracy: {adaboost_val_accuracy:.4f}, Test Accuracy: {adaboost_accuracy:.4f}, Time: {adaboost_time:.2f}s")
    except Exception as e:
        print(f"AdaBoost with MultinomialNB failed: {e}")
        
    # 3. Voting Classifier with different Naive Bayes variants
    start_time = time.time()
    voting = VotingClassifier(estimators=[
        ('mnb', MultinomialNB()),
        ('cnb', ComplementNB()),
        ('bnb', BernoulliNB())
    ], voting='soft')
    
    voting.fit(X_train, y_train)
    
    # Validation accuracy
    y_val_pred = voting.predict(X_val)
    voting_val_accuracy = accuracy_score(y_val, y_val_pred)
    
    # Test accuracy
    y_pred = voting.predict(X_test)
    voting_accuracy = accuracy_score(y_test, y_pred)
    voting_time = time.time() - start_time
    results['Voting (MNB, CNB, BNB)'] = {'test_accuracy': voting_accuracy, 'val_accuracy': voting_val_accuracy, 'time': voting_time}
    print(f"Voting Classifier - Val Accuracy: {voting_val_accuracy:.4f}, Test Accuracy: {voting_accuracy:.4f}, Time: {voting_time:.2f}s")
    
    # Find the best method based on validation accuracy
    best_method = max(results.items(), key=lambda x: x[1]['val_accuracy'])
    print(f"\nBest method based on validation: {best_method[0]} with validation accuracy: {best_method[1]['val_accuracy']:.4f}")
    
    return results, best_method[0]

# --- Run the basic model ---
print("\n=== Basic MultinomialNB Model ===")
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate on validation set
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred) * 100
print(f"Naive Bayes Validation Accuracy: {val_accuracy:.2f}%")

# Evaluate on test set
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Naive Bayes Test Accuracy: {test_accuracy:.2f}%")
print(classification_report(y_test, y_pred))

# --- Run hyperparameter tuning ---
best_nb = tune_naive_bayes()

# --- Try ensemble methods ---
ensemble_results, best_method = evaluate_ensemble_methods()

# Save the best model
output_dir = '../model_params'
os.makedirs(output_dir, exist_ok=True)

# Save the parameters of the base model
params = {
    'feature_names': feature_names,
    'label_mapping': label_mapping,
    'class_log_prior': model.class_log_prior_,
    'feature_log_prob': model.feature_log_prob_,
    'classes': model.classes_,
}

with open(f'{output_dir}/naive_bayes_params.pkl', 'wb') as f:
    pickle.dump(params, f)

print(f"Model parameters saved to {output_dir}/naive_bayes_params.pkl")

# Save the best model from our experiments
with open(f'{output_dir}/best_naive_bayes_model.pkl', 'wb') as f:
    if best_method == 'Base MultinomialNB':
        pickle.dump(best_nb, f)
    else:
        if best_method == 'Bagging with MultinomialNB':
            best_model = BaggingClassifier(
                estimator=MultinomialNB(**best_nb.get_params()),
                n_estimators=10,
                max_samples=0.8,
                max_features=0.8,
                bootstrap=True,
                bootstrap_features=False,
                random_state=42
            )
            best_model.fit(X_train, y_train)
        elif best_method == 'Voting (MNB, CNB, BNB)':
            best_model = VotingClassifier(estimators=[
                ('mnb', MultinomialNB(**best_nb.get_params())),
                ('cnb', ComplementNB()),
                ('bnb', BernoulliNB())
            ], voting='soft')
            best_model.fit(X_train, y_train)
        pickle.dump(best_model, f)

print(f"Best model saved to {output_dir}/best_naive_bayes_model.pkl")

# --- 5-Fold Cross-Validation Function ---
def k_fold_cv(model, X, y, k=5):
    """
    Performs k-fold cross-validation on the given model using the data X and y.
    
    Parameters:
      model: An untrained instance of a scikit-learn estimator.
      X: Features array.
      y: Labels array.
      k: Number of folds for cross-validation (default is 5).
      
    Returns:
      scores: A list of accuracy scores for each fold.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
        X_train_cv, X_val_cv = X[train_index], X[val_index]
        y_train_cv, y_val_cv = y[train_index], y[val_index]
        
        # Clone the model to ensure each fold gets a fresh instance.
        model_cv = clone(model)
        model_cv.fit(X_train_cv, y_train_cv)
        y_pred_cv = model_cv.predict(X_val_cv)
        fold_accuracy = accuracy_score(y_val_cv, y_pred_cv)
        scores.append(fold_accuracy)
        print(f"Fold {fold} Validation Accuracy: {fold_accuracy * 100:.2f}%")
    
    mean_accuracy = np.mean(scores)
    print(f"Mean 5-Fold Cross Validation Accuracy: {mean_accuracy * 100:.2f}%")
    return scores

# --- Continue with original k-fold cross-validation ---
print("\nPerforming 5-Fold Cross Validation on the entire dataset:")
k_fold_cv(MultinomialNB(), X, y, k=5)

# --- Perform k-fold CV on the best model as well ---
print("\nPerforming 5-Fold Cross Validation with the best model:")
if best_method == 'Base MultinomialNB':
    k_fold_cv(best_nb, X, y, k=5)
else:
    if best_method == 'Bagging with MultinomialNB':
        best_model_cv = BaggingClassifier(
            estimator=MultinomialNB(),
            n_estimators=10, 
            random_state=42
        )
    elif best_method == 'Voting (MNB, CNB, BNB)':
        best_model_cv = VotingClassifier(estimators=[
            ('mnb', MultinomialNB()),
            ('cnb', ComplementNB()),
            ('bnb', BernoulliNB())
        ], voting='soft')
    k_fold_cv(best_model_cv, X, y, k=5)

# --- Advanced Improvement Strategies ---
def try_advanced_improvements():
    print("\n=== Trying Advanced Improvement Strategies ===")
    results = {}
    
    # Strategy 1: Adjust training/test split ratio for more training data
    print("\n1. Adjusting train/test split ratio")
    X_train_large, X_test_small, y_train_large, y_test_small = train_test_split(
        X, y, test_size=0.2, random_state=42  # Using 80/20 split instead of 70/30
    )
    
    voting = VotingClassifier(estimators=[
        ('mnb', MultinomialNB(alpha=0.5)),
        ('cnb', ComplementNB(alpha=0.5)),
        ('bnb', BernoulliNB(alpha=0.5))
    ], voting='soft')
    
    voting.fit(X_train_large, y_train_large)
    y_pred = voting.predict(X_test_small)
    accuracy = accuracy_score(y_test_small, y_pred)
    results['80/20 Split + Voting'] = accuracy
    print(f"Accuracy with 80/20 split: {accuracy:.4f}")
    
    # Strategy 2: Weighted Voting Classifier
    print("\n2. Using Weighted Voting Classifier")
    weighted_voting = VotingClassifier(estimators=[
        ('mnb', MultinomialNB(alpha=0.5)),
        ('cnb', ComplementNB(alpha=0.5)),
        ('bnb', BernoulliNB(alpha=0.5))
    ], voting='soft', weights=[2, 1, 1])  # Give more weight to MultinomialNB
    
    weighted_voting.fit(X_train, y_train)
    
    # Validation accuracy
    y_val_pred = weighted_voting.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Weighted voting validation accuracy: {val_accuracy:.4f}")
    
    # Test accuracy
    y_pred = weighted_voting.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    results['Weighted Voting'] = {'test_accuracy': test_accuracy, 'val_accuracy': val_accuracy}
    print(f"Weighted voting test accuracy: {test_accuracy:.4f}")
    
    # Strategy 3: Feature Selection
    print("\n3. Feature Selection")
    from sklearn.feature_selection import SelectKBest, chi2
    
    # Select top k features
    k_values = [50, 100, 150, 200]
    for k in k_values:
        selector = SelectKBest(chi2, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_val_selected = selector.transform(X_val)
        X_test_selected = selector.transform(X_test)
        
        voting = VotingClassifier(estimators=[
            ('mnb', MultinomialNB(alpha=0.5)),
            ('cnb', ComplementNB(alpha=0.5)),
            ('bnb', BernoulliNB(alpha=0.5))
        ], voting='soft')
        
        voting.fit(X_train_selected, y_train)
        
        # Validation accuracy
        y_val_pred = voting.predict(X_val_selected)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        
        # Test accuracy
        y_pred = voting.predict(X_test_selected)
        test_accuracy = accuracy_score(y_test, y_pred)
        results[f'Feature Selection (k={k})'] = {'test_accuracy': test_accuracy, 'val_accuracy': val_accuracy}
        print(f"Feature Selection (k={k}) - Val Accuracy: {val_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Strategy 4: Stacking with a meta-classifier
    print("\n4. Stacking with Meta-classifier")
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import StackingClassifier
    
    estimators = [
        ('mnb', MultinomialNB(alpha=0.5)),
        ('cnb', ComplementNB(alpha=0.5)),
        ('bnb', BernoulliNB(alpha=0.5))
    ]
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5
    )
    
    stacking.fit(X_train, y_train)
    
    # Validation accuracy
    y_val_pred = stacking.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    # Test accuracy
    y_pred = stacking.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    results['Stacking with LR'] = {'test_accuracy': test_accuracy, 'val_accuracy': val_accuracy}
    print(f"Stacking - Val Accuracy: {val_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Strategy 5: Bagging with larger ensemble
    print("\n5. Bagging with Larger Ensemble")
    bagging_large = BaggingClassifier(
        estimator=MultinomialNB(alpha=0.5),
        n_estimators=50,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True,
        bootstrap_features=False,
        random_state=42
    )
    bagging_large.fit(X_train, y_train)
    
    # Validation accuracy
    y_val_pred = bagging_large.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    # Test accuracy
    y_pred = bagging_large.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    results['Bagging (n=50)'] = {'test_accuracy': test_accuracy, 'val_accuracy': val_accuracy}
    print(f"Bagging with 50 estimators - Val Accuracy: {val_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Strategy 6: Try adding TF-IDF transformation
    print("\n6. Using TF-IDF Features")
    from sklearn.feature_extraction.text import TfidfTransformer
    
    # We're simulating TF-IDF on the bag-of-words features we already have
    # In a real scenario, you might want to go back to raw text and use TfidfVectorizer
    tfidf = TfidfTransformer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)
    
    voting = VotingClassifier(estimators=[
        ('mnb', MultinomialNB(alpha=0.5)),
        ('cnb', ComplementNB(alpha=0.5)),
        ('bnb', BernoulliNB(alpha=0.5))
    ], voting='soft')
    
    voting.fit(X_train_tfidf, y_train)
    
    # Validation accuracy
    y_val_pred = voting.predict(X_val_tfidf)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    # Test accuracy
    y_pred = voting.predict(X_test_tfidf)
    test_accuracy = accuracy_score(y_test, y_pred)
    results['TF-IDF + Voting'] = {'test_accuracy': test_accuracy, 'val_accuracy': val_accuracy}
    print(f"TF-IDF + Voting - Val Accuracy: {val_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Find best strategy based on validation accuracy
    best_strategy = max([(k, v['val_accuracy']) for k, v in results.items() if isinstance(v, dict)], key=lambda x: x[1])
    print(f"\nBest strategy by validation: {best_strategy[0]} with validation accuracy: {best_strategy[1]:.4f}")
    
    return results, best_strategy

# After existing ensemble evaluation, add:
print("\nTrying advanced improvement strategies to reach 90%+ accuracy...")
advanced_results, best_strategy = try_advanced_improvements()

# Update the best_model saving section to include the best advanced strategy if it's better
best_advanced_accuracy = best_strategy[1]
current_best_accuracy = ensemble_results[best_method]['val_accuracy'] if best_method in ensemble_results else 0

if best_advanced_accuracy > current_best_accuracy:
    print(f"\nUsing advanced strategy '{best_strategy[0]}' as it achieved better validation accuracy: {best_advanced_accuracy:.4f} vs {current_best_accuracy:.4f}")
    
    # Implement code to save the best advanced model (specifics depend on which strategy wins)
    # This would replace the existing best model saving code or be added to it
    
    # For example, if the best strategy is TF-IDF + Voting:
    if best_strategy[0] == 'TF-IDF + Voting':
        from sklearn.feature_extraction.text import TfidfTransformer
        
        tfidf = TfidfTransformer()
        X_train_tfidf = tfidf.fit_transform(X_train)
        
        best_model = VotingClassifier(estimators=[
            ('mnb', MultinomialNB(alpha=0.5)),
            ('cnb', ComplementNB(alpha=0.5)),
            ('bnb', BernoulliNB(alpha=0.5))
        ], voting='soft')
        
        best_model.fit(X_train_tfidf, y_train)
        
        # Save both the TF-IDF transformer and the model
        best_model_package = {
            'tfidf': tfidf,
            'model': best_model,
            'feature_names': feature_names,
            'label_mapping': label_mapping
        }
        
        with open(f'{output_dir}/best_advanced_nb_model.pkl', 'wb') as f:
            pickle.dump(best_model_package, f)
            
        print(f"Best advanced model saved to {output_dir}/best_advanced_nb_model.pkl")

# Add a function to visualize all results
def visualize_results(ensemble_results, advanced_results):
    print("\n=== Performance Comparison of All Methods ===")
    import matplotlib.pyplot as plt
    
    # Combine all results for validation accuracy
    val_results = {}
    for method, data in ensemble_results.items():
        if isinstance(data, dict) and 'val_accuracy' in data:
            val_results[method] = data['val_accuracy']
    
    for method, data in advanced_results.items():
        if isinstance(data, dict) and 'val_accuracy' in data:
            val_results[method] = data['val_accuracy']
    
    # Sort by accuracy
    sorted_results = sorted(val_results.items(), key=lambda x: x[1], reverse=True)
    methods = [item[0] for item in sorted_results]
    accuracies = [item[1] for item in sorted_results]
    
    # Create the plot for validation accuracy
    plt.figure(figsize=(12, 8))
    bars = plt.bar(methods, accuracies, color='skyblue')
    plt.xlabel('Method')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Performance Comparison of All Methods')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add accuracy values on top of bars
    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                 f"{accuracy:.4f}", ha='center', va='bottom', rotation=0)
    
    # Add a horizontal line at 0.9 for the target
    plt.axhline(y=0.9, color='r', linestyle='-', label='90% Target')
    plt.legend()
    
    # Save the figure
    plt.savefig(f'{output_dir}/model_validation_performance.png', dpi=300, bbox_inches='tight')
    print(f"Validation performance comparison chart saved to {output_dir}/model_validation_performance.png")
    
    # Also create a test accuracy comparison chart
    test_results = {}
    for method, data in ensemble_results.items():
        if isinstance(data, dict) and 'test_accuracy' in data:
            test_results[method] = data['test_accuracy']
    
    for method, data in advanced_results.items():
        if isinstance(data, dict) and 'test_accuracy' in data:
            test_results[method] = data['test_accuracy']
    
    # Sort by accuracy
    sorted_results = sorted(test_results.items(), key=lambda x: x[1], reverse=True)
    methods = [item[0] for item in sorted_results]
    accuracies = [item[1] for item in sorted_results]
    
    # Create the plot for test accuracy
    plt.figure(figsize=(12, 8))
    bars = plt.bar(methods, accuracies, color='lightgreen')
    plt.xlabel('Method')
    plt.ylabel('Test Accuracy')
    plt.title('Test Performance Comparison of All Methods')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add accuracy values on top of bars
    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                 f"{accuracy:.4f}", ha='center', va='bottom', rotation=0)
    
    # Add a horizontal line at 0.9 for the target
    plt.axhline(y=0.9, color='r', linestyle='-', label='90% Target')
    plt.legend()
    
    # Save the figure
    plt.savefig(f'{output_dir}/model_test_performance.png', dpi=300, bbox_inches='tight')
    print(f"Test performance comparison chart saved to {output_dir}/model_test_performance.png")

# Call visualization function at the end
try:
    visualize_results(ensemble_results, advanced_results)
except Exception as e:
    print(f"Could not create visualization: {e}")
