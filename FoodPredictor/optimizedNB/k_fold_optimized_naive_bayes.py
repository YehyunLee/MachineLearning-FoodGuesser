import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
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

# --- Improved K-Fold Cross-Validation Function ---
def k_fold_cv_with_test(model_creator, X, y, k=5):
    """
    Performs k-fold cross-validation and calculates validation and test accuracy.
    For each fold, the data is split into training (70%), validation (15%), and test (15%) sets.
    
    Parameters:
      model_creator: Function that creates and returns a new model instance
      X: Features array
      y: Labels array
      k: Number of folds (default=5)
      
    Returns:
      val_scores: Validation accuracies for each fold
      test_scores: Test accuracies for each fold
    """
    # Use stratified K-fold to maintain class distribution across folds
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    val_scores = []
    test_scores = []
    
    print(f"\n=== {k}-Fold Cross-Validation with Validation and Test Sets ===")
    
    for fold, (train_val_index, test_index) in enumerate(skf.split(X, y), 1):
        # Split the non-test data into train and validation
        X_train_val, X_test_fold = X[train_val_index], X[test_index]
        y_train_val, y_test_fold = y[train_val_index], y[test_index]
        
        # Further split train_val into train and validation (roughly 70:15 ratio from original dataset)
        # This maintains the 70:15:15 split ratio in each fold
        X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
            X_train_val, y_train_val, test_size=0.177, random_state=42  # 0.177 is ~15/85 to get our desired 70:15:15 split
        )
        
        # Create a fresh model instance for this fold
        model = model_creator()
        
        # Train on the training set
        start_time = time.time()
        model.fit(X_train_fold, y_train_fold)
        train_time = time.time() - start_time
        
        # Evaluate on validation set
        y_val_pred = model.predict(X_val_fold)
        val_accuracy = accuracy_score(y_val_fold, y_val_pred)
        val_scores.append(val_accuracy)
        
        # Evaluate on test set
        y_test_pred = model.predict(X_test_fold)
        test_accuracy = accuracy_score(y_test_fold, y_test_pred)
        test_scores.append(test_accuracy)
        
        print(f"Fold {fold}/{k} - Training: {X_train_fold.shape[0]} samples, "
              f"Validation: {X_val_fold.shape[0]} samples, Test: {X_test_fold.shape[0]} samples")
        print(f"Fold {fold}/{k} - Validation Accuracy: {val_accuracy:.4f}, "
              f"Test Accuracy: {test_accuracy:.4f}, Time: {train_time:.2f}s")
    
    # Calculate and display mean and standard deviation
    mean_val_accuracy = np.mean(val_scores)
    std_val_accuracy = np.std(val_scores)
    mean_test_accuracy = np.mean(test_scores)
    std_test_accuracy = np.std(test_scores)
    
    print(f"\nMean Validation Accuracy: {mean_val_accuracy:.4f} (±{std_val_accuracy:.4f})")
    print(f"Mean Test Accuracy: {mean_test_accuracy:.4f} (±{std_test_accuracy:.4f})")
    
    return val_scores, test_scores

# --- Models to evaluate with K-fold CV ---
def evaluate_models_with_kfold():
    """
    Evaluate multiple Naive Bayes models using K-fold CV and compare their performance.
    """
    print("\n=== Evaluating Models with K-fold Cross-Validation ===")
    
    models = {
        "MultinomialNB": lambda: MultinomialNB(),
        "MultinomialNB (alpha=0.5)": lambda: MultinomialNB(alpha=0.5),
        "ComplementNB": lambda: ComplementNB(),
        "BernoulliNB": lambda: BernoulliNB(),
        "Bagging with MultinomialNB": lambda: BaggingClassifier(
            estimator=MultinomialNB(),
            n_estimators=10,
            max_samples=0.8,
            max_features=0.8,
            bootstrap=True,
            bootstrap_features=False,
            random_state=42
        ),
        "Voting (MNB, CNB, BNB)": lambda: VotingClassifier(estimators=[
            ('mnb', MultinomialNB()),
            ('cnb', ComplementNB()),
            ('bnb', BernoulliNB())
        ], voting='soft')
    }
    
    results = {}
    
    for model_name, model_creator in models.items():
        print(f"\nEvaluating {model_name}...")
        val_scores, test_scores = k_fold_cv_with_test(model_creator, X, y, k=5)
        results[model_name] = {
            'val_scores': val_scores,
            'mean_val': np.mean(val_scores),
            'test_scores': test_scores,
            'mean_test': np.mean(test_scores)
        }
    
    # Find the best model based on validation accuracy
    best_model = max(results.items(), key=lambda x: x[1]['mean_val'])
    print(f"\nBest model based on mean validation accuracy: {best_model[0]} with {best_model[1]['mean_val']:.4f}")
    
    return results, best_model[0]

# --- Advanced Strategies with K-fold CV ---
def evaluate_advanced_strategies_with_kfold():
    """
    Evaluate advanced modeling strategies using K-fold CV.
    """
    print("\n=== Evaluating Advanced Strategies with K-fold Cross-Validation ===")
    
    # Define advanced strategies
    strategies = {
        "TF-IDF + Voting": lambda: {
            'preprocess': lambda X_train, X_val, X_test: (
                tfidf_transformer.fit_transform(X_train),
                tfidf_transformer.transform(X_val),
                tfidf_transformer.transform(X_test)
            ),
            'model': VotingClassifier(estimators=[
                ('mnb', MultinomialNB(alpha=0.5)),
                ('cnb', ComplementNB(alpha=0.5)),
                ('bnb', BernoulliNB(alpha=0.5))
            ], voting='soft')
        },
        "Feature Selection (k=100)": lambda: {
            'preprocess': lambda X_train, X_val, X_test: (
                selector.fit_transform(X_train, y_train_fold),
                selector.transform(X_val),
                selector.transform(X_test)
            ),
            'model': VotingClassifier(estimators=[
                ('mnb', MultinomialNB(alpha=0.5)),
                ('cnb', ComplementNB(alpha=0.5)),
                ('bnb', BernoulliNB(alpha=0.5))
            ], voting='soft')
        },
        "Weighted Voting": lambda: {
            'preprocess': lambda X_train, X_val, X_test: (X_train, X_val, X_test),
            'model': VotingClassifier(estimators=[
                ('mnb', MultinomialNB(alpha=0.5)),
                ('cnb', ComplementNB(alpha=0.5)),
                ('bnb', BernoulliNB(alpha=0.5))
            ], voting='soft', weights=[2, 1, 1])
        }
    }
    
    # Custom k-fold CV for advanced strategies that require preprocessing
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_selection import SelectKBest, chi2
    
    results = {}
    
    # Define the k-fold splitter
    k = 5
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    # Evaluate each strategy
    for strategy_name in strategies:
        print(f"\nEvaluating {strategy_name}...")
        val_scores = []
        test_scores = []
        
        for fold, (train_val_index, test_index) in enumerate(skf.split(X, y), 1):
            # Split data for this fold
            X_train_val, X_test_fold = X[train_val_index], X[test_index]
            y_train_val, y_test_fold = y[train_val_index], y[test_index]
            
            # Further split train_val into train and validation
            X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
                X_train_val, y_train_val, test_size=0.177, random_state=42
            )
            
            # Create the strategy components for this fold
            if strategy_name == "TF-IDF + Voting":
                tfidf_transformer = TfidfTransformer()
                strategy_components = strategies[strategy_name]()
                
                # Apply preprocessing
                X_train_proc, X_val_proc, X_test_proc = strategy_components['preprocess'](
                    X_train_fold, X_val_fold, X_test_fold
                )
                
                # Get the model and train it
                model = strategy_components['model']
                model.fit(X_train_proc, y_train_fold)
                
                # Evaluate
                val_accuracy = accuracy_score(y_val_fold, model.predict(X_val_proc))
                test_accuracy = accuracy_score(y_test_fold, model.predict(X_test_proc))
                
            elif strategy_name == "Feature Selection (k=100)":
                selector = SelectKBest(chi2, k=100)
                strategy_components = strategies[strategy_name]()
                
                # Apply preprocessing
                X_train_proc = selector.fit_transform(X_train_fold, y_train_fold)
                X_val_proc = selector.transform(X_val_fold)
                X_test_proc = selector.transform(X_test_fold)
                
                # Get the model and train it
                model = strategy_components['model']
                model.fit(X_train_proc, y_train_fold)
                
                # Evaluate
                val_accuracy = accuracy_score(y_val_fold, model.predict(X_val_proc))
                test_accuracy = accuracy_score(y_test_fold, model.predict(X_test_proc))
                
            else:  # Weighted Voting or other strategies that don't need special preprocessing
                strategy_components = strategies[strategy_name]()
                X_train_proc, X_val_proc, X_test_proc = strategy_components['preprocess'](
                    X_train_fold, X_val_fold, X_test_fold
                )
                
                # Get the model and train it
                model = strategy_components['model']
                model.fit(X_train_proc, y_train_fold)
                
                # Evaluate
                val_accuracy = accuracy_score(y_val_fold, model.predict(X_val_proc))
                test_accuracy = accuracy_score(y_test_fold, model.predict(X_test_proc))
            
            val_scores.append(val_accuracy)
            test_scores.append(test_accuracy)
            
            print(f"Fold {fold}/{k} - Validation Accuracy: {val_accuracy:.4f}, "
                  f"Test Accuracy: {test_accuracy:.4f}")
        
        # Calculate and display mean and standard deviation
        mean_val_accuracy = np.mean(val_scores)
        std_val_accuracy = np.std(val_scores)
        mean_test_accuracy = np.mean(test_scores)
        std_test_accuracy = np.std(test_scores)
        
        print(f"{strategy_name} - Mean Validation Accuracy: {mean_val_accuracy:.4f} (±{std_val_accuracy:.4f})")
        print(f"{strategy_name} - Mean Test Accuracy: {mean_test_accuracy:.4f} (±{std_test_accuracy:.4f})")
        
        results[strategy_name] = {
            'val_scores': val_scores,
            'mean_val': mean_val_accuracy,
            'std_val': std_val_accuracy,
            'test_scores': test_scores,
            'mean_test': mean_test_accuracy,
            'std_test': std_test_accuracy
        }
    
    # Find the best strategy based on validation accuracy
    best_strategy = max(results.items(), key=lambda x: x[1]['mean_val'])
    print(f"\nBest advanced strategy: {best_strategy[0]} with mean validation accuracy: {best_strategy[1]['mean_val']:.4f}")
    
    return results, best_strategy[0]

# --- Visualize results from K-fold CV ---
def visualize_kfold_results(base_results, advanced_results=None):
    """
    Visualize the results from K-fold cross-validation.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.ticker import MaxNLocator
    
    output_dir = '../model_params'
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine results
    all_results = {}
    for model_name, metrics in base_results.items():
        all_results[model_name] = metrics
    
    if advanced_results:
        for strategy_name, metrics in advanced_results.items():
            all_results[strategy_name] = metrics
    
    # Create a DataFrame for easier visualization
    results_df = pd.DataFrame({
        'Model': [model for model in all_results],
        'Mean Validation Accuracy': [all_results[model]['mean_val'] for model in all_results],
        'Mean Test Accuracy': [all_results[model]['mean_test'] for model in all_results]
    })
    
    # Sort by validation accuracy
    results_df = results_df.sort_values('Mean Validation Accuracy', ascending=False)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot validation and test accuracies
    x = np.arange(len(results_df))
    width = 0.35
    
    val_bars = ax.bar(x - width/2, results_df['Mean Validation Accuracy'], 
                     width, label='Validation Accuracy', color='skyblue')
    test_bars = ax.bar(x + width/2, results_df['Mean Test Accuracy'], 
                      width, label='Test Accuracy', color='lightgreen')
    
    # Add some text for labels, title and custom x-axis tick labels
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation and Test Accuracy by Model (5-fold CV)')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
    ax.legend()
    
    # Add accuracy values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(val_bars)
    autolabel(test_bars)
    
    # Add a horizontal line at 0.9 for the target
    ax.axhline(y=0.9, color='r', linestyle='--', label='90% Target')
    ax.legend()
    
    fig.tight_layout()
    
    # Save the figure
    plt.savefig(f'{output_dir}/kfold_model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"K-fold model comparison chart saved to {output_dir}/kfold_model_comparison.png")
    
    # Also create a box plot to show distribution of scores across folds
    plt.figure(figsize=(14, 8))
    
    # Prepare data for boxplot
    val_data = [all_results[model]['val_scores'] for model in results_df['Model']]
    
    plt.boxplot(val_data, labels=results_df['Model'], patch_artist=True)
    plt.title('Distribution of Validation Accuracies Across 5 Folds')
    plt.xlabel('Model')
    plt.ylabel('Validation Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the boxplot
    plt.savefig(f'{output_dir}/kfold_validation_distribution.png', dpi=300, bbox_inches='tight')
    print(f"K-fold validation distribution chart saved to {output_dir}/kfold_validation_distribution.png")

# --- Main execution ---
if __name__ == "__main__":
    # First evaluate basic models with k-fold CV
    base_results, best_base_model = evaluate_models_with_kfold()
    
    # Then evaluate advanced strategies with k-fold CV
    advanced_results, best_advanced_strategy = evaluate_advanced_strategies_with_kfold()
    
    # Compare the best basic model with the best advanced strategy
    print("\n=== Final Comparison of Best Models ===")
    print(f"Best Base Model: {best_base_model} - Validation Accuracy: {base_results[best_base_model]['mean_val']:.4f}")
    print(f"Best Advanced Strategy: {best_advanced_strategy} - Validation Accuracy: {advanced_results[best_advanced_strategy]['mean_val']:.4f}")
    
    # Determine the overall best model
    if advanced_results[best_advanced_strategy]['mean_val'] > base_results[best_base_model]['mean_val']:
        print(f"\nOverall Best Model: {best_advanced_strategy} (Advanced) with validation accuracy: {advanced_results[best_advanced_strategy]['mean_val']:.4f}")
        best_overall = best_advanced_strategy
    else:
        print(f"\nOverall Best Model: {best_base_model} (Base) with validation accuracy: {base_results[best_base_model]['mean_val']:.4f}")
        best_overall = best_base_model
    
    # Visualize the results
    visualize_kfold_results(base_results, advanced_results)
    
    # Train the best model on the entire dataset and save it
    print("\n=== Training Best Model on Entire Dataset ===")
    output_dir = '../model_params'
    os.makedirs(output_dir, exist_ok=True)
    
    if best_overall == "TF-IDF + Voting":
        from sklearn.feature_extraction.text import TfidfTransformer
        
        tfidf = TfidfTransformer()
        X_tfidf = tfidf.fit_transform(X)
        
        best_model = VotingClassifier(estimators=[
            ('mnb', MultinomialNB(alpha=0.5)),
            ('cnb', ComplementNB(alpha=0.5)),
            ('bnb', BernoulliNB(alpha=0.5))
        ], voting='soft')
        
        best_model.fit(X_tfidf, y)
        
        # Save both the TF-IDF transformer and the model
        best_model_package = {
            'tfidf': tfidf,
            'model': best_model,
            'feature_names': feature_names,
            'label_mapping': label_mapping
        }
        
        with open(f'{output_dir}/best_kfold_nb_model.pkl', 'wb') as f:
            pickle.dump(best_model_package, f)
    
    elif best_overall == "Feature Selection (k=100)":
        from sklearn.feature_selection import SelectKBest, chi2
        
        selector = SelectKBest(chi2, k=100)
        X_selected = selector.fit_transform(X, y)
        
        best_model = VotingClassifier(estimators=[
            ('mnb', MultinomialNB(alpha=0.5)),
            ('cnb', ComplementNB(alpha=0.5)),
            ('bnb', BernoulliNB(alpha=0.5))
        ], voting='soft')
        
        best_model.fit(X_selected, y)
        
        # Save both the selector and the model
        best_model_package = {
            'selector': selector,
            'model': best_model,
            'feature_names': feature_names,
            'label_mapping': label_mapping
        }
        
        with open(f'{output_dir}/best_kfold_nb_model.pkl', 'wb') as f:
            pickle.dump(best_model_package, f)
    
    else:
        # For models that don't require special preprocessing
        if best_overall == "MultinomialNB":
            best_model = MultinomialNB()
        elif best_overall == "MultinomialNB (alpha=0.5)":
            best_model = MultinomialNB(alpha=0.5)
        elif best_overall == "ComplementNB":
            best_model = ComplementNB()
        elif best_overall == "BernoulliNB":
            best_model = BernoulliNB()
        elif best_overall == "Bagging with MultinomialNB":
            best_model = BaggingClassifier(
                estimator=MultinomialNB(),
                n_estimators=10,
                max_samples=0.8,
                max_features=0.8,
                bootstrap=True,
                bootstrap_features=False,
                random_state=42
            )
        elif best_overall == "Voting (MNB, CNB, BNB)":
            best_model = VotingClassifier(estimators=[
                ('mnb', MultinomialNB()),
                ('cnb', ComplementNB()),
                ('bnb', BernoulliNB())
            ], voting='soft')
        elif best_overall == "Weighted Voting":
            best_model = VotingClassifier(estimators=[
                ('mnb', MultinomialNB(alpha=0.5)),
                ('cnb', ComplementNB(alpha=0.5)),
                ('bnb', BernoulliNB(alpha=0.5))
            ], voting='soft', weights=[2, 1, 1])
        
        best_model.fit(X, y)
        
        # Save the model
        best_model_package = {
            'model': best_model,
            'feature_names': feature_names,
            'label_mapping': label_mapping
        }
        
        with open(f'{output_dir}/best_kfold_nb_model.pkl', 'wb') as f:
            pickle.dump(best_model_package, f)
    
    print(f"Best model from K-fold CV saved to {output_dir}/best_kfold_nb_model.pkl")
    print(f"Best model: {best_overall} with mean validation accuracy: {advanced_results[best_advanced_strategy]['mean_val']:.4f}" if best_overall in advanced_results else 
          f"Best model: {best_overall} with mean validation accuracy: {base_results[best_base_model]['mean_val']:.4f}")
