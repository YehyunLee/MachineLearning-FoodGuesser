import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, learning_curve
import pickle
import os
import sys
sys.path.append('../utils')
from preprocess import preprocess

# Path to dataset and model params
DATASET_PATH = '../data/cleanedWithScript/manual_cleaned_data_universal.csv'
BASE_MODEL_PARAMS = '../model_params/naive_bayes_params.pkl'
TUNED_MODEL_PARAMS = '../model_params/k_fold_naive_bayes_params.pkl'
TUNED_MODEL_PATH = '../model_params/k_fold_naive_bayes_model.pkl'
OUTPUT_DIR = '../statistics/naive_bayes'

def ensure_dir_exists(path):
    os.makedirs(path, exist_ok=True)

def load_model_params(param_file):
    with open(param_file, 'rb') as f:
        return pickle.load(f)

def load_full_model(model_file):
    with open(model_file, 'rb') as f:
        return pickle.load(f)

def load_and_prepare_data():
    """Load and prepare data for analysis."""
    print("Loading and preparing data...")
    df = preprocess(DATASET_PATH, normalize_and_onehot=False, mode="full")
    
    # Identify label columns
    label_cols = [col for col in df.columns if col.startswith("Label")]
    
    # Extract features and labels
    X = df.drop(["id"] + label_cols, axis=1)
    feature_names = X.columns.tolist()
    X = X.values
    
    # Convert one-hot labels to integer indices
    y = np.argmax(df[label_cols].values, axis=1)
    
    # Save the label mapping for prediction
    label_mapping = {i: col.replace('Label_', '') for i, col in enumerate(label_cols)}
    
    # Get distinct classes
    classes = np.unique(y)
    
    return X, y, feature_names, label_mapping, classes, label_cols

def plot_confusion_matrices(X_test, y_test, base_model, tuned_model, label_mapping):
    """Generate and plot confusion matrices for base and tuned models."""
    print("Generating confusion matrices...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Base model confusion matrix
    y_pred_base = base_model.predict(X_test)
    cm_base = confusion_matrix(y_test, y_pred_base)
    
    # Tuned model confusion matrix
    y_pred_tuned = tuned_model.predict(X_test)
    cm_tuned = confusion_matrix(y_test, y_pred_tuned)
    
    # Plot confusion matrices
    class_names = [label_mapping[i] for i in sorted(label_mapping.keys())]
    
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Base Model Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    sns.heatmap(cm_tuned, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Tuned Model Confusion Matrix')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrices.png", dpi=300)
    
    return y_pred_base, y_pred_tuned

def plot_learning_curves(X, y, base_params, tuned_params):
    """Plot learning curves to show how model performance varies with training data size."""
    print("Generating learning curves...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Base model - Create new model with same parameters
    base_model = MultinomialNB(alpha=1.0, fit_prior=True)
    
    # Tuned model - Create new model with tuned parameters
    tuned_model = MultinomialNB(
        alpha=tuned_params['best_params']['alpha'], 
        fit_prior=tuned_params['best_params']['fit_prior']
    )
    
    # Learning curve for base model
    train_sizes, train_scores, val_scores = learning_curve(
        base_model, X, y, train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1) * 100
    train_std = np.std(train_scores, axis=1) * 100
    val_mean = np.mean(val_scores, axis=1) * 100
    val_std = np.std(val_scores, axis=1) * 100
    
    axes[0].plot(train_sizes, train_mean, label='Training score')
    axes[0].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    axes[0].plot(train_sizes, val_mean, label='Cross-validation score')
    axes[0].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    axes[0].set_title('Base Model Learning Curve')
    axes[0].set_xlabel('Training Examples')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend(loc='lower right')
    axes[0].grid(True)
    
    # Learning curve for tuned model
    train_sizes, train_scores, val_scores = learning_curve(
        tuned_model, X, y, train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1) * 100
    train_std = np.std(train_scores, axis=1) * 100
    val_mean = np.mean(val_scores, axis=1) * 100
    val_std = np.std(val_scores, axis=1) * 100
    
    axes[1].plot(train_sizes, train_mean, label='Training score')
    axes[1].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    axes[1].plot(train_sizes, val_mean, label='Cross-validation score')
    axes[1].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    axes[1].set_title('Tuned Model Learning Curve')
    axes[1].set_xlabel('Training Examples')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend(loc='lower right')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/learning_curves.png", dpi=300)

def analyze_feature_importance(base_params, tuned_params, feature_names, label_mapping):
    """Analyze and visualize the most important features for each class."""
    print("Analyzing feature importance...")
    
    # Get feature log probabilities from both models
    base_feature_log_prob = base_params['feature_log_prob']
    tuned_feature_log_prob = tuned_params['feature_log_prob']
    
    num_classes = base_feature_log_prob.shape[0]
    num_top_features = 15  # Number of top features to display
    
    fig, axes = plt.subplots(num_classes, 2, figsize=(18, num_classes * 5))
    
    for class_idx in range(num_classes):
        class_name = label_mapping[class_idx]
        
        # Base model - Get top features
        base_feature_importance = base_feature_log_prob[class_idx]
        top_indices = np.argsort(base_feature_importance)[-num_top_features:]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = [base_feature_importance[i] for i in top_indices]
        
        # Sort for better visualization
        sorted_indices = np.argsort(top_importances)
        top_features = [top_features[i] for i in sorted_indices]
        top_importances = [top_importances[i] for i in sorted_indices]
        
        axes[class_idx, 0].barh(top_features, top_importances)
        axes[class_idx, 0].set_title(f'Base Model - Top Features for {class_name}')
        axes[class_idx, 0].set_xlabel('Log Probability')
        
        # Tuned model - Get top features
        tuned_feature_importance = tuned_feature_log_prob[class_idx]
        top_indices = np.argsort(tuned_feature_importance)[-num_top_features:]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = [tuned_feature_importance[i] for i in top_indices]
        
        # Sort for better visualization
        sorted_indices = np.argsort(top_importances)
        top_features = [top_features[i] for i in sorted_indices]
        top_importances = [top_importances[i] for i in sorted_indices]
        
        axes[class_idx, 1].barh(top_features, top_importances)
        axes[class_idx, 1].set_title(f'Tuned Model - Top Features for {class_name}')
        axes[class_idx, 1].set_xlabel('Log Probability')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=300)

def analyze_class_distribution(y, y_pred_base, y_pred_tuned, label_mapping):
    """Analyze and visualize class distribution and model performance by class."""
    print("Analyzing class distribution...")
    
    # Class distribution in dataset
    classes = np.unique(y)
    class_counts = {label_mapping[cls]: np.sum(y == cls) for cls in classes}
    
    # Calculate metrics per class
    base_precision = precision_score(y, y_pred_base, average=None)
    base_recall = recall_score(y, y_pred_base, average=None)
    base_f1 = f1_score(y, y_pred_base, average=None)
    
    tuned_precision = precision_score(y, y_pred_tuned, average=None)
    tuned_recall = recall_score(y, y_pred_tuned, average=None)
    tuned_f1 = f1_score(y, y_pred_tuned, average=None)
    
    # Create a DataFrame for visualization
    metrics_df = pd.DataFrame({
        'Class': [label_mapping[cls] for cls in classes],
        'Count': [class_counts[label_mapping[cls]] for cls in classes],
        'Base Precision': base_precision * 100,
        'Base Recall': base_recall * 100,
        'Base F1': base_f1 * 100,
        'Tuned Precision': tuned_precision * 100,
        'Tuned Recall': tuned_recall * 100,
        'Tuned F1': tuned_f1 * 100
    })
    
    # Save metrics as CSV
    metrics_df.to_csv(f"{OUTPUT_DIR}/class_metrics.csv", index=False)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Class distribution
    axes[0, 0].pie(metrics_df['Count'], labels=metrics_df['Class'], autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Class Distribution in Dataset')
    axes[0, 0].axis('equal')
    
    # Precision comparison
    index = np.arange(len(classes))
    bar_width = 0.35
    axes[0, 1].bar(index, metrics_df['Base Precision'], bar_width, label='Base Model')
    axes[0, 1].bar(index + bar_width, metrics_df['Tuned Precision'], bar_width, label='Tuned Model')
    axes[0, 1].set_xlabel('Class')
    axes[0, 1].set_ylabel('Precision (%)')
    axes[0, 1].set_title('Precision by Class')
    axes[0, 1].set_xticks(index + bar_width / 2)
    axes[0, 1].set_xticklabels(metrics_df['Class'])
    axes[0, 1].legend()
    
    # Recall comparison
    axes[1, 0].bar(index, metrics_df['Base Recall'], bar_width, label='Base Model')
    axes[1, 0].bar(index + bar_width, metrics_df['Tuned Recall'], bar_width, label='Tuned Model')
    axes[1, 0].set_xlabel('Class')
    axes[1, 0].set_ylabel('Recall (%)')
    axes[1, 0].set_title('Recall by Class')
    axes[1, 0].set_xticks(index + bar_width / 2)
    axes[1, 0].set_xticklabels(metrics_df['Class'])
    axes[1, 0].legend()
    
    # F1 comparison
    axes[1, 1].bar(index, metrics_df['Base F1'], bar_width, label='Base Model')
    axes[1, 1].bar(index + bar_width, metrics_df['Tuned F1'], bar_width, label='Tuned Model')
    axes[1, 1].set_xlabel('Class')
    axes[1, 1].set_ylabel('F1 Score (%)')
    axes[1, 1].set_title('F1 Score by Class')
    axes[1, 1].set_xticks(index + bar_width / 2)
    axes[1, 1].set_xticklabels(metrics_df['Class'])
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/class_distribution.png", dpi=300)
    
    return metrics_df

def compare_alpha_impact():
    """Compare model performance with different alpha values."""
    print("Analyzing impact of alpha parameter...")
    
    X, y, feature_names, label_mapping, classes, _ = load_and_prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Test different alpha values
    alpha_values = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    accuracies = []
    log_losses = []
    
    for alpha in alpha_values:
        model = MultinomialNB(alpha=alpha)
        model.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        accuracies.append(accuracy_score(y_test, y_pred) * 100)
        log_losses.append(log_loss(y_test, y_pred_proba))
    
    # Plot results
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plotting Accuracy
    ax1.plot(alpha_values, accuracies, 'b-', label='Accuracy')
    ax1.set_xlabel('Alpha Value')
    ax1.set_ylabel('Accuracy (%)', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_xscale('log')
    
    # Plotting Log Loss on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(alpha_values, log_losses, 'r-', label='Log Loss')
    ax2.set_ylabel('Log Loss', color='r')
    ax2.tick_params('y', colors='r')
    
    # Adding legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.title('Impact of Alpha Parameter on Model Performance')
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/alpha_impact.png", dpi=300)
    
    # Save data as CSV
    alpha_df = pd.DataFrame({
        'Alpha': alpha_values,
        'Accuracy (%)': accuracies,
        'Log Loss': log_losses
    })
    alpha_df.to_csv(f"{OUTPUT_DIR}/alpha_comparison.csv", index=False)
    
    return alpha_df

def generate_summary_statistics():
    """Generate summary statistics and save them to a file."""
    print("Generating summary statistics...")
    
    try:
        base_params = load_model_params(BASE_MODEL_PARAMS)
        tuned_params = load_model_params(TUNED_MODEL_PARAMS)
        
        # Create a summary statistics dictionary
        summary = {
            "Base Model": {
                "Accuracy (%)": base_params.get('metrics', {}).get('accuracy', 'N/A'),
                "Precision (%)": base_params.get('metrics', {}).get('precision', 'N/A'),
                "Recall (%)": base_params.get('metrics', {}).get('recall', 'N/A'),
                "F1 Score (%)": base_params.get('metrics', {}).get('f1_score', 'N/A'),
                "Log Loss": base_params.get('metrics', {}).get('log_loss', 'N/A'),
                "Alpha": 1.0,
                "Fit Prior": True
            },
            "Tuned Model": {
                "Accuracy (%)": tuned_params.get('test_metrics', {}).get('accuracy', 'N/A'),
                "Precision (%)": tuned_params.get('test_metrics', {}).get('precision', 'N/A'),
                "Recall (%)": tuned_params.get('test_metrics', {}).get('recall', 'N/A'),
                "F1 Score (%)": tuned_params.get('test_metrics', {}).get('f1_score', 'N/A'),
                "Log Loss": tuned_params.get('test_metrics', {}).get('log_loss', 'N/A'),
                "Alpha": tuned_params.get('best_params', {}).get('alpha', 'N/A'),
                "Fit Prior": tuned_params.get('best_params', {}).get('fit_prior', 'N/A')
            },
            "Cross Validation (Tuned Model)": {
                "Avg Accuracy (%)": tuned_params.get('avg_accuracy', 'N/A'),
                "Std Accuracy (%)": tuned_params.get('std_accuracy', 'N/A'),
                "Avg Log Loss": tuned_params.get('avg_log_loss', 'N/A'),
                "Std Log Loss": tuned_params.get('std_log_loss', 'N/A'),
                "Fold Accuracies (%)": tuned_params.get('fold_accuracies', 'N/A'),
                "Fold Log Losses": tuned_params.get('fold_log_losses', 'N/A')
            }
        }
        
        # Convert to DataFrames for easier formatting
        base_df = pd.DataFrame(summary["Base Model"], index=[0])
        tuned_df = pd.DataFrame(summary["Tuned Model"], index=[0])
        cv_df = pd.DataFrame({k: [v] if not isinstance(v, list) else [str(v)] 
                             for k, v in summary["Cross Validation (Tuned Model)"].items()})
        
        # Save to CSV files
        base_df.to_csv(f"{OUTPUT_DIR}/base_model_summary.csv", index=False)
        tuned_df.to_csv(f"{OUTPUT_DIR}/tuned_model_summary.csv", index=False)
        cv_df.to_csv(f"{OUTPUT_DIR}/cv_summary.csv", index=False)
        
        # Create a comprehensive LaTeX table
        with open(f"{OUTPUT_DIR}/model_comparison_latex.txt", "w") as f:
            f.write("\\begin{table}[H]\n")
            f.write("\\centering\n")
            f.write("\\caption{Comparison of Naive Bayes Models}\n")
            f.write("\\label{tab:nb_comparison}\n")
            f.write("\\begin{tabular}{lcc}\n")
            f.write("\\toprule\n")
            f.write("\\textbf{Metric} & \\textbf{Base Model} & \\textbf{Tuned Model} \\\\\n")
            f.write("\\midrule\n")
            f.write(f"Accuracy (\\%) & {summary['Base Model']['Accuracy (%)']} & {summary['Tuned Model']['Accuracy (%)']} \\\\\n")
            f.write(f"Precision (\\%) & {summary['Base Model']['Precision (%)']} & {summary['Tuned Model']['Precision (%)']} \\\\\n")
            f.write(f"Recall (\\%) & {summary['Base Model']['Recall (%)']} & {summary['Tuned Model']['Recall (%)']} \\\\\n")
            f.write(f"F1 Score (\\%) & {summary['Base Model']['F1 Score (%)']} & {summary['Tuned Model']['F1 Score (%)']} \\\\\n")
            f.write(f"Log Loss & {summary['Base Model']['Log Loss']} & {summary['Tuned Model']['Log Loss']} \\\\\n")
            f.write(f"Alpha & {summary['Base Model']['Alpha']} & {summary['Tuned Model']['Alpha']} \\\\\n")
            f.write(f"Fit Prior & {summary['Base Model']['Fit Prior']} & {summary['Tuned Model']['Fit Prior']} \\\\\n")
            f.write("\\midrule\n")
            f.write(f"K-fold CV Accuracy (\\%) & - & {summary['Cross Validation (Tuned Model)']['Avg Accuracy (%)']} $\\pm$ {summary['Cross Validation (Tuned Model)']['Std Accuracy (%)']} \\\\\n")
            f.write(f"K-fold CV Log Loss & - & {summary['Cross Validation (Tuned Model)']['Avg Log Loss']} $\\pm$ {summary['Cross Validation (Tuned Model)']['Std Log Loss']} \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
            
        return summary
    except Exception as e:
        print(f"Error in generate_summary_statistics: {e}")
        return None

def main():
    """Main function to generate all statistics."""
    # Create output directory
    ensure_dir_exists(OUTPUT_DIR)
    
    # Load and prepare data
    X, y, feature_names, label_mapping, classes, label_cols = load_and_prepare_data()
    
    # Load model parameters
    try:
        base_params = load_model_params(BASE_MODEL_PARAMS)
        tuned_params = load_model_params(TUNED_MODEL_PARAMS)
        tuned_model = load_full_model(TUNED_MODEL_PATH)
        
        # Create a base model with default parameters
        base_model = MultinomialNB(alpha=1.0, fit_prior=True)
        base_model.feature_log_prob_ = base_params['feature_log_prob']
        base_model.class_log_prior_ = base_params['class_log_prior']
        base_model.classes_ = base_params['classes']
        
        # Split the data for testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Re-fit the base model (since we only have parameters)
        base_model.fit(X_train, y_train)
        
        # Generate confusion matrices
        y_pred_base, y_pred_tuned = plot_confusion_matrices(X_test, y_test, base_model, tuned_model, label_mapping)
        
        # Generate learning curves
        plot_learning_curves(X, y, base_params, tuned_params)
        
        # Analyze feature importance
        analyze_feature_importance(base_params, tuned_params, feature_names, label_mapping)
        
        # Analyze class distribution and per-class metrics
        metrics_df = analyze_class_distribution(y_test, y_pred_base, y_pred_tuned, label_mapping)
        
        # Analyze impact of alpha parameter
        alpha_df = compare_alpha_impact()
        
        # Generate summary statistics
        summary = generate_summary_statistics()
        
        print("\nStatistics generation complete! Files saved to:", OUTPUT_DIR)
        
    except Exception as e:
        print(f"Error in main: {e}")
    
if __name__ == "__main__":
    main()
