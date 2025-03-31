"""
Prediction script for CSC311 Food Predictor project.
This script takes a CSV file as input and returns a list of food predictions.
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'universalDataCleaning'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from preprocessForAllModels import preprocess as clean_data
from builtin_preprocess import preprocess as feature_extraction

# Define allowed prediction classes
ALLOWED_PREDICTIONS = ['Pizza', 'Shawarma', 'Sushi']

# Load model parameters
def load_model_params():
    params_path = os.path.join(os.path.dirname(__file__), 'model_params', 'naive_bayes_params.pkl')
    with open(params_path, 'rb') as f:
        return pickle.load(f)

model_params = load_model_params()

def naive_bayes_predict(X):
    """
    Implements Naive Bayes prediction without using sklearn
    """
    # Ensure X has the same features as model (add missing columns with zeros)
    model_features = model_params['feature_names']
    
    # Create a new DataFrame with only the features the model knows about
    aligned_X = pd.DataFrame(0, index=range(len(X)), columns=model_features)
    
    # Fill in values for features that exist in both X and model_features
    common_features = set(X.columns).intersection(set(model_features))
    for feature in common_features:
        try:
            # Try to ensure numeric values and proper alignment
            values = pd.to_numeric(X[feature], errors='coerce').fillna(0).values
            aligned_X[feature] = values
        except Exception as e:
            # Continue with other features if one fails
            continue
    
    # Get parameters
    classes = model_params['classes']
    class_log_prior = model_params['class_log_prior']
    feature_log_prob = model_params['feature_log_prob']
    
    # Calculate log probabilities for each class
    joint_log_likelihood = np.zeros((aligned_X.shape[0], len(classes)))
    
    for i, c in enumerate(classes):
        # For each class, calculate the joint log likelihood
        joint_log_likelihood[:, i] = class_log_prior[i]
        joint_log_likelihood[:, i] += np.dot(aligned_X.values, feature_log_prob[i])
    
    # Return the class with highest probability
    predicted_indices = np.argmax(joint_log_likelihood, axis=1)
    predicted_labels = [model_params['label_mapping'][idx] for idx in predicted_indices]
    
    return predicted_labels

def predict_all(filename):
    """
    Make predictions for the data in filename
    """
    # First, read the original file to know how many rows we should return
    original_df = pd.read_csv(filename)
    original_row_count = len(original_df)
    
    try:
        # Step 1: Clean the data using preprocessForAllModels
        cleaned_df = clean_data(filename, return_df=True)
        
        # Step 2: Extract features using builtin_preprocess (no sklearn)
        # Important: set drop_na=False to keep all rows
        preprocessed_df = feature_extraction(None, normalize_and_onehot=False, mode="full", df_in=cleaned_df, drop_na=False)
        
        # Step 3: Get feature columns (exclude id and label columns)
        feature_cols = [col for col in preprocessed_df.columns if not col.startswith('Label_') and col != 'id']
        features = preprocessed_df[feature_cols]
        
        # Step 4: Make predictions
        predictions = naive_bayes_predict(features)
        
        # Ensure predictions are valid
        # predictions = [pred if pred in ALLOWED_PREDICTIONS else 'Pizza' for pred in predictions]
        
        # Sanity check on prediction count
        if len(predictions) != original_row_count:
            print(f"Warning: Prediction count ({len(predictions)}) doesn't match row count ({original_row_count})")
            # If somehow we have fewer predictions, pad with default
            # if len(predictions) < original_row_count:
            #     predictions.extend(['Pizza'] * (original_row_count - len(predictions)))
            # # If we have more predictions, truncate
            # predictions = predictions[:original_row_count]
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        # If anything fails, return default predictions for all rows
        # predictions = ['Pizza'] * original_row_count
    
    print(f"Final predictions: {len(predictions)} for {original_row_count} rows")
    return predictions

def check_prediction_answers(filename):
    """
    Calculate the accuracy of the predictions against the true labels.
    """
    try:
        # Load the data
        df = pd.read_csv(filename)
        
        # Get the true labels
        true_labels = df["Label"].tolist()
        
        # Get the predictions
        predictions = predict_all(filename)
        
        # Ensure that the number of true labels and predictions match
        if len(true_labels) != len(predictions):
            print(f"Warning: Number of true labels ({len(true_labels)}) does not match number of predictions ({len(predictions)}).")
            min_length = min(len(true_labels), len(predictions))
            true_labels = true_labels[:min_length]
            predictions = predictions[:min_length]
        
        # Calculate the number of correct predictions
        correct_predictions = sum(1 for true, pred in zip(true_labels, predictions) if str(true) == str(pred))
        
        # Calculate the accuracy
        accuracy = correct_predictions / len(true_labels) if len(true_labels) > 0 else 0.0
        
        print(f"Accuracy: {accuracy:.4f}")
        return accuracy
    
    except Exception as e:
        print(f"Error during accuracy calculation: {e}")
        return 0.0

# For testing the script
if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        # predictions = predict_all(test_file)
        # print("Sample predictions:", predictions[:10], "...")
        
        # Check accuracy if the file has labels
        try:
            df = pd.read_csv(test_file)
            if "Label" in df.columns:
                check_prediction_answers(test_file)
        except:
            pass
    else:
        print("Usage: python pred.py <test_file.csv>")
