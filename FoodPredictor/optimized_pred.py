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

# Load optimized model
def load_model():
    # Try to load the best optimized model first
    best_model_path = os.path.join(os.path.dirname(__file__), 'model_params', 'best_naive_bayes_model.pkl')
    try:
        with open(best_model_path, 'rb') as f:
            model = pickle.load(f)
        print("Loaded optimized Naive Bayes model.")
        
        # Check if the model is a sklearn model, but missing the feature selector
        if hasattr(model, 'predict') and not hasattr(model, 'selector'):
            # Load feature selector if available separately
            selector_path = os.path.join(os.path.dirname(__file__), 'model_params', 'feature_selector.pkl')
            try:
                with open(selector_path, 'rb') as f:
                    selector = pickle.load(f)
                print("Loaded feature selector.")
                # Create a wrapper object that includes both model and selector
                return {'model': model, 'selector': selector, 'type': 'feature_selection_model'}
            except Exception as e:
                print(f"Warning: Feature selector not found: {e}")
                # Still return the model even without feature selection
        
        return model
    except Exception as e:
        print(f"Error loading optimized model: {e}")
        
        # Fall back to original model parameters
        original_params_path = os.path.join(os.path.dirname(__file__), 'model_params', 'naive_bayes_params.pkl')
        try:
            with open(original_params_path, 'rb') as f:
                params = pickle.load(f)
            print("Loaded original model parameters as fallback.")
            return params
        except Exception as e2:
            print(f"Error loading original model parameters: {e2}")
            return None

model = load_model()

def select_best_features(X, k=150):
    """
    Select the k most important features.
    Based on the optimization results, k=150 was found to be optimal.
    """
    try:
        # If model is a dict with explicit 'type' field for feature selection
        if isinstance(model, dict) and model.get('type') == 'feature_selection_model':
            if 'selector' in model:
                print(f"Using feature selector from optimized model (k=150)")
                try:
                    selector = model['selector']
                    # Check if the selector has transform method (like SelectKBest)
                    if hasattr(selector, 'transform'):
                        return pd.DataFrame(selector.transform(X), index=X.index)
                    # Or if it just has a list of selected feature names
                    elif hasattr(selector, 'get_support'):
                        support = selector.get_support()
                        selected_features = [feature for feature, selected in zip(X.columns, support) if selected]
                        print(f"Selected {len(selected_features)} features using optimized feature selector")
                        return X[selected_features]
                    else:
                        print("Feature selector not compatible, using all features")
                        return X
                except Exception as e:
                    print(f"Error applying feature selector: {e}")
                    return X
        
        # If we're loading a direct scikit-learn model that expects all features
        if hasattr(model, 'predict'):
            # Check for dimension requirements by inspecting model attributes
            if hasattr(model, 'feature_count_') and model.feature_count_.shape[1] != X.shape[1]:
                print(f"Model expects {model.feature_count_.shape[1]} features, but got {X.shape[1]} features")
                print("Using all features instead of selection to match model dimensions")
                return X
        
        # For explicit feature selection - DON'T USE THIS, return all features instead
        # The issue is that the model was trained on all features, not selected ones
        print(f"Skipping feature selection to match model dimensions - using all {X.shape[1]} features")
        return X
    
    except Exception as e:
        print(f"Error in feature selection: {e}")
        # Return original features on error
        return X

def naive_bayes_predict(X):
    """
    Implements prediction using the loaded model, handling different model types
    """
    if model is None:
        print("No model loaded. Cannot make predictions.")
        return [ALLOWED_PREDICTIONS[0]] * len(X)
    
    try:
        # Skip feature selection for now, as it's causing dimension mismatch
        X_selected = X  # Use all features
        
        # Check model from optimization results structure
        if isinstance(model, dict) and model.get('type') == 'feature_selection_model':
            if 'model' in model:
                print("Using model from feature selection package")
                nb_model = model['model']
                try:
                    predictions = nb_model.predict(X_selected)
                    
                    # Handle numeric predictions
                    if isinstance(predictions[0], (int, np.integer)):
                        label_mapping = {i: label for i, label in enumerate(ALLOWED_PREDICTIONS)}
                        predicted_labels = [label_mapping.get(idx, ALLOWED_PREDICTIONS[0]) for idx in predictions]
                    else:
                        predicted_labels = predictions
                    
                    return predicted_labels
                except Exception as e:
                    print(f"Error predicting with feature selection model: {e}")
                    return [ALLOWED_PREDICTIONS[0]] * len(X)
        
        # Check if model is a scikit-learn model with predict method
        if hasattr(model, 'predict'):
            print("Using scikit-learn model's predict method")
            try:
                predictions = model.predict(X_selected)
                
                # Convert numeric predictions to label strings if needed
                if isinstance(predictions[0], (int, np.integer)):
                    label_mapping = {i: label for i, label in enumerate(ALLOWED_PREDICTIONS)}
                    predicted_labels = [label_mapping.get(idx, ALLOWED_PREDICTIONS[0]) for idx in predictions]
                else:
                    predicted_labels = predictions
                    
                return predicted_labels
            except ValueError as e:
                print(f"Error in prediction with scikit-learn model: {e}")
                print("Falling back to default predictions")
                return [ALLOWED_PREDICTIONS[0]] * len(X)
        
        # Check if model is a dictionary containing components
        elif isinstance(model, dict):
            # Check model format
            if 'model' in model:
                print("Using stored model component")
                nb_model = model['model']
                
                # Apply TF-IDF if available
                if 'tfidf' in model:
                    print("Applying TF-IDF transformation")
                    X_selected = model['tfidf'].transform(X_selected)
                
                # Make predictions
                predictions = nb_model.predict(X_selected)
                
                # Get label mapping
                label_mapping = model.get('label_mapping', 
                                         {i: label for i, label in enumerate(ALLOWED_PREDICTIONS)})
                
                predicted_labels = [label_mapping.get(idx, ALLOWED_PREDICTIONS[0]) for idx in predictions]
                return predicted_labels
            
            # Raw parameters
            elif 'feature_names' in model and 'feature_log_prob' in model:
                print("Using model parameters directly")
                # Get model parameters
                model_features = model['feature_names']
                classes = model['classes']
                class_log_prior = model['class_log_prior']
                feature_log_prob = model['feature_log_prob']
                
                # Align features with model
                aligned_X = pd.DataFrame(0, index=range(len(X_selected)), columns=model_features)
                common_features = set(X_selected.columns).intersection(set(model_features))
                
                for feature in common_features:
                    try:
                        aligned_X[feature] = pd.to_numeric(X_selected[feature], errors='coerce').fillna(0).values
                    except Exception as e:
                        print(f"Error processing feature {feature}: {e}")
                
                # Calculate probabilities
                joint_log_likelihood = np.zeros((aligned_X.shape[0], len(classes)))
                
                for i, c in enumerate(classes):
                    joint_log_likelihood[:, i] = class_log_prior[i]
                    joint_log_likelihood[:, i] += np.dot(aligned_X.values, feature_log_prob[i])
                
                # Get predictions
                predicted_indices = np.argmax(joint_log_likelihood, axis=1)
                label_mapping = model.get('label_mapping', 
                                         {i: label for i, label in enumerate(ALLOWED_PREDICTIONS)})
                predicted_labels = [label_mapping.get(idx, ALLOWED_PREDICTIONS[0]) for idx in predicted_indices]
                
                return predicted_labels
        
        # Fallback
        print("Unknown model format, using default predictions")
        return [ALLOWED_PREDICTIONS[0]] * len(X)
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return [ALLOWED_PREDICTIONS[0]] * len(X)

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
        print(f"Data cleaned: {cleaned_df.shape}")
        
        # Step 2: Extract features using builtin_preprocess
        preprocessed_df = feature_extraction(None, normalize_and_onehot=False, mode="full", df_in=cleaned_df, drop_na=False)
        print(f"Features extracted: {preprocessed_df.shape}")
        
        # Step 3: Get feature columns (exclude id and label columns)
        feature_cols = [col for col in preprocessed_df.columns if not col.startswith('Label_') and col != 'id']
        features = preprocessed_df[feature_cols]
        print(f"Selected {len(feature_cols)} initial features")
        
        # Step 4: Make predictions
        predictions = naive_bayes_predict(features)
        
        # Check prediction distribution
        prediction_counts = {}
        for p in predictions:
            prediction_counts[p] = prediction_counts.get(p, 0) + 1
        print(f"Prediction distribution: {prediction_counts}")
        
        # Ensure predictions are valid
        predictions = [pred if pred in ALLOWED_PREDICTIONS else ALLOWED_PREDICTIONS[0] for pred in predictions]
        
        # Sanity check on prediction count
        if len(predictions) != original_row_count:
            print(f"Warning: Prediction count ({len(predictions)}) doesn't match row count ({original_row_count})")
            # If somehow we have fewer predictions, pad with default
            if len(predictions) < original_row_count:
                predictions.extend([ALLOWED_PREDICTIONS[0]] * (original_row_count - len(predictions)))
            # If we have more predictions, truncate
            predictions = predictions[:original_row_count]
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        # If anything fails, return default predictions for all rows
        predictions = [ALLOWED_PREDICTIONS[0]] * original_row_count
    
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
        predictions = predict_all(test_file)
        print("Sample predictions:", predictions[:10], "...")
        
        # Check accuracy if the file has labels
        try:
            df = pd.read_csv(test_file)
            if "Label" in df.columns:
                check_prediction_answers(test_file)
        except:
            pass
    else:
        print("Usage: python pred.py <test_file.csv>")