import os
import pickle
import sys
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2

"""
This script extracts feature importance information from the optimized model
and saves it as a separate feature selector for use with pred.py.
"""

# Function to load the optimized model
def load_optimized_model():
    model_path = os.path.join('model_params', 'best_naive_bayes_model.pkl')
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Loaded optimized model: {type(model).__name__}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Function to create and save a feature selector
def create_feature_selector(k=150):
    """Create a feature selector based on the training data"""
    try:
        # We need to re-import the same data that was used for optimization
        # This assumes we're using the same training data path as in optimized_naive_bayes.py
        sys.path.append('utils')
        from preprocess import preprocess
        
        DATASET_PATH = 'data/cleanedWithScript/manual_cleaned_data_universal.csv'
        
        # Load data in full mode to retrieve bag-of-words
        df = preprocess(DATASET_PATH, normalize_and_onehot=False, mode="full")
        
        # Identify label columns (assumed to be the one-hot encoded ones)
        label_cols = [col for col in df.columns if col.startswith("Label")]
        
        # Extract features by dropping "id" and label cols
        X = df.drop(["id"] + label_cols, axis=1)
        
        # Convert one-hot labels to integer indices
        y = np.argmax(df[label_cols].values, axis=1)
        
        print(f"Creating feature selector with k={k} best features")
        selector = SelectKBest(chi2, k=k)
        selector.fit(X, y)
        
        # Get the selected feature names
        feature_mask = selector.get_support()
        selected_features = X.columns[feature_mask].tolist()
        
        print(f"Selected {len(selected_features)} features out of {X.shape[1]}")
        
        # Save the selector
        os.makedirs('model_params', exist_ok=True)
        with open('model_params/feature_selector.pkl', 'wb') as f:
            pickle.dump(selector, f)
        print("Feature selector saved to model_params/feature_selector.pkl")
        
        # Also save just the feature names for simplicity
        with open('model_params/selected_features.pkl', 'wb') as f:
            pickle.dump(selected_features, f)
        print("Selected feature names saved to model_params/selected_features.pkl")
        
        # Create a modified version of the optimized model that can work with selected features
        model = load_optimized_model()
        if hasattr(model, 'predict'):
            print("Creating a model package that combines the selector and model")
            
            # Create a new full workflow
            model_package = {
                'selector': selector,
                'selected_features': selected_features,
                'model': model,
                'label_mapping': {i: label for i, label in enumerate(df.filter(regex='^Label').columns)},
                'type': 'feature_selection_with_model'
            }
            
            # Save the combined model
            with open('model_params/integrated_model.pkl', 'wb') as f:
                pickle.dump(model_package, f)
            print("Integrated model saved to model_params/integrated_model.pkl")
            
            # Now modify pred.py to load this model or add text instructions
            print("\nIMPORTANT: To use this model, you need to:")
            print("1. Either modify pred.py to load 'integrated_model.pkl' instead of 'best_naive_bayes_model.pkl'")
            print("2. OR run: cp model_params/integrated_model.pkl model_params/best_naive_bayes_model.pkl")
            
        return True
    except Exception as e:
        print(f"Error creating feature selector: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    model = load_optimized_model()
    
    # Print model structure for debugging
    if model is not None:
        print("\nModel Information:")
        if isinstance(model, dict):
            print(f"Model is a dictionary with keys: {list(model.keys())}")
        else:
            print(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
    
    # Create and save feature selector
    success = create_feature_selector(k=150)
    
    if success:
        print("\nFeature selector successfully created.")
        print("You can now use pred.py with this feature selector.")
    else:
        print("\nFailed to create feature selector.")
