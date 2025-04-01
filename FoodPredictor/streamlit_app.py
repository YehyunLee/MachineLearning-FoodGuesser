"""
Streamlit app for the Food Predictor model
Allows users to input data directly through a UI and get predictions
without needing to create/upload CSV files.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import pickle

# Configure paths to access prediction modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'universalDataCleaning'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Import necessary functions from your existing code
from preprocessForAllModels import preprocess as clean_data
from builtin_preprocess import preprocess as feature_extraction
from pred import naive_bayes_predict, load_model_params

# Set page configuration
st.set_page_config(
    page_title="AI Food Guesser",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS to improve the app's appearance
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stButton > button {
        width: 100%;
    }
    .title {
        text-align: center;
        color: #FF5733;
    }
    .prediction {
        font-size: 2rem;
        text-align: center;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .pizza {
        background-color: #FFCCCB;
    }
    .shawarma {
        background-color: #FFFFCC;
    }
    .sushi {
        background-color: #CCFFCC;
    }
    </style>
    """, unsafe_allow_html=True)

def generate_unique_id():
    """Generate a random unique ID for the input data."""
    return np.random.randint(100000, 999999)

def main():
    """Main function to create the Streamlit app."""
    
    # Title and intro
    st.markdown("<h1 class='title'>🍽️ AI Food Guesser 🍽️</h1>", unsafe_allow_html=True)
    st.markdown("""
    Tell us about your food preferences, and we'll guess whether it's Pizza, Shawarma, or Sushi!
    """)
    
    # Create a form for user input
    with st.form("food_prediction_form"):
        st.subheader("Food Information")
        
        # Q1: Complexity
        q1 = st.slider("From a scale 1 to 5, how complex is it to make this food?", 
                      min_value=1, max_value=5, value=3,
                      help="1 is the most simple, and 5 is the most complex")
        
        # Q2: Number of ingredients
        q2 = st.number_input("How many ingredients would you expect this food item to contain?", 
                            min_value=1, max_value=100, value=10)
        
        # Q3: Setting
        q3_options = ["Week day lunch", "Week day dinner", "Weekend lunch", 
                     "Weekend dinner", "At a party", "Late night snack"]
        q3 = st.multiselect("In what setting would you expect this food to be served?", 
                          options=q3_options,
                          default=["Week day dinner"])
        
        # Q4: Price
        q4 = st.slider("How much would you expect to pay for one serving of this food item?", 
                      min_value=1.0, max_value=50.0, value=10.0, step=0.5)
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        # Q5: Movie
        with col1:
            q5 = st.text_input("What movie do you think of when thinking of this food item?", 
                              value="")
        
        # Q6: Drink
        with col2:
            q6 = st.text_input("What drink would you pair with this food item?", 
                              value="")
        
        # Q7: Who it reminds of
        q7_options = ["Friends", "Teachers", "Strangers", "Parents", "Siblings"]
        q7 = st.multiselect("When you think about this food item, who does it remind you of?", 
                          options=q7_options,
                          default=["Friends"])
        
        # Q8: Hot sauce
        q8_options = ["A lot (hot)", "I will have some of this food item with my hot sauce",
                     "A moderate amount (medium)", "A little (mild)", "None"]
        q8 = st.selectbox("How much hot sauce would you add to this food item?", 
                         options=q8_options,
                         index=3)
        
        # Submit button
        submit = st.form_submit_button("Guess Food Type!")
    
    # Process the prediction when the form is submitted
    if submit:
        with st.spinner('Making prediction...'):
            # Convert the input to a DataFrame in the format expected by the model
            input_data = pd.DataFrame({
                "id": [generate_unique_id()],
                "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)": [str(q1)],
                "Q2: How many ingredients would you expect this food item to contain?": [str(q2)],
                "Q3: In what setting would you expect this food to be served? Please check all that apply": [','.join(q3)],
                "Q4: How much would you expect to pay for one serving of this food item?": [str(q4)],
                "Q5: What movie do you think of when thinking of this food item?": [q5],
                "Q6: What drink would you pair with this food item?": [q6],
                "Q7: When you think about this food item, who does it remind you of?": [','.join(q7)],
                "Q8: How much hot sauce would you add to this food item?": [q8],
                "Label": ["unknown"]  # Placeholder label, will be ignored
            })
            
            # Save the input data to a temporary file for preprocessing
            temp_file = "temp_input.csv"
            input_data.to_csv(temp_file, index=False)
            
            # Apply the preprocessing pipeline
            # Clean the data using preprocessForAllModels
            cleaned_df = clean_data(temp_file, return_df=True)
            
            # Feature extraction using builtin_preprocess
            preprocessed_df = feature_extraction(None, normalize_and_onehot=False, 
                                                mode="full", df_in=cleaned_df, drop_na=False)
            
            # Clean up the temporary file
            try:
                os.remove(temp_file)
            except:
                pass
                
            # Extract features and make prediction
            feature_cols = [col for col in preprocessed_df.columns 
                            if not col.startswith('Label_') and col != 'id']
            features = preprocessed_df[feature_cols]
            prediction = naive_bayes_predict(features)[0]
            
            # Display the prediction with some style
            st.success("Prediction complete!")
            css_class = prediction.lower()
            st.markdown(f"""
                <div class='prediction {css_class}' style='color: black;'>
                    Our model guesses your food is: <strong>{prediction}</strong>
                </div>
            """, unsafe_allow_html=True)
            
            # Display food image based on prediction
            if prediction == "Pizza":
                st.image("https://images.unsplash.com/photo-1565299624946-b28f40a0ae38", 
                         caption="Delicious Pizza")
            elif prediction == "Shawarma":
                st.image("https://images.unsplash.com/photo-1529006557810-274b9b2fc783?q=80&w=2952&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
                         caption="Tasty Shawarma")
            elif prediction == "Sushi":
                st.image("https://images.unsplash.com/photo-1617196035154-1e7e6e28b0db", 
                         caption="Fresh Sushi")
# Run the app
if __name__ == "__main__":
    main()
