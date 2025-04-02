import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


# def process_intervals(x: str):
#     if x.isdigit():
#         return x
#     interval = x.split('-')
#     return interval[0]


# def min_max_scaling(series: pd.Series):
#     return (series - series.min()) / (series.max() - series.min())
    

def preprocess(file_path, normalize_and_onehot=False, mode="full", df_in=None, drop_na=False):
    # If a DataFrame is provided, use it; otherwise read from file_path.
    if df_in is not None:
        df = df_in.copy()
    else:
        df = pd.read_csv(file_path, dtype=str)

    # Rename columns
    df.rename(columns={
        "Q2: How many ingredients would you expect this food item to contain?": "Q2 Cleaned",
        "Q4: How much would you expect to pay for one serving of this food item?": "Q4 Cleaned",
        "Q5: What movie do you think of when thinking of this food item?": "Q5 Cleaned",
        "Q6: What drink would you pair with this food item?": "Q6 Cleaned"
    }, inplace=True)
    
    # Convert all columns to string
    df = df.astype(str)
    
    # Record initial row count before handling missing values
    initial_rows = len(df)
    
    # Replace N/A with a special marker instead of dropping
    df.replace("N/A", "MISSING_VALUE", inplace=True)
    
    # Only drop rows if specifically requested
    if drop_na:
        df.replace("MISSING_VALUE", pd.NA, inplace=True)
        df.dropna(inplace=True)
        dropped_rows = initial_rows - len(df)
        print(f"Dropped {dropped_rows} rows out of {initial_rows} due to missing values.")
    
    # Bag-of-Words for Q5 and Q6 - handle missing values
    vectorizer_q5 = CountVectorizer(max_features=100)  # Limit to top 100 features
    vectorizer_q6 = CountVectorizer(max_features=50)   # Limit to top 50 features

    # Replace MISSING_VALUE with empty strings for vectorization
    q5_texts = df["Q5 Cleaned"].replace("MISSING_VALUE", "")
    q6_texts = df["Q6 Cleaned"].replace("MISSING_VALUE", "")
    
    # Handle empty vocabulary case
    try:
        bow_q5 = pd.DataFrame(vectorizer_q5.fit_transform(q5_texts).toarray(), 
                            columns=vectorizer_q5.get_feature_names_out())
    except ValueError:  # Empty vocabulary
        bow_q5 = pd.DataFrame(0, index=df.index, columns=['no_features_q5'])
    
    try:
        bow_q6 = pd.DataFrame(vectorizer_q6.fit_transform(q6_texts).toarray(), 
                            columns=vectorizer_q6.get_feature_names_out())
    except ValueError:  # Empty vocabulary
        bow_q6 = pd.DataFrame(0, index=df.index, columns=['no_features_q6'])
    
    print(f"Shape of bow_q5: {bow_q5.shape}")
    print(f"Shape of bow_q6: {bow_q6.shape}")

    if mode == "full":
        # Process numerical features
        numerical_cols = ["Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
                          "Q2 Cleaned", "Q4 Cleaned"]
        
        # Convert to numeric, coercing errors to NaN
        for col in numerical_cols:
            df[col] = pd.to_numeric(df[col].replace("MISSING_VALUE", pd.NA), errors='coerce')
        
        # Fill missing values with median or 0 if all are missing
        for col in numerical_cols:
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0
            df[col].fillna(median_val, inplace=True)

        # Scale features (no need to drop NA rows first)
        scaler = MinMaxScaler()
        # Handle case where min==max (all values identical)
        if df[numerical_cols].nunique().min() <= 1:
            normalized_numerical = pd.DataFrame(0.5, index=df.index, columns=numerical_cols)
        else:
            normalized_numerical = pd.DataFrame(scaler.fit_transform(df[numerical_cols]), columns=numerical_cols)
        print(f"Shape of normalized numerical features: {normalized_numerical.shape}")

        # One-hot encode all categorical features (including Label)
        categorical_cols = ["Q3: In what setting would you expect this food to be served? Please check all that apply",
                            "Q7: When you think about this food item, who does it remind you of?",
                            "Q8: How much hot sauce would you add to this food item?", "Label"]
        
        # Process categorical columns - replace missing values with 'unknown'
        for col in categorical_cols:
            df[col] = df[col].replace("MISSING_VALUE", "unknown")
        
        # Use OneHotEncoder with handle_unknown='ignore'
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_categorical = pd.DataFrame(encoder.fit_transform(df[categorical_cols]),
                                       columns=encoder.get_feature_names_out(categorical_cols))
        print(f"Shape of encoded categorical features: {encoded_categorical.shape}")
        
        final_df = pd.concat([df["id"], normalized_numerical, bow_q5, bow_q6, encoded_categorical], axis=1)
    elif mode == "softmax":
        # One-hot encode only the Label column
        df["Label"] = df["Label"].replace("MISSING_VALUE", "unknown")
        
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_label = pd.DataFrame(encoder.fit_transform(df[["Label"]]),
                                  columns=encoder.get_feature_names_out(["Label"]))
        final_df = pd.concat([df["id"], bow_q5, bow_q6, encoded_label], axis=1)
    else:
        raise ValueError("Unsupported mode. Use mode='full' or mode='softmax'.")
    
    # Replace any remaining NaNs with zeros instead of dropping rows
    final_df.fillna(0, inplace=True)

    # shuffle the rows so that the order of the data is not biased
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)  # drop here is for resetting the index

    print(f"Preprocessed data shape: {final_df.shape}")
    print(f"Preprocessed data columns: {final_df.columns}")
    return final_df
