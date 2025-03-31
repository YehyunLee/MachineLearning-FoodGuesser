import pandas as pd
import numpy as np
from collections import Counter


def preprocess(file_path, normalize_and_onehot=False, mode="full", df_in=None, drop_na=True):
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
    
    # Record initial row count before dropping missing values
    initial_rows = len(df)
    
    # Replace N/A with a special marker instead of dropping
    df.replace("N/A", "MISSING_VALUE", inplace=True)
    
    # Only drop rows if specifically requested
    if drop_na:
        df.replace("MISSING_VALUE", pd.NA, inplace=True)
        df.dropna(inplace=True)
        dropped_rows = initial_rows - len(df)
        print(f"Dropped {dropped_rows} rows out of {initial_rows} due to missing values.")
    
    # Bag-of-Words for Q5 and Q6 - handle missing values by treating them as empty strings
    def bag_of_words(column, max_features):
        # Replace MISSING_VALUE with empty string
        processed_column = column.replace("MISSING_VALUE", "")
        all_words = ' '.join(processed_column).split()
        most_common = Counter(all_words).most_common(max_features)
        vocab = [word for word, _ in most_common]
        bow = []
        for text in processed_column:
            word_count = Counter(text.split())
            bow.append([word_count.get(word, 0) for word in vocab])
        return pd.DataFrame(bow, columns=vocab if vocab else ['no_features'])

    bow_q5 = bag_of_words(df["Q5 Cleaned"], max_features=100)
    bow_q6 = bag_of_words(df["Q6 Cleaned"], max_features=50)
    print(f"Shape of bow_q5: {bow_q5.shape}")
    print(f"Shape of bow_q6: {bow_q6.shape}")

    if mode == "full":
        # Process numerical features - replace missing with median values
        numerical_cols = ["Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
                          "Q2 Cleaned", "Q4 Cleaned"]
        
        # Convert to numeric, coercing errors to NaN
        for col in numerical_cols:
            df[col] = pd.to_numeric(df[col].replace("MISSING_VALUE", np.nan), errors='coerce')
        
        # Fill missing values with median or 0 if all are missing
        for col in numerical_cols:
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0
            df[col].fillna(median_val, inplace=True)
            
        # No need to drop rows with missing numerical values anymore
        
        def min_max_scale(column):
            min_val, max_val = column.min(), column.max()
            if min_val == max_val:  # Avoid division by zero
                return pd.Series(0.5, index=column.index)
            return (column - min_val) / (max_val - min_val)

        normalized_numerical = pd.DataFrame({col: min_max_scale(df[col]) for col in numerical_cols})
        print(f"Shape of normalized numerical features: {normalized_numerical.shape}")

        # One-hot encode all categorical features (including Label)
        categorical_cols = ["Q3: In what setting would you expect this food to be served? Please check all that apply",
                            "Q7: When you think about this food item, who does it remind you of?",
                            "Q8: How much hot sauce would you add to this food item?", "Label"]

        def one_hot_encode(column):
            # Replace MISSING_VALUE with a default value
            processed_column = column.replace("MISSING_VALUE", "unknown")
            unique_values = processed_column.unique()
            one_hot = np.zeros((len(processed_column), len(unique_values)), dtype=int)
            for i, value in enumerate(processed_column):
                one_hot[i, np.where(unique_values == value)[0][0]] = 1
            return pd.DataFrame(one_hot, columns=[f"{column.name}_{val}" for val in unique_values])

        encoded_categorical = pd.concat([one_hot_encode(df[col]) for col in categorical_cols], axis=1)
        print(f"Shape of encoded categorical features: {encoded_categorical.shape}")
        
        final_df = pd.concat([df["id"], normalized_numerical, bow_q5, bow_q6, encoded_categorical], axis=1)
    elif mode == "softmax":
        # One-hot encode only the Label column
        def one_hot_encode(column):
            processed_column = column.replace("MISSING_VALUE", "unknown")
            unique_values = processed_column.unique()
            one_hot = np.zeros((len(processed_column), len(unique_values)), dtype=int)
            for i, value in enumerate(processed_column):
                one_hot[i, np.where(unique_values == value)[0][0]] = 1
            return pd.DataFrame(one_hot, columns=[f"{column.name}_{val}" for val in unique_values])

        encoded_label = one_hot_encode(df["Label"])
        final_df = pd.concat([df["id"], bow_q5, bow_q6, encoded_label], axis=1)
    else:
        raise ValueError("Unsupported mode. Use mode='full' or mode='softmax'.")
    
    # DON'T drop NA values at the end
    # final_df.dropna(inplace=True)
    
    # Replace any remaining NaNs with zeros
    final_df.fillna(0, inplace=True)

    print(f"Preprocessed data shape: {final_df.shape}")
    print(f"Preprocessed data columns: {final_df.columns}")
    return final_df