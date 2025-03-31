import pandas as pd
import numpy as np
from collections import Counter


def preprocess(file_path, normalize_and_onehot=False, mode="full"):
    # Load data
    df = pd.read_csv(file_path, dtype=str)
    
    # Record initial row count before dropping missing values
    initial_rows = len(df)
    
    # Drop rows with invalid values (N/A)
    df.replace("N/A", pd.NA, inplace=True)
    df.dropna(inplace=True)
    dropped_rows = initial_rows - len(df)
    print(f"Dropped {dropped_rows} rows out of {initial_rows} due to missing values.")
    
    # Bag-of-Words for Q5 and Q6
    def bag_of_words(column, max_features):
        all_words = ' '.join(column).split()
        most_common = Counter(all_words).most_common(max_features)
        vocab = [word for word, _ in most_common]
        bow = []
        for text in column:
            word_count = Counter(text.split())
            bow.append([word_count.get(word, 0) for word in vocab])
        return pd.DataFrame(bow, columns=vocab)

    bow_q5 = bag_of_words(df["Q5 Cleaned"], max_features=100)
    bow_q6 = bag_of_words(df["Q6 Cleaned"], max_features=50)
    print(f"Shape of bow_q5: {bow_q5.shape}")
    print(f"Shape of bow_q6: {bow_q6.shape}")

    if mode == "full":
        # Process numerical features
        numerical_cols = ["Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
                          "Q2 Cleaned", "Q4 Cleaned"]
        for col in numerical_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=numerical_cols, inplace=True)

        def min_max_scale(column):
            min_val, max_val = column.min(), column.max()
            return (column - min_val) / (max_val - min_val)

        normalized_numerical = pd.DataFrame({col: min_max_scale(df[col]) for col in numerical_cols})
        print(f"Shape of normalized numerical features: {normalized_numerical.shape}")

        # One-hot encode all categorical features (including Label)
        categorical_cols = ["Q3: In what setting would you expect this food to be served? Please check all that apply",
                            "Q7: When you think about this food item, who does it remind you of?",
                            "Q8: How much hot sauce would you add to this food item?", "Label"]

        def one_hot_encode(column):
            unique_values = column.unique()
            one_hot = np.zeros((len(column), len(unique_values)), dtype=int)
            for i, value in enumerate(column):
                one_hot[i, np.where(unique_values == value)[0][0]] = 1
            return pd.DataFrame(one_hot, columns=[f"{column.name}_{val}" for val in unique_values])

        encoded_categorical = pd.concat([one_hot_encode(df[col]) for col in categorical_cols], axis=1)
        print(f"Shape of encoded categorical features: {encoded_categorical.shape}")
        
        final_df = pd.concat([df["id"], normalized_numerical, bow_q5, bow_q6, encoded_categorical], axis=1)
    elif mode == "softmax":
        # One-hot encode only the Label column
        def one_hot_encode(column):
            unique_values = column.unique()
            one_hot = np.zeros((len(column), len(unique_values)), dtype=int)
            for i, value in enumerate(column):
                one_hot[i, np.where(unique_values == value)[0][0]] = 1
            return pd.DataFrame(one_hot, columns=[f"{column.name}_{val}" for val in unique_values])

        encoded_label = one_hot_encode(df["Label"])
        final_df = pd.concat([df["id"], bow_q5, bow_q6, encoded_label], axis=1)
    else:
        raise ValueError("Unsupported mode. Use mode='full' or mode='softmax'.")
        
    final_df.dropna(inplace=True)  # Ensure no NaN values remain

    print(f"Preprocessed data shape: {final_df.shape}")
    print(f"Preprocessed data columns: {final_df.columns}")
    return final_df