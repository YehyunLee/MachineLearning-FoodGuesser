"""
Data Analysis Script for Food Predictor Project
This script analyzes the distribution of features in the food dataset
after cleaning and preprocessing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from collections import Counter

# Add project paths
sys.path.append('universalDataCleaning')
sys.path.append('utils')

# Import preprocessing functions
from preprocessForAllModels import preprocess as clean_data
from preprocess import preprocess as feature_extraction

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 6)})

# Define paths
DATA_PATH = 'data/cleanedWithScript/manual_cleaned_data_universal.csv'
OUTPUT_DIR = 'analysis_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define question mapping for shorter titles
QUESTION_MAPPING = {
    "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)": "Q1: Complexity",
    "Q2 Cleaned": "Q2: Ingredients",
    "Q3: In what setting would you expect this food to be served? Please check all that apply": "Q3: Setting",
    "Q4 Cleaned": "Q4: Price",
    "Q5 Cleaned": "Q5: Movie",
    "Q6 Cleaned": "Q6: Drink",
    "Q7: When you think about this food item, who does it remind you of?": "Q7: Association",
    "Q8: How much hot sauce would you add to this food item?": "Q8: Hot Sauce",
    "Label": "Food Type"
}

def load_and_process_data():
    """Load, clean, and preprocess the data."""
    print("Loading and preprocessing data...")
    
    # Load and clean the data
    cleaned_df = clean_data(DATA_PATH, return_df=True)
    
    # Process the data
    processed_df = feature_extraction(None, normalize_and_onehot=False, mode="full", df_in=cleaned_df)
    
    return cleaned_df, processed_df

def analyze_data_schema(df):
    """Analyze and print the schema of the dataframe."""
    print("\n===== DATA SCHEMA =====")
    
    # Get data types and non-null counts
    schema_info = pd.DataFrame({
        'Data Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    
    # Add descriptions based on column names
    descriptions = {
        'id': 'Unique identifier for each response',
        'Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)': 
            'Complexity rating (1-5)',
        'Q2 Cleaned': 'Estimated number of ingredients',
        'Q3: In what setting would you expect this food to be served? Please check all that apply':
            'Setting where food is served (categorical, multiple selection)',
        'Q4 Cleaned': 'Expected price for one serving',
        'Q5 Cleaned': 'Movie associated with the food (free text)',
        'Q6 Cleaned': 'Drink paired with the food (free text)',
        'Q7: When you think about this food item, who does it remind you of?':
            'Person association (categorical, multiple selection)',
        'Q8: How much hot sauce would you add to this food item?':
            'Hot sauce preference (categorical)',
        'Label': 'Food type (Pizza, Shawarma, Sushi)'
    }
    schema_info['Description'] = [descriptions.get(col, '') for col in df.columns]
    
    print(schema_info)
    return schema_info

def analyze_categorical_distribution(df, column):
    """Analyze the distribution of a categorical column."""
    if column not in df.columns:
        print(f"Column {column} not found in dataframe")
        return
        
    # Count values
    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = [column, 'Count']
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=column, y='Count', data=value_counts)
    
    # Use short title
    short_title = QUESTION_MAPPING.get(column, column)
    plt.title(f"Distribution of {short_title}")
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Use simple filename
    simple_name = short_title.split(':')[0].strip().lower()
    plt.savefig(f"{OUTPUT_DIR}/{simple_name}_distribution.png")
    plt.close()
    
    return value_counts

def analyze_numerical_distribution(df, column):
    """Analyze the distribution of a numerical column."""
    if column not in df.columns:
        print(f"Column {column} not found in dataframe")
        return
        
    # Convert to numeric
    df[column] = pd.to_numeric(df[column], errors='coerce')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Get short title
    short_title = QUESTION_MAPPING.get(column, column)
    
    # Histogram
    sns.histplot(df[column].dropna(), kde=True, ax=ax1)
    ax1.set_title(f"Histogram of {short_title}")
    
    # Box plot
    sns.boxplot(y=df[column].dropna(), ax=ax2)
    ax2.set_title(f"Box Plot of {short_title}")
    
    plt.tight_layout()
    
    # Use simple filename
    simple_name = short_title.split(':')[0].strip().lower()
    plt.savefig(f"{OUTPUT_DIR}/{simple_name}_distribution.png")
    plt.close()
    
    # Calculate statistics
    stats = df[column].describe()
    
    return stats

def analyze_label_correlations(df):
    """Analyze how features correlate with the target label."""
    print("\n===== LABEL CORRELATIONS =====")
    
    # Group numerical features by label
    numerical_cols = [
        "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
        "Q2 Cleaned", 
        "Q4 Cleaned"
    ]
    
    # Convert to numeric
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Group by label and get mean
    label_means = df.groupby('Label')[numerical_cols].mean()
    
    # Create bar plots for numerical features by label
    plt.figure(figsize=(12, 8))
    label_means.plot(kind='bar')
    plt.title("Average Numerical Features by Food Type")
    plt.ylabel("Average Value")
    
    # Rename x-axis labels to shorter versions
    plt.gca().set_xticklabels(plt.gca().get_xticklabels(), rotation=0)
    
    # Rename the legend labels
    handles, labels = plt.gca().get_legend_handles_labels()
    short_labels = [QUESTION_MAPPING.get(label, label) for label in labels]
    plt.legend(handles, short_labels)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/numerical_by_label.png")
    plt.close()
    
    # Analyze categorical distributions by label
    categorical_cols = [
        "Q3: In what setting would you expect this food to be served? Please check all that apply",
        "Q7: When you think about this food item, who does it remind you of?",
        "Q8: How much hot sauce would you add to this food item?"
    ]
    
    for col in categorical_cols:
        # Create a larger figure to accommodate labels
        plt.figure(figsize=(18, 12))
        
        short_col = QUESTION_MAPPING.get(col, col)
        simple_name = short_col.split(':')[0].strip().lower()
        
        # Special handling for multi-select categorical columns (Q3 and Q7)
        if col in ["Q3: In what setting would you expect this food to be served? Please check all that apply",
                   "Q7: When you think about this food item, who does it remind you of?"]:
            
            # Define the main categories for each question
            if col == "Q3: In what setting would you expect this food to be served? Please check all that apply":
                main_categories = [
                    "Week day lunch", "Week day dinner", "Weekend lunch", 
                    "Weekend dinner", "At a party", "Late night snack"
                ]
            else:  # Q7
                main_categories = [
                    "Friends", "Teachers", "Strangers", "Parents", "Siblings"
                ]
            
            # Create a new DataFrame to hold binary indicators for each category
            binary_df = pd.DataFrame(index=df.index)
            for category in main_categories:
                # If the category appears in the response string, mark it as 1
                binary_df[category] = df[col].str.contains(category).astype(int)
            
            # Group by label and calculate proportion for each category
            grouped = binary_df.groupby(df['Label']).mean()
            
            # Create a grouped bar plot instead of stacked
            ax = grouped.plot(kind='bar', figsize=(18, 10))
            plt.title(f"{short_col} by Food Type", fontsize=16)
            plt.ylabel("Proportion", fontsize=14)
            plt.xticks(rotation=0, fontsize=14)
            
            # Format legend
            plt.legend(title=None, fontsize=12, loc='upper right')
            
        else:
            # Regular processing for other columns (like Q8)
            # Create a cross-tabulation
            cross_tab = pd.crosstab(df['Label'], df[col])
            cross_tab_norm = cross_tab.div(cross_tab.sum(axis=1), axis=0)
            
            # Plot normalized stacked bar chart
            ax = cross_tab_norm.plot(kind='bar', stacked=True)
            plt.title(f"{short_col} by Food Type", fontsize=16)
            plt.ylabel("Proportion", fontsize=14)
            plt.xticks(rotation=0, fontsize=14)
            
            # For other questions, just place the legend outside
            plt.legend(title=short_col, bbox_to_anchor=(1.05, 1), 
                      loc='upper left', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{simple_name}_by_label.png", bbox_inches='tight')
        plt.close()
    
    # Analyze text features (Q5, Q6) by label
    text_cols = ["Q5 Cleaned", "Q6 Cleaned"]
    
    for col in text_cols:
        plt.figure(figsize=(15, 12))
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        short_col = QUESTION_MAPPING.get(col, col)
        simple_name = short_col.split(':')[0].strip().lower()
        
        fig.suptitle(f"Most Common Words in {short_col} by Food Type")
        
        for i, label in enumerate(['Pizza', 'Shawarma', 'Sushi']):
            # Get texts for this label
            texts = df[df['Label'] == label][col].str.lower().dropna()
            
            # Create word count
            all_words = ' '.join(texts).split()
            word_counts = Counter(all_words)
            
            # Get top 15 words
            top_words = pd.DataFrame(word_counts.most_common(15), columns=['Word', 'Count'])
            
            # Plot
            sns.barplot(x='Count', y='Word', data=top_words, ax=axes[i])
            axes[i].set_title(f"{label}")
            
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{simple_name}_by_label.png")
        plt.close()
    
    return label_means

def analyze_processed_data(processed_df):
    """Analyze the processed dataframe after feature extraction."""
    print("\n===== PROCESSED DATA ANALYSIS =====")
    
    # Get a list of all columns
    all_columns = processed_df.columns.tolist()
    
    # Identify feature groups more accurately
    id_cols = ['id']
    numerical_features = [col for col in all_columns if col.startswith('Q1:') or col.startswith('Q2 ') or col.startswith('Q4 ')]
    
    # For bag-of-words features, use a more direct approach
    label_features = [col for col in all_columns if col.startswith('Label_')]
    categorical_features = [col for col in all_columns if col.startswith('Q3:') or col.startswith('Q7:') or col.startswith('Q8:')]
    
    # All remaining features are likely from bag-of-words
    remaining_cols = set(all_columns) - set(id_cols) - set(numerical_features) - set(label_features) - set(categorical_features)
    
    # Heuristic: Among remaining columns, identify Q5 vs Q6 bag of words
    # Q5 typically has more features (100 vs 50) and different word patterns
    # For now, we'll just split them in half and print details for further debugging
    bow_features = list(remaining_cols)
    
    # Print feature examples for debugging
    print("\nExample features in remaining columns (potential bag-of-words):")
    for i, col in enumerate(sorted(bow_features)[:10]):
        print(f"{i+1}. {col}")
    
    # Try to estimate Q5 vs Q6 features using word patterns
    # Words like "movie", "film", etc. typically appear in Q5, while drink-related words appear in Q6
    q5_indicators = ['movie', 'film', 'watch', 'scene', 'actor', 'show', 'series']
    q6_indicators = ['drink', 'water', 'soda', 'coffee', 'tea', 'beer', 'juice', 'wine', 'milk', 'coke']
    
    bow_q5_features = []
    bow_q6_features = []
    
    for col in bow_features:
        # Check for indicators in column name
        is_q5 = any(indicator in col.lower() for indicator in q5_indicators)
        is_q6 = any(indicator in col.lower() for indicator in q6_indicators)
        
        if is_q5:
            bow_q5_features.append(col)
        elif is_q6:
            bow_q6_features.append(col)
        else:
            # If we can't determine based on word pattern, default to Q5 
            # (since it has more features in your preprocessing)
            bow_q5_features.append(col)
    
    # If our heuristic failed, just split by count (Q5 should have about 2x as many features as Q6)
    if len(bow_q5_features) == 0 or len(bow_q6_features) == 0:
        total_bow = len(bow_features)
        split_index = int(total_bow * 2/3)  # Assuming Q5 has about 2/3 of the features
        bow_q5_features = bow_features[:split_index]
        bow_q6_features = bow_features[split_index:]
    
    print(f"\nBOW Q5 count: {len(bow_q5_features)}")
    print(f"BOW Q6 count: {len(bow_q6_features)}")
    
    # Create summary table
    feature_summary = pd.DataFrame({
        'Feature Type': ['Bag-of-Words (Q5)', 'Bag-of-Words (Q6)', 'One-Hot Categorical', 'One-Hot Labels', 'Numerical'],
        'Count': [len(bow_q5_features), len(bow_q6_features), len(categorical_features), len(label_features), len(numerical_features)]
    })
    
    print(feature_summary)
    
    # Create pie chart of feature types
    plt.figure(figsize=(10, 8))
    plt.pie(feature_summary['Count'], labels=feature_summary['Feature Type'], autopct='%1.1f%%')
    plt.title('Feature Distribution by Type')
    plt.savefig(f"{OUTPUT_DIR}/features_distribution.png")
    plt.close()
    
    return feature_summary

def main():
    """Main function to run all analyses."""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load raw and processed data
    cleaned_df, processed_df = load_and_process_data()
    
    # Analyze data schema
    analyze_data_schema(cleaned_df)
    
    # Analyze distributions
    print("\n===== ANALYZING DISTRIBUTIONS =====")
    
    # Label distribution
    analyze_categorical_distribution(cleaned_df, 'Label')
    
    # Numerical features
    numerical_cols = [
        "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
        "Q2 Cleaned", 
        "Q4 Cleaned"
    ]
    for col in numerical_cols:
        stats = analyze_numerical_distribution(cleaned_df, col)
        print(f"\nStatistics for {QUESTION_MAPPING.get(col, col)}:\n{stats}")
    
    # Categorical features
    categorical_cols = [
        "Q3: In what setting would you expect this food to be served? Please check all that apply",
        "Q7: When you think about this food item, who does it remind you of?",
        "Q8: How much hot sauce would you add to this food item?"
    ]
    for col in categorical_cols:
        counts = analyze_categorical_distribution(cleaned_df, col)
        print(f"\nValue counts for {QUESTION_MAPPING.get(col, col)}:\n{counts}")
    
    # Text features
    text_cols = ["Q5 Cleaned", "Q6 Cleaned"]
    for col in text_cols:
        # Word cloud for text features
        plt.figure(figsize=(12, 8))
        
        short_col = QUESTION_MAPPING.get(col, col)
        simple_name = short_col.split(':')[0].strip().lower()
        
        # Count word frequencies
        all_texts = ' '.join(cleaned_df[col].dropna()).lower()
        word_counts = Counter(all_texts.split())
        
        # Get top 20 words
        top_words = pd.DataFrame(word_counts.most_common(20), columns=['Word', 'Count'])
        
        # Create bar chart
        sns.barplot(x='Count', y='Word', data=top_words)
        plt.title(f"Top 20 Words in {short_col}")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{simple_name}_top_words.png")
        plt.close()
    
    # Analyze label correlations
    analyze_label_correlations(cleaned_df)
    
    # Analyze processed data
    analyze_processed_data(processed_df)
    
    print(f"\nAnalysis complete. Results saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
