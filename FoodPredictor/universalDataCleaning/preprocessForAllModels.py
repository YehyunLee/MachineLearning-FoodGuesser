import pandas as pd
import re

# Mapping for correcting Excel date conversion issue
MONTH_MAP = {
    'Jan': '1', 'Feb': '2', 'Mar': '3', 'Apr': '4', 'May': '5', 'Jun': '6',
    'Jul': '7', 'Aug': '8', 'Sep': '9', 'Oct': '10', 'Nov': '11', 'Dec': '12'
}

# Function to fix Excel date conversion issue in Q2
def convert_date_format(value):
    if isinstance(value, str) and '-' in value:
        parts = value.split('-')
        if len(parts) == 2 and parts[1] in MONTH_MAP:
            return f"{MONTH_MAP[parts[1]]}-{parts[0]}"
    return value

# Function to clean Q2 (number of ingredients)
def clean_ingredient_count(value):
    if isinstance(value, str):
        value = value.lower().strip()
        value = re.sub(r'[^0-9.\-]', '', value)  # Keep only digits, dots, and dashes

        if '-' in value:  # Handle range format like "3.5-5"
            parts = value.split('-')
            if len(parts) == 2 and all(p.replace('.', '', 1).isdigit() for p in parts):  # Ensure both parts are valid numbers
                p1, p2 = float(parts[0]), float(parts[1])
                mean_val = (p1 + p2) / 2
                return f"{mean_val:.1f}"
        elif value.replace('.', '', 1).isdigit():  # Allow single float or integer values
            return f"{float(value):.1f}"  # Convert "5.00" → "5.0"

    return "N/A"

# Function to clean price (Q4)
def clean_price(value):
    if isinstance(value, str):
        value = re.sub(r'[^0-9.\-]', '', value)  # modified: allow dash for range check
        if '-' in value:  # modified to handle range format like "3.00-5.50"
            parts = value.split('-')
            if len(parts) == 2 and all(p.replace('.', '', 1).isdigit() for p in parts):
                p1, p2 = float(parts[0]), float(parts[1])
                mean_val = (p1 + p2) / 2
                return f"{mean_val:.1f}"
        elif value.replace('.', '', 1).isdigit():
            return f"{float(value):.1f}"
    return "N/A"

# Function to clean movie titles (Q5)
def clean_movie_title(value):
    if isinstance(value, str):
        value = re.sub(r'[^a-zA-Z0-9 ]', '', value)  # Remove special characters
        value = re.sub(r'\b\d+\b', '', value)  # Remove standalone numbers (except if the entire title is a number)
        value = re.sub(r'\s+', ' ', value).strip().lower()  # Remove extra spaces
        return value if value else "N/A"
    return "N/A"

# Function to clean drink names (Q6)
def clean_drink(value):
    if isinstance(value, str):
        value = re.sub(r'[^a-zA-Z ]', '', value).strip().lower()  # Remove numbers and special characters
        return value if value else "N/A"
    return "N/A"

# Function to clean multi-option categorical columns (Q3, Q7)
def clean_multi_option(value, valid_options):
    if isinstance(value, str):
        options = [opt.strip() for opt in value.split(',')]
        valid = [opt for opt in options if opt in valid_options]
        return ",".join(valid) if valid else "N/A"
    return "N/A"

# Function to clean Q8 (hot sauce preference)
def clean_hot_sauce(value):
    valid_options = [
        "A lot (hot)", "I will have some of this food item with my hot sauce",
        "A moderate amount (medium)", "A little (mild)", "None"
    ]
    return value if value in valid_options else "N/A"

# Main preprocessing function
def preprocess(file_path, output_path):
    # Load data
    df = pd.read_csv(file_path, dtype=str)

    # Identify if it's raw or manually cleaned
    is_manual_cleaned = "Q2 Cleaned" in df.columns

    # Column mapping based on input type
    col_map = {
        "Q1": "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
        "Q2": "Q2 Cleaned" if is_manual_cleaned else "Q2: How many ingredients would you expect this food item to contain?",
        "Q3": "Q3: In what setting would you expect this food to be served? Please check all that apply",
        "Q4": "Q4 Cleaned" if is_manual_cleaned else "Q4: How much would you expect to pay for one serving of this food item?",
        "Q5": "Q5 Cleaned" if is_manual_cleaned else "Q5: What movie do you think of when thinking of this food item?",
        "Q6": "Q6 Cleaned" if is_manual_cleaned else "Q6: What drink would you pair with this food item?",
        "Q7": "Q7: When you think about this food item, who does it remind you of?",
        "Q8": "Q8: How much hot sauce would you add to this food item?",
        "Label": "Label"
    }

    # Apply cleaning functions
    df["id"] = df["id"].astype(str).str.strip()
    df[col_map["Q1"]] = df[col_map["Q1"]].str.strip()
    df[col_map["Q2"]] = df[col_map["Q2"]].apply(convert_date_format).apply(clean_ingredient_count)
    df[col_map["Q3"]] = df[col_map["Q3"]].apply(lambda x: clean_multi_option(x, [
        "Week day lunch", "Week day dinner", "Weekend lunch", "Weekend dinner", "At a party", "Late night snack"
    ]))
    df[col_map["Q4"]] = df[col_map["Q4"]].apply(clean_price)
    df[col_map["Q5"]] = df[col_map["Q5"]].apply(clean_movie_title)
    df[col_map["Q6"]] = df[col_map["Q6"]].apply(clean_drink)
    df[col_map["Q7"]] = df[col_map["Q7"]].apply(lambda x: clean_multi_option(x, [
        "Friends", "Teachers", "Strangers", "Parents", "Siblings"
    ]))
    df[col_map["Q8"]] = df[col_map["Q8"]].apply(clean_hot_sauce)

    # Rename to standard raw format
    final_df = df.rename(columns=col_map)[["id"] + list(col_map.values())]

    # convert all columns to string
    final_df = final_df.astype(str)

    # Save cleaned file
    final_df.to_csv(output_path, index=False, quoting=1) # quoting=1 to avoid double quotes around strings
    print(f"Cleaned data saved to {output_path}")


# preprocess("../data/raw/cleaned_data_combined.csv", "../data/cleanedWithScript/cleaned_data_universal.csv")
# preprocess("../data/raw/cleaned_data_combined_modified.csv", "../data/cleanedWithScript/cleaned_data_modified_universal.csv")
# preprocess("../data/manuallyCleaned/manual_cleaned_data.csv", "../data/cleanedWithScript/manual_cleaned_data_universal.csv")
