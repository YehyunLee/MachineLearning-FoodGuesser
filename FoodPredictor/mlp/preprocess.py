import pandas as pd


def process_intervals(x: str):
    if x.isdigit():
        return x
    interval = x.split('-')
    return interval[0]


def min_max_scaling(series: pd.Series):
    return (series - series.min()) / (series.max() - series.min())
    

def preprocess(path):
    df = pd.read_csv(path).astype(str)
    new_df = pd.DataFrame()

    # Q1
    new_df['complexity'] = pd.to_numeric(df.iloc[:,1], errors='coerce')

    # Q2
    new_df['ingredients'] = pd.to_numeric(df.iloc[:,3].apply(process_intervals), errors='coerce')

    # Q3
    new_df['weekday_lunch'] = df.iloc[:,4].apply(lambda x: 1 if 'Week day lunch' in x else 0)
    new_df['weekday_dinner'] = df.iloc[:,4].apply(lambda x: 1 if 'Week day dinner' in x else 0)
    new_df['weekend_lunch'] = df.iloc[:,4].apply(lambda x: 1 if 'Weekend lunch' in x else 0)
    new_df['weekend_dinner'] = df.iloc[:,4].apply(lambda x: 1 if 'Weekend dinner' in x else 0)
    new_df['party'] = df.iloc[:,4].apply(lambda x: 1 if 'At a party' in x else 0)
    new_df['late_night_snack'] = df.iloc[:,4].apply(lambda x: 1 if 'Late night snack' in x else 0)

    # Q4
    new_df['price'] = pd.to_numeric(df.iloc[:,6].apply(process_intervals), errors='coerce')

    # # Q7
    new_df['friends'] = df.iloc[:,11].apply(lambda x: 1 if 'Friends' in x else 0)
    new_df['teachers'] = df.iloc[:,11].apply(lambda x: 1 if 'Teachers' in x else 0)
    new_df['strangers'] = df.iloc[:,11].apply(lambda x: 1 if 'Strangers' in x else 0)
    new_df['parents'] = df.iloc[:,11].apply(lambda x: 1 if 'Parents' in x else 0)
    new_df['siblings'] = df.iloc[:,11].apply(lambda x: 1 if 'Siblings' in x else 0)
    
    # Q8
    new_df = pd.concat([new_df, pd.get_dummies(df.iloc[:,12])], axis=1)

    # Label
    new_df = pd.concat([new_df, pd.get_dummies(df.iloc[:,13])], axis=1)

    # drop rows with at least one NaN
    new_df = new_df.dropna()

    # normalize all values between 0 and 1
    numeric_df = new_df.select_dtypes(include=[float, int])
    new_df[numeric_df.columns] = (numeric_df - numeric_df.min()) / (numeric_df.max() - numeric_df.min())

    return new_df.astype('float32')
    