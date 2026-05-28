import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def handle_missing_values(df):
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(method='ffill')
    return df

def remove_outliers(df, column, method='zscore', z_threshold=3):
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataframe.")
    if method == 'zscore':
        z_scores = np.abs(stats.zscore(df[column]))
        df = df[z_scores <= z_threshold]
        print(f"Outlier removal (zscore): removed {len(df) - len(df[z_scores <= z_threshold])} rows, remaining {len(df)}")
    elif method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[column] >= Q1 - 1.5*IQR) & (df[column] <= Q3 + 1.5*IQR)]
    else:
        raise ValueError("Method must be 'zscore' or 'iqr'.")
    if df.empty:
        raise ValueError("No data left after outlier removal.")
    return df

def feature_engineering(df, target_column, lags=[1]):
    # Create time features
    df['hour'] = df['time'].dt.hour
    df['day'] = df['time'].dt.day
    df['month'] = df['time'].dt.month
    df['weekday'] = df['time'].dt.weekday

    # Create lag features for target column
    for lag in lags:
        df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)

    # Drop only rows where target lag is NaN
    lag_cols = [f'{target_column}_lag_{lag}' for lag in lags]
    df.dropna(subset=lag_cols, inplace=True)

    if df.empty:
        raise ValueError("No data left after lag feature creation. Reduce lags.")

    print(f"Feature engineering complete: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def scale_features(df, columns, method='minmax'):
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Method must be 'minmax' or 'standard'.")
    df[columns] = scaler.fit_transform(df[columns])
    return df

def preprocess_energy_data(df, target_column, feature_columns=None, lags=[1,2], outlier_method='zscore', scale_method='minmax'):
    df = handle_missing_values(df)
    df = remove_outliers(df, column=target_column, method=outlier_method)
    df = feature_engineering(df, target_column=target_column, lags=lags)

    if feature_columns is None:
        feature_columns = df.select_dtypes(include=['float64','int64']).columns.tolist()
        if target_column in feature_columns:
            feature_columns.remove(target_column)

    df = scale_features(df, columns=feature_columns, method=scale_method)
    print(f"Preprocessing successful: {df.shape}")
    return df
