import joblib
import pandas as pd
from pathlib import Path

def forecast_energy(model_file, df, target_column, steps=5):
    if not Path(model_file).exists():
        raise FileNotFoundError(f"{model_file} not found.")
    model = joblib.load(model_file)

    last_row = df.iloc[-1]
    forecasts = []

    for step in range(steps):
        features = last_row.drop(['time', target_column]).values.reshape(1, -1)
        pred = model.predict(features)[0]
        forecasts.append(pred)
        for lag in range(1, 3):  # update lag features
            last_row[f'{target_column}_lag_{lag}'] = last_row.get(f'{target_column}_lag_{lag-1}', pred)
        last_row[target_column] = pred

    forecast_df = pd.DataFrame({target_column: forecasts})
    return forecast_df
