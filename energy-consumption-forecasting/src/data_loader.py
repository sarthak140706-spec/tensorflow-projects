import pandas as pd
from pathlib import Path

def load_energy_data():
    path = Path("C:/Users/ASUS/OneDrive/Desktop/energy-consumption-forecasting/data/energy_dataset.csv")

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)

    # Parse time safely
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["time"] = df["time"].dt.tz_localize(None)

    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"Dataset loaded: {df.shape}")
    return df
