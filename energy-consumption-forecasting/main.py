from src.data_loader import load_energy_data
from src.preprocessing import preprocess_energy_data
from src.train import train_pipeline
from src.predict import forecast_energy

TARGET_COLUMN = "total load actual"  # column from your dataset
MODEL_TYPE = "xgboost"

def main():
    # Load dataset
    df = load_energy_data()

    # Preprocess dataset
    df = preprocess_energy_data(df, target_column=TARGET_COLUMN)

    # Train, evaluate, and save model
    model, MAE, MSE, R2 = train_pipeline(df, target_column=TARGET_COLUMN, model_type=MODEL_TYPE)

    print("\nFinal Metrics:")
    print(f"MAE: {MAE}")
    print(f"MSE: {MSE}")
    print(f"R²: {R2}")

    # Forecast next 5 steps
    forecast_df = forecast_energy(f"models/{MODEL_TYPE}_energy_model.pkl", df, target_column=TARGET_COLUMN, steps=5)
    print("\nNext 5-step forecast:")
    print(forecast_df)

if __name__ == "__main__":
    main()
