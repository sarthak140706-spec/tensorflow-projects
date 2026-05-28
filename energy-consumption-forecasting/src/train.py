import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

def split_data(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_column, 'time'])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train, model_type='xgboost'):
    if model_type=='linear':
        model = LinearRegression()
    elif model_type=='random_forest':
        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    elif model_type=='xgboost':
        model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)
    else:
        raise ValueError("Invalid model_type")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    MAE = mean_absolute_error(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    R2 = r2_score(y_test, y_pred)
    print(f"MAE: {MAE}, MSE: {MSE}, R2: {R2}")
    return MAE, MSE, R2

def save_model(model, model_name):
    model_path = Path("models")
    model_path.mkdir(exist_ok=True)
    file_path = model_path / f"{model_name}.pkl"
    joblib.dump(model, file_path)
    print(f"Model saved at {file_path}")

def train_pipeline(df, target_column, model_type='xgboost'):
    X_train, X_test, y_train, y_test = split_data(df, target_column)
    model = train_model(X_train, y_train, model_type)
    MAE, MSE, R2 = evaluate_model(model, X_test, y_test)
    save_model(model, f"{model_type}_energy_model")
    return model, MAE, MSE, R2
