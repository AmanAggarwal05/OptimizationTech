import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


# Step 1: Load and Clean Data
def load_and_clean_data(file_path):
    dataset = pd.read_csv(file_path)
    print("Columns in the dataset:", dataset.columns)
    dataset.columns = dataset.columns.str.strip().str.lower()
    column_mapping = {
        'open price': 'open',
        'high price': 'high',
        'low price': 'low',
        'close price': 'close',
        'volume traded': 'volume'
    }
    dataset = dataset.rename(columns=column_mapping)
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in dataset.columns:
            raise KeyError(f"Missing required column: {col}")
    for col in required_columns:
        dataset[col] = pd.to_numeric(dataset[col].replace({',': ''}, regex=True), errors='coerce')
    dataset = dataset.dropna(subset=required_columns)
    return dataset


# Step 2: Feature Extraction Functions
def extract_features(data):
    features = pd.DataFrame()
    windows = [2, 3, 5, 10]
    for w in windows:
        features[f'SMA_{w}'] = data['close'].rolling(window=w).mean()
        features[f'EMA_{w}'] = data['close'].ewm(span=w, adjust=False).mean()
    features = features.dropna()
    return features


# Align Features and Target
def extract_features_and_target(data):
    features = extract_features(data)
    target = data.loc[features.index, 'close']  # Align target to features
    return features, target


# Step 3: Normalize Features and Target
def normalize_features(features):
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(features)
    normalized_df = pd.DataFrame(normalized, columns=features.columns)
    return normalized_df, scaler

def normalize_target(target):
    target_scaler = MinMaxScaler()
    target_scaled = target_scaler.fit_transform(target.values.reshape(-1, 1))
    return target_scaled, target_scaler


# Step 4: Train Neural Network
def train_neural_network(X, y):
    model = Sequential([
        Dense(64, activation='relu', input_dim=X.shape[1]),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=100, batch_size=12, verbose=1, validation_data=(X_test, y_test), callbacks=[early_stopping])
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Neural Network RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return model, history, rmse, mae


# Step 5: Visualize Top Features
def plot_top_features(normalized_features):
    feature_variances = normalized_features.var().sort_values(ascending=False)
    top_features = feature_variances.index[:4]
    print("\nTop 4 Features by Variance:")
    print(top_features)
    for feature in top_features:
        plt.figure(figsize=(10, 5))
        plt.plot(normalized_features[feature], label=feature, color='blue')
        plt.title(f"Trend of {feature}")
        plt.xlabel("Index")
        plt.ylabel("Normalized Value")
        plt.legend()
        plt.show()


# Main Function
def main():
    file_path = "consolidated_coin_data.csv"
    crypto_name = "bitcoin"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    print(f"Analyzing {crypto_name} cryptocurrency data...")
    data = load_and_clean_data(file_path)
    features, target = extract_features_and_target(data)
    normalized_features, scaler = normalize_features(features)
    target_scaled, target_scaler = normalize_target(target)
    plot_top_features(normalized_features)
    _, history, nn_rmse, nn_mae = train_neural_network(normalized_features, target_scaled)
    print(f"\nNeural Network RMSE: {nn_rmse:.4f}")
    print(f"Neural Network MAE: {nn_mae:.4f}")


if __name__ == "__main__":
    main()
