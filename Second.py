import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler


# Step 1: Load and Clean Data
def load_and_clean_data(file_path):
    """
    Load and clean the dataset, ensuring required columns exist and are properly formatted.
    """
    dataset = pd.read_csv(file_path)

    # Print columns for debugging
    print("Columns in the dataset:", dataset.columns)

    # Normalize column names
    dataset.columns = dataset.columns.str.strip().str.lower()

    # Map column names if necessary
    column_mapping = {
        'open price': 'open',
        'high price': 'high',
        'low price': 'low',
        'close price': 'close',
        'volume traded': 'volume'
    }
    dataset = dataset.rename(columns=column_mapping)

    # Required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in dataset.columns:
            raise KeyError(f"Missing required column: {col}")

    # Convert columns to numeric
    for col in required_columns:
        dataset[col] = pd.to_numeric(dataset[col].replace({',': ''}, regex=True), errors='coerce')

    # Drop rows with missing values in required columns
    dataset = dataset.dropna(subset=required_columns)
    return dataset


# Step 2: Feature Extraction Functions
def calculate_sma(data, window):
    return data['close'].rolling(window=window).mean()


def calculate_ema(data, window):
    return data['close'].ewm(span=window, adjust=False).mean()


def calculate_adx(data, window):
    high, low, close = data['high'], data['low'], data['close']
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    atr = tr.rolling(window=window).mean()
    plus_di = 100 * (plus_dm / atr).rolling(window=window).mean()
    minus_di = 100 * (minus_dm / atr).rolling(window=window).mean()
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(window=window).mean()


def calculate_macd(data, short_window, long_window, signal_window):
    short_ema = data['close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal


def calculate_roc(data, window):
    return 100 * (data['close'].diff(window) / data['close'].shift(window))


def calculate_cci(data, window):
    tp = (data['high'] + data['low'] + data['close']) / 3
    sma = tp.rolling(window=window).mean()
    mean_dev = (tp - sma).abs().rolling(window=window).mean()
    cci = (tp - sma) / (0.015 * mean_dev)
    return cci


# Step 3: Feature Extraction Wrapper
def extract_features(data):
    """
    Extract features from the dataset using technical indicators.
    """
    features = pd.DataFrame()
    windows = [2, 3, 5, 10]

    for w in windows:
        features[f'SMA_{w}'] = calculate_sma(data, w)
        features[f'EMA_{w}'] = calculate_ema(data, w)
        features[f'ADX_{w}'] = calculate_adx(data, w)
        features[f'ROC_{w}'] = calculate_roc(data, w)
        features[f'CCI_{w}'] = calculate_cci(data, w)

    macd, signal = calculate_macd(data, 12, 26, 9)
    features['MACD'] = macd
    features['Signal'] = signal
    features = features.dropna()  # Remove NaN rows
    return features


# Step 4: Normalize Features
def normalize_features(features):
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(features)
    normalized_df = pd.DataFrame(normalized, columns=features.columns)
    return normalized_df, scaler


# Step 5: Simulated ACO with Visualization
def apply_aco(features, target, crypto_name):
    """
    Simulated ACO feature selection and visualization.
    """
    selected_features = features[['SMA_2', 'ADX_3', 'MACD', 'CCI_10']]  # Example selection

    # Visualize ACO feature importance
    importance = [1, 2, 3, 4]  # Simulated importance values
    plt.figure(figsize=(8, 6))
    plt.bar(selected_features.columns, importance, color='teal', alpha=0.7)
    plt.title(f"ACO Feature Selection - {crypto_name}")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.show()

    return selected_features


# Step 6: Train SVM Model
def train_svm(X, y):
    """
    Train an SVM model with a Gaussian kernel.
    """
    svm_model = SVR(kernel='rbf', C=1.0, gamma='scale')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Plot True vs. Predicted Prices
    plt.figure(figsize=(8, 6))
    plt.plot(y_test.values, label="True Prices", color="blue")
    plt.plot(y_pred, label="Predicted Prices", color="orange")
    plt.legend()
    plt.title("True vs. Predicted Close Prices")
    plt.show()
    return svm_model


# Main Process
def main():
    file_path = "consolidated_coin_data.csv"  # Replace with your file path
    crypto_name = "Bitcoin"  # Change as needed

    # Load and clean data
    data = load_and_clean_data(file_path)

    # Extract features
    features = extract_features(data)
    target = data['close'].iloc[len(data) - len(features):]  # Target is 'close' price

    # Normalize features
    normalized_features, scaler = normalize_features(features)
    print("Normalized Data Sample:")
    print(normalized_features.head())

    # Plot normalized features
    plt.figure(figsize=(12, 6))
    for col in normalized_features.columns[:5]:  # Plot the first 5 normalized features
        plt.plot(normalized_features[col], label=col)
    plt.title("Normalized Feature Trends")
    plt.xlabel("Index")
    plt.ylabel("Normalized Values")
    plt.legend()
    plt.show()

    # Feature selection with ACO
    selected_features = apply_aco(normalized_features, target, crypto_name)

    # Train SVM model
    train_svm(selected_features, target)


# Execute the script
if __name__ == "__main__":
    main()
