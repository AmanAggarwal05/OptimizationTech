import pandas as pd
# the csv data is cleaned
# and processed using this library and also helps
# using the datafram operations to create technical indicator features like SMA,EMA,etc.
import numpy as np
# primarily used in handking of the arrays
# , in our project we are used to perform the mathematical
# calculations used in ADX calculation
import matplotlib.pyplot as plt
#plotting the trends of features to analyze the behavior of
# the cryptocurrency ,  and therefore used to generate descriptive plots
from sklearn.model_selection import train_test_split
# this is to divide the dataset and the
# testing the subsets
# splits feature data and target values into training and testing for model training
from sklearn.metrics import mean_squared_error , mean_absolute_error
# this is the value we have obtained averahe absolute difference
# between predicted and actual values , this is a unit of measurement to
# get the margin of error in the prediction model
from sklearn.preprocessing import MinMaxScaler
# this is imported for neural network and is scaled from 0 to 1 such
# to ensure uniformity and improving neural network training
from tensorflow.keras.model import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
# Sequential - defines a linear stack of layers for the neural  network
# Dense - it is used to connect the layers of the networks
# Adam - a popular potimizer for the training neural networks , combining
# the benefits of the momentum and adaptive learning rates
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


# Step 5: Neural Network Model
def train_neural_network(X, y):
    """
    Train a Neural Network for regression.
    """
    model = Sequential([
        Dense(64, activation='relu', input_dim=X.shape[1]),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    history = model.fit(X_train, y_train, epochs=100, batch_size=12, verbose=1, validation_data=(X_test, y_test))

    # Evaluate Model
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Neural Network RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    return model, history, rmse, mae


# Step 6: Visualize Top 4 Features
def plot_top_features(normalized_features):
    """
    Plot trends for the top 4 features based on variance.
    """
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


# Main Process
def main():
    file_path = "consolidated_coin_data.csv"  # Replace with your file path
    crypto_name = "bitcoin"  # Change as needed

    # Load and clean data
    data = load_and_clean_data(file_path)

    # Extract features
    features = extract_features(data)
    target = data['close'].iloc[len(data) - len(features):]  # Target is 'close' price

    # Normalize features
    normalized_features, scaler = normalize_features(features)

    # Plot top 4 features
    plot_top_features(normalized_features)

    # Train Neural Network
    _, history, nn_rmse, nn_mae = train_neural_network(normalized_features, target)

    # Display performance metrics
    print(f"\nNeural Network RMSE: {nn_rmse:.4f}")
    print(f"Neural Network MAE: {nn_mae:.4f}")


# Execute the script
if __name__ == "__main__":
    main()
