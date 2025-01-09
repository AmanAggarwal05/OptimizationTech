from tensorflow.keras.callbacks import Callback


# Custom Callback for Early Stopping Based on MAE
class EarlyStoppingByMAE(Callback):
    def __init__(self, target_mae=1.0):
        super(EarlyStoppingByMAE, self).__init__()
        self.target_mae = target_mae

    def on_epoch_end(self, epoch, logs=None):
        current_mae = logs.get("val_mae")
        if current_mae and current_mae <= self.target_mae:
            print(f"\nStopping training early as MAE has reached {current_mae:.4f} at epoch {epoch + 1}")
            self.model.stop_training = True


# Train Neural Network with Early Stopping
def train_neural_network(X, y, target_mae=1.0):
    """
    Train a Neural Network for regression with early stopping when MAE is near the target.
    """
    model = Sequential([
        Dense(64, activation='relu', input_dim=X.shape[1]),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the model with early stopping
    early_stopping = EarlyStoppingByMAE(target_mae=target_mae)
    history = model.fit(
        X_train, y_train,
        epochs=200,  # Increased epochs for better convergence
        batch_size=32,
        verbose=1,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )

    # Evaluate Model
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\nFinal Neural Network RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Print Weights and Biases
    print("\nModel Weights and Biases:")
    for layer_idx, layer in enumerate(model.layers):
        weights, biases = layer.get_weights()
        print(f"Layer {layer_idx + 1} Weights Shape: {weights.shape}, Biases Shape: {biases.shape}")
        print(f"Weights:\n{weights}")
        print(f"Biases:\n{biases}")

    return model, history, rmse, mae
