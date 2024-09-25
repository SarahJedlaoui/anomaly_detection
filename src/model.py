import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

# LSTM autoencoder model
def build_model(input_shape=(10, 1)):
    """
    Builds and returns an LSTM autoencoder model.

    Args:
        input_shape (tuple): The shape of the input data (time_steps, features). 
                             Default is (10, 1), meaning 10 time steps with 1 feature.

    Returns:
        model: A compiled LSTM autoencoder model.

    Raises:
        ValueError: If input_shape is not a tuple or is improperly defined.
    """
    try:
        model = Sequential([
            Input(shape=input_shape),
            LSTM(64, activation='relu', return_sequences=False),
            RepeatVector(input_shape[0]),
            LSTM(64, activation='relu', return_sequences=True),
            TimeDistributed(Dense(1))
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    except Exception as e:
        raise ValueError(f"Error building the model: {e}")

# Prepare data for LSTM model
def prepare_data(data, time_steps):
    """
    Transforms a 1D data array into a 3D format suitable for LSTM input.

    Args:
        data (np.ndarray): 1D array of time series data.
        time_steps (int): Number of time steps for the LSTM input.

    Returns:
        np.ndarray: 3D array of shape (samples, time_steps, features), where features = 1.

    Raises:
        ValueError: If data length is less than the required time steps.
    """
    if len(data) < time_steps:
        raise ValueError(f"Insufficient data: Length of data ({len(data)}) is less than time_steps ({time_steps}).")
    
    X = []
    for i in range(len(data) - time_steps + 1):
        X.append(data[i:i + time_steps])
    
    return np.array(X)

# Scale data to [0, 1]
def scale_data(data):
    """
    Scales the input data to the range [0, 1].

    Args:
        data (np.ndarray): 1D array of data to be scaled.

    Returns:
        scaled_data (np.ndarray): Scaled data in the range [0, 1].
        scaler (MinMaxScaler): Fitted MinMaxScaler object for inverse transformation.

    Raises:
        ValueError: If data cannot be reshaped into a 2D array.
    """
    try:
        scaler = MinMaxScaler()
        data = data.reshape(-1, 1)  # Ensure data is 2D
        return scaler.fit_transform(data), scaler
    except Exception as e:
        raise ValueError(f"Error scaling data: {e}")

# Train model with the data
def train_model(model, data, epochs=20, batch_size=32):
    """
    Trains the LSTM autoencoder on the prepared data.

    Args:
        model (tf.keras.Model): Compiled LSTM autoencoder model.
        data (np.ndarray): 1D time series data for training.
        epochs (int): Number of training epochs. Default is 20.
        batch_size (int): Batch size for training. Default is 32.

    Returns:
        model (tf.keras.Model): Trained model.

    Raises:
        ValueError: If data is improperly formatted for training.
    """
    try:
        time_steps = model.input_shape[1]
        X_train = prepare_data(data, time_steps)
        model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, shuffle=False)
        return model
    except ValueError as ve:
        raise ValueError(f"Training data preparation error: {ve}")
    except Exception as e:
        raise ValueError(f"Error during model training: {e}")

if __name__ == "__main__":
    try:
        # Example usage with mock data
        mock_data = np.sin(np.linspace(0, 100, 1000))  # Sine wave as mock normal data
        
        # Build the LSTM model
        model = build_model()
        
        # Scale the data
        scaled_data, _ = scale_data(mock_data)
        
        # Train the model
        train_model(model, scaled_data)
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"Unexpected error: {e}")
