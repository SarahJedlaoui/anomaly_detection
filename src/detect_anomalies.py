import numpy as np

def detect_anomalies(model, data, threshold):
    """
    Detects anomalies in the data based on the model's reconstruction error.

    Args:
        model (tf.keras.Model): The trained autoencoder model used for anomaly detection.
        data (np.ndarray): 3D array of data to be analyzed (samples, time_steps, features).
        threshold (float): Error threshold above which anomalies are flagged.

    Returns:
        anomalies (np.ndarray): Boolean array where True indicates an anomaly.
        mse (np.ndarray): Mean squared error between the data and the model predictions.

    Raises:
        ValueError: If data is empty, None, or not properly shaped for prediction.
        RuntimeError: If an error occurs during model prediction.
    """
    if data is None or len(data) == 0:
        raise ValueError("Data cannot be None or empty.")
    
    try:
        # Predict using the model
        predictions = model.predict(data)
        
        # Ensure predictions and data have compatible shapes
        if data.shape != predictions.shape:
            raise ValueError(f"Data shape {data.shape} and predictions shape {predictions.shape} do not match.")
        
        # Calculate mean squared error
        mse = np.mean(np.power(data - predictions, 2), axis=(1, 2))
        anomalies = mse > threshold
        
        return anomalies, mse
    
    except Exception as e:
        raise RuntimeError(f"Error during anomaly detection: {e}")

def calculate_threshold(data, model, percentile=95):
    """
    Calculates an anomaly detection threshold based on reconstruction error percentiles.

    Args:
        data (np.ndarray): 3D array of data used for training (samples, time_steps, features).
        model (tf.keras.Model): Trained autoencoder model.
        percentile (int): The percentile to use for threshold calculation. Default is 95.

    Returns:
        float: The calculated anomaly detection threshold.

    Raises:
        ValueError: If the data is empty or improperly formatted.
        RuntimeError: If an error occurs during model prediction.
    """
    if data is None or len(data) == 0:
        raise ValueError("Training data cannot be None or empty.")
    
    try:
        # Predict using the model
        predictions = model.predict(data)
        
        # Calculate mean squared error
        mse = np.mean(np.power(data - predictions, 2), axis=(1, 2))
        
        # Calculate the threshold based on the specified percentile
        return np.percentile(mse, percentile)
    
    except Exception as e:
        raise RuntimeError(f"Error during threshold calculation: {e}")

if __name__ == "__main__":
    try:
        # Example usage with mock data
        from src.model import build_model, prepare_data
        
        # Create random mock data (100 samples, 10 time_steps, 1 feature)
        mock_data = np.random.random((100, 10, 1))
        
        # Build the autoencoder model
        mock_model = build_model(input_shape=(10, 1))
        
        # Calculate threshold for anomaly detection
        threshold = calculate_threshold(mock_data, mock_model)
        print(f"Anomaly detection threshold: {threshold}")
        
        # Detect anomalies in the data
        anomalies, mse = detect_anomalies(mock_model, mock_data, threshold)
        print(f"Anomalies: {anomalies}")
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except RuntimeError as re:
        print(f"RuntimeError: {re}")
    except Exception as e:
        print(f"Unexpected error: {e}")
