from src.data_stream import get_yfinance_data_stream, get_yfinance_data
from src.model import build_model, prepare_data, scale_data, train_model
from src.detect_anomalies import detect_anomalies, calculate_threshold
from src.visualize import visualize

def main():
    """
    Main function to build the LSTM model, train it on historical data, calculate an anomaly
    detection threshold, and then detect anomalies in a real-time data stream.
    
    Workflow:
    1. Build an LSTM autoencoder model for anomaly detection.
    2. Retrieve and prepare historical training data.
    3. Train the model on scaled training data.
    4. Calculate the anomaly detection threshold using training reconstruction errors.
    5. Stream real-time data and visualize anomaly detection in real-time.
    
    Error Handling:
    - Handles cases where training data is unavailable or real-time data stream fails.
    - Prints appropriate messages and exits if critical errors occur.
    """
    try:
        # Step 1: Build LSTM model
        time_steps = 10
        print("Building LSTM model...")
        model = build_model(input_shape=(time_steps, 1))

        # Step 2: Retrieve and prepare training data
        print("Retrieving training data for AAPL...")
        training_data = get_yfinance_data("AAPL", period="1mo", interval="5m")
        
        if training_data.size == 0:
            print("No training data available. Exiting...")
            exit(1)  # Exit if no training data is found

        # Scale the data
        print("Scaling training data...")
        scaled_training_data, scaler = scale_data(training_data)

        # Step 3: Train the model
        print("Training model...")
        train_model(model, scaled_training_data, epochs=20)

        # Step 4: Calculate anomaly detection threshold
        print("Calculating anomaly detection threshold...")
        X_train = prepare_data(scaled_training_data, time_steps)
        threshold = calculate_threshold(X_train, model)
        print(f"Anomaly detection threshold set to: {threshold}")

        # Step 5: Retrieve real-time data stream and detect anomalies
        print("Starting real-time data stream and visualization...")
        stream = get_yfinance_data_stream("AAPL", period="5d", interval="5m")
        visualize(stream, model, scaler, threshold)

    except ValueError as ve:
        print(f"ValueError occurred: {ve}")
    except RuntimeError as re:
        print(f"RuntimeError occurred: {re}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)  # Exit if an unexpected error occurs


if __name__ == "__main__":
    main()
