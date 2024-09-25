import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from src.model import prepare_data, scale_data
from src.detect_anomalies import detect_anomalies, calculate_threshold

def visualize(stream, model, scaler, threshold, time_steps=10):
    """
    Real-time visualization of data stream and anomalies.

    Args:
        stream (generator): A generator that yields the data points from the stream.
        model (tf.keras.Model): The trained autoencoder model for anomaly detection.
        scaler (sklearn.preprocessing.MinMaxScaler): Scaler used for transforming the data.
        threshold (float): Anomaly detection threshold based on reconstruction error.
        time_steps (int): Number of time steps used by the LSTM model (default=10).

    Raises:
        ValueError: If the data transformation fails.
        RuntimeError: If an error occurs during anomaly detection.
    """
    fig, ax = plt.subplots()
    data = []
    anomalies = []
    error_history = []

    # Create plot lines for data and anomaly points
    line, = ax.plot([], [], label='Data')
    anomaly_points, = ax.plot([], [], 'ro', label='Anomalies')

    def update(frame):
        """Update function for real-time plot."""
        try:
            value = next(stream)  # Retrieve next data point from stream
        except StopIteration:
            print("End of data stream reached.")
            return line, anomaly_points
        except Exception as e:
            print(f"Error while accessing stream: {e}")
            return line, anomaly_points

        data.append(value)  # Append new data point to data list

        # Ensure we have enough data points to feed into the LSTM model
        if len(data) >= time_steps:
            try:
                # Prepare the most recent time_steps of data for model input
                sequence = np.array(data[-time_steps:]).reshape(-1, 1)
                scaled_data = scaler.transform(sequence)
                input_data = prepare_data(scaled_data, time_steps)

                # Detect anomalies in the current data window
                anomaly, error = detect_anomalies(model, input_data, threshold)
                error_history.append(error)
                
                print(f"Data Point: {value}, Error: {error[-1]}, Anomaly Detected: {anomaly[-1]}")
            
            except ValueError as e:
                print(f"Data transformation error: {e}")
                return line, anomaly_points
            except Exception as e:
                print(f"Anomaly detection error: {e}")
                return line, anomaly_points

            # Update the plot with new data
            line.set_data(range(len(data)), data)
            ax.relim()
            ax.autoscale_view()

            # Highlight anomaly points on the plot
            if anomaly and anomaly[-1]:
                anomalies.append(len(data) - 1)
                anomaly_points.set_data(anomalies, [data[i] for i in anomalies])

        # Adjust plot limits
        ax.set_xlim(0, len(data) + 10)
        ax.set_ylim(min(data[-time_steps:]) - 1, max(data[-time_steps:]) + 1)

        return line, anomaly_points

    # Animate the plot with real-time updates
    ani = FuncAnimation(fig, update, blit=True, interval=200)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    try:
        # Import necessary modules for data stream and model
        from src.data_stream import get_yfinance_data_stream
        from src.model import build_model, scale_data, train_model

        # Step 1: Retrieve real-time data stream (e.g., Apple stock data)
        stream = get_yfinance_data_stream("AAPL")

        # Step 2: Build and train the LSTM model
        model = build_model()
        training_data = np.sin(np.linspace(0, 100, 1000))  # Example sine wave data for training
        scaled_data, scaler = scale_data(training_data)
        train_model(model, scaled_data)

        # Step 3: Analyze training errors and set the anomaly detection threshold
        training_input = prepare_data(scaled_data, time_steps=model.input_shape[1])
        threshold = calculate_threshold(training_input, model)
        print(f"Threshold set to: {threshold}")

        # Step 4: Start the real-time visualization
        visualize(stream, model, scaler, threshold)
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except RuntimeError as re:
        print(f"RuntimeError: {re}")
    except Exception as e:
        print(f"Unexpected error: {e}")
