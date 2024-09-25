import yfinance as yf
import numpy as np
import time

def get_yfinance_data(symbol, period="5d", interval="5m"):
    """
    Retrieve historical stock data from Yahoo Finance for a given symbol.

    Args:
        symbol (str): The stock symbol/ticker to retrieve data for (e.g., 'AAPL' for Apple).
        period (str): The time period for which to retrieve data (e.g., '5d' for 5 days).
        interval (str): The frequency of the data (e.g., '5m' for 5-minute intervals).

    Returns:
        np.ndarray: A NumPy array of 'Close' prices over the specified period.
        If an error occurs or no data is available, an empty array is returned.

    Raises:
        ValueError: If no data is found for the given symbol.
    """
    try:
        stock_data = yf.download(tickers=symbol, period=period, interval=interval)
        if stock_data.empty:
            raise ValueError(f"No data found for symbol: {symbol}")
        return stock_data['Close'].values  # Use 'Close' prices for analysis
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return np.array([])  # Return an empty array on error
    except yf.shared._exceptions.RemoteDataError as rde:
        print(f"RemoteDataError: Unable to retrieve data for {symbol} due to a network issue.")
        return np.array([])
    except Exception as e:
        print(f"Unexpected error retrieving data: {e}")
        return np.array([])  # Catch-all for unexpected errors


def get_yfinance_data_stream(symbol, period="5d", interval="5m"):
    """
    Generator to simulate a real-time data stream using Yahoo Finance historical data.

    Args:
        symbol (str): The stock symbol/ticker to retrieve data for (e.g., 'AAPL').
        period (str): The time period for which to retrieve data (default is '5d').
        interval (str): The frequency of data points (e.g., '5m' for 5-minute intervals).

    Yields:
        float: The next price in the data stream.
    
    Raises:
        StopIteration: If no data is available, the generator stops.
    """
    data = get_yfinance_data(symbol, period, interval)
    
    if data.size == 0:
        print("No data available for streaming. Ensure valid symbol and internet connection.")
        return  # Stop the generator if no data is available
    
    for price in data:
        yield price
        try:
            time.sleep(0.1)  # Simulate real-time delay
        except KeyboardInterrupt:
            print("Data stream interrupted by user.")
            return
        except Exception as e:
            print(f"Error during data streaming: {e}")
            return
