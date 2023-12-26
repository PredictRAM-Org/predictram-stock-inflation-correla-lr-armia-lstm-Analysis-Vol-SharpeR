import os
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from pmdarima import auto_arima
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Function to prepare data for LSTM
def prepare_data_for_lstm(data, look_back=1):
    x, y = [], []
    for i in range(len(data) - look_back):
        x.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(x), np.array(y)

# Function to build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to predict future prices using LSTM
def predict_future_lstm(last_observed_price, model, min_max_scaler, num_steps=1):
    predicted_prices = []
    input_data = last_observed_price.reshape(1, -1, 1)

    for _ in range(num_steps):
        predicted_price = model.predict(input_data)
        predicted_prices.append(predicted_price[0, 0])
        input_data = np.append(input_data[:, 1:, :], predicted_price.reshape(1, 1, 1), axis=1)

    return min_max_scaler.inverse_transform(np.array(predicted_prices).reshape(1, -1))[0]

# Load CPI data
cpi_data = pd.read_excel("CPI.xlsx")
cpi_data['Date'] = pd.to_datetime(cpi_data['Date'])
cpi_data.set_index('Date', inplace=True)

# Streamlit UI
st.title("Stock-CPI Correlation Analysis with Expected Inflation, Price Prediction, and Sentiment Analysis")

# User input for uploading Excel file with stocks name column
uploaded_file = st.file_uploader("Upload Excel file with stocks name column", type=["xlsx", "xls"])
if uploaded_file is not None:
    stocks_data = pd.read_excel(uploaded_file)
else:
    st.warning("Please upload an Excel file.")
    st.stop()

# Select data range for training models
data_range = st.selectbox("Select Data Range for Model Training:", ["6 months", "1 year", "3 years", "5 years"])

# Filter data based on the selected range
end_date = pd.to_datetime('today')
if data_range == "6 months":
    start_date = end_date - pd.DateOffset(months=6)
elif data_range == "1 year":
    start_date = end_date - pd.DateOffset(years=1)
elif data_range == "3 years":
    start_date = end_date - pd.DateOffset(years=3)
else:
    start_date = end_date - pd.DateOffset(years=5)

# Filter CPI data
filtered_cpi_data = cpi_data.loc[start_date:end_date]

# User input for expected CPI inflation
expected_inflation = st.number_input("Enter Expected Upcoming CPI Inflation:", min_value=0.0, step=0.01)

# Train models
if st.button("Train Models"):
    st.write(f"Training models with data range: {data_range}, expected CPI inflation: {expected_inflation}...")

    correlations = []
    future_prices_lr_list = []
    future_prices_arima_list = []
    latest_actual_prices = []
    future_price_lstm_list = []
    stock_names = []
    volatilities = []
    sharpe_ratios = []

    for index, row in stocks_data.iterrows():
        stock_name = row['Stock']

        # Fetch stock data and filter based on selected date range
        stock_file_path = os.path.join("stock_folder", f"{stock_name}.xlsx")
        if os.path.exists(stock_file_path):
            selected_stock_data = pd.read_excel(stock_file_path)
            selected_stock_data['Date'] = pd.to_datetime(selected_stock_data['Date'])
            selected_stock_data.set_index('Date', inplace=True)
            filtered_stock_data = selected_stock_data.loc[start_date:end_date]

            # Merge stock and CPI data on Date
            merged_data = pd.merge(filtered_stock_data, filtered_cpi_data, left_index=True, right_index=True, how='inner')

            # Handle NaN values in CPI column
            if merged_data['CPI'].isnull().any():
                st.write(f"Warning: NaN values found in 'CPI' column for {stock_name}. Dropping NaN values.")
                merged_data = merged_data.dropna(subset=['CPI'])

            # Calculate CPI change
            merged_data['CPI Change'] = merged_data['CPI'].pct_change()

            # Drop NaN values after calculating percentage change
            merged_data = merged_data.dropna()

            # Show correlation between 'Close' column and 'CPI Change'
            correlation_close_cpi = merged_data['Close'].corr(merged_data['CPI Change'])
            correlation_actual = merged_data['Close'].corr(merged_data['CPI'])

            st.write(f"Correlation between 'Close' and 'CPI Change' for {stock_name}: {correlation_close_cpi}")
            st.write(f"Actual Correlation between 'Close' and 'CPI' for {stock_name}: {correlation_actual}")

            # Train Linear Regression model
            model_lr = LinearRegression()
            X_lr = merged_data[['CPI']]
            y_lr = merged_data['Close']
            model_lr.fit(X_lr, y_lr)

            # Train ARIMA model using auto_arima
            model_arima = auto_arima(y_lr, seasonal=False, suppress_warnings=True)

            # Train LSTM model
            min_max_scaler = MinMaxScaler()
            scaled_data = min_max_scaler.fit_transform(y_lr.values.reshape(-1, 1))
            x_train, y_train = prepare_data_for_lstm(scaled_data, look_back=3)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            model_lstm = build_lstm_model(x_train.shape[1])
            model_lstm.fit(x_train, y_train, epochs=50, batch_size=32)

            # Predict future prices based on Linear Regression
            future_prices_lr = model_lr.predict([[expected_inflation]])
            st.write(f"Predicted Price Change for Future Inflation (Linear Regression): {future_prices_lr[0]}")

            # Predict future prices based on ARIMA
            arima_predictions = model_arima.predict(1)
            if isinstance(arima_predictions, pd.Series):
                future_prices_arima = arima_predictions.iloc[0]
            else:
                future_prices_arima = arima_predictions[0]
            st.write(f"Predicted Price Change for Future Inflation (ARIMA): {future_prices_arima}")

            # Predict future prices using LSTM
            last_observed_price = scaled_data[-3:]  # Use the last 3 observations for prediction
            future_price_lstm = predict_future_lstm(last_observed_price, model_lstm, min_max_scaler)
            st.write(f"Predicted Stock Price for Future Inflation (LSTM): {future_price_lstm}")

            # Display the latest actual price
            latest_actual_price = merged_data['Close'].iloc[-1]
            st.write(f"Latest Actual Price for {stock_name}: {latest_actual_price}")

            # Calculate volatility and Sharpe ratio
            daily_returns = merged_data['Close'].pct_change().dropna()
            volatility = daily_returns.std()
            annualized_volatility = volatility * np.sqrt(252)  # Assuming 252 trading days in a year
            average_daily_return = daily_returns.mean()
            annualized_return = average_daily_return * 252
            risk_free_rate = 0.02  # You can adjust the risk-free rate as needed
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

            st.write(f"Volatility for {stock_name}: {annualized_volatility}")
            st.write(f"Sharpe Ratio for {stock_name}: {sharpe_ratio}")

            correlations.append(correlation_close_cpi)
            future_prices_lr_list.append(future_prices_lr[0])
            future_prices_arima_list.append(future_prices_arima)
            latest_actual_prices.append(latest_actual_price)
            future_price_lstm_list.append(future_price_lstm)
            stock_names.append(stock_name)
            volatilities.append(annualized_volatility)
            sharpe_ratios.append(sharpe_ratio)

    # Create a DataFrame for results
    results_data = {
        'Stock': stock_names,
        'Correlation with CPI Change': correlations,
        'Predicted Price Change (Linear Regression)': future_prices_lr_list,
        'Predicted Price Change (ARIMA)': future_prices_arima_list,
        'Latest Actual Price': latest_actual_prices,
        'Predicted Stock Price (LSTM)': future_price_lstm_list,
        'Volatility': volatilities,
        'Sharpe Ratio': sharpe_ratios
    }
    results_df = pd.DataFrame(results_data)

    # Display results in descending order of correlation
    st.write("\nResults Sorted by Correlation:")
    sorted_results_df = results_df.sort_values(by='Correlation with CPI Change', ascending=False)
    st.table(sorted_results_df)

    # Add a download button for the sorted results
    if st.button("Download Results Sorted by Correlation"):
        # Prepare the file for download
        output_filename = "results_sorted_by_correlation.xlsx"
        output_path = os.path.join("output", output_filename)  # Adjust the path as needed

        # Save the sorted results to Excel
        sorted_results_df.to_excel(output_path, index=False, sheet_name='Results')

        # Provide a link for download
        st.success(f"Download [Results Sorted by Correlation](sandbox:/output/{output_filename})")
