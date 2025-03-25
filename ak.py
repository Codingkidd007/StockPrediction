import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_stock_data():
    file_path = "Quote-Equity-ADANIGREEN-EQ-25-02-2025-to-25-03-2025.csv"
    stock = pd.read_csv(file_path)

    # Strip spaces from column names
    stock.columns = stock.columns.str.strip()
    print("Columns in CSV:", stock.columns)  # Debugging line to check column names

    # Convert 'Date' column to datetime
    stock['Date'] = pd.to_datetime(stock['Date'], format='%d-%b-%Y')
    stock['Days'] = (stock['Date'] - stock['Date'].min()).dt.days

    # Ensure 'close' column is numeric (remove commas if needed)
    stock['Close'] = pd.to_numeric(stock['close'].astype(str).str.replace(',', ''), errors='coerce')

    # Feature engineering: Moving Average
    stock['MA_5'] = stock['Close'].rolling(window=5).mean()
    stock['MA_10'] = stock['Close'].rolling(window=10).mean()
    stock = stock.dropna()

    return stock[['Days', 'Close', 'MA_5', 'MA_10']]


def train_model(stock):
    X = stock[['Days', 'MA_5', 'MA_10']]
    y = stock['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_train, X_test, y_train, y_test, y_pred


def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse


def visualize_predictions(stock, model):
    plt.figure(figsize=(10, 5))
    plt.scatter(stock['Days'], stock['Close'], color='blue', label='Actual Prices')
    plt.plot(stock['Days'], model.predict(stock[['Days', 'MA_5', 'MA_10']]), color='red', label='Predicted Prices')
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction using Linear Regression')
    plt.legend()
    st.pyplot(plt)


def main():
    st.title("Stock Price Prediction System")
    st.write("Upload your stock CSV file to get insights and predictions.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        stock = pd.read_csv(uploaded_file)
        stock.columns = stock.columns.str.strip()
        stock['Date'] = pd.to_datetime(stock['Date'], format='%d-%b-%Y')
        stock['Days'] = (stock['Date'] - stock['Date'].min()).dt.days
        stock['Close'] = pd.to_numeric(stock['close'].astype(str).str.replace(',', ''), errors='coerce')
        stock['MA_5'] = stock['Close'].rolling(window=5).mean()
        stock['MA_10'] = stock['Close'].rolling(window=10).mean()
        stock = stock.dropna()

        model, X_train, X_test, y_train, y_test, y_pred = train_model(stock)

        mae, mse, rmse = evaluate_model(y_test, y_pred)
        st.write(f"**Mean Absolute Error:** {mae}")
        st.write(f"**Mean Squared Error:** {mse}")
        st.write(f"**Root Mean Squared Error:** {rmse}")

        visualize_predictions(stock, model)


if __name__ == "__main__":
    main()
