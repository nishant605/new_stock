import os
# This MUST be the first line of code in the file
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import sys
import numpy as np
# Fix for NumPy 2.0/Pickle mismatch
sys.modules['numpy._core'] = np

import pandas as pd
import streamlit as st
import pickle
import plotly.express as px
import tensorflow as tf

from tensorflow.keras.models import load_model

try:
    df = pd.read_pickle('Copy of df_clean.pkl')
except FileNotFoundError:
    st.error("Data file 'Copy of df_clean.pkl' not found in root directory.")

excluded_name = {
    'HINDALC0',    # Old symbol for HINDALCO
    'HEROHONDA',   # Old symbol for HEROMOTOCO
    'HINDLEVER',   # Old symbol for HINDUNILVR
    'INFOSYSTCH',  # Old symbol for INFY
    'TISCO',       # Old symbol for TATASTEEL
    'MUNDRAPORT',  # Old symbol for ADANIPORTS
    'SESAGOA',     # Old symbol for VEDL
    'ZEETELE',     # Old symbol for ZEEL
    'UTIBANK',     # Old symbol for AXISBANK
    'SSLT',        # Old symbol for VEDL (Sterlite)
    'KOTAKMAH',    # Old symbol for KOTAKBANK
    'TELCO',       # Old symbol for TATAMOTORS
    'BHARTI',      # Old symbol for BHARTIARTL
    'UNIPHOS',     # Old symbol (UPL predecessor)
}
all_stocks_raw = sorted(df['Symbol'].unique())
list_of_stocks = [s for s in all_stocks_raw if s not in excluded_name]


st.title('Stock Price Prediction')

features = [
    'Open', 'High', 'Low', 'Close', 'VWAP', 'Volume',
    'Deliverable Volume',
    'return', 'MA10', 'MA50', 'volatility','MACD', 'High_Low_Ratio', 'Close_Open_Ratio'
]

def feature_creation(df, stock_name):
    data = df[df['Symbol'] == stock_name].copy()
    data.sort_values('Date', inplace=True)

    data['return'] = data['Close'].pct_change()
    data['MA10'] = data['Close'].rolling(10).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    data['volatility'] = data['Close'].rolling(10).std()

    data = data.dropna().reset_index(drop=True)

    ema12 = data['Close'].ewm(span=12, adjust=False).mean() #Calculates the 12-day Exponential Moving Average (EMA) of the closing price.
    ema26 = data['Close'].ewm(span=26, adjust=False).mean() #Means EMA considers 12 previous periods.
    data['MACD'] = ema12 - ema26 #MACD is the difference between short-term EMA and long-term EMA.

    data['High_Low_Ratio'] = data['High'] / (data['Low'] + 1e-10)
    data['Close_Open_Ratio'] = data['Close'] / (data['Open'] + 1e-10)

    data['target'] = data['Close'].pct_change().shift(-1)
    data['current_close'] = data['Close'] #why not use future close price as target?
    #because we want to predict the return, not the price itself. By using percentage change, we can model the relative movement of the stock price,
    #which is more stable and easier for the model to learn compared to absolute price values.

    data = data.dropna().reset_index(drop=True)

    return data

st.subheader('Select a stock to predict its next-day price')

selected_stock = st.selectbox('Select Stock', list_of_stocks)

if "predict_clicked" not in st.session_state:
    st.session_state.predict_clicked = False

if st.button('Predict'):
    st.session_state.predict_clicked = True

if st.session_state.predict_clicked:
    st.divider()

    data = df[df['Symbol'] == selected_stock]
    st.write(f'last 10 days data of {selected_stock}')
    st.dataframe(data.tail(10))

    data = feature_creation(data, selected_stock)
    X = data.drop(['target','Symbol','Date'], axis=1)

    st.divider()

    st.subheader('Historical Close Price')

    fig = px.line(data, x='Date', y='Close', title=f'{selected_stock} Close Price Over Time')
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader('Predicted Next-Day Close Price')

    tf.keras.backend.clear_session()
    model_path = f'models/gru_model_{selected_stock}.keras'
    model = load_model(model_path, compile=False)
    #model = load_model(f'models/gru_model_{selected_stock}.keras')

    scaler_x = pickle.load(open(f'models/scaler_X_{selected_stock}.pkl', 'rb'))
    scaler_y = pickle.load(open(f'models/scaler_y_{selected_stock}.pkl', 'rb'))

    X_scaled = scaler_x.transform(data[features])
    X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1]) #GRU expect this(samples, timesteps, features)
    y_pred_scaled = model.predict(X_scaled)

    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test = scaler_y.inverse_transform(data['target'].values.reshape(-1, 1)) #reshape(-1,1) → converts 1D array to column format.

    last_price = data["current_close"].iloc[-1]
    predicted_price = last_price * (1 + y_pred[-1][0]) #Predicted Price = Current Price × (1 + Predicted Return)

    st.metric(
    "Predicted Close Price for Tomorrow",
    f"₹{predicted_price:.2f}")

    st.divider()

    st.subheader('Actual vs Predicted Close Price')

    train_size = int(len(data) * 0.8)
    X_test = X_scaled[train_size:]
    y_test = data['target'].values[train_size:]
    dates_test = data['Date'][train_size:]

    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    actual_price = data["current_close"].iloc[train_size:] * (1 + y_test.flatten())
    predicted_price_series = data["current_close"].iloc[train_size:] * (1 + y_pred.flatten()) #Calculates predicted stock price using predicted return

    comparison_df = pd.DataFrame({
        'Date': dates_test,
        'Actual Close Price': actual_price,
        'Predicted Close Price': predicted_price_series
    })

    fig2 = px.line(comparison_df, x='Date', y=['Actual Close Price', 'Predicted Close Price'], 
                   title=f'{selected_stock} Actual vs Predicted Close Price')
    fig2.update_traces(selector=dict(name="Predicted Price"), line=dict(dash="dash"))
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    st.subheader('Model Performance Metrics')

    mae = np.mean(np.abs(predicted_price_series - actual_price))
    mape = np.mean(np.abs((predicted_price_series - actual_price) / actual_price)) * 100 #(predicted - actual) / actual
    direction_accuracy = np.mean((np.sign(np.diff(actual_price)) == np.sign(np.diff(predicted_price_series)))) * 100 #Calculates the percentage of times the model correctly predicted the direction of price movement (up or down) by comparing the signs of the differences between consecutive actual and predicted prices.

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"₹{mae:.2f}")
    col2.metric("MAPE", f"{mape:.2f}%")
    col3.metric("Direction Accuracy", f"{direction_accuracy:.2f}%")

    st.divider()

    st.subheader('Multiple Day Predictions')
    recent_data = data.copy()
    days = st.slider('Select number of days to predict',1,7,1)

    if st.button('Predict Multiple Days'):
        feature_price = []
        last_price = data["current_close"].iloc[-1]
        current_features = X_scaled[-1].reshape(1, 1, -1) #last row of X_scaled

        for day in range(days):
            pred_scaled = model.predict(current_features)
            pred_return = scaler_y.inverse_transform(pred_scaled)[0][0] #predicted return for the day
            next_price = last_price * (1 + pred_return) #Predicted Price = Current
            feature_price.append(next_price)
            
            new_row = recent_data.iloc[-1].copy()
            new_row['Open'] = last_price
            new_row['Close'] = next_price
            new_row['High'] = next_price * 1.005   # small approximation
            new_row['Low'] = next_price * 0.995
            new_row['VWAP'] = next_price
            new_row['return'] = pred_return

            # Append and recalculate rolling features properly
            recent_data = pd.concat([recent_data, new_row.to_frame().T], ignore_index=True)
            recent_data['MA10'] = recent_data['Close'].rolling(10).mean()
            recent_data['MA50'] = recent_data['Close'].rolling(50).mean()
            recent_data['volatility'] = recent_data['Close'].rolling(10).std()
            ema12 = recent_data['Close'].ewm(span=12, adjust=False).mean()
            ema26 = recent_data['Close'].ewm(span=26, adjust=False).mean()
            recent_data['MACD'] = ema12 - ema26
            recent_data['High_Low_Ratio'] = recent_data['High'] / (recent_data['Low'] + 1e-10)
            recent_data['Close_Open_Ratio'] = recent_data['Close'] / (recent_data['Open'] + 1e-10)
            recent_data = recent_data.fillna(method='ffill')

            last_price = next_price

        future_dates = pd.date_range(
        start=data["Date"].iloc[-1],
        periods=days+1,
        freq="B")[1:]

        future_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Price": feature_price})

        st.dataframe(future_df)

        fig_future = px.line(
        future_df,
        x="Date",
        y="Predicted Price",
        title=f"{selected_stock} Next {days} Days Prediction")

        st.plotly_chart(fig_future, use_container_width=True)
