import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title='Stock Price Prediction', page_icon=':chart_with_upwards_trend:',layout='wide')

st.title('Stock Price Prediction')

st.markdown("""This project predicts the **next-day stock price** using historical stock market data 
and **Deep Learning models**.  

The system analyzes past price movements and learns patterns to forecast future prices.""")

st.divider()

st.subheader('Project Overview')

st.markdown("""
**Project Title:**  
Stock Price Prediction using Deep Learning

**Description:**  
This system predicts the next-day stock price using historical stock market data.  
Machine learning and deep learning models analyze past trends to forecast future prices.

**Models Used:**
- GRU (Gated Recurrent Unit)
- LSTM (Long Short-Term Memory)
- XGBoost
""")

st.divider()

st.subheader('Supported Stocks')

st.write("""
This project supports **NIFTY50 stocks**, including companies like:
- Reliance Industries
- TCS
- Infosys
- HDFC Bank
- ICICI Bank
- ITC
- Adani Ports
- Asian Paints
and other NIFTY50 companies.
""")

st.divider()

st.subheader('Data Source')

st.write("""
Stock market data is collected from **historical NSE datasets** containing:

- Date
- Open price
- High price
- Low price
- Close price
- Volume
- Deliverable quantity
""")

st.divider()
st.caption('This project is developed by Nishant')