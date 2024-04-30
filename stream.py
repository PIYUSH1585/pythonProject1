import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Function to preprocess the dataset
def preprocess_data(df):
    # Remove commas from numerical values
    df = df.replace(',', '', regex=True)
    # Convert numerical columns to float, excluding 'End Date'
    for col in df.columns:
        if col != 'End Date':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna()  # Drop rows with NaN values after conversion

# Function to train the model
def train_model(df, currency_pair):
    X = df['End Date'].dt.strftime('%m%d%Y').astype(int).values.reshape(-1, 1)  # Convert date to integer representation
    y = df[currency_pair]  # Select the target column for training

    model = LinearRegression()
    model.fit(X, y)

    return model

# Function to predict currency exchange rate
def predict_currency_exchange_rate(model, current_date, year):
    predictions = []
    end_of_year = datetime(year, 12, 31)
    while current_date <= end_of_year:
        # Predict exchange rate for each day in the year
        next_date = current_date + timedelta(days=1)
        next_date_int = int(next_date.strftime('%m%d%Y'))
        prediction = model.predict([[next_date_int]])
        predictions.append((next_date, prediction[0]))
        current_date = next_date
    return predictions

# Load data
@st.cache_data
def load_data():
    # Load your dataset here
    # Replace 'exchange_rate_data.csv' with the path to your dataset
    df = pd.read_csv('exchange_rate_data.csv')
    return df

def main():
    st.title('Currency Exchange Rate Prediction')

    # Load data
    df = load_data()

    # Preprocess the data
    df = preprocess_data(df)

    # Filter data after 2/17/2017
    df = df[df['End Date'] > '2017-02-17']

    # Currency selection sidebar
    currency_pair = st.sidebar.selectbox('Select Currency Pair', df.columns[1:])

    # Option to show graph in sidebar
    show_graph = st.sidebar.checkbox('Show Graph')

    # Convert 'End Date' column to datetime objects
    df['End Date'] = pd.to_datetime(df['End Date'])

    # Train the model
    st.subheader('Model Training')
    model = train_model(df, currency_pair)

    # Predict currency exchange rate for 2017
    st.subheader(f'Predicted Exchange Rate for 2017 - {currency_pair}')
    current_date = datetime(2017, 1, 1)
    predictions = predict_currency_exchange_rate(model, current_date, 2017)
    for prediction_date, exchange_rate in predictions:
        st.write(f'{prediction_date.strftime("%Y-%m-%d")}: {exchange_rate}')

    # Plot graph if selected
    if show_graph:
        dates = [prediction[0] for prediction in predictions]
        exchange_rates = [prediction[1] for prediction in predictions]
        fig, ax = plt.subplots()
        ax.plot(dates, exchange_rates)
        ax.set_xlabel('Date')
        ax.set_ylabel('Exchange Rate')
        ax.set_title(f'Predicted Exchange Rate for 2017 - {currency_pair}')
        st.pyplot(fig)

if __name__ == "__main__":
    main()
    