import numpy as np              # importing required modules for data download, scaling and training
import pandas as pd
import yfinance as yf                   #type: ignore       (required to avoid warning if packages aren't recognised but still loaded)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, LSTM, Dropout         #type: ignore          #importing keras classes needed for building LSTM-based neural network
from tensorflow.keras.models import Sequential          #type: ignore

def fetch_stock_data(symbol, num_days=2000):
    '''Function to download stock data of Yahoo finance from any company of choice
    Storing data in a Pandas dataframe and dropping rows with null values 
    '''         
    df = yf.download(symbol, period=f"{num_days}d")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df

def preprocess_data(df):
   #Function used to scale the data between 0 and 1, used to improve efficiency of neural network
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    return scaled_df, scaler


def dataset_maker(scaled_df):
    '''Dividing data into training and testing data
    Generating data pairs by taking 50 days' info as input and the following day's info as output'''
    train_data, train_labels = [], []
    for i in range(len(scaled_df) - 50):
        x_array = np.array(scaled_df.iloc[i:i+50, :])
        y_array = np.array(scaled_df.iloc[i+50, :])
        train_data.append(x_array)
        train_labels.append(y_array)
    #Tajing 80% of the generated data-label pairs as training data and 80% as testing data
    train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=0)
    return train_data, train_labels, test_data, test_labels

def train_model(train_data, train_labels):
    '''Building an LSTM-based model to capture patterns inn stock prices over time'''
    model = Sequential([LSTM(50, return_sequences=True), LSTM(50, return_sequences=True), LSTM(50), Dense(5), Dropout(0.2), Dense(5)])
    model.compile(optimizer='adamw', loss='mse')            #choosing optimiser and loss function

    model.fit(train_data, train_labels, batch_size=30, epochs=5)            #training model on training data
    return model


symbol = 'AAPL'             #taking Apple's data for training
df = fetch_stock_data(symbol)

#saving dataframe to .csv file for reference and visibility as to what data is like                
df.to_csv('stock_info.csv', index=True)
df = pd.read_csv('stock_info.csv')
df = df.drop(columns=['Date'], axis=1)              #removing date column, which was also written as a coluumn in the .csv file

scaled_df, scaler = preprocess_data(df)         #scaling the data values between 0 and 1

train_data, train_labels, test_data, test_labels = dataset_maker(scaled_df)     #generating data-label pairs

train_data = np.array(train_data)               #converting to numpy arrays
train_labels = np.array(train_labels)

test_data = np.array(test_data)
test_labels = np.array(test_labels)

model = train_model(train_data, train_labels)       #training model on chosen data by calling function
model.evaluate(test_data, test_labels)          #evaluating model performance; gives an insight on model's capabilities, helps tune hyperparameters
model.save('predictor_model.keras')             #saving model for use in stock price predictor program