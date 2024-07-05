'''Importing various moduules and libraries needed to download, store and manipulate data, plot graphs, send emails, verify imput,
and create temporary memory buffers to sve images'''
import numpy as np
import pandas as pd
import yfinance as yf                   #type: ignore
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model          #type: ignore
import matplotlib.pyplot as plt
import smtplib
import re
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import io

def fetch_stock_data(symbol, num_of_days):
    '''Used to download 50 days of data pertaining to any company of user's choice(most recent days available)
    Dropping all rows with null values, then ensuring that there at leaast 50 records with no null values by downloading more if necessary
    Using a try except block to ensure that errors are handled gracefully and info is provided if something is wrong with the download'''
    try:
        df = yf.download(symbol, period=f"{num_of_days}d")
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        while len(df) < 51:
            num_of_days = num_of_days + 1
            df = yf.download(symbol, period=f"{num_of_days}d")
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        return df
    except Exception as e:
        print(f"Error while downloading data : {e}")
        return df

def preprocess_data(df):
    '''Scaling values of the dataframe between 0 and 1, as the ML model trained operates on values between 0 and 1 and produces corresponding output
    Returning scaler object to use an inverse transform on model output to remove scaling and identify actual values of output'''
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    return scaled_df, scaler

def get_last_date(df):      #used to get the last date for which info is stored in the dataframe
    last_date = df.index[-1]
    return last_date

def validate_date(date):            #used to ensure that the dates entered by the user are of the form 'YYYY-MM-DD'
    proper_date = r'^\d{4}-\d{2}-\d{2}$'        #regular string which is used to match 4 digits-2 digits-2 digits

    if re.match(proper_date, date):         #using match function to validate date
        return True
    else:
        return False

def validate_email(recipient_email):            #used to perform basic validation check on entered email addresses
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'   #ensuring it is of the form <text>@<text>.<text with at least 2 characters>
    if re.match(pattern, recipient_email):
        return True
    else:
        return False

def send_alert(recipient_email, subject, body, sender_email="just3xp@gmail.com", sender_password="xyrk insy htfr ebap"):
    '''Used to generate an email, login to account whose credentials are given in header,
    and send the alert notification to user's mail address using smtplib'''
    message = MIMEMultipart()           #building message
    message['From'] = sender_email      #building message header
    message['To'] = recipient_email
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))         #adding message body

    try:                #using try except to handle errors gracefully in case mail fails to send(Example: mail address no longer exists)
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)

            #send email
            server.sendmail(sender_email, recipient_email, message.as_string())
    except Exception as e:
        print(f"Error while sending email!\n{e}")

def send_summary(recipient_email, subject, body, file_obj, sender_email="just3xp@gmail.com", sender_password="xyrk insy htfr ebap"):
    '''This is used to send an email containing the stock summary chosen by the user'''
    message = MIMEMultipart()       #building message
    message['From'] = sender_email          #adding message header
    message['To'] = recipient_email
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))         #adding body which contains the summary
    message.attach(MIMEImage(file_obj.read(), name='plots.png'))        #reading from file object which has plot stored in it
    
    try:                #using try except to graefully handle errors without terminating program
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)

            #send email
            server.sendmail(sender_email, recipient_email, message.as_string())
    except Exception as e:
        print(f"Error while sending email!\n{e}")



def stock_report(report_choice, df):
    '''Used to decide which type of summary user has asked for and calculate metrics and return them to user'''
    if report_choice == 'w':            #calculating means and standard deviations of each info metric for 7 days
        w_means = []
        for i in range(5):
            w_means.append(np.array(df.iloc[-7:, i]).mean())
        w_stds = []
        for i in range(5):
            w_stds.append(np.array(df.iloc[-7:, i]).std())
        return w_means, w_stds
    else:                               #calculating means and standard deviations of each info metric for 30 days
        m_means = []
        for i in range(5):
            m_means.append(np.array(df.iloc[-30:, i]).mean())
        m_stds = []
        for i in range(5):
            m_stds.append(np.array(df.iloc[-30:, i]).std())
        return m_means, m_stds
        
def main(df):
    '''Function is called if data is successfully fetched from yahoo finance
    Gives user the option to predict the stock prices for any day upto 3 days after the last day for which info is available
    Extending the duration any further may cause inaccuracies, as the predictions made for earlier days will be used as info for later days,
    causing error buildup
    '''
    last_date = get_last_date(df)           #identifying allowed date range
    min_allowed_date = last_date + pd.Timedelta(days=1)
    max_allowed_date = last_date + pd.Timedelta(days=3)

    scaled_df, scaler = preprocess_data(df)         #scaling data for passing into ML model

    print(f"Minimum allowed date : {min_allowed_date}")     #informing user regarding allowed dates
    print(f"Maximum allowed date : {max_allowed_date}")

    #accepting date from user
    date_of_choice = input("Enter date for which you would like to predict stock prices(format : YYYY-MM-DD) : ")
    while not validate_date(date_of_choice):        #ensuring valid date format
        print("Enter valid date format")
        date_of_choice = input("Enter date for which you would like to predict stock prices(format : YYYY-MM-DD) : ")

    date_of_choice = pd.to_datetime(date_of_choice)         #converting to datetime object

    while date_of_choice > max_allowed_date or date_of_choice < min_allowed_date:   #ensuring date falls within allowed range, asking user to enter another date otherwise
        print("Date entered is either completed or too far in the future to predict")
        date_of_choice = input("Enter date for which you would like to predict stock prices(format : YYYY-MM-DD) : ")

        while not validate_date(date_of_choice):        #validating user's input format
            print("Enter valid date format")
            date_of_choice = input("Enter date for which you would like to predict stock prices(format : YYYY-MM-DD) : ")

        date_of_choice = pd.to_datetime(date_of_choice)
                
    input_data = np.array(df.iloc[-50:])        #making predictions for first allowed day, as it is necessary regardless of day user chooses
    input_data = np.reshape(input_data, (1, 50, 5))
    predictions_1 = model.predict(input_data)
    predictions_1 = scaler.inverse_transform(predictions_1)     #undoing scaling to get actual values of predictions

    if date_of_choice == min_allowed_date:      #if user chooses first allowed date
        predicted_values = predictions_1[0]  #extracting the predicted values for the date
    elif date_of_choice == min_allowed_date + pd.Timedelta(days=1): #if user chooses second allowed date, we need to use predictions for the first day to predict the required ones
        input_data = np.concatenate((np.array(df.iloc[-49:]), predictions_1), axis=0)
        predictions_2 = model.predict(np.array(input_data).reshape(1, 50, 5))
        predictions_2 = scaler.inverse_transform(predictions_2)
        predicted_values = predictions_2[0]     #extracting predicted values
    else:
        input_data = np.concatenate((np.array(df.iloc[-49:]), predictions_1), axis=0)   #similarly, for 3rd date, predictions for 1st and 2nd dates are needed
        predictions_2 = model.predict(input_data.reshape(1, 50, 5))
        predictions_2 = scaler.inverse_transform(predictions_2)
        input_data = np.concatenate((np.array(df.iloc[-48:]), predictions_1, predictions_2), axis=0)
        predictions_3 = model.predict(input_data.reshape(1, 50, 5))
        predictions_3 = scaler.inverse_transform(predictions_3)
        predicted_values = predictions_3[0]     #extracting predicted values

    '''Due to inaccuracies in prediction by the model, if high price is lower than opening price, clearly the high price is the opening price itself
    So, we modify high and low price values according to opening and closing prices'''
    for price in [predicted_values[0], predicted_values[3]]:    
        if price > predicted_values[1]:
            predicted_values[1] = price
        if price < predicted_values[2]:
            predicted_values[2] = price

    print(f"Opening price = {predicted_values[0]}")         #printing predicted data
    print(f"High price = {predicted_values[1]}")
    print(f"Low price = {predicted_values[2]}")
    print(f"Closing price = {predicted_values[3]}")
    print(f"Volume of shares traded = {predicted_values[4]}")
    want_graph = input("Enter yes to view a graphical representation along with printed data, no to skip : ")
    
    while want_graph not in ['yes', 'no']:           #validating input
        print("Enter valid input")
        want_graph = input("Enter yes to view a graphical representation along with printed data, no to skip : ")
    if want_graph == 'yes':
        plt.figure(figsize=(12, 8))         #plotting a graph to represent trends
        #plotting a graph representing opening and closing prices, while also plotting moving averages for further analysis
        plt.plot(df.index, df['Open'], label='Open', color='blue', marker='o')
        plt.plot(df.index, df['Open'].rolling(10).mean(), label='10 day moving average - Open', color = 'purple', linestyle='--')
        plt.plot(df.index, df['Close'], label='Close', color='orange', marker='o')
        plt.plot(df.index, df['Close'].rolling(10).mean(), label='10 day moving average - Close', color = 'pink', linestyle='--')
        
        plt.title('Stock Metrics Over Time')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.grid(True)          #enabling grid for better understanding


        #plotting the predictions for Open and Close on the same graph to give a visual representation on increase/decrease
        plt.scatter(date_of_choice, predicted_values[0], color='blue', label='Predicted open', marker='x')
        plt.scatter(date_of_choice, predicted_values[3], color='orange', label='Predicted close', marker='x')
        plt.legend()            #adding legend
        plt.show()              #showing graphs

    #asking user to choose between the kind of stock summary they wish for
    report_choice = input("Enter w if you wish for a week's stock summary, m if you wish for a month's summary : ")
    while report_choice not in ['w', 'm']:      #validating input
        print("Enter valid input")
        report_choice = input("Enter w if you wish for a week's stock summary, m if you wish for a month's summary : ")
    req_means, req_stds = stock_report(report_choice, df)       #obtaining required summary info

    #asking if user would like the summaries to be mailed
    want_mail = input("Enter yes if you would like an email containing the report, no if you wouldn't : ")
    while want_mail not in ['yes', 'no']:           #validating input
        print("Enter valid input")
        want_mail = input("Enter yes if you would like an email containing the report, no if you wouldn't : ")
    if want_mail == 'yes':
        recipient_email = input("Enter an email address : ")        #asking for email ID if user opts for mail
        while not validate_email(recipient_email):      #performing basic validation on email address entered
            print("Enter valid email!")
            recipient_email = input("Enter an email address : ")

        subject = ""            #defining subject of email according to summary chosen by user
        if report_choice == 'w':
            subject = "7 days' stock summary"
        else:
            subject = "30 days' stock summary"


        #preparing body of email to display content of summary
        body = f"Mean opening price : {req_means[0]}\n"\
            f"Mean closing price : {req_means[3]}\n"\
            f"Mean high price : {req_means[1]}\n"\
            f"Mean low price : {req_means[2]}\n"\
            f"Mean volume of shares traded : {req_means[4]}\n"\
            "\n"\
            f"Standard deviation of opening price : {req_stds[0]}\n"\
            f"Standard deviation of high price : {req_stds[1]}\n"\
            f"Standard deviation of low price : {req_stds[2]}\n"\
            f"Standard deviation of closing price : {req_stds[3]}\n"\
            f"Standard deviation of volume of shares traded : {req_stds[4]}\n"
        
        file_obj = io.BytesIO()     #preparing a file-like object to store plots in memory(RAM) and avoid permament local storage 
        if report_choice == 'w':
            plt.plot(df.index[-7:], df.iloc[-7:, 0], color='blue', label='Opening price')   #plot for opening and closing prices
            plt.plot(df.index[-7:], df.iloc[-7:, 3], color='green', label='Closing price')

            plt.xlabel("Dates of 7 days'")
            plt.ylabel("Prices for 7 days")
            plt.grid(True)      #enabling grid

            #plotting predicted values for opening and closing prices
            plt.scatter(date_of_choice, predicted_values[0], color='blue', label='Predicted open', marker='x')
            plt.scatter(date_of_choice, predicted_values[3], color='green', label='Predicted close', marker='x')
            plt.legend()

            plt.savefig(file_obj, format='png')     #saving plot to file-like object

            file_obj.seek(0)            #resetting pointer to enable reading while sending email

            send_summary(recipient_email, subject, body, file_obj)      #sending email
            print("Stock summary has been emailed to given address, along with a graph showing predicted prices alongside some statistics")
        else:
            plt.figure(figsize=(12, 8))     #preparing similar plot for monthly summary(Here, since there are enough days, moving averages are added to plot)
            plt.plot(df.index[-30:], df.iloc[-30:, 0], color='blue', label='Opening price', marker='o')
            plt.plot(df.index[-30:], df.iloc[-30:, 0].rolling(5).mean(), label='5 day moving average - Open', color = 'purple', linestyle='--')
            plt.plot(df.index[-30:], df.iloc[-30:, 3], color='green', label='Closing price', marker='o')
            plt.plot(df.index[-30:], df.iloc[-30:, 3].rolling(5).mean(), label='5 day moving average - Close', color = 'pink', linestyle='--')
            
            plt.xlabel("Dates of 30 days'")
            plt.ylabel("Prices for 30 days")
            plt.grid(True)

            plt.scatter(date_of_choice, predicted_values[0], color='blue', label='Predicted open', marker='x')
            plt.scatter(date_of_choice, predicted_values[3], color='green', label='Predicted close', marker='x')
            plt.legend()

            plt.savefig(file_obj, format='png')            #similar use of file-like object

            file_obj.seek(0)

            send_summary(recipient_email, subject, body, file_obj)          #sending mail
            print("Stock summary has been emailed to given address, along with a graph showing predicted opening and closing prices alongside past trends")           

    else:       #if user opts out of email, then the info is printed onto the terminal itself
        summary = f"Mean opening price : {req_means[0]}\n"\
            f"Mean closing price : {req_means[3]}\n"\
            f"Mean high price : {req_means[1]}\n"\
            f"Mean low price : {req_means[2]}\n"\
            f"Mean volume of shares traded : {req_means[4]}\n"\
            "\n"\
            f"Standard deviation of opening price : {req_stds[0]}\n"\
            f"Standard deviation of high price : {req_stds[1]}\n"\
            f"Standard deviation of low price : {req_stds[2]}\n"\
            f"Standard deviation of closing price : {req_stds[3]}\n"\
            f"Standard deviation of volume of shares traded : {req_stds[4]}\n"
        print("Since you have opted out of receiving an email, summary is printed below")
        print(summary)
        
    #user has the option to choose whether to receive an email as a reminder that the predicted close price has exceeded some alert value set by them
    print("We can send you an email if the predicted closing price has exceeded an alert price of your choice")
    #asking if user would like to receive a email
    want_mail = input("Enter yes if you would like an email, enter no if you wouldn't : ")
    while want_mail not in ['yes', 'no']:       #validating input
        print("Enter valid input")
        want_mail = input("Enter yes if you would like an email, enter no if you wouldn't : ")
    if want_mail == 'yes':      #asking for alert price if user opts in
        alert_price = float(input("Enter alert price of your choice : "))

        recipient_email = input("Enter email address : ")       #getting mail address

        while not validate_email(recipient_email):      # performing basic validation on email address
            print("Enter valid email!")
            recipient_email = input("Enter an email address : ")

        if predicted_values[3] > alert_price:       #sending mail if conditions are met
            subject = "Alert: Closing price exceeded alert price"
            body = f"The predicted closing price({predicted_values[3]}) exceeds the alert price."

            send_alert(recipient_email, subject, body)
    print("Thank you for using our service!")
model = load_model('predictor_model.keras', custom_objects={'mse' : 'mse'})     #loading model for use(ADD MODEL PATH ON LOCAL MACHINE, UNLESS MODEL IS SAVED IN WORKING DIRECTORY)

print("Welcome to the stock price predictor")
#asking user to choose a company
symbol = input("Enter the ticker symbol for company whose stocks you wish to monitor(Eg. : AAPL for Apple) : ")

df = fetch_stock_data(symbol, num_of_days=50)       #fetching data from yahoo finance for chosen company

if df.empty:        #ensuring that if there is an issue with downloading, user is informed
    print("Sorry, there was an issue with downloading data, please try again")
else:               #if there are no issues, program proceeds and main() function is called
    main(df)