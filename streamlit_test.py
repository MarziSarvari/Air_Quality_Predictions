import pandas as pd
import os
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

import math
import time



def extract_sampling_point_id(samplingpoint):
    return samplingpoint.str.replace('CZ/', '', regex=False)

@st.cache_data
def load_data():
    data_folder = 'data'
    dfs = []

    for file_name in os.listdir(data_folder):
        if file_name.endswith(".parquet"):
            file_path = os.path.join(data_folder, file_name)
            df = pd.read_parquet(file_path)
            dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    additional_data = pd.read_csv('data\DataExtract.csv')   
    df['Sampling Point Id'] = extract_sampling_point_id(df['Samplingpoint'])
    # Merge the 'B-G Namespace' column from additional_data into df based on the 'Sampling Point Id' column
    df = pd.merge(df, additional_data[['Sampling Point Id', 'Air Quality Station Name', 'Air Pollutant', 'Longitude','Latitude',
        'Altitude', 'Altitude Unit', 'Air Quality Station Area','Measurement Method']], on='Sampling Point Id', how='left')
    df['City Name'] = df['Air Quality Station Name'].str.split('-').str[0]
    df = df.drop("Sampling Point Id", axis=1)
    df['Date'] = pd.to_datetime(df['Start'])
    df = df[['Value','Air Pollutant','Date','City Name']]
    df = df[df['Value']>=0]
    df = df.dropna()
    return df

df = load_data()


# List of air pollutants
pollutants = ['NO2', 'PM10', 'O3', 'PM2.5', 'SO2']
# Create separate DataFrames for each pollutant


# Now, you have separate DataFrames for each pollutant, with all other columns retained

Cities = ['Praha', 'Brno', 'Kucharovice', 'Mikulov',
       'Znojmo', 'Ceske Budejovice', 'Churanov', 'Hojna Voda',
       'Kocelovice', 'Prachatice', 'Tabor', 'Moravska Trebova ',
       'Pardubice', 'Pardubice Dukla', 'Svratouch', 'Hradec Kralove',
       'Krkonose', 'Polom', 'Trutnov ', 'Kostelni Myslova', 'Kosetice',
       'Trebic', 'Cheb', 'Prebuz', 'Sokolov', 'Ceska Lipa', 'Frydlant',
       'Horni Vitkov', 'Liberec', 'Sous', 'Uhelna', 'Jesenik', 'Olomouc',
       'Prerov', 'Prostejov', 'Kamenny Ujezd', 'Plzen', 'Primda',
       'Beroun', 'Kladno', 'Mlada Boleslav', 'Ondrejov', 'Rozdalovice',
       'Bily Kriz', 'Cervena hora', 'Cesky Tesin', 'Frydek', 'Havirov',
       'Ostrava', 'Opava', 'Studenka', 'Trinec', 'Vernovice', 'Chomutov',
       'Decin', 'Doksany', 'Krupka', 'Lom', 'Litomerice', 'Medenec',
       'Most', 'Rudolice v Horach', 'Sneznik', 'Teplice', 'Tusimice',
       'Usti n.L.', 'Stitna n.Vlari', 'Tesnovice', 'Uherske Hradiste',
       'Valasske Mezirici', 'Zlin']


message_placeholder = st.empty()

st.title('Air Quality Data Filter')

# Sidebar for filtering
st.sidebar.header('Filters')
selected_city = st.sidebar.selectbox('Select a City', Cities)
selected_pollutant = st.sidebar.selectbox('Select a Pollutant', pollutants)
text_to_display = selected_pollutant + " plot for " + selected_city
# Filter the data based on user input using str.contains
filtered_data = df[df['City Name'].str.contains(selected_city, case=False) & df['Air Pollutant'].str.contains(selected_pollutant, case=False)]
filtered_data = filtered_data.reset_index(drop=True)
if(len(filtered_data)>0):

    message_placeholder.text("Deleting Outliers...")

    value_column = filtered_data['Value']

    # Convert the 'Value' column to float
    value_column = value_column.astype(float)

    # Calculate the quartiles
    Q1 = value_column.quantile(0.1)
    Q3 = value_column.quantile(0.9)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = filtered_data[(filtered_data['Value'] < lower_bound) | (filtered_data['Value'] > upper_bound)]
    filtered_data = filtered_data[~filtered_data.index.isin(outliers.index)]
    filtered_data = filtered_data[['Date','Value']]
    filtered_data = filtered_data.set_index(pd.to_datetime(filtered_data['Date'], format='%Y-%m-%d %H:%M:%S'), drop=True)
    filtered_data = filtered_data.drop('Date', axis=1)

    filtered_data = filtered_data.resample('H').mean()
    message_placeholder.empty()

    message_placeholder.text("Training the model... Please wait.")

    train_data, test_data = filtered_data[0:int(len(filtered_data)*0.98)], filtered_data[int(len(filtered_data)*0.98):]


    train_arima = train_data['Value']
    test_arima = test_data['Value']


    history = [x for x in train_arima]
    y = test_arima
    # make first prediction
    predictions = list()
    model = ARIMA(history, order=(1,1,0))
    model_fit = model.fit()
    message_placeholder.empty()
    message_placeholder.text("Predicting Results... Please wait.")
    progress_bar = st.progress(0)  # Initialize the progress bar

    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
    history.append(y[0])

    total_iterations = len(y)
    for i in range(1, len(y)):
        # predict
        progress = i / total_iterations
        progress_bar.progress(progress)

        model = ARIMA(history, order=(1,1,0))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        # invert transformed prediction
        predictions.append(yhat)
        # observation
        obs = y[i]
        history.append(obs)
    progress_bar.empty()
    message_placeholder.empty()

    import pandas as pd
    import numpy as np

    # Define date_size and the tail_size for train_data
    date_size = int(len(filtered_data) * 0.1)
    tail_size = date_size - len(predictions)

    # Create date indices for train_data
    date_index = filtered_data.index[-date_size:]
    # Create the DataFrame for train_data
    train_data_df = pd.DataFrame({
        'Date': date_index,
        'train_data': pd.concat([train_data['Value'].tail(tail_size), pd.Series([np.nan] * (date_size - tail_size))], ignore_index=True)

    })

    # Create the DataFrame for y with NaN values and y values
    y_df = pd.DataFrame({
        'Date': date_index,
        'Real Data': pd.concat([pd.Series([np.nan] * tail_size), pd.Series(y)], ignore_index=True)
    })

    # Create the DataFrame for predictions with NaN values and prediction values
    predictions_df = pd.DataFrame({
        'Date': date_index,
        'Predictions': pd.concat([pd.Series([np.nan] * tail_size), pd.Series(predictions)], ignore_index=True)
    })
    
    combined_df = pd.concat([train_data_df, y_df, predictions_df], axis=1)
    # Drop the 'Date' column
    combined_df.drop('Date', axis=1, inplace=True)
    # Set the 'Date' column as the index
    combined_df.index = pd.to_datetime(date_index) 
    line_chart = st.line_chart(combined_df, use_container_width=True, color=['#008000', '#FF0000', '#0000FF'])

    # Calculate the minimum and maximum values of the filtered data
    min_value = filtered_data['Value'].min()
    max_value = filtered_data['Value'].max()

    # Display the minimum and maximum values in your Streamlit app
    st.write("Minimum Value in Filtered Data:", min_value)
    st.write("Maximum Value in Filtered Data:", max_value)
   
    try:
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        rmse = math.sqrt(mse)  # Use the calculated MSE for RMSE
    
        # Display the performance metrics in your Streamlit app
        st.write("Performance Metrics:")
        st.write("Mean Squared Error (MSE):", mse)
        st.write("Mean Absolute Error (MAE):", mae)
        st.write("Root Mean Squared Error (RMSE):", rmse)
    except ValueError as e:
        st.write("Error calculating performance metrics:", str(e))
else:
    st.write("There is no data for "+text_to_display)    