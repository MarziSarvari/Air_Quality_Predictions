import pandas as pd
import os
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import altair as alt

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
    df = df[['Value','Air Pollutant','Date','City Name','Air Quality Station Name']]
    df = df[df['Value']>=0]
    df = df.dropna()
    return df


df = load_data()





# Set the page title
st.title("Czech Air Quality Prediction App")



pollutants = ['NO2', 'PM10', 'O3', 'PM2.5', 'SO2', 'CO']
# Create separate DataFrames for each pollutant
#text_to_display = "NO2 plot for Praha" 

Stations=['Praha 6-Brevnov', 'Praha 8-Karlin', 'Praha 8-Kobylisy',
       'Praha 2-Legerova', 'Praha 4-Libus', 'Praha 10-Prumyslova',
       'Praha 1-n. Republiky', 'Praha 2-Riegrovy sady',
       'Praha 5-Stodulky', 'Praha 6-Suchdol', 'Praha 10-Vrsovice',
       'Praha 9-Vysocany', 'Brno - Detska nemocnice', 'Brno-Arboretum',
       'Brno-Lisen', 'Brno-Uvoz', 'Brno-Turany', 'Kucharovice',
       'Mikulov-Sedlec', 'Znojmo', 'Ceske Budejovice', 'Churanov',
       'Hojna Voda', 'Kocelovice', 'Prachatice', 'Tabor',
       'Moravska Trebova - Piaristicka.', 'Pardubice-Rosice',
       'Pardubice Dukla', 'Svratouch', 'Hradec Kralove-Brnen',
       'Hradec Kralove-observator', 'Krkonose-Rychory', 'Polom',
       'Trutnov - Tkalcovska', 'Kostelni Myslova', 'Kosetice', 'Trebic',
       'Cheb', 'Prebuz', 'Sokolov', 'Ceska Lipa', 'Frydlant',
       'Horni Vitkov', 'Liberec-Rochlice', 'Sous', 'Uhelna',
       'Jesenik-lazne', 'Olomouc-Hejcin', 'Prerov', 'Prostejov',
       'Kamenny Ujezd', 'Plzen-Slovany', 'Plzen-stred', 'Plzen-Lochotin',
       'Plzen-Doubravka', 'Primda', 'Beroun', 'Kladno-stred mesta',
       'Kladno-Svermov', 'Mlada Boleslav', 'Ondrejov',
       'Rozdalovice-Ruska', 'Bily Kriz', 'Cervena hora', 'Cesky Tesin',
       'Frydek-Mistek', 'Havirov', 'Ostrava-Ceskobratrsk',
       'Ostrava-Fifejdy', 'Ostrava-Poruba CHMU', 'Ostrava-Privoz',
       'Opava-Katerinky', 'Ostrava-Zabreh', 'Studenka', 'Trinec-Kanada',
       'Trinec-Kosmos', 'Vernovice', 'Chomutov', 'Decin', 'Doksany',
       'Krupka', 'Lom', 'Litomerice', 'Medenec', 'Most',
       'Rudolice v Horach', 'Sneznik', 'Teplice', 'Tusimice',
       'Usti n.L.-Vseboricka', 'Usti n.L.-Kockov', 'Usti n.L.-mesto',
       'Stitna n.Vlari', 'Tesnovice', 'Uherske Hradiste',
       'Valasske Mezirici', 'Zlin']

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

st.sidebar.header('Filters')
# Add filters for city and pollutants in the sidebar
selected_city = st.sidebar.selectbox('Select a City', Cities)
selected_station = st.sidebar.selectbox('Select a Station', Stations)

selected_pollutant = st.sidebar.selectbox('Select a Pollutant', pollutants)

checkbox_state = st.sidebar.toggle("Deleting Outliers")
prediction_granularity = st.sidebar.radio("Predict based on: ",["City", "Station"])

# Create a button in the sidebar to get predictions
get_predictions = st.sidebar.button("Get Predictions")


selected_date = st.sidebar.date_input("Select a date")
chart_days_number = st.sidebar.number_input('Number of days', min_value  = 2, max_value = 30, step=1)

get_charts = st.sidebar.button("Get Charts")

# Create a button in the sidebar to reveal or hide the "About the Site" content
show_about_site = st.sidebar.button("About the Site")
message_placeholder = st.empty()

# Create a session state to store the visibility state
if 'about_site_visible' not in st.session_state:
    st.session_state.about_site_visible = False
    st.session_state.prediction_visible = False
    st.session_state.chart_visible = False


# Check if the "About the Site" button is clicked
if show_about_site:
    # Toggle the visibility state
    st.session_state.about_site_visible = not st.session_state.about_site_visible
    # Reset the visibility state of the "About the Site" content
    st.session_state.about_site_visible = False
    st.session_state.chart_visible = False
    st.session_state.prediction_visible = False
    st.session_state.chart_visible = False

    st.write("Introducing the Czech Air Quality Prediction App, a powerful tool that empowers you to take control of your environment.")
    st.write("Our app leverages the latest data from the European Environmental Agency's Air Quality Download Service to deliver customized, real-time air quality predictions for locations across the Czech Republic, from January 1, 2023 and it is updated to the latest data.")
    st.write("Key Features:")
    st.write("🌆 City-Based and Station-Based Filtering: Tailor your predictions to your location of interest. Select your city, and get accurate forecasts that matter to you.")
    st.write("🏭 Pollutant-Specific Insights: Choose the pollutants you want to monitor. Whether it's particulate matter, ozone, or nitrogen dioxide, our app provides predictions tailored to your preferences.")
    st.write("🧹 Outlier Removal: Clean data is essential for accurate predictions. Our app identifies and removes outliers, ensuring the highest data quality.")
    st.write("📊 Time Series Predictions Along With Charts: See the future. Our app uses ARIMA models to generate time series predictions for air pollutant levels, enabling you to plan your activities, make informed decisions, and take precautions.")
    st.write("📏 Performance Metrics: Stay informed with comprehensive metrics, including Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE), giving you insights into the accuracy of our predictions.")
    st.write("🕒 Hourly Basis Predictions: Get predictions on an hourly basis, allowing you to track changes in air quality throughout the day.")
    st.write("The Czech Air Quality Prediction App is designed with you in mind. It puts the power of environmental insight in your hands, so you can confidently navigate your day, make informed choices, and contribute to a cleaner, healthier future.")
    st.write("Experience the difference. Try our app today, and enjoy it!")

       
if get_charts:
    st.session_state.about_site_visible = False
    st.session_state.prediction_visible = False
    if prediction_granularity == 'City':
        text_to_display = "Plot for " + selected_city + " city from " + str(selected_date)
        # Filter the data based on user input using str.contains
        filtered_data = df[df['City Name'].str.contains(selected_city, case=False)]
        filtered_data = filtered_data.reset_index(drop=True)
    else:
        text_to_display = "Plot for " + selected_station + " station from " + str(selected_date)
        # Filter the data based on user input using str.contains
        filtered_data = df[df['Air Quality Station Name'].str.contains(selected_station, case=False)]
        filtered_data = filtered_data.reset_index(drop=True)
    
    filtered_data = filtered_data[['Date', 'Air Pollutant', 'Value']]
    selected_date = pd.to_datetime(selected_date)

    
    #filtered_data = filtered_data.resample('H').mean()
    end_date = selected_date + pd.DateOffset(days=chart_days_number)
    selected_data = filtered_data[(filtered_data['Date'] >= selected_date) & (filtered_data['Date'] <= end_date)]
    selected_data = selected_data.groupby(['Air Pollutant', pd.Grouper(key='Date', freq='H')])['Value'].mean().reset_index()
    chart = alt.Chart(selected_data).mark_line().encode(
        x='Date',
        y='Value',
        color='Air Pollutant'
    ).properties(
        width=900,
        height=600
    )
    midnight_dates = pd.date_range(start=selected_date, end=end_date, freq='D', normalize=True)
    midnight_df = pd.DataFrame({'Midnight': midnight_dates})
    midnight_df['Midnight'] = midnight_df['Midnight'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Add vertical lines at midnight based on the midnight_df
    midnight_lines = alt.Chart(midnight_df).mark_rule(strokeDash=[2, 2], color='gray').encode(
        x='Midnight:T'
    )
    st.subheader(text_to_display)
    st.altair_chart(chart+midnight_lines)  # Display the chart

# Check if the "Get Predictions" button is clicked
if get_predictions:
    # Reset the visibility state of the "About the Site" content
    st.session_state.about_site_visible = False
    st.session_state.chart_visible = False
    if prediction_granularity == 'City':
        text_to_display = selected_pollutant + " plot for " + selected_city + " city"
        # Filter the data based on user input using str.contains
        filtered_data = df[df['City Name'].str.contains(selected_city, case=False) & df['Air Pollutant'].str.contains(selected_pollutant, case=False)]
        filtered_data = filtered_data.reset_index(drop=True)
    else:
        text_to_display = selected_pollutant + " plot for " + selected_station + " station"
        # Filter the data based on user input using str.contains
        filtered_data = df[df['Air Quality Station Name'].str.contains(selected_station, case=False) & df['Air Pollutant'].str.contains(selected_pollutant, case=False)]
        filtered_data = filtered_data.reset_index(drop=True)
    
    if(len(filtered_data)>0):
        if checkbox_state:
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
        date_index = filtered_data.index[-(date_size-1):]
        # Create the DataFrame for train_data
        train_data_df = pd.DataFrame({
            'Date': date_index,
            'train_data': pd.concat([train_data['Value'].tail(tail_size), pd.Series([np.nan] * (date_size - tail_size-1))], ignore_index=True)

        })

        # Create the DataFrame for y with NaN values and y values
        y_df = pd.DataFrame({
            'Date': date_index,
            'Real Data': pd.concat([pd.Series([np.nan] * (tail_size-1)), pd.Series(y)], ignore_index=True)
        })

        # Create the DataFrame for predictions with NaN values and prediction values
        predictions_df = pd.DataFrame({
            'Date': date_index,
            'Predictions': pd.concat([pd.Series([np.nan] * (tail_size-1)), pd.Series(predictions)], ignore_index=True)
        })

        combined_df = pd.concat([train_data_df, y_df, predictions_df], axis=1)
        # Drop the 'Date' column
        combined_df.drop('Date', axis=1, inplace=True)
        # Set the 'Date' column as the index
        combined_df.index = pd.to_datetime(date_index) 
        st.subheader(text_to_display)
        line_chart = st.line_chart(combined_df, use_container_width=True, color=['#008000', '#FF0000', '#0000FF'])

            
        try:
            mse = mean_squared_error(y, predictions)
            mae = mean_absolute_error(y, predictions)
            rmse = math.sqrt(mse)  # Use the calculated MSE for RMSE
            mean_y = np.mean(filtered_data['Value'])
            std_y = np.std(filtered_data['Value'])
            min_value = filtered_data['Value'].min()
            max_value = filtered_data['Value'].max()

            metrics = {
                "Metric": ["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)", "Mean ", "Standard Deviation", "Min", "Max"],
                "Value": [mse, mae, rmse, mean_y, std_y, min_value, max_value]
            }

            # Create a DataFrame from the dictionary
            df = pd.DataFrame(metrics)


            st.markdown("""
            <style>
            table {
                border-collapse: collapse;
                width: 50%;
            }

            th, td {
                text-align: left;
                padding: 8px;
            }

            th {
                background-color: #f2f2f2;
            }

            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            </style>
            """, unsafe_allow_html=True)

            # Display the styled table
            st.table(df)

        except ValueError as e:
            st.write("Error calculating performance metrics:", str(e))
        with st.spinner('Generating future predictions...'):
            filtered_data['Value'] = filtered_data['Value'].astype(float)
            train_arima = filtered_data['Value']
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            model = SARIMAX(train_arima, order=(1, 0, 1), seasonal_order=(2, 1, 1, 12))
            results = model.fit()
            period = 6
            #predictions = results.predict(len(train_arima), len(train_arima)+ period ).rename('SARIMA Predictions')
            predictions = [np.nan] * period

            for i in range(0,(period)):
                # Perform prediction for each step and update progress bar
                prediction_step = results.predict(len(train_arima) + i, len(train_arima) + i + 1)
                predictions[i] = prediction_step.iloc[0]


            tail_size = 60
            # Calculate the number of points to plot (5% of the tail)
            last_rows = filtered_data.tail(tail_size)  # Assuming you want the last 48 rows
            last_filtered_hours = filtered_data.tail(tail_size)


        # Create an index for the next 48 hours
        next_48_hours_index = pd.date_range(start=last_filtered_hours.index[1] + pd.Timedelta(hours=1), periods=(tail_size+period), freq='H')

        real_data = pd.DataFrame({
            'Date': next_48_hours_index,
            'Real Data': pd.concat([last_filtered_hours['Value'], pd.Series([np.nan] * (period))], ignore_index=True)

        })
        last_real_data = last_filtered_hours.iloc[-1]

        # Create the DataFrame for y with NaN values and y values
        predictions_future = pd.DataFrame({
            'Date': next_48_hours_index,
            'Predictions': pd.concat([pd.Series([np.nan] * (tail_size-1)),pd.Series(last_real_data['Value']), pd.Series(predictions)], ignore_index=True)
        })

        plotting_data = pd.concat([real_data, predictions_future], axis=1)
        plotting_data.drop('Date', axis=1, inplace=True)
        # Set the 'Date' column as the index
        plotting_data.index = pd.to_datetime(next_48_hours_index) 
        st.subheader(text_to_display+ " for Future")
        # Create a line chart using Streamlit for the entire dataset
        st.line_chart(plotting_data, use_container_width=True, color=['#008000', '#FF0000'])

    else:
        st.write("There is no data for "+text_to_display)    


# Display the "About the Site" content in the main content area based on the visibility state
#if st.session_state.about_site_visible:
    





