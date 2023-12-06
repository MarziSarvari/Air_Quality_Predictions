# Air Quality Prediction Project Setup Guide

## Pre-Setup: Data Preparation

1. Visit [EEADMZ1 Downloads WebApp](https://eeadmz1-downloads-webapp.azurewebsites.net/)
2. Choose Country: Select "Czech Republic (Czechia)"
3. Cities: Keep as "All"
4. Pollutants: Choose "No2, Pm10, O3, Pm2.5, So2, CO"
5. Datasets: Leave as "All"
6. Click "Download" (Wait for download completion)
7. Click "Metadata" and select "Czechia" for country.
8. Download CSV file

## Pre-Setup Part 2: Project Initialization

1. Go to [Air Quality Predictions GitHub Repository](https://github.com/MarziSarvari/Air_Quality_Predictions)
2. Download "Streamlit_site.py" and "requirements.txt"
3. Place downloads in a short file path
4. Create a new Python project with a short file path
5. Move "Streamlit_site.py" and "requirements.txt" into this project
6. Install libraries from "requirements.txt"

## Setup

1. Ensure all required files are available
2. Create a folder named "data" inside your Python project
3. Move the downloaded files from the EU Environment Agency's "ParquetFiles" (specifically from the "E2a" folder) into the "data" folder within your project
4. Copy and paste the "DataExtract.csv" file from the EU Environment Agency into the "data" folder as well

## Running the Program Locally

1. Open a terminal
2. Navigate to your project directory
3. Run the command: `streamlit run Streamlit_site.py`

Upon execution, a window should open in your primary browser.

## Accessing Online Deployment

This project is also hosted online at [16.171.194.83](http://16.171.194.83).  Please note that the server might be slow. Access the deployed version by visiting the provided IP address. Another simplified version of project is also hosted online at [airquality-cz-predictions.streamlit.app](https://airquality-cz-predictions.streamlit.app). This version is the same but with limited data consiisting only Brno.

## Usage

1. Open the deployed site
2. Click "Get Predictions" to utilize the project functionality
