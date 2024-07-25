import streamlit as st
import matplotlib.dates as mdates
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
st.set_page_config(layout="wide")

folder_path = '/local/s3377954/remote_ssh_files/log_analysis/cached_plot_data'

# Load dataset classes from .pkl files in the specified directory
def load_datasets(folder_path):
    datasets = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl'):
            with open(os.path.join(folder_path, filename), 'rb') as file:
                datasets[filename] = pickle.load(file)
    return datasets


# Function to plot the data
def plot_data(dataset, variant_ids, start_date, end_date):
    fig, ax = plt.subplots(figsize=(15, 8))  # Increase the size of the plot

    for variant_id in variant_ids:
        x_data = dataset.data_store['variants'][variant_id][0]['x']
        y_data = dataset.data_store['variants'][variant_id][0]['y']

        # Convert to pandas Series for easy date range filtering
        data = pd.Series(y_data, index=x_data)
        filtered_data = data[start_date:end_date]

        ax.plot(filtered_data.index, filtered_data.values, label=variant_id)

    ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator())  # Add all months to the graph
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y')) # Add months to the graph
    plt.xticks(rotation=90)  # Rotate x-axis labels 90 degrees
    st.pyplot(fig)


# Load datasets
datasets = load_datasets(folder_path)

# Streamlit interface
st.title("Dataset Dashboard")

# Dataset selection
dataset_name = st.selectbox("Select Dataset", list(datasets.keys()))
selected_dataset = datasets[dataset_name]

# Variant selection
variant_ids = list(selected_dataset.data_store['variants'].keys())
min_variant, max_variant = st.slider("Select range of variants", 1, len(variant_ids), (1, 5))
selected_variant_ids = variant_ids[min_variant - 1:max_variant]

# Date range selection
x_data = pd.Series(selected_dataset.data_store['variants'][variant_ids[0]][0]['x'])
start_date = st.date_input("Start Date", min(x_data))
end_date = st.date_input("End Date", max(x_data))

# Ensure valid date range
if start_date > end_date:
    st.error("Error: End date must fall after start date.")
else:
    # Plot the data
    plot_data(selected_dataset, selected_variant_ids, start_date, end_date)