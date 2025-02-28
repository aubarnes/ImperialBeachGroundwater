'''
Written by Kian Bagheri

CTD Processing Step 2:
(First, make sure to run through CTD Processing Step 1 to compute density profiles)
Filter the CTD profiles to only take the values as the CTD descends into the well
'''

import os
import pandas as pd

# Specify the folder where your CSV files are located
folder_path = '../data/'

# Specify the folder where you want to save processed CSV files
processed_folder_path = '../data/'

# Specify the folder where you want to save the summary CSV file
mean_csv_folder_path = '../data/'

# Create directories if they don't exist
os.makedirs(processed_folder_path, exist_ok=True)
os.makedirs(mean_csv_folder_path, exist_ok=True)

# Get a list of all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Create an empty dictionary to store the mean density values by date for each well
well_data = {}

# Loop through each CSV file in the folder
for file in csv_files:
    file_path = os.path.join(folder_path, file)

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Get the maximum depth for the file
    max_depth = df['Depth (m)'].max()

    # Find the index of the first occurrence of the max depth
    max_depth_index = df[df['Depth (m)'] == max_depth].index[0]

    # Keep only the rows up to and including the first occurrence of max depth
    filtered_df = df.iloc[:max_depth_index + 1]

    # Apply additional filters for density and valid depths (if necessary)
    filtered_df = filtered_df[(filtered_df['Density (kg/m³)'] >= 990) & (filtered_df['Depth (m)'] > 0)]

    # Create a new filename by appending '_processed' to the original filename
    new_file_path = os.path.join(processed_folder_path, file.replace('.csv', '_processed.csv'))

    # Save the cleaned DataFrame to the new CSV file
    filtered_df.to_csv(new_file_path, index=False)

    # Extract the date and well name from the filename
    file_name = file.replace('.csv', '')  # Remove '.csv' to get the base filename
    date = file_name.split('_')[0]  # Date is before the first underscore
    well_name = file_name.split('_')[1]  # Well name is between the first and second underscore

    # Group by well name and calculate the mean density for each well
    mean_density = filtered_df['Density (kg/m³)'].mean()

    # Add the mean density values to the well_data dictionary
    if date not in well_data:
        well_data[date] = {}  # Initialize a dictionary for the specific date if not already present
    well_data[date][well_name] = mean_density  # Store the mean density for each well

# Convert the well_data dictionary into a DataFrame
well_data_df = pd.DataFrame(well_data).T  # Transpose to have dates as rows and wells as columns

# Sort the DataFrame by date
well_data_df = well_data_df.sort_index()

# Calculate the average for each well column and add it as the last row
well_data_df.loc['Average'] = well_data_df.mean()

# Save the summary table with the average row to a new CSV file in the mean CSV folder
summary_file_path = os.path.join(mean_csv_folder_path, 'mean_density_by_well_and_date.csv')
well_data_df.to_csv(summary_file_path)

