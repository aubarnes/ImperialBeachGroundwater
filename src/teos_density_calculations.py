'''
Written by Kian Bagheri

CTD Processing Step 1:
Compute density profiles from each CTD cast file
After this step, run through CTD Processing Step 2 to filter values
'''

import gsw
import numpy as np
import pandas as pd
import os

# Function to process a single CSV file
def process_file(file_path, output_folder):
    try:
        # Load CSV, skipping header rows and removing rows with missing essential values
        df = pd.read_csv(file_path, skiprows=31).dropna(
            subset=['Salinity (PSU)', 'Temperature (°C)', 'Pressure (psi)', 'Date / Time'])

        # Extract relevant columns as numpy arrays
        SA = np.array(df['Salinity (PSU)'])
        CT = np.array(df['Temperature (°C)'])
        P = np.array(df['Pressure (psi)']) * 0.689475728  # Convert psi to dbar

        # Compute density
        r1 = gsw.density.rho(SA, CT, P)

        g = 9.81
        P_1 = P * 10000
        delta_P = np.diff(P_1)

        # Initialize depth array
        depth = np.zeros_like(r1)
        for i in range(1, len(r1)):
            depth[i] = depth[i - 1] + (delta_P[i - 1] / (r1[i] * g))

        # Extract well name and date from the filename
        well, date = os.path.basename(file_path).split('_')[:2]

        # Save CSV file with calculated density and depth for this file
        output_df = pd.DataFrame({
            'Density (kg/m³)': r1,
            'Depth (m)': depth
        })

        # Define the output directory
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Output file path
        output_file = os.path.join(output_folder, f"{well}_{date}_density_depth.csv")
        output_df.to_csv(output_file, index=False)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return


# Define input and output directories
input_folder = # Define input and output directories
output_folder = #

# Process each CSV file in the folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.csv'):
        process_file(os.path.join(input_folder, file_name), output_folder)
