import numpy as np
import csv

# Load the NumPy array from the .npy file
array = np.load('trajectory22April.npy')

# Specify the path for the CSV file
csv_file = 'converted_array.csv'

# Write the array data to the CSV file
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(array)

