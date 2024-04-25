import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


# Example usage
csv_file = 'levine.csv'  # Replace 'data.csv' with your CSV file name
data = np.loadtxt(csv_file, delimiter=',', skiprows=1)  # Assuming the first row is a header
x_data = data[:, 0]
y_data = data[:, 1]

# Create a CubicSpline object
spline = CubicSpline(x_data, y_data)

# Now you can evaluate the spline at any desired x value
x_values = np.linspace(min(x_data), max(x_data), 100)  # Example: 100 points between the min and max x values
y_values = spline(x_values)

plt.plot(x_data, y_data, 'o', label='Waypoints')
plt.plot(x_values, y_values, label='Spline')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Spline Interpolation')
plt.grid(True)
plt.show()
