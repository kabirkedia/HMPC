import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline

# Read the CSV file into a DataFrame
df = pd.read_csv('levine.csv')

# Interpolate x and y coordinates
cs_x = CubicSpline(df.iloc[:, 0], df.iloc[:, 1])

# Calculate derivative to get velocities (dx/dt, dy/dt)
dx_dt = cs_x(df.index, 1)
dy_dt = cs_y(df.index, 1)

# Calculate curvature (k = (dx/dt * d2y/dt2 - d2x/dt2 * dy/dt) / ((dx/dt)^2 + (dy/dt)^2)^(3/2))
d2x_dt2 = cs_x(df.index, 2)
d2y_dt2 = cs_y(df.index, 2)
curvature = (dx_dt * d2y_dt2 - d2x_dt2 * dy_dt) / ((dx_dt**2 + dy_dt**2)**1.5)

# Calculate velocity (v = 1 / radius)
v = 1 / curvature

# Create a NumPy array with x, y, yaw, and velocity
data = np.vstack((cs_x(df.index), cs_y(df.index), v, df[3])).T

# Save the data to a .npy file
np.save('traj.npy', data)
