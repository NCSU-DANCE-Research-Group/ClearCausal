import numpy as np
import pandas as pd
from sklearn_som.som import SOM
from sklearn.preprocessing import StandardScaler

# Sample data generation: Random walk with anomalies
np.random.seed(42)
data_length = 1000
data = np.random.normal(0, 0.5, data_length).cumsum() + 50
# Inject anomalies
data[100] += 15
data[200] -= 15
data[800] += 20

data = np.array([1, 1, 1, 2, 1, 1, 3, 1, 2, 0, 1, 2, 2, 1, 2, 1, 2, 2, 3, 0, 2, 2, 1, 1, 103, 61, 111, 91, 97, 111, 127, 49, 148, 60, 1, 2, 2, 2])
data = data.reshape(-1, 1)



# Standardizing the series
scaler = StandardScaler()
series_scaled = scaler.fit_transform(data)

# Create and train SOM
som = SOM(m=20, n=20, dim=1, max_iter=200)
som.fit(series_scaled)

# Finding the BMUs and their distances
bmus = np.array([som.find_bmu(x) for x in series_scaled])  # Find BMU for each data point
bmu_indices = bmus[:, 0]  # BMU index in the map
bmu_distances = bmus[:, 1]  # Distance to the BMU

# Detecting outliers
threshold = np.percentile(bmu_distances, 80)  # Threshold as 95th percentile of distances
outliers = np.where(bmu_distances > threshold)[0]  # Indices of outliers

# Results
print("Indices of potential anomalies:", outliers)

# print the anomaly value
print("Anomaly value: ", data[outliers])
