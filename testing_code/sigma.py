import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigma_rule_anomaly_detection(time_series, threshold=3):
    mean = np.mean(time_series)
    std = np.std(time_series)
    anomalies = []
    for i, value in enumerate(time_series):
        z_score = (value - mean) / std
        if np.abs(z_score) > threshold:
            anomalies.append(i)
    return anomalies

def rolling_window_anomaly_detection(time_series, window_size=10, threshold=3):
    anomalies = []
    for i in range(len(time_series) - window_size + 1):
        window = time_series[i:i+window_size]
        mean = np.mean(window)
        std = np.std(window)
        for j in range(window_size):
            z_score = (window[j] - mean) / std
            if np.abs(z_score) > threshold:
                anomalies.append(i+j)
    return anomalies

# Generate a time series data with anomalies
np.random.seed(0)
time_series = np.concatenate((np.random.normal(0, 1, 100), np.array([10, 15, 20])))

# Detect anomalies using sigma rule method
anomalies_sigma_rule = sigma_rule_anomaly_detection(time_series)

# Detect anomalies using rolling window method
anomalies_rolling_window = rolling_window_anomaly_detection(time_series)

# Plot the time series data and highlight the anomalies
plt.plot(time_series, '-b')
plt.plot(anomalies_sigma_rule, time_series[anomalies_sigma_rule], 'ro', label='sigma rule anomalies')
plt.plot(anomalies_rolling_window, time_series[anomalies_rolling_window], 'go', label='rolling window anomalies')
plt.legend()
plt.show()
