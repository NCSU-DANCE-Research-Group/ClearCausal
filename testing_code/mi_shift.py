from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt

def calculate_mi(time_series_1, time_series_2):
    # print(f"time_series_1: {time_series_1}")
    # print(f"time_series_2: {time_series_2}")
    try:
        assert(len(time_series_1) == len(time_series_2))
    except AssertionError:
        print(f"len1: {len(time_series_1)}, len2: {len(time_series_2)}")
        return -1
    # Calculate mutual information
    mutual_info = normalized_mutual_info_score(time_series_1, time_series_2)
    print(f"time_series_1: {time_series_1}, time_series_2: {time_series_2}, mi_res: {mutual_info:.2f}")
    return mutual_info

time_series_1 = [0, 1, 2, 3, 2, 1, 0]
time_series_2 = [1, 2, 3, 4, 3, 2, 1]
mi_res = calculate_mi(time_series_1, time_series_2)

time_series_1 = [1, 2, 3, 2, 1, 0, -1]
time_series_2 = [1, 2, 3, 4, 3, 2, 1]
mi_res = calculate_mi(time_series_1, time_series_2)

# Plot the graph
time_series_a = [1, 2, 3, 4, 3, 2, 1]
time_series_b = [0, 1, 2, 3, 2, 1, 0]
time_series_c = [1, 2, 3, 2, 1, 0, -1]

# Create x-axis values assuming each time series has the same length
x = range(len(time_series_a))

# Plot the time series
plt.plot(x, time_series_a, label='Time Series A')
plt.plot(x, time_series_b, label='Time Series B')
plt.plot(x, time_series_c, label='Time Series C')

# Add labels and title
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Time Series Data')

# Add a legend
plt.legend()

# Display the plot
plt.show()
