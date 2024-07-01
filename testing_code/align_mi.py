import numpy as np
from scipy import signal

# Generate two example time series data
x = np.sin(np.linspace(0, 10, 100))  # Original time series
y = np.sin(np.linspace(1, 11, 100))  # Shifted time series

# Calculate the cross-correlation between the two time series
cross_correlation = signal.correlate(x, y, mode='full')

# Find the time shift with the maximum cross-correlation
time_shift = np.argmax(cross_correlation) - (len(x) - 1)

# Shift the time series to align them
if time_shift > 0:
    x_aligned = np.concatenate((np.zeros(time_shift), x[:len(x) - time_shift]))
    y_aligned = y
else:
    x_aligned = x
    y_aligned = np.concatenate((np.zeros(-time_shift), y[-time_shift:]))

# Calculate mutual information between the aligned time series
def mutual_information(x, y, bins=10):
    counts, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    pxy = counts / np.sum(counts)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    pxy_nonzero = pxy[pxy > 0]
    px_reshaped = px[pxy > 0].reshape((-1, 1))
    mutual_info = np.sum(pxy_nonzero * np.log2(pxy_nonzero / (px_reshaped * py)))
    return mutual_info

mi = mutual_information(x_aligned, y_aligned)
print("Mutual Information:", mi)
