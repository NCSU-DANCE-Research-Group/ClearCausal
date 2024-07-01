import numpy as np
from fastdtw import fastdtw
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import mutual_info_score

def mutual_information(x, y, bins=10):
    counts, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = counts / np.sum(counts)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    pxy_nonzero = pxy[px > 0][:, py > 0]
    px_reshaped = px[px > 0]
    py_reshaped = py[py > 0]
    epsilon = np.finfo(float).eps  # Smallest positive float value
    mutual_info = np.sum(pxy_nonzero * np.log2(pxy_nonzero / (px_reshaped[:, np.newaxis] * py_reshaped) + epsilon))
    return mutual_info

# Generate two example time series data
x = np.array([1, 2, 3, 4, 3, 2, 1])  # Original time series
y = np.array([0, 1, 2, 3, 2, 1, 0])  # Shifted time series

x = np.array([11146, 9047, 11277, 0, 11120, 8666, 11061, 0, 11622, 9299, 9457, 9346, 9739, 11202, 11112, 9662, 8933, 9843, 9806, 9261, 0, 0, 0, 0, 0, 265015, 8313, 10420])
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -3, 3, 26, 49, 49, 26, 3, -4, 0]) #email
# y = np.array([4, 4, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2.]) # currency

mi = mutual_information(x, y)
print("Original Mutual Information:", mi)
mi_sklearn = mutual_info_score(x, y)
print("Original Mutual Information (sklearn):", mi_sklearn)
# use the normalized_mutual_info_score function
mi_sklearn = normalized_mutual_info_score(x, y)
print("Original Mutual Information (sklearn):", mi_sklearn)

# Align the time series using Dynamic Time Warping (DTW)
distance, path = fastdtw(x.flatten(), y.flatten())
print("Distance:", distance)
print("Path:", path)
aligned_x = np.array([x[i] for _, i in path])
aligned_y = np.array([y[j] for _, j in path])

print(aligned_x)
print(aligned_y)

# Calculate mutual information between the aligned time series
mi = mutual_information(aligned_x, aligned_y)
print("Mutual Information:", mi)
mi_sklearn = mutual_info_score(aligned_x, aligned_y)
print("Mutual Information (sklearn):", mi_sklearn)
# use the normalized_mutual_info_score function
mi_sklearn = normalized_mutual_info_score(aligned_x, aligned_y)
print("Mutual Information (sklearn):", mi_sklearn)
