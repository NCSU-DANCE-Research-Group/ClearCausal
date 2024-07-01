from dtaidistance import dtw
import numpy as np

y = np.random.randint(0, 10, 10)
y1 = y[1:]
print("y: ", y)
print("y1: ", y1)

dist = dtw.distance(y, y1)
print("Distance between y and y1: ", dist)

dist, path = dtw.warping_paths(y, y1)
print("Distance between y and y1: ", dist)
alignment = dtw.best_path(path)
print("alignment: ", alignment)

aligned_y = [y[i] for i, j in alignment]
aligned_y1 = [y1[j] for i, j in alignment]
print("aligned_y: ", aligned_y)
print("aligned_y1: ", aligned_y1)
