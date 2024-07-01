import dtaidistance
from fastdtw import fastdtw
import dtwalign
import numpy as np

# checkout service avg span duration
# email service CPU percentage
checkout = [11, 9, 8, 10, 11, 17, 10, 9, 8, 12, 9, 9, 31, 0, 9, 10, 9, 8, 8, 11, 97094, 76592, 56988, 1021, 11, 9, 9, 11]
email = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,44,57,57,12,0,0,0]

# find the distance between the two
dist = dtaidistance.dtw.distance(checkout, email)
print("Distance between checkout and email: ", dist)

distance, path = fastdtw(checkout, email)
print("Distance:", distance)
print("Path:", path)

if not isinstance(checkout, np.ndarray):
    checkout = np.array(checkout)
if not isinstance(email, np.ndarray):
    email = np.array(email)
res = dtwalign.dtw(checkout, email)
print(res.distance)
print(res.normalized_distance)
print(type(res.path))
# convert res.path to list of lists
path = res.path.tolist()
print(path)