from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
import numpy as np

USE_DISCRETE_MI = True # True: use discrete MI, False: use continuous MI

def calculate_mi(time_series_1, time_series_2):
    # print(f"time_series_1: {time_series_1}")
    # print(f"time_series_2: {time_series_2}")
    try:
        assert(len(time_series_1) == len(time_series_2))
    except AssertionError:
        print(f"len1: {len(time_series_1)}, len2: {len(time_series_2)}")
        return -1
    print(f"time_series_1: {time_series_1}, time_series_2: {time_series_2}")
    # Calculate mutual information
    if USE_DISCRETE_MI:
        mutual_info = normalized_mutual_info_score(time_series_1, time_series_2)
    else:
        # reshape to 2D array
        time_series_1 = np.array(time_series_1).reshape(-1, 1)
        time_series_2 = np.array(time_series_2)
        num_repeat = 1
        for i in range(num_repeat):
            mutual_info_sum = mutual_info_regression(time_series_1, time_series_2, random_state=i)[0]
        mutual_info = mutual_info_sum / num_repeat
    print(f"mi_res: {mutual_info}")
    return mutual_info

time_series_1 = [1, 2, 3, 4]
time_series_2 = [1, 2, 3, 4]
mi_res = calculate_mi(time_series_1, time_series_2)

time_series_1 = [1, 2, 3, 4]
time_series_2 = [2, 3, 4, 5]
mi_res = calculate_mi(time_series_1, time_series_2)

time_series_1 = [1, 2, 3, 4]
time_series_2 = [-1, -2, -3, -4]
mi_res = calculate_mi(time_series_1, time_series_2)

time_series_1 = [1, 2, 3, 4]
time_series_2 = [1, 3, 2, 4]
mi_res = calculate_mi(time_series_1, time_series_2)

time_series_1 = [1, 2, 3, 4]
time_series_2 = [1, 3, 4, 2]
mi_res = calculate_mi(time_series_1, time_series_2)

time_series_1 = [1, 2, 3, 4]
time_series_2 = [4, 3, 2, 1]
mi_res = calculate_mi(time_series_1, time_series_2)

time_series_1 = [1, 2, 3, 4, 5]
time_series_2 = [1, 2, 3, 4, 5]
mi_res = calculate_mi(time_series_1, time_series_2)

time_series_1 = [1, 2, 3, 4, 5]
time_series_2 = [5, 4, 3, 2, 1]
mi_res = calculate_mi(time_series_1, time_series_2)

time_series_1 = [1, 0, 1, 1, 0]
time_series_2 = [1, 1, 1, 0, 0]
mi_res = calculate_mi(time_series_1, time_series_2)

time_series_1 = [1, 0, 1, 1, 0]
time_series_2 = [1, 0, 1, 1, 0]
mi_res = calculate_mi(time_series_1, time_series_2)

time_series_1 = [25, 27, 26, 24, 28]
time_series_2 = [100, 120, 110, 90, 130]
mi_res = calculate_mi(time_series_1, time_series_2)

time_series_1 = [27, 24, 26, 25, 28]
time_series_2 = [120, 90, 110, 100, 130]
mi_res = calculate_mi(time_series_1, time_series_2)

time_series_1 = [27, 24, 26, 25, 28]
time_series_2 = [90, 120, 110, 100, 130]
mi_res = calculate_mi(time_series_1, time_series_2)

# time_series_1 = [38523.87, 50327.65, 51768.71, 46879.83, 35866.2, 59342.86, 17648.5, 90638.17, 51156.21, 41870.8, 38733.1, 0.0, 287040.38]
# time_series_2 = [7, 6, 5, 6, 6, 6, 6, 6, 6, 5, 4, 4, 3]
# mi_res = calculate_mi(time_series_1, time_series_2)

time_series_1 = [38523, 50327, 51768, 46879, 35866, 59342, 17648, 90638, 51156, 41870, 38733, 0, 287040]
time_series_2 = [7, 6, 5, 6, 6, 6, 6, 6, 6, 5, 4, 4, 3]
mi_res = calculate_mi(time_series_1, time_series_2)

# time_series_1 = [38523.87, 50327.65, 51768.71, 46879.83, 35866.2, 59342.86, 17648.5, 90638.17, 51156.21, 41870.8, 38733.1, 0.0, 287040.38]
# time_series_2 = [7, 6, 4, 4, 3, 5, 6, 6, 6, 6, 6, 6, 5]
# mi_res = calculate_mi(time_series_1, time_series_2)

time_series_1 = [38523, 50327, 51768, 46879, 35866, 59342, 17648, 90638, 51156, 41870, 38733, 0, 287040]
time_series_2 = [7, 6, 4, 4, 3, 5, 6, 6, 6, 6, 6, 6, 5]
mi_res = calculate_mi(time_series_1, time_series_2)

time_series_1 = [38523, 50327, 51768, 46879, 35866, 59342, 17648, 90638, 51156, 41870, 38733, 0, 287040]
time_series_2 = [6, 4, 4, 3, 7, 5, 6, 6, 6, 6, 6, 6, 5]
mi_res = calculate_mi(time_series_1, time_series_2)


num_digits = 6 # mimimum number of digits to get a good estimate of MI is 6
time_series_1 = [i for i in range(num_digits)]
time_series_2 = [i for i in range(num_digits)]
mi_res = calculate_mi(time_series_1, time_series_2)

time_series_1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
time_series_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
mi_res = calculate_mi(time_series_1, time_series_2)


time_series_1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
time_series_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
mi_res = calculate_mi(time_series_1, time_series_2)

time_series_1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
time_series_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
mi_res = calculate_mi(time_series_1, time_series_2)

time_series_1 = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
time_series_2 = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
mi_res = calculate_mi(time_series_1, time_series_2)

time_series_1 = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
time_series_2 = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
mi_res = calculate_mi(time_series_1, time_series_2)

time_series_1 = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
time_series_2 = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
mi_res = calculate_mi(time_series_1, time_series_2)