from collections import Counter
from sklearn.metrics import normalized_mutual_info_score
from sklearn.feature_selection import mutual_info_regression
import numpy as np
from setting import DATASET, ROOT_CAUSE_FUNCTION_ANALYSIS, BUGID

USE_DISCRETE_MI = True # True: use discrete MI, False: use continuous MI

def calculate_mi(time_series_1, time_series_2, debug=False):
    # print(f"time_series_1: {time_series_1}")
    # print(f"time_series_2: {time_series_2}")
    try:
        assert(len(time_series_1) == len(time_series_2))
    except AssertionError:
        print(f"len1: {len(time_series_1)}, len2: {len(time_series_2)}")
        return -1
    if debug:
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
    if debug:
        print(f"mi_res: {mutual_info}")
    return mutual_info

# start alginment of the anomalies
# To avoid the wrong anomalies, we need to add a continuous check
# If there are less than three continuous anomalies, we will ignore them
# If there are more than three continuous anomalies, we will start matching from the first anomaly

def check_continuous(data, min_continuous=1):
    # check if there are more than min_continuous anomalies
    # if yes, return the first anomaly
    # if no, return -1
    # if BUGID in {2}:
    #     min_continuous = 3 #1
    # if BUGID in {4, 5}:
    #     min_continuous = 2
    #     if ROOT_CAUSE_FUNCTION_ANALYSIS and BUGID == 4:
    #         min_continuous = 4
    #     if ROOT_CAUSE_FUNCTION_ANALYSIS and BUGID == 5:
    #         min_continuous = 4
    continuous = 0
    for i in range(len(data)):
        if data[i] == 1:
            continuous += 1
        else:
            continuous = 0
        if continuous >= min_continuous:
            return i - min_continuous + 1
    return -1

def add_data(data, num_point_before=10, num_point_after=10):
    # add num_point_before data before the first anomaly
    # add num_point_after data after the last anomaly
    # return the new data
    index = check_continuous(data)
    if index == -1:
        return data
    data_new = []
    actual_num_point_before = min(num_point_before, index)
    actual_num_point_after = min(num_point_after, len(data) - index)
    for i in range(max(0, index - num_point_before), index):
        data_new.append(data[i])
    for i in range(index, min(len(data), index + num_point_after)):
        data_new.append(data[i])
    return data_new, actual_num_point_before, actual_num_point_after

def align_anomaly(data1, data2, min_continuous=3, num_point_before=10, num_point_after=5):
    # check if satify the continuous condition
    # if not, we will ignore the anomaly
    # if yes, we will start matching from the first anomaly
    # return the aligned data
    index1 = check_continuous(data1, min_continuous)
    index2 = check_continuous(data2, min_continuous)
    data1_new = []
    data2_new = []
    if index1 == -1 or index2 == -1:
        # if there is no anomaly that satisifies our constraint, return the original data
        return data1, data2
    # add the data before the first anomaly and after the last anomaly
    data1_new, actual_num_point_before1, actual_num_point_after1 = add_data(data1, num_point_before, num_point_after)
    data2_new, actual_num_point_before2, actual_num_point_after2 = add_data(data2, num_point_before, num_point_after)
    # print_debug = False
    # if len(data1_new) != len(data2_new):
    #     print_debug = True
    #     print(f"len1: {len(data1_new)}, len2: {len(data2_new)}")
    #     print(f"actual_num_point_before1: {actual_num_point_before1}, actual_num_point_before2: {actual_num_point_before2}")
    #     print(f"actual_num_point_after1: {actual_num_point_after1}, actual_num_point_after2: {actual_num_point_after2}")
    #     print(f"data1_new: {data1_new}")
    #     print(f"data2_new: {data2_new}")
    # make sure the length of data1_new and data2_new are the same, remove the extra data
    if actual_num_point_before1 > actual_num_point_before2:
        data1_new = data1_new[actual_num_point_before1 - actual_num_point_before2:]
    elif actual_num_point_before1 < actual_num_point_before2:
        data2_new = data2_new[actual_num_point_before2 - actual_num_point_before1:]
    if actual_num_point_after1 > actual_num_point_after2:
        data1_new = data1_new[:-actual_num_point_after1 + actual_num_point_after2]
    elif actual_num_point_after1 < actual_num_point_after2:
        data2_new = data2_new[:-actual_num_point_after2 + actual_num_point_after1]
    # if print_debug:
    #     print(f"data1_new: {data1_new}")
    #     print(f"data2_new: {data2_new}")
    return data1_new, data2_new

if __name__ == "__main__":
    # using anomaly detection of 90 percentile
    # checkout = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
    # email = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]

    # using anomaly detection of 85 percentile 
    checkout = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    email = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
    # 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0
    calculate_mi(checkout, email, debug=True)
    checkout_new, email_new = align_anomaly(checkout, email, num_point_before=10, num_point_after=5)
    calculate_mi(checkout_new, email_new, debug=True)
    res_counter = Counter()
    for num_point_before in range(1, 30):
        for num_point_after in range(1, 30):
            checkout_new, email_new = align_anomaly(checkout, email, num_point_before=num_point_before, num_point_after=num_point_after)
            res = calculate_mi(checkout_new, email_new)
            res_counter[res] += 1
    # for key in res_counter:
    #     print(f"{key}: {res_counter[key]}")
    # print key from the smallest to the largest
    for key in sorted(res_counter):
        print(f"{key}: {res_counter[key]}")
