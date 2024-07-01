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
    print(f"time_series_1: {time_series_1},\ntime_series_2: {time_series_2},\nmi_res: {mutual_info:.2f}\n")
    return mutual_info

checkout = [10805.2, 9810.5, 10871.25, 11698.125, 12760.0, 9982.5, 10033.0, 11060.25, 9895.8, 10997.5, 9974.5, 10609.666666666666, 11619.0, 8111.0, 10943.5, 0.0, 0.0, 11659.0, 0.0, 12582.75, 11985.6, 0.0, 0.0, 0.0, 2760861.0, 523589.5, 0.0, 12670.0]
# convert all values to int
checkout = [int(x) for x in checkout]
email = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -3, 2, 27, 51, 48, 21, 1, 0]
currency = [4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2]

cut_num_values = 20
checkout = checkout[cut_num_values:]
email = email[cut_num_values:]
currency = currency[cut_num_values:]
print(f"After removing the first {cut_num_values} values: checkout vs email")
mi_res = calculate_mi(checkout, email)
print(f"After removing the first {cut_num_values} values: checkout vs currency")
mi_res = calculate_mi(checkout, currency)

print("Manual labelling: checkout vs email")
time_series_1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
time_series_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
mi_res = calculate_mi(time_series_1, time_series_2)

print("Manual labelling: checkout vs currency")
time_series_1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
time_series_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
mi_res = calculate_mi(time_series_1, time_series_2)

print("KNN labelling: checkout vs email")
time_series_1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
time_series_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
mi_res = calculate_mi(time_series_1, time_series_2)

print("KNN labelling: checkout vs currency")
time_series_1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
time_series_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
mi_res = calculate_mi(time_series_1, time_series_2)
