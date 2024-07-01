import os
from collections import defaultdict
import sys
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import normalize
from setting import IGNORE_ERROR, interval_sec, DATASET, BUGID, ROOT_CAUSE_FUNCTION_ANALYSIS
from get_dependency_graph import load_dependency, get_neigh_services
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
from scipy import signal
from scipy.stats import rankdata, kendalltau
import math
from common import save_data
from combine_theme_result import combine_theme_result
from change_point import detect_response_time_change_point
from matplotlib import pyplot as plt
from moving_average import moving_average_with_padding
import dtaidistance
import dtwalign
from fastdtw import fastdtw
from anomaly_detection import anomaly_detection
from sklearn.feature_selection import mutual_info_regression
from simple_alignment import align_anomaly

if IGNORE_ERROR:
    def warn(*args, **kwargs):
        pass


    import warnings

    warnings.warn = warn

# "knn" or "meanstd" or "percentile" or None
ANOMALY_DETECTION_MODEL = "som"  # None percentile "som" ae
# True: remove non-priority pods, False: do not remove non-priority pods
DELETE_NON_PRIORITY_PODS = True

# True: align anomalies, False: do not align anomalies
ALIGN_ANOMALY = True  # 

DEBUG = False  # True: print debug info, False: do not print debug info
USE_DTW = False  # True: use DTW before MI, False: use MI only
FILTERING_MODE = None  # "moving_average" or "low_pass_filtering" or None
USE_DISCRETE_MI = True  # True: use discrete MI, False: use continuous MI

#FIXME: Duplicate in anomaly_detection.py
MINUTE_BEFORE_CHANGE_POINT = 10  # 12 minutes
MINUTE_AFTER_CHANGE_POINT = 3  # 3 minutes
if BUGID == 5:
    MINUTE_BEFORE_CHANGE_POINT = 13  # 12 minutes
    MINUTE_AFTER_CHANGE_POINT = 7  # 2 minutes

USE_DISCRETE_FEATURES = False  # True: use discrete features, False: use auto continuous features

FAULTY_SERVICE_NAME = None
ROOT_CAUSE_NAME = None
if ROOT_CAUSE_FUNCTION_ANALYSIS:
    if DATASET == "onlineboutique":
        if BUGID == 1:
            FAULTY_SERVICE_NAME = "emailservice"
            ROOT_CAUSE_NAME = "SendOrderConfirmation"
        elif BUGID == 10:
            FAULTY_SERVICE_NAME = "paymentservice"
            ROOT_CAUSE_NAME = "charge"
    elif DATASET == "socialnetwork":
        if BUGID == 2:
            FAULTY_SERVICE_NAME = "social-graph-service"
            ROOT_CAUSE_NAME = "GetFollowers"
        elif BUGID == 5:
            FAULTY_SERVICE_NAME = "post-storage-service"
            ROOT_CAUSE_NAME = "StorePost"
        elif BUGID == 6:
            FAULTY_SERVICE_NAME = "text-service"
            ROOT_CAUSE_NAME = "ComposeText"
        elif BUGID == 8:
            FAULTY_SERVICE_NAME = "user-mention-service"
            ROOT_CAUSE_NAME = "ComposeUserMentions"
    elif DATASET == "mediamicroservices":
        if BUGID == 3:
            FAULTY_SERVICE_NAME = "movie-review-service"
            ROOT_CAUSE_NAME = "UploadMovieReview"
        elif BUGID == 4:
            FAULTY_SERVICE_NAME = "user-review-service"
            ROOT_CAUSE_NAME = "UploadUserReview"
        elif BUGID == 7:
            FAULTY_SERVICE_NAME = "review-storage-service"
            ROOT_CAUSE_NAME = "StoreReview"
        elif BUGID == 9:
            FAULTY_SERVICE_NAME = "rating-service"
            ROOT_CAUSE_NAME = "UploadRating"
else:
    if DATASET == "onlineboutique":
        if BUGID == 1:
            FAULTY_SERVICE_NAME = "checkoutservice"
            ROOT_CAUSE_NAME = "emailservice"
        elif BUGID == 10:
            FAULTY_SERVICE_NAME = "checkoutservice"
            ROOT_CAUSE_NAME = "paymentservice"
    elif DATASET == "socialnetwork":
        if BUGID == 2:
            FAULTY_SERVICE_NAME = "home-timeline-service"
            ROOT_CAUSE_NAME = "social-graph-service"
        elif BUGID == 5:
            FAULTY_SERVICE_NAME = "compose-post-service"
            ROOT_CAUSE_NAME = "post-storage-service"
        elif BUGID == 6:
            FAULTY_SERVICE_NAME = "compose-post-service"
            ROOT_CAUSE_NAME = "text-service"
        elif BUGID == 8:
            FAULTY_SERVICE_NAME = "text-service"
            ROOT_CAUSE_NAME = "user-mention-service"
    elif DATASET == "mediamicroservices":
        if BUGID == 3:
            FAULTY_SERVICE_NAME = "compose-review-service"
            ROOT_CAUSE_NAME = "movie-review-service"
        elif BUGID == 4:
            FAULTY_SERVICE_NAME = "compose-review-service"
            ROOT_CAUSE_NAME = "user-review-service"
        elif BUGID == 7:
            FAULTY_SERVICE_NAME = "compose-review-service"
            ROOT_CAUSE_NAME = "review-storage-service"
        elif BUGID == 9:
            FAULTY_SERVICE_NAME = "movie-id-service"
            ROOT_CAUSE_NAME = "rating-service"


def calculate_mi(time_series_1, time_series_2):
    # print(f"time_series_1: {time_series_1}")
    # print(f"time_series_2: {time_series_2}")
    # try:
    #     assert(len(time_series_1) == len(time_series_2))
    # except AssertionError:
    #     print(f"len1: {len(time_series_1)}, len2: {len(time_series_2)}")
    #     return -1
    # Calculate mutual information
    if USE_DISCRETE_MI:
        mutual_info = normalized_mutual_info_score(time_series_1, time_series_2)
    else:
        # reshape to 2D array
        time_series_1 = np.array(time_series_1).reshape(-1, 1)
        time_series_2 = np.array(time_series_2)
        # discrete_features = [False]
        # if ANOMALY_DETECTION_MODEL is not None:
        #     discrete_features = [True]
        discrete_features = 'auto'
        if USE_DISCRETE_FEATURES:
            discrete_features = [True]
        num_repeat = 1
        for i in range(num_repeat):
            mutual_info_sum = \
            mutual_info_regression(time_series_1, time_series_2, discrete_features=discrete_features, random_state=i)[0]
        mutual_info = mutual_info_sum / num_repeat
    return mutual_info


def align_fastdtw(time_series_1, time_series_2):
    # print(f"time_series_1: {time_series_1}")
    # print(f"time_series_2: {time_series_2}")
    try:
        assert (len(time_series_1) == len(time_series_2))
    except AssertionError:
        print(f"len1: {len(time_series_1)}, len2: {len(time_series_2)}")
        return -1
    x = np.array(time_series_1)
    y = np.array(time_series_2)
    distance, path = fastdtw(x.flatten(), y.flatten())
    aligned_x = np.array([x[i] for _, i in path])
    aligned_y = np.array([y[j] for _, j in path])
    return aligned_x, aligned_y


def align_dtw2(time_series_1, time_series_2):
    y = np.array(time_series_1)
    y1 = np.array(time_series_2)
    dist, path = dtaidistance.dtw.warping_paths(y, y1)
    alignment = dtaidistance.dtw.best_path(path)
    aligned_y = [y[i] for i, j in alignment]
    aligned_y1 = [y1[j] for i, j in alignment]
    return aligned_y, aligned_y1


def align_dtw(x, y):
    # check if x is not a numpy array
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    res = dtwalign.dtw(x, y)
    x_path = res.path[:, 0]
    y_path = res.path[:, 1]
    x_aligned = x[x_path]
    y_aligned = y[y_path]
    if DEBUG:
        print(res.distance)
        print(res.normalized_distance)
        print(res.path)
        print(len(x))
        print(len(y))
        print(x.shape)
        print(y.shape)
        print(len(x_aligned))
        print(len(y_aligned))
        plt.plot(x_aligned, label="Checkout Average Span")
        plt.plot(y_aligned, label="Email CPU Percentage")
        plt.legend()
        plt.ylim(0, 100)
        plt.show()
    return x_aligned, y_aligned


def reorder_services(services: list, root_cause_name: str):
    # define a custom sorting function
    def sort_key(key):
        if root_cause_name in key:
            return 0  # put emailservice keys first
        else:
            return 1  # sort other keys alphabetically

    # sort the keys of the defaultdict using the custom function
    sorted_services = sorted(services, key=sort_key)
    return sorted_services


def reorder_metric(metric, root_cause_name):
    # define a custom sorting function
    def sort_key(key):
        if root_cause_name in key:
            return 0  # put emailservice keys first
        else:
            return 1  # sort other keys alphabetically

    # sort the keys of the defaultdict using the custom function
    sorted_keys = sorted(metric.keys(), key=sort_key)

    # create a new defaultdict with the sorted keys
    sorted_d = defaultdict(str, {k: metric[k] for k in sorted_keys})
    return sorted_d


def are_both_non_stationary(data1, data2, debug=False):
    result1 = adfuller(data1)
    result2 = adfuller(data2)
    # return result1[0] > result1[4]['5%'] and result2[0] > result2[4]['5%']
    if debug:
        if result1[1] <= 0.05:
            print("data1 is stationary")
            print(f"result1: {result1[1]}")
        if result2[1] <= 0.05:
            print("data2 is stationary")
            print(f"result2: {result2[1]}")
    return result1[1] > 0.05 and result2[1] > 0.05


def convert_int(metric):
    for key in metric:
        for i in range(len(metric[key])):
            metric[key][i] = int(metric[key][i])


def preprocess_metric_moving_average(metric, window_size=3):
    for key in metric:
        metric[key] = moving_average_with_padding(metric[key], window_size)


def preprocess_metric_low_pass_filtering(metric):
    b, a = signal.butter(3, 0.5)
    # b, a = signal.butter(3, 0.5)

    for key in metric:
        metric[key] = signal.filtfilt(b, a, metric[key])


def preprocess_multiply(metric, value):
    for key in metric:
        for i in range(len(metric[key])):
            metric[key][i] *= value


def adjust_zero(metric):
    for key in metric:
        # sample window 19>=val <= 24
        for i in range(len(metric[key])):
            # e.g 1 1 1 10 0 5
            if metric[key][i] == 0:
                metric[key][i] = metric[key][i - 1]
    return



def preprocess_metric(metric, metric_file):
    if metric_file == "CPU_percentage_pod":
        preprocess_multiply(metric, 100)  # 100
    # any metric beginning with avg_span_duration
    elif metric_file.startswith("avg_span_duration"):
        preprocess_multiply(metric, 1 / 100000)
    elif metric_file.startswith("operation_name"):
        preprocess_multiply(metric, 1000)
    else:
        pass

    # replace sudden 0 with previous val in window >9.555
    adjust_zero(metric)

    if FILTERING_MODE == "low_pass_filtering":
        preprocess_metric_low_pass_filtering(metric)
    elif FILTERING_MODE == "moving_average":
        preprocess_metric_moving_average(metric)
    else:
        pass
    if USE_DISCRETE_MI:
        # Force convertting to int
        convert_int(metric)


def read_text_file_node(file_path):
    is_header = True
    mapping = dict()
    values = defaultdict(list)
    counter = 0
    with open(file_path, "r") as fin:
        for line in fin:
            data = line.strip().split(",")
            if is_header:
                for i, col in enumerate(data):
                    mapping[col] = i
                is_header = False
            else:
                if len(data) >= 2:
                    counter += 1
                    values["values"].append(float(data[mapping["value"]]))
                    values["timestamps"].append(data[mapping["timestamp"]])
    return values


# def get_services_in_namespace(namespace, file_path="data", file="CPU_percentage_pod.csv"):
#     services = []
#     with open(file_path,"r") as fin:
#         for line in fin:
#             data = line.strip().split(",")
#             if data[0] == namespace:
#                 services.append(data[1])
#     return services

# def parse_list_string(list_string):
#     try:
#         # Use ast.literal_eval to safely parse the string into a list object
#         parsed_list = ast.literal_eval(list_string)
#         return parsed_list
#     except (SyntaxError, ValueError) as e:
#         print(f"Error parsing list string: {e}")
#         return []

def read_text_file_pod(file_path, file, start_time, end_time, mode=None, cared_namespace=None, cared_service=None,
                       debug_pods=None):
    is_header = True
    mapping = dict()
    timestamps = defaultdict(list)
    values = defaultdict(list)
    priority_pods = set()
    # read header
    with open(file_path, "r") as fin:
        for line in fin:
            data = line.strip().split(",")
            if is_header:
                for i, col in enumerate(data):
                    mapping[col] = i
                is_header = False
                if file == "operation_duration":
                    mapping["timestamp"] = mapping["end_time"]
            else:
                try:
                    if file in {"CPU_percentage_pod", "memory", "network_receive", "network_transmit"}:
                        timestamp = int(float(data[mapping["timestamp"]]) * 10 ** 6)
                        # print(timestamp)
                    else:
                        timestamp = int(data[mapping["timestamp"]])
                except ValueError:
                    print(f"file: {file}")
                    raise
                except KeyError:
                    print(f"file: {file}")
                    raise
                # Ignore the data outside the specified time range
                if timestamp < start_time or timestamp > end_time:
                    continue
                key = f"{file}"
                if "namespace" in mapping:
                    namespace = data[mapping["namespace"]]
                else:
                    namespace = None
                pod = data[mapping["pod"]]
                key = pod
                if debug_pods is not None and pod not in debug_pods:
                    continue
                if mode == "wnamespace":
                    if namespace == cared_namespace:
                        priority_pods.add(pod)
                    elif DELETE_NON_PRIORITY_PODS:
                        continue
                if mode == "wdependency":
                    if any(pod.startswith(pod_name) for pod_name in cared_service):
                        priority_pods.add(pod)
                    elif DELETE_NON_PRIORITY_PODS:
                        continue

                # try: 
                values[key].append(float(data[mapping["value"]]))
                # except ValueError:
                #     # need to parse if the value is like [0,0]
                #     # the first value is the value we need, average span duration
                #     # the second value is the number of span counts
                #     print(parse_list_string(data[mapping["value"]]))
                #     values[key].append(float(parse_list_string(data[mapping["value"]])[0]))
                #     raise
    return values, priority_pods


def normalization(inputlist):  # no actually normalization
    tmp = np.array(inputlist)
    return tmp


def normalization_real_unused(inputlist):
    tmp = np.array(inputlist)
    norm = normalize(tmp[:, np.newaxis], axis=0).ravel()
    return norm


def pod_mi_analysis_span_metric_node(avg_span_durations, metric):
    pods = list(avg_span_durations.keys())
    metric_values = metric["values"]
    for i in range(len(pods)):
        mi = 0
        mi = calculate_mi(avg_span_durations, metric_values)
        if mi > 0:
            print(f"{pods[i]}, {mi}")

        # Export the raw data to a csv file


def export_raw_data_to_csv(file_name, metric):
    folder = "export"
    os.makedirs(folder, exist_ok=True)
    file_path = f"./{folder}/{file_name}.csv"
    with open(file_path, "w") as fout:
        for key in metric:
            fout.write(f"{key},")
            fout.write(",".join([str(val) for val in metric[key]]))
            fout.write("\n")


# Tie breaking for the candidate with the same correlation value
# This will utilize the dependency graph to rank the candidate, the leaf node will be ranked higher
def tie_breaking(service_correlation, ranking, service_callcount_map, leaf_nodes, neigh_services_step_map):
    # Get the distance of the faulty service to each candidate
    # The candidate with the largest distance will be ranked higher
    # Find all the ties of ranking
    # print(service_distance_map)
    ranking_mapping = defaultdict(list)
    service_full_mapping = dict()
    for i, service_full in enumerate(service_correlation):
        parts = service_full.split("-")
        # remove the last two from parts
        service = "-".join(parts[:-2])
        service_full_mapping[service] = service_full
        ranking_mapping[ranking[i]].append((service, i))
    sorted_ranking = sorted(ranking_mapping.keys())
    # print(ranking_mapping)
    new_ranking = [0 for _ in range(len(ranking))]
    # Break the tie
    curr = 1
    for rank in sorted_ranking:
        services_pairs = ranking_mapping[rank]
        print(services_pairs)
        print(service_callcount_map)
        # Sort the list by the call count in descending order
        services_pairs.sort(key=lambda x: service_callcount_map[x[0]], reverse=True)
        new_service_pairs = []
        # Rank the leaf nodes higher by placing them to the top of the list
        for service, index in services_pairs:
            if service not in leaf_nodes:
                new_service_pairs.append((service, index))
            else:
                new_service_pairs.insert(0, (service, index))

        # Find the ties in the new service pairs list
        i = 0
        while i < len(new_service_pairs):
            service, index = new_service_pairs[i]
            tie = [service]
            while i + 1 < len(new_service_pairs):
                # We can compare the next service to see if they are the same in correclation value
                next_service, next_index = new_service_pairs[i + 1]
                if service_correlation[service_full_mapping[service]] == service_correlation[
                    service_full_mapping[next_service]]:
                    # They are the same, we need to break the tie
                    tie.append(next_service)
                    i += 1  # i moves to track number of ties
                else:
                    break
            # We have a tie if the length of tie is greater than 1
            if len(tie) > 1:
                print(f"tie: {tie}")
                # We need to break the tie by ranking the nodes that have the earliest change point
                # We need to find the change point of each service in the tie
                # change_points = []
                # for service in tie:
                #     change_points.append(service_change_point_map[service])
                # Sort the change points in ascending order        
                # # The neighbor closer to the symptom will be ranked higher
                service_step_index = []
                # neigh_services_step_map
                for service, index in new_service_pairs:  # services_pairs:  new_service_pairs
                    distance = neigh_services_step_map[service]
                    service_step_index.append((distance, index))
                service_step_index.sort(key=lambda x: (x[0], -x[1]), reverse=False)  # True
                # new_service_pairs.sort(key=lambda x: service_step_index[x[0]], reverse=False)
                # # Update the ranking
                for distance, index in service_step_index:
                    new_ranking[index] = curr
                    curr += 1
                    # i += 1  # Move to the next service, already moved counting ties
            else:
                # Produce the new ranking
                for service, index in new_service_pairs:
                    new_ranking[index] = curr
                    curr += 1
                i += 1  # Move to the next service
        
        # # Rank the leaf nodes higher
        # for service, index in services_pairs:
        #     if service in leaf_nodes:
        #         new_ranking[index] = curr
        #         curr += 1
        # # Rank the non-leaf nodes lower
        # for service, index in services_pairs:
        #     if service not in leaf_nodes:
        #         new_ranking[index] = curr
        #         curr += 1

        # service_distance_index = []
        # # Need to break the tie
        # # Find the distance of the faulty service to each candidate
        # # The candidate with the largest distance will be ranked higher. (lower, actually?) 
        # for service, index in services_pairs:
        #     distance = service_distance_map[service]
        #     service_distance_index.append((distance, index))
        # # print(service_distance_index)
        # # Sort the list by the distance
        # service_distance_index.sort(key=lambda x: (x[0], -x[1]), reverse=False)  # True
        # # print(service_distance_index)
        # # Update the ranking
        # for distance, index in service_distance_index:
        #     new_ranking[index] = curr
        #     curr += 

    print(new_ranking)
    return new_ranking


def draw_graph(y, file, xlabel: str, ylabel: str, anomly_labels=None):
    fontsize = 24
    markers = ['o', 'X', 's', 'D', '1', '^', '<', '>', 'P', 'v', '*', '2', 'd', 'x', '3', '4', 'H', '+', 'h', 0, 1, 2,
               3, 4, 5, 6, 7, 8, 9, 10, 11]
    # make a new graph
    # make a new graph
    fig, ax = plt.subplots()
    for i, key in enumerate(y.keys()):
        # create an x-axis label using the length of the y-axis
        x = [interval_sec / 60 * i for i in range(len(y[key]))]
        try:
            plt.plot(x, y[key], label=f"{key}", marker=markers[i % (len(markers))], markersize=10)
        except IndexError:
            print(f"i={i}, len(markers)= f{len(markers)}, len(keys)={len(x.keys())}, keys={x.keys()}")
            raise
        if anomly_labels is not None and key in anomly_labels and anomly_labels[key] is not None:
            for i, anomaly in enumerate(anomly_labels[key]):
                if anomaly:
                    # add dot with edge color of black and inside white to the original value, markersize=12
                    plt.plot(x[i], y[key][i], marker='s', markersize=12, markerfacecolor='white', markeredgewidth=2,
                             markeredgecolor='black')
        plt.xticks(rotation=0, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend(prop={'size': fontsize}, loc='upper left')
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.xlim(left=0)
        # if file == "evidence/checkoutservice":
        #     plt.ylim(bottom=0)
        # y axis should only show 0 or 1 and nothing in between when plotting the anomalies
        if file == "evidence/all_anomalies":
            plt.ylim(bottom=0, top=1)
        plt.ylim(bottom=0)
        # plt.pause(0.5)
        fig.set_size_inches(25, 7)
        plt.tight_layout()
        file_path = f"image/{file}/{key}.png"
        # make directories to the file path if not exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, format='png', dpi=100)
        plt.close()


def correlation_analysis_span_metric_pod(faulty_avg_span, metric, algorithm, priority_pods, debug_pods=None,
                                         analyze_root_cause_function=False, metric_file=None):
    algorithm = algorithm.lower()
    ignore_pval = True  # put this to true to ignore the p-value
    pods = list(metric.keys())
    res = dict()
    faulty_service = list(faulty_avg_span.keys())[0]
    max_val = 0
    if debug_pods is not None:
        all_data = dict()
        all_anomaly_labels = dict()
    for i in range(len(pods)):
        line = f"{pods[i]},"
        line += ",".join([str(val) for val in metric[pods[i]]])
        # print(line)
        val = 0
        time_series_1 = normalization(faulty_avg_span[faulty_service])
        time_series_2 = normalization(metric[pods[i]])
        anomaly_labels_1 = None
        anomaly_labels_2 = None
        high_percentile = 95  # 90
        low_percentile = 10  # 0
        if USE_DTW:
            time_series_1, time_series_2 = align_dtw(time_series_1, time_series_2)
        # time_series_1_new = [round(val, 2)/1000 for val in time_series_1]
        if ANOMALY_DETECTION_MODEL is not None:
            # percentile initialized above
            anomaly_labels_1 = anomaly_detection(time_series_1, ANOMALY_DETECTION_MODEL, high_percentile,
                                                 low_percentile, "faulty_avg_span")
            print("time_series_2", time_series_2)
            anomaly_labels_2 = anomaly_detection(time_series_2, ANOMALY_DETECTION_MODEL, high_percentile,
                                                 low_percentile, metric_file)
        if debug_pods is not None:
            # print the time series 1 but round to 2 decimal places	
            raw_data_1 = {f"{faulty_service}": time_series_1}
            raw_data_2 = {f"{pods[i]}": time_series_2}
            # all_data[faulty_service] = time_series_1
            all_data[pods[i]] = time_series_2
            all_anomaly_labels[pods[i]] = anomaly_labels_2
            print(f"time_series_1, data: {list(time_series_1)}")
            if not analyze_root_cause_function:
                ylabel_1 = "Average Span Duration (milliseconds)"
                if BUGID not in {2, 3}:
                    ylabel_2 = "CPU Percentage (%)"
                else:
                    ylabel_2 = "Memory"
            else:
                if BUGID not in {2, 3}:
                    ylabel_1 = "CPU Percentage (%)"
                else:
                    ylabel_2 = "Memory"
                ylabel_2 = "Average Function Execution Time (microseconds)"
            # ylabel_1 = "Average Span Duration (milliseconds)" if not analyze_root_cause_function else "CPU Percentage (%)"
            draw_graph(raw_data_1, f"evidence/{faulty_service}", "Time (min)", ylabel_1,
                       {f"{faulty_service}": anomaly_labels_1})
            print(f"Pod: {pods[i]}, data: {time_series_2}")
            # ylabel_2 = "CPU Percentage (%)" if not analyze_root_cause_function else "Average Function Execution Time (microseconds)"
            draw_graph(raw_data_2, f"evidence/{pods[i]}", "Time (min)", ylabel_2, {f"{pods[i]}": anomaly_labels_2})
        # check the length of the time series
        if len(time_series_1) != len(time_series_2):
            print(f"Length of time series 1: {len(time_series_1)}")
            print(f"Length of time series 2: {len(time_series_2)}")
        # print(f"Time series 1: {time_series_1}")
        # print(f"Time series 2: {time_series_2}")
        if ANOMALY_DETECTION_MODEL is not None:
            time_series_1 = anomaly_labels_1
            time_series_2 = anomaly_labels_2
            if ALIGN_ANOMALY:
                time_series_1, time_series_2 = align_anomaly(time_series_1, time_series_2)
            if debug_pods is not None:
                # print the actual values of the time series
                print(f"Time series 1: {time_series_1}")
                print(f"Time series 2: {time_series_2}")
        if algorithm == "mi":
            val = calculate_mi(time_series_1, time_series_2)
        elif algorithm == "pearson":
            val, p_value = pearsonr(time_series_1, time_series_2)
            if not ignore_pval:
                if p_value > 0.05:
                    val = 0
            val = abs(val)
        elif algorithm == "spearman":
            val, p_value = spearmanr(time_series_1, time_series_2)
            if not ignore_pval:
                if p_value > 0.05:
                    val = 0
            val = abs(val)
        elif algorithm == "cointegration":
            # comment out the cointegration
            # # check if the two time series are both non-stationary
            # # if not, then they are not cointegrated
            # if are_both_non_stationary(time_series_1, time_series_2, debug=False):
            #     # check if the two time series are cointegrated
            #     t, p_value, crit_value = coint(time_series_1, time_series_2)
            #     if p_value > 0.05:
            #         val = 0
            #     else:
            #         val = 1
            # else:
            #     # print("Not both are non-stationary")
            #     val = 0
            val = 0
        elif algorithm == "kendalltau":
            val, p_value = kendalltau(time_series_1, time_series_2)
            if not ignore_pval:
                if p_value > 0.05:
                    val = 0
            val = abs(val)
        max_val = max(max_val, val)
        res[pods[i]] = val

    if algorithm == "mi" and debug_pods is not None:
        draw_graph(all_data, f"evidence/all_pods", "Time (min)", "CPU Percentage (%)", all_anomaly_labels)
        if ANOMALY_DETECTION_MODEL is not None:
            draw_graph(all_anomaly_labels, f"evidence/all_anomalies", "Time (min)", "CPU Percentage Anomaly", None)

    # sort based on descending order of the MI
    # res = dict(sorted(res.items(), key=lambda item: item[1], reverse=True))
    # compute the ranking of the numbers in ascending order
    values = list(res.values())
    print(f"unique value: {len(set(values))}")
    for i in range(len(values)):
        if math.isnan(values[i]):
            values[i] = 0
        if pods[i] in priority_pods:
            values[i] += max(max_val, 0) + 0.1
    # print(values)
    ranking_list = len(values) - rankdata(values, method='max') + 1
    for i, key in enumerate(res):
        print(f"{key},{res[key]:.4f},{ranking_list[i]}")
    return res, ranking_list


def correlation_analysis_span_metric(symptom_file, metric_files, modes, algorithms, debug_pods=None,
                                     faulty_service_name="checkoutservice", root_cause_name="emailservice",
                                     analyze_root_cause_function=False):
    # Folder Path
    path = "data"  # "data"
    path = os.path.join(os.getcwd(), path)

    # # change point detection in seconds
    change_point = detect_response_time_change_point()
    # determine the start and end timestamps using the change point
    # start time should be 4 minute before the change point
    # end time should be 1 minute after the change point
    start_time = (change_point - MINUTE_BEFORE_CHANGE_POINT * 60) * 10 ** 6
    
    end_time = (change_point + MINUTE_AFTER_CHANGE_POINT * 60) * 10 ** 6

    # start_time = 0
    # end_time = float("inf")

    for mode in modes:
        cared_namespace = None
        cared_services = None

        faulty_service = None

        if mode == "wnamespace":
            cared_namespace = "default"
        elif mode == "wdependency":
            edges = load_dependency()
            neigh_service_map, leaf_nodes, neigh_services_step_map = get_neigh_services(edges, faulty_service_name, num_step=-1)
            cared_services = list(neigh_service_map.keys())
            cared_services = reorder_services(cared_services, root_cause_name)
            # service_distance_map

        # file = "avg_span_duration_ignore_other_services"
        file_name = f"{symptom_file}.csv"
        curr_path = os.path.join(path, file_name)
        print(curr_path)
        avg_span_durations, _ = read_text_file_pod(curr_path, symptom_file, start_time, end_time, None, None, None,
                                                   debug_pods)

        faulty_avg_span = None
        for service in avg_span_durations:
            if service.startswith(faulty_service_name):
                faulty_service = service
                faulty_avg_span = {f"{service}": avg_span_durations[service]}
                export_raw_data_to_csv(f"raw_{symptom_file}", faulty_avg_span)
                break
        # convert_int(faulty_avg_span)
        print(faulty_avg_span)
        preprocess_metric(faulty_avg_span, symptom_file)
        print(faulty_avg_span)

        res_folder = f"res_{mode}"
        # create the folder if it does not exist
        os.makedirs(res_folder, exist_ok=True)

        for metric_file in metric_files:
            file_name = f"{metric_file}.csv"
            curr_path = os.path.join(path, file_name)
            # print(f"\n{curr_path}")
            metric, priority_pods = read_text_file_pod(curr_path, metric_file, start_time, end_time, mode,
                                                       cared_namespace, cared_services, debug_pods)
            if analyze_root_cause_function:
                # remove the operation name that is not related to the faulty service
                for key in list(metric.keys()):
                    if faulty_service_name not in key:
                        del metric[key]
            # check the length of the pods in the metric
            print(f"len(metric): {len(metric)}")
            if mode == "pure":
                # check the length of the priority pods
                print(f"len(priority_pods): {len(priority_pods)}")
            preprocess_metric(metric, metric_file)  # comment it for now, will uncomment it later
            metric = reorder_metric(metric, root_cause_name)
            if mode == "wdependency":
                export_raw_data_to_csv(f"raw_{metric_file}", metric)

            res_header = ["service_name"]
            res_file = f"correlation_{metric_file}"
            res_correlation = defaultdict(list)

            for algorithm_index in range(len(algorithms)):
                algorithm = algorithms[algorithm_index]
                print(f"Used algorithm: {algorithm}")
                res_header.append(algorithm)
                res_header.append("ranking")

                service_correlation, ranking = correlation_analysis_span_metric_pod(faulty_avg_span, metric, algorithm,
                                                                                    priority_pods, debug_pods,
                                                                                    analyze_root_cause_function, metric_file)

                if mode == "wdependency":
                    new_ranking = tie_breaking(service_correlation, ranking, neigh_service_map, leaf_nodes, neigh_services_step_map)
                    print(f"Before the tie breaking:")
                    for i, key in enumerate(service_correlation):
                        print(f"{key},{service_correlation[key]:.4f},{ranking[i]}")
                    print(f"After the tie breaking:")
                    for i, key in enumerate(service_correlation):
                        print(f"{key},{service_correlation[key]:.4f},{new_ranking[i]}")
                    ranking = new_ranking

                for i, key in enumerate(service_correlation):
                    res_correlation[key].append(f"{service_correlation[key]:.4f},{ranking[i]}")
            print("\n")

            # save the results to the csv file
            save_data(res_file, res_header, is_header=True, folder=res_folder)
            for key in res_correlation:
                res_correlation[key].insert(0, key)
                save_data(res_file, res_correlation[key], is_header=False, folder=res_folder)

    if len(modes) > 1:
        combine_theme_result()


if __name__ == "__main__":
    # check the command line arguments, and retrieve the first argument
    if len(sys.argv) >= 2:
        DELETE_NON_PRIORITY_PODS = sys.argv[1].lower() == "true"
        print(f"Using the command line args, DELETE_NON_PRIORITY_PODS: {DELETE_NON_PRIORITY_PODS}")
    else:
        print("Please specify the value for the DELETE_NON_PRIORITY_PODS")
        exit(1)
    if ROOT_CAUSE_FUNCTION_ANALYSIS:
        if BUGID in {2, 3}:
            symptom_file = "memory"
        else:
            symptom_file = "CPU_percentage_pod"
        metric_files = ["operation_duration"]
        modes = ["pure"]
    else:
        symptom_file = "avg_span_duration_offline"
        metric_files = ["CPU_percentage_pod", "memory", "network_receive", "network_transmit", "operation_duration"]
        modes = ["pure", "wnamespace", "wdependency"]
    algorithms = ["MI", "pearson", "spearman", "cointegration", "kendalltau"]  # "cointegration", : issues with ae, som
    # debug_pods = ["alertmanager-main-1", "checkoutservice", "paymentservice-79dffbb687-rm9lp", "emailservice-58d767d8bc-95tdv"]
                #   "post-storage-service-86547bbf4f-h7cwz",'post-storage-service-86547bbf4f-h7cwz', 'media-service-5c4df49d86-5txlp', 'url-shorten-service-58d5c7c666-jkmbp', 'user-timeline-service-6c656f56-lgcdc', 'home-timeline-service-7888d6bcb8-9xlxc', 'social-graph-service-b64ccf865-5vt5x', 'unique-id-service-66656c57c8-5n27m', 'user-mention-service-5957dc9c5f-p6vqf', 'text-service-664f6d7c-44svr', 'compose-post-service-7b6c5fc858-vfxsd', 'user-service-69479c58fc-kgzqt']
    debug_pods = None # ["url-shorten-mongodb-7f6fc7999b-rqnvl", "url-shorten-mongodb","post-storage-service-86547bbf4f-h7cwz", "post-storage-service", "compose-post-service-7b6c5fc858-vfxsd", "compose-post-service"]  # ["checkoutservice", "emailservice", "productcatalogservice"]  # None ["checkoutservice-55fdd74c4f-dcpjx", "emailservice-696bc9f5df-42pn5",  "productcatalogservice-546b57ff5-f8gjs"]

    correlation_analysis_span_metric(symptom_file, metric_files, modes, algorithms, debug_pods, FAULTY_SERVICE_NAME,
                                     ROOT_CAUSE_NAME, ROOT_CAUSE_FUNCTION_ANALYSIS)
