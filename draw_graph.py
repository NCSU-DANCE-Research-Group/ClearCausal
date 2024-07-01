import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from setting import interval_sec
from parse_time import parse_start_end_time

def select_values(values, indices):
    return [values[index] for index in indices]

def draw_graph(x:dict, y:dict, file:str, xlabel:str, ylabel:str):
    fontsize = 24
    markers = ['o', 'X', 's', 'D', '1', '^', '<', '>', 'P', 'v', '*', '2', 'd', 'x', '3', '4', 'H', '+', 'h', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    for i, key in enumerate(x.keys()):
        # make a new graph
        fig, ax = plt.subplots()
        if file != "response_time_stats_history":
            x[key] = [interval_sec / 60 * j for j, val in enumerate(x[key])]
        try:
            plt.plot(x[key], y[key], label=f"{key}", marker=markers[i%(len(markers))], markersize=10)
        except IndexError:
            print(f"i={i}, len(markers)= f{len(markers)}, len(keys)={len(x.keys())}, keys={x.keys()}")
            raise
        # Manually set the xticks for every two data points and rotate the xticks for 0 degrees
        if file != "response_time_stats_history":
            xticks = [x[key][j] for j in range(0, len(x[key]), 2)]
            plt.xticks(xticks, rotation=0, fontsize=fontsize)
        plt.xticks(rotation=0, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend(prop={'size': fontsize}, loc='upper left')
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        fig.set_size_inches(25, 7)
        plt.tight_layout()
        file_path = f"image/{file}/{key}.png"
        # make directories to the file path if not exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, format='png', dpi=100)
        plt.close()

def sigma_rule_anomaly_detection(time_series, threshold=1):
    mean = np.mean(time_series)
    std = np.std(time_series)
    anomalies = []
    for i, value in enumerate(time_series):
        z_score = (value - mean) / std
        if np.abs(z_score) > threshold:
            anomalies.append(i)
    return anomalies

def rolling_window_anomaly_detection(time_series, window_size=2, threshold=1):
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


def read_text_file(file_path, file):
    is_header = True
    mapping = dict()
    timestamps = defaultdict(list)
    values = defaultdict(list)
    # cared_pods = ["email", "checkout", "productcatalog", "frontend", "payment", "shipping", "currency", "cartservice", "jaeger"]
    # # if file == "avg_span_duration":
    # #     cared_pods = ["checkout"]
    # if file == "CPU_percentage_pod":
    #     cared_pods.extend(["session-db", "rabbitmq", "front-end", "carts"])
    # read_all_files = {"operation_duration", "CPU_percentage_pod", "response_time_stats_history", }
    # read header
    with open(file_path,"r") as fin:
        for line in fin:
            data = line.strip().split(",")
            if is_header:
                for i, col in enumerate(data):
                    mapping[col] = i
                is_header = False
                if "timestamp" not in mapping:
                    if file == "operation_duration":
                        mapping["timestamp"] = mapping["end_time"]
                    elif file in {"response_time_stats_history", "downsampled_response_time"}:
                        mapping["timestamp"] = mapping["Timestamp"]
                        mapping["pod"] = mapping["Name"]
                        mapping["value"] = mapping["99.99%"]
                    elif file == "span_count":
                        mapping["timestamp"] = mapping["end_time"]
                        mapping["value"] = mapping["count"]     
            else:
                key = f"{file}"
                if len(data) > 2:
                    if file not in ["disk_read_bytes_node", "CPU_percentage_node", "memory_node"]:
                        # namespace = data[mapping["namespace"]]
                        pod = data[mapping["pod"]]
                        # if file not in read_all_files and not any(pod.startswith(pod_name) for pod_name in cared_pods):
                        #     continue
                        key = pod
                    # else:
                    #     key = ",".join(data[:-2])
                timestamps[key].append(float(data[mapping["timestamp"]]))
                try:
                    values[key].append(float(data[mapping["value"]]))
                except ValueError:
                    values[key].append(0)
    xlabel = "Time (minutes)"
    ylabel = f"{file}"
    # # preprocess the data by ignoring the operations that has lower value in the 11th data point
    # if file == "operation_duration":
    #     all_operations = list(values.keys())
    #     for key in all_operations:
    #         if values[key][10] < 1 * 10**7:
    #             # Ignore this operation
    #             del timestamps[key]
    #             del values[key]
    if file == "response_time_stats_history":
        start_time, end_time = parse_start_end_time()
        # convert timestamps to the number of minutes from the start
        for i, name in enumerate(timestamps.keys()):
            new_timestamps = []
            new_values = []
            # make sure that the values have the same length
            for j, timestamp in enumerate(timestamps[name]):
                # convert to the number of minutes from the start
                # only within the range of start and end time
                if start_time <= timestamp <= end_time:
                    new_timestamps.append((timestamp - start_time)/60)
                    new_values.append(values[name][j])
            timestamps[name] = new_timestamps
            values[name] = new_values
        ylabel = "Response Time (milliseconds)"
    elif file == "CPU":
        ylabel = "CPU Usage Seconds (seconds)"
    elif file == "CPU_percentage_pod":
        # multiple each value by 100 to convert to the percentage
        # print(values.keys())
        for key in values.keys():
            values[key] = [val * 100 for val in values[key]]
        ylabel = "CPU Usage Percentage (%)"
    elif file == "memory":
        # convert to megabytes
        for key in values.keys():
            values[key] = [val / 10**6 for val in values[key]]
        ylabel = "Active Memory Usage (megabytes)"
    elif file.startswith("avg_span_duration"):
        # convert to milliseconds
        for key in values.keys():
            values[key] = [val / 1000 for val in values[key]]
        ylabel = "Average Span Duration (milliseconds)"
    elif file == "operation_duration":
        # convert to milliseconds
        for key in values.keys():
            values[key] = [val / 1000 for val in values[key]]
        ylabel = "Function Execution Time (milliseconds)"
    elif file == "span_count":
        ylabel = "Span Count"
    draw_graph(timestamps, values, file, xlabel, ylabel)


if __name__ == "__main__":
    # Folder Path
    path = "data"
    path = os.path.join(os.getcwd(), path)

    # ["CPU_percentage_node", "memory_node"]
    # ["avg_span_duration", "memory", "CPU_percentage_pod", "restart_total", "network_receive", "network_transmit", "memory_node", "avg_span_duration_ignore_other_services", "operation_duration", "response_time_stats_history", "span_count", "downsampled_response_time", "fs_write", "fs_read"]
    for file in ["avg_span_duration_online", "memory", "CPU_percentage_pod", "restart_total", "network_receive", "network_transmit", "response_time_stats_history", "avg_span_duration_offline", "operation_duration", "span_count", "downsampled_response_time", "fs_write", "fs_read"]:
        file_name = f"{file}.csv"
        curr_path = os.path.join(path, file_name)
        print(curr_path)
        read_text_file(curr_path, file)
