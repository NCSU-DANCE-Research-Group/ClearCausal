import os
import numpy as np
from parse_time import parse_start_end_time
from common import get_timestamps, Event
from setting import interval_sec

def find_change_point_percentile(response_times):
    min_change_point_val = 9000
    threshold = np.percentile(response_times, 97.5)
    for i, response_time in enumerate(response_times):
        if response_time > min_change_point_val and response_time > threshold:
            return response_time, i  # Return the value and the index of the first data point that exceeds the 90th percentile

    return None, None  # Return None if no data point exceeds the 90th percentile

def find_change_point(response_times):
    # find the index of the first data point that is the largest value
    max_index = np.argmax(response_times)
    return response_times[max_index], max_index

def get_response_times_timestamps():
    start_time, end_time = parse_start_end_time()
    timestamp = None
    response_times = []
    timestamps = []

    # read in the data from a csv file
    with open("data/response_time_stats_history.csv", "r") as fin:
        # Read the header and save to a dictionary
        header = fin.readline().strip().split(",")
        mapping = dict()
        for i, col in enumerate(header):
            mapping[col] = i
        for line in fin:
            data = line.strip().split(",")
            # Get the timestamp and value from the data
            timestamp = float(data[mapping["Timestamp"]])
            if timestamp < start_time or timestamp > end_time:
                continue
            timestamps.append(timestamp)
            # Handle the case where the value is not a float
            try:
                value = float(data[mapping["99.99%"]])
            except ValueError:
                continue
            response_times.append(value)
    return response_times, timestamps

def downsampling():
    response_times, timestamps = get_response_times_timestamps()
    target_timestamps = get_timestamps("avg_span_duration_online.csv")
    record = []
    for end_time in target_timestamps:
        sampling_rate = interval_sec * 10**6  # interval_sec seconds
        start_time = end_time - sampling_rate
        record.append(Event(start_time, end_time, 0, 0))
    for i, timestamp in enumerate(timestamps):
        for event in record:
            if event.start_time <= timestamp * 10**6 <= event.end_time:
                event.count += 1
                event.total_duration += response_times[i]
                break
    for event in record:
        if event.count != 0:
            event.total_duration /= event.count
    # export the data to a csv file
    with open("data/downsampled_response_time.csv", "w") as fout:
        fout.write("Timestamp,99.99%,Name\n")
        for event in record:
            fout.write(f"{event.end_time},{event.total_duration},downsampled_aggregated\n")
    return record

def detect_response_time_change_point():
    response_times, timestamps = get_response_times_timestamps()
    response_time, index = find_change_point(response_times)
    timestamp = 0
    if index is not None:
        timestamp = timestamps[index]
        print(f"A big jump was detected at timestamp {timestamp} with value {response_time}")
    else:
        print("No change point was detected")
    return int(timestamp)


if __name__ == "__main__":
    change_point = detect_response_time_change_point()
    print(change_point)
    # Check if the file exists, if not then do the downsampling to create the file
    if not os.path.exists("data/downsampled_response_time.csv"):
        downsampling()
