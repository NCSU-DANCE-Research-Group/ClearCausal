import numpy as np
from parse_time import parse_start_end_time
from setting import DATASET

class JumpDetector:
    def __init__(self, window_size, threshold_factor, min_anomaly_value=4000):
        self.window_size = window_size
        self.threshold_factor = threshold_factor
        self.data = []
        self.min_anomaly_value = min_anomaly_value
    
    def add_data(self, new_data):
        self.data.append(new_data)
        if len(self.data) > self.window_size:
            self.data.pop(0)
        
        if len(self.data) < self.window_size:
            return -1
        
        diffs = np.diff(self.data)
        threshold = np.std(diffs) * self.threshold_factor
        if abs(diffs[-1]) > threshold and self.data[-1] > self.min_anomaly_value:
            return len(self.data) - 1
        else:
            return -1

def detect_response_time_change_point():
    window_size = 5
    threshold_factor = 2
    min_anomaly_value = 4000
    if DATASET == "socialnetwork":
        min_anomaly_value = 6000
    jump_detector = JumpDetector(window_size, threshold_factor, min_anomaly_value)
    start_time, end_time = parse_start_end_time()
    timestamp = None

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
            # Handle the case where the value is not a float
            try:
                value = float(data[mapping["99.99%"]])
            except ValueError:
                continue
            # Add the data to the jump detector
            jump_index = jump_detector.add_data(value)
            if jump_index != -1:
                print(f"A big jump was detected at timestamp {timestamp} with value {value}")
                break
    return int(timestamp) if timestamp is not None else 0


if __name__ == "__main__":
    change_point = detect_response_time_change_point()
    print(change_point)
