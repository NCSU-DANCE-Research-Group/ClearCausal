import requests
import os
import csv

def retrieve_api(url, query=None):
    try:
        if query is not None:
            print(f"Your query was: {query}")
            response = requests.get(url, params=query)
        else:
            response = requests.get(url)
        response.raise_for_status()
        raw_response = response.json()
        return raw_response
    except requests.exceptions.HTTPError as errh:
        print(errh)
    except requests.exceptions.ConnectionError as errc:
        print(errc)
    except requests.exceptions.Timeout as errt:
        print(errt)
    except requests.exceptions.RequestException as err:
        print(err)

def save_data(filename, data, is_header=False, folder='data'):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{filename}.csv")
    if is_header:
        mode = 'w'
    else:
        mode = 'a+'
    with open(path, mode) as fout:
        for i in range(len(data)):
            data[i] = str(data[i])
        line = ",".join(data) + "\n"
        fout.write(line)

def get_timestamps(file_name)->list:
    root_data_folder = 'data/'
    unique_timestamps = set()

    with open(os.path.join(root_data_folder, file_name)) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            unique_timestamps.add(int(row['timestamp']))

    # sort the timestamps
    unique_timestamps = sorted(unique_timestamps)
    # print(len(unique_timestamps))
    # for timestamp in unique_timestamps:
    #     print(timestamp)
    # print(type(unique_timestamps))
    return unique_timestamps

class Event:
    def __init__(self, start_time, end_time, count, total_duration):
        self.start_time = start_time
        self.end_time = end_time
        self.count = count
        self.total_duration = total_duration