from collections import defaultdict
import json
import os
from common import get_timestamps, Event
from setting import interval_sec


def get_span_duration(end_time, debug=False):
    folder = 'data/traces/'
    res = defaultdict(float)
    # Iteratve through the files in the folder
    for json_file in os.listdir(folder):
        service = json_file.split('_')[0]
        print(json_file)
        # Open the file containing JSON data
        with open(os.path.join(folder, json_file), 'r') as file:
            # Load the JSON data from the file
            traces = json.load(file)

            duration_total = 0
            count = 0
            durations = []
            num_error = 0
            num_warning = 0
            num_error_warning = 0
            clean_total_duration = 0

            for trace in traces:
                # trace_id = trace['traceID']
                spans = trace['spans']
                # trace_warnings = trace['warnings']
                for span in spans:
                    duration = span['duration']
                    span_warnings = span['warnings']
                    has_warning = False
                    if span_warnings is not None:
                        has_warning = True
                    tags = span['tags']
                    duration_total += duration
                    count += 1
                    has_error = False
                    for tag in tags:
                        if tag['key'] == 'error' and tag['value']:
                            has_error = True
                            break
                    durations.append(duration)
                    if has_error and has_warning:
                        num_error_warning += 1
                    elif has_error:
                        num_error += 1
                    elif has_warning:
                        num_warning += 1
                    else:
                        clean_total_duration += duration
            if count != 0:
                avg_duration = duration_total / count
                if debug:
                    print(f'average duration of service {service}: {avg_duration}, count: {count}')
            else:
                avg_duration = 0
            res[service] = avg_duration
    return res

def get_span_count(timestamps, ignore_other_services=True, debug=False):
    folder = 'data/traces/'
    record = defaultdict(list)
    USE_SPAN_START_TIME = True
    # force an order of services
    services = []
    for json_file in os.listdir(folder):
        service = json_file.split('_')[0]
        print(service)
        services.append(service)
        for end_time in timestamps:
            sampling_rate = interval_sec * 10**6  # interval_sec seconds
            start_time = end_time - sampling_rate
            record[service].append(Event(start_time, end_time, 0, 0))
        # Open the file containing JSON data
        with open(os.path.join(folder, json_file), 'r') as file:
            # Load the JSON data from the file
            traces = json.load(file)

            duration_total = 0
            count = 0

            for trace in traces:
                processes = trace['processes']
                service_processid = None
                for process_id in processes:
                    service_name = processes[process_id]['serviceName']
                    if service_name != service:
                        continue
                    else:
                        service_processid = process_id
                        break
                # trace_id = trace['traceID']
                spans = trace['spans']
                if len(spans) == 1 and 'health' in spans[0]['operationName'].lower(): 
                    continue
                for span in spans:
                    span_start_time = span['startTime']
                    duration = span['duration']
                    span_end_time = span_start_time + duration
                    # # Ignore the spans that are not in the specified time range
                    # if span_end_time < start_time or span_start_time > end_time:
                    #     continue
                    # span_warnings = span['warnings']
                    # operation_name = span['operationName'].lower()
                    if ignore_other_services and span['processID'] != service_processid:
                        # print(service, operation_name)
                        continue
                    duration_total += duration
                    count += 1
                    for event in record[service]:
                        # use the start time of the span to determine which interval it belongs to
                        # if not(span_end_time < event.start_time or span_start_time > event.end_time):
                        if USE_SPAN_START_TIME:
                            if event.start_time <= span_start_time <= event.end_time:
                                event.count += 1
                                event.total_duration += duration
                                break
                        else:
                            # use the end time of the span to determine which interval it belongs to
                            if event.start_time <= span_end_time <= event.end_time: 
                                event.count += 1
                                event.total_duration += duration
                                break
    for service in services:
        for event in record[service]:
            if event.count != 0:
                event.total_duration /= event.count
    # export record to csv
    with open('data/span_count.csv', 'w') as f:
        f.write('pod,start_time,end_time,count,avg_duration\n')
        for service in services:
            for event in record[service]:
                if debug and event.count > 0:
                    print("service: ", service, "start_time: ", event.start_time, "end_time: ", event.end_time, "count: ", event.count, "avg_duration: ", event.total_duration)
                f.write(f'{service},{event.start_time},{event.end_time},{event.count},{event.total_duration}\n')
    return record

def get_span_duration_by_operation(timestamps, operations, debug=False):
    record = defaultdict(list)
    ignore_other_services = True
    for end_time in timestamps:
        sampling_rate = interval_sec * 10**6  # interval_sec seconds
        start_time = end_time - sampling_rate
        for operation in operations:
            record[operation].append(Event(start_time, end_time, 0, 0))
    folder = 'data/traces/'
    # Iteratve through the files in the folder
    for json_file in os.listdir(folder):
        service = json_file.split('_')[0]
        print(json_file)
        # Open the file containing JSON data
        with open(os.path.join(folder, json_file), 'r') as file:
            # Load the JSON data from the file
            traces = json.load(file)
            for trace in traces:
                processes = trace['processes']
                service_processid = None
                for process_id in processes:
                    service_name = processes[process_id]['serviceName']
                    if service_name != service:
                        continue
                    else:
                        service_processid = process_id
                        break
                spans = trace['spans']
                # trace_warnings = trace['warnings']
                for span in spans:
                    if ignore_other_services and span['processID'] != service_processid:
                        # print(service, operation_name)
                        continue
                    span_start_time = span['startTime']
                    duration = span['duration']
                    # span_warnings = span['warnings']
                    operation_name = span['operationName']
                    operation_name = process_operation_name(service, operation_name)
                    # Search for the corresponding event
                    for event in record[operation_name]:
                        if event.start_time <= span_start_time <= event.end_time:
                            event.count += 1
                            event.total_duration += duration
                            break
    for operation in operations:
        for event in record[operation]:
            if event.count != 0:
                event.total_duration /= event.count
    # export record to csv
    with open('data/operation_duration.csv', 'w') as f:
        f.write('pod,start_time,end_time,count,value\n')
        for operation in operations:
            for event in record[operation]:
                f.write(f'{operation},{event.start_time},{event.end_time},{event.count},{event.total_duration}\n')
    return record

def process_operation_name(service, operation_name):
    prefix_to_remove = ['hipstershop.']
    if service not in operation_name.lower():
        if operation_name[0] != '/':
            operation_name = '/' + operation_name
        operation_name = service + operation_name
    else: # the operation name contains the service name
        operation_name = operation_name.lower()
        if operation_name[0] == '/':
            operation_name = operation_name[1:]
        for prefix in prefix_to_remove:
            operation_name = operation_name.split(prefix)[-1]
    return operation_name

def get_all_operation():
    folder = 'data/traces/'
    unique_operation = set()
    ignore_healthchecking = False
    ignore_other_services = True
    # Iteratve through the files in the folder
    for json_file in os.listdir(folder):
        print(json_file)
        
        # Open the file containing JSON data
        with open(os.path.join(folder, json_file), 'r') as file:
            # Load the JSON data from the file
            traces = json.load(file)
            service = json_file.split('_')[0]
            # Print the loaded data
            print(len(traces))
            for trace in traces:
                spans = trace['spans']
                processes = trace['processes']
                service_processid = None
                for process_id in processes:
                    service_name = processes[process_id]['serviceName']
                    if service_name != service:
                        continue
                    else:
                        service_processid = process_id
                        break
                for span in spans:
                    if ignore_other_services and span['processID'] != service_processid:
                        # print(service, operation_name)
                        continue
                    operation_name = span['operationName']
                    # Ignore those health check operations
                    if ignore_healthchecking and "health" in operation_name.lower():
                        continue
                    operation_name = process_operation_name(service, operation_name)
                    unique_operation.add(operation_name)
    print(len(unique_operation))
    print(unique_operation)
    return unique_operation


if __name__ == "__main__":
    timestamps = get_timestamps("avg_span_duration_online.csv")
    debug = True
    # print(timestamps)
    get_span_count(timestamps, debug=debug)
    # print("Be sure to uncomment the following lines!")
    unique_operation = get_all_operation()
    get_span_duration_by_operation(timestamps, unique_operation, debug=debug)
    
