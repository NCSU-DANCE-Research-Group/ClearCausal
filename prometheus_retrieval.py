import os
from common import retrieve_api, save_data
from setting import interval_sec
from span_duration import SpanDuration
import time
import json

total_mins = 20  # minutes


def save_traces(services, start_time, end_time):
    port = 16686  # jaeger
    host = 'localhost'
    print(start_time, end_time)
    for service in services:
        url = f"http://{host}:{port}/api/traces?service={service}&start={start_time}&end={end_time}&limit=100000"
        raw_response = retrieve_api(url, None)
        if raw_response is None:
            print(f"No result returned for the query: {query}")
        else:
            # save the raw response to a file
            filename = f"{service}_{start_time}_{end_time}.json"
            folder = "data/traces"
            os.makedirs(folder, exist_ok=True)
            file_path = os.path.join(folder, filename)
            with open(file_path, 'w') as fout:
                json.dump(raw_response['data'], fout)


def convert_timestamp(linux_timestamp):
    return int(10**6 * linux_timestamp)


def parse_result(query, metric_keys, filename, is_first=False, limit=None, debug=False):
    port = 9090
    host = 'localhost'
    url = f"http://{host}:{port}/api/v1/query"

    raw_response = retrieve_api(url, query)
    if raw_response is None:
        print(f"No result returned for the query: {query}")
    else:
        items = raw_response['data']['result']
        header_list = metric_keys.copy()
        header_list.extend(['timestamp', 'value'])
        print("\t".join(header_list))
        if is_first:
            save_data(filename, header_list)
        counter = 0
        for item in items:
            counter += 1
            metric = item['metric']
            if debug:
                print(metric)
            row = []
            try:
                for key in metric_keys:
                    row.append(metric[key])
                raw_value = item['value']
                row.extend(raw_value)
                print(row)
                save_data(filename, row)
            except KeyError as e:
                print(f"No value found in item {item}")
            if limit is not None and counter >= limit:
                break


if __name__ == "__main__":
    counter = 0
    num_iters = int(total_mins * 60 / interval_sec)
    span_duration = SpanDuration()
    start_time = time.time()
    while counter < num_iters:
        is_first = (counter == 0)
        span_end_time = round(time.time()*10**6)

        # # Memory usage percentage
        # query = {'query': 'node_memory_Active_bytes/node_memory_MemTotal_bytes*100'}
        # parse_result(query, [], 'memory_node', is_first=is_first)

        # Memory usage by pod, namespace
        query = {
            'query': 'sum(container_memory_usage_bytes{container!=""}) by (namespace, pod)'}
        parse_result(query, ['namespace', 'pod'], 'memory', is_first=is_first)

        # # File system avilable percentage
        # query={'query':'node_filesystem_avail_bytes/node_filesystem_size_bytes*100'}
        # parse_result(query, ['namespace', 'pod', 'mountpoint'], 'fs_available', is_first=is_first)

        # # CPU percentage by node name
        # query = {
        #     'query': f'100 - (avg by (instance) (irate(node_cpu_seconds_total{{mode="idle"}}[{interval_sec}s]) * 100) * on(instance) group_left(nodename) (node_uname_info))'}
        # parse_result(query, ['instance', 'nodename'],
        #              'CPU_percentage_node', is_first=is_first)

        # CPU percentage by pod
        query = {
            'query': 'sum(rate(container_cpu_usage_seconds_total{}[1m])) by (namespace, pod)'}
        parse_result(query, ["namespace", "pod"],
                     'CPU_percentage_pod', is_first=is_first)

        # Network transmited by pod, namespace
        query = {
            'query': 'sum(container_network_transmit_bytes_total) by (namespace, pod)'}
        parse_result(query, ['namespace', 'pod'],
                     'network_transmit', is_first=is_first)

        # Network received by pod, namespace
        query = {
            'query': 'sum(container_network_receive_bytes_total) by (namespace, pod)'}
        parse_result(query, ['namespace', 'pod'],
                     'network_receive', is_first=is_first)

        # # CPU seconds by pod, namespace
        # # query={'query':f'sum(rate(container_cpu_usage_seconds_total[{interval_sec}s])) by (namespace, pod)'}
        # query = {
        #     'query': 'sum(container_cpu_usage_seconds_total{container!=""}) by (namespace, pod)'}
        # parse_result(query, ['namespace', 'pod'],
        #              'CPU_usage_second', is_first=is_first)

        # FS read bytes by pod, namespace
        query={'query':'sum(container_fs_reads_bytes_total) by (namespace, pod)'}
        parse_result(query, ['namespace', 'pod'], 'fs_read', is_first=is_first)

        # FS write bytes by pod, namespace
        query={'query':'sum(container_fs_writes_bytes_total) by (namespace, pod)'}
        parse_result(query, ['namespace', 'pod'], 'fs_write', is_first=is_first)

        # # Disk read bytes total
        # query={'query':f'sum(rate(node_disk_read_bytes_total[{interval_sec}s])) by (device, instance) * on(instance) group_left(nodename) (node_uname_info)'}
        # parse_result(query, ['device', 'instance'], 'disk_read_bytes_node', is_first=is_first, limit=1)

        # Restart count
        query = {'query': 'kube_pod_container_status_restarts_total'}
        parse_result(query, ['container', 'instance', 'namespace',
                     'pod'], 'restart_total', is_first=is_first)

        # Collect the average span durations
        res = span_duration.get_span_duration_new(
            end_time=span_end_time, ignore_other_services=True, debug=False)
        filename = "avg_span_duration_online"
        header_list = ["timestamp", "pod", "value", "count"]
        if is_first:
            save_data(filename, header_list)
        for service in res:
            avg_span_duration, count = res[service]
            row = [span_end_time, service, avg_span_duration, count]
            save_data(filename, row)

        time.sleep(interval_sec - ((time.time() - start_time) % interval_sec))
        counter += 1
    end_time = time.time()
    # save all the traces of all the services
    start_time = start_time - 60  # start time is 1 min earlier than the first data point
    save_traces(span_duration.get_services(), convert_timestamp(
        start_time), convert_timestamp(end_time))
    # save the state time and end time of the traces as a text file
    with open("data/trace_start_end_time.txt", "w") as fout:
        fout.write("start_time, end_time\n")
        fout.write(f"{start_time}, {end_time}\n")
