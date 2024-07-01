from common import retrieve_api, get_timestamps, save_data
from get_dependency_graph import print_dependency, get_all_service
import time
from collections import defaultdict
from setting import interval_sec, DATASET


class SpanDuration():
    def __init__(self):
        lookback = 86400000  # 1 day
        edges = print_dependency(lookback)
        self.services = get_all_service(edges)
        # self.services = ['frontend']
        self.count_start_before = defaultdict(int)
        self.count_end_after = defaultdict(int)
        self.duration_end_after = defaultdict(int)

    def get_services(self):
        return self.services
    
    def exam_span(self, end_time, debug=False):
        sampling_rate = interval_sec * 10**6  # interval_sec seconds
        start_time = end_time - sampling_rate
        jaeger_port = 16686
        host = 'localhost'
        # http://localhost:16686/api/traces?service=emailservice&start=&end=&prettyPrint=false&limit=100000
        url = f"http://{host}:{jaeger_port}/api/traces"
        if debug:
            print(f"Querying start_time: {start_time}, end_time: {end_time}")
        for service in self.services:
            query = {'service': service, 'start': start_time, 'end': end_time,
                     'prettyPrint': 'false', 'limit': 100000}  # 'error':'true'
            raw_response = retrieve_api(url, query)
            traces = None
            count1 = 0 
            if raw_response is None:
                print(f"service {service} has no trace")
            else:
                traces = raw_response['data']
            for trace in traces:
                trace_id = trace['traceID']
                spans = trace['spans']
                if len(spans) == 1 and 'health' in spans[0]['operationName'].lower(): 
                    continue
                for span in spans:
                    span_start_time = span['startTime']
                    # print(f"span_start_time: {span_start_time}, start_time: {start_time}, end_time: {end_time}")
                    duration = span['duration']
                    operation_name = span['operationName'].lower()
                    if service not in operation_name:
                        # print(service, operation_name)
                        continue

                    if span_start_time < start_time:
                        print(f"{trace_id} span_start_time {span_start_time} is before the specified start. start_time: {start_time}")
                        # 0/0
                        self.count_start_before[service] += 1
                    if span_start_time > end_time:
                        print(f"{trace_id} span_start_time {span_start_time} is after the specified end. end_time: {end_time}, duration: {duration}")
                        # 0/0
                        self.count_end_after[service] += 1
                    if span_start_time + duration > end_time:
                        print(f"{trace_id} span_start_time {span_start_time} + duration {duration} is more than the end time. end_time: {end_time}")
                        # 0/0
                        self.duration_end_after[service] += 1
                    
                    # assert(span_start_time >= start_time)
                    # assert(span_start_time <= end_time)
                    # assert(span_start_time + duration <= end_time)

    
    def calculate_average_span(self, start_time, end_time, service, traces, ignore_other_services=False, debug=False):
        avg_duration = 0
        duration_total = 0
        count = 0
        # durations = []
        num_error = 0
        num_warning = 0
        num_error_warning = 0
        clean_total_duration = 0
        if traces is None:
            return avg_duration
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
            # trace_warnings = trace['warnings']
            # Ignore the health checking traces
            if len(spans) == 1 and 'health' in spans[0]['operationName'].lower(): 
                continue
            for span in spans:
                span_start_time = span['startTime']
                duration = span['duration']
                span_end_time = span_start_time + duration
                # Ignore the spans that are not in the specified time range
                if span_end_time < start_time or span_start_time > end_time:
                    continue
                span_warnings = span['warnings']
                # operation_name = span['operationName'].lower()
                if ignore_other_services and span['processID'] != service_processid:
                    # print(service, operation_name)
                    continue
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
                # durations.append(duration)
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
                print(
                    f'start time: {start_time}, average duration of service {service}: {avg_duration}, count: {count}')
        return avg_duration, count
            

    def get_span_duration(self, debug=False):
        # Does not set the limit of traces
        # This query is not correct, because the starttime does not work. The traces can have a start time before the provided start time
        now_time = time.time()
        now = round(now_time*1000)
        sampling_rate = interval_sec * 1000  # interval_sec seconds
        earliest_start_time = now - sampling_rate
        jaeger_port = 16686
        host = 'localhost'
        # http://localhost:16686/api/traces?service=emailservice&startTime=&prettyPrint=false
        url = f"http://{host}:{jaeger_port}/api/traces"
        res = defaultdict(float)
        for service in self.services:
            query = {'service': service, 'startTime': earliest_start_time,
                     'prettyPrint': 'false'}  # 'error':'true'
            raw_response = retrieve_api(url, query)
            duration_total = 0
            count = 0
            # durations = []
            num_error = 0
            num_warning = 0
            num_error_warning = 0
            clean_total_duration = 0
            if raw_response is None:
                print(f"service {service} has no trace")
            else:
                traces = raw_response['data']
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
                        # durations.append(duration)
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
                    print(
                        f'start time: {earliest_start_time}, average duration of service {service}: {avg_duration}, count: {count}')
            else:
                avg_duration = 0
            res[service] = avg_duration
        return res, earliest_start_time

    def get_span_duration_new(self, end_time, ignore_other_services=True, debug=False):
        sampling_rate = interval_sec * 10**6  # interval_sec seconds
        start_time = end_time - sampling_rate
        jaeger_port = 16686
        host = 'localhost'
        # http://localhost:16686/api/traces?service=emailservice&start=&end=&prettyPrint=false&limit=100000
        url = f"http://{host}:{jaeger_port}/api/traces"
        res = defaultdict(float)
        if debug:
            print(f"Querying start_time: {start_time}, end_time: {end_time}")
        for service in self.services:
            query = {'service': service, 'start': start_time, 'end': end_time,
                     'prettyPrint': 'false', 'limit': 100000}  # 'error':'true'
            raw_response = retrieve_api(url, query)
            traces = None
            if raw_response is None:
                print(f"service {service} has no trace")
            else:
                traces = raw_response['data']
            avg_span, count = self.calculate_average_span(start_time, end_time, service, traces, ignore_other_services, debug)
            res[service] = [avg_span, count]        
        return res


if __name__ == "__main__":
    timestamps = get_timestamps("avg_span_duration_online.csv")
    span_duration = SpanDuration()
    if DATASET == 'onlineboutique':
        span_duration.services = ['cartservice', 'productcatalogservice', 'emailservice','shippingservice', 'recommendationservice', 'adservice', 'frontend', 'currencyservice', 'checkoutservice', 'paymentservice']
    elif DATASET == 'socialnetwork':
        span_duration.services = ['compose-post-service', 'user-mention-service', 'social-graph-service', 'text-service', 'url-shorten-service', 'nginx-web-server', 'home-timeline-service', 'unique-id-service', 'user-service', 'media-service', 'post-storage-service', 'user-timeline-service']
    elif DATASET == 'mediamicroservices':
        span_duration.services = ['movie-id-service', 'user-service', 'review-storage-service', 'plot-service', 'nginx-web-server', 'rating-service', 'cast-info-service', 'movie-review-service', 'movie-info-service', 'text-service', 'user-review-service', 'compose-review-service', 'unique-id-service']

    # filename = "avg_span_duration_old"
    # header_list = ["timestamp", "pod", "value"]
    # save_data(filename, header_list)
    # # Before ignore other services
    # for timestamp in timestamps:
    #     res = span_duration.get_span_duration_new(ignore_other_services=False,
    #         end_time=timestamp, debug=False)
    #     print(res)
    #     for service in res:
    #         row = [timestamp, service, res[service]]
    #         save_data(filename, row)

    filename = "avg_span_duration_offline"
    header_list = ["timestamp", "pod", "value", "count"]
    save_data(filename, header_list)
    # After ignore other services
    for timestamp in timestamps:
        res = span_duration.get_span_duration_new(ignore_other_services=True,
            end_time=timestamp, debug=False)
        print(res)
        for service in res:
            avg_duration, count = res[service]
            row = [timestamp, service, avg_duration, count]
            save_data(filename, row)
    
    for timestamp in timestamps:
        span_duration.exam_span(end_time=timestamp, debug=True)
    print("Service, count_start_before, count_end_after, duration_end_after")
    for service in span_duration.services:
        print(f"{service},{span_duration.count_start_before[service]},{span_duration.count_end_after[service]},{span_duration.duration_end_after[service]}")
    
    # 1681931721200502
    # 1681931781200502

    # 1681931700000000
    # 1681931760000000
