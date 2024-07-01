from common import retrieve_api
from get_dependency_graph import print_dependency, get_all_service

lookback = 86400000
edges = print_dependency(lookback)
services = get_all_service(edges)
# services = ['frontend']

for service in services:
    jaeger_port = 16686
    host ='localhost'
    lookback='3m'
    url = f"http://{host}:{jaeger_port}/api/traces"
    query = {'service':service, 'lookback':lookback, 'prettyPrint':'false', 'error':'true'}
    raw_response = retrieve_api(url, query)
    if raw_response is None:
        print(f"service {service} has no trace")
    else:
        traces = raw_response['data']
        for trace in traces:
            trace_id = trace['traceID']
            spans = trace['spans']
            processes = trace['processes']
            trace_warnings = trace['warnings']
            if trace_warnings is None:
                if True or trace_id == '':
                    for span in spans:
                        span_trace_id = span['traceID']
                        span_id = span['spanID']
                        start_time = span['startTime']
                        tags = span['tags']
                        logs = span['logs']
                        operation_name = span['operationName']
                        references = span['references']
                        duration = span['duration']
                        process_id = span['processID']
                        span_warnings = span['warnings']
                        # print(service, trace_id, span_id, operation_name, span_warnings, tags)
                        # if span_warnings is not None:
                        #     print(service, trace_id, span_id, span_warnings, tags)
                            # pass
                        for tag in tags:
                            if tag['key'] == 'error' and tag['value']:
                                print(f"service: {service}, trace_id: {trace_id}, span_id: {span_id}, operation_name: {operation_name}, start_time: {start_time}, tags: {tags}, span_warnings: {span_warnings}")
                        # print(tags)
                        # print(span.keys())
                        # 0/0
            else:
                print(trace_id, trace_warnings, spans, processes)
            # 0/0
