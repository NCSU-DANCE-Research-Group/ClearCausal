from common import retrieve_api, save_data
from prometheus_retrieval_docker_compose import parse_result
import time

# from get_dependency_graph import print_dependency, get_all_service

port = 9090
host ='localhost'
url = f"http://{host}:{port}/api/v1/query"

interval_sec = 30.0 # seconds
total_mins = 10 # minutes
num_iters = int(total_mins * 60 / interval_sec)

counter = 0
start_time = time.time()

# query={'query':'sum(container_memory_usage_bytes{container!=""}) by (namespace, pod)'}
# parse_result(query, ["namespace", "pod"], 'test', is_first=True, debug=True)

# sum(rate(container_cpu_usage_seconds_total{container!=""})) by (node)
# query={'query':'sum(rate(container_cpu_usage_seconds_total{container!=""})) by (namespace, pod)'}
# query={'query':'sum (rate(container_cpu_usage_seconds_total{}[1m])) by (pod, namespace)'} # working
# sum (rate (container_cpu_usage_seconds_total{image!=""}[1m])) by (pod_name)
# query={'query':'sum(rate(container_cpu_usage_seconds_total{}[40s])) by (namespace, pod)'}
# sum(rate(container_cpu_usage_seconds_total{}[1h])) by (pod_name, namespace)
# query={'query':'sum(container_network_receive_bytes_total) by (namespace, pod)'} # working 
# query={'query':'sum(container_network_transmit_bytes_total) by (namespace, pod)'} # working
# query={'query':'sum(container_fs_writes_bytes_total) by (namespace, pod)'} # working
# query={'query':'sum(container_fs_reads_bytes_total) by (namespace, pod)'} # working
# query = {'query': 'rate(container_cpu_usage_seconds_total{container!~"POD|"}[50s])'} # working
# parse_result(query, ["namespace", "pod"], 'test', is_first=True, debug=False)
query = {'query': 'sum(rate(container_cpu_usage_seconds_total{}[1m])) by (namespace, pod)'}
parse_result(query, ["namespace", "pod"], 'test', is_first=True, debug=False)
