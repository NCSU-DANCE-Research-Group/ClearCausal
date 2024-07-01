import time
from collections import defaultdict, deque
from common import retrieve_api
import json
import os
from setting import DATASET

def save_to_file(data, file_name='temp.txt'):
    with open(file_name, "w") as fout:
        fout.write(str(data) + '\n')

def bfs(edges, start):
    graph = defaultdict(list)
    for edge in edges:
        parent = edge['parent']
        child = edge['child']
        # call_count = edge['callCount']
        graph[child].append(parent)
    queue = deque([start])
    visited = {start}
    parents = [start]
    while len(queue):
        node = queue.popleft()
        for parent in graph[node]:
            if parent not in visited:
                visited.add(parent)
                queue.append(parent)
                parents.append(parent)
    return parents

def get_neigh_services(edges, start, num_step=-1):
    graph = defaultdict(list)
    for edge in edges:
        parent = edge['parent']
        child = edge['child']
        call_count = edge['callCount']
        # graph[child].append(parent)
        graph[parent].append((child, call_count))
    queue = deque([start])
    visited = {start}
    nei_services = defaultdict(int)
    nei_services[start] = 0
    curr_step = 0
    nei_services_step = defaultdict(int)
    nei_services_step[start] = curr_step
    while len(queue):
        if num_step > 0 and curr_step >= num_step:
            break
        length = len(queue)
        curr_step += 1
        for i in range(length):
            node = queue.popleft()
            for nei, call_count in graph[node]:
                if nei not in visited:
                    visited.add(nei)
                    queue.append(nei)
                    nei_services_step[nei] = curr_step
                    nei_services[nei] = call_count
    leaf_nodes = set()
    for node in visited:
        # find if the node is a leaf node
        if len(graph[node]) == 0:
            leaf_nodes.add(node)
    return nei_services, leaf_nodes, nei_services_step

def get_all_service(edges):
    services = set()
    for edge in edges:
        parent = edge['parent']
        child = edge['child']
        # call_count = edge['call_count']
        services.add(parent)
        services.add(child)
    return services

def load_dependency(file_name=f'dependency_graph_{DATASET}.txt'):
    # Open the file and read the contents
    with open(file_name, 'r') as f:
        data = f.read()
    # Replace single quotes with double quotes
    data = data.replace("'", '"')
    # Parse the JSON data
    edges = json.loads(data)
    return edges

def print_dependency(lookback=86400000, file_name=f'dependency_graph_{DATASET}.txt'):
    # Jaeger doc: https://www.jaegertracing.io/docs/1.38/apis/
    # Can be retrieved from Query Service at /api/dependencies endpoint. The GET request expects two parameters:
    # endTs (number of milliseconds since epoch) - the end of the time interval
    # lookback (in milliseconds) - the length the time interval (i.e. start-time + lookback = end-time).
    # The returned JSON is a list of edges represented as tuples (caller, callee, count)
    if os.path.isfile(file_name):
        edges = load_dependency(file_name)
    else:
        curr_time = int(round(time.time() * 1000))
        query = {'endTs':curr_time, 'lookback':lookback}
        url = 'http://localhost:16686/api/dependencies'
        raw_response = retrieve_api(url, query)
        edges = raw_response['data']
        save_to_file(edges, file_name)
    return edges

if __name__ == "__main__":
    ms_per_day = 86400000
    num_day = 1
    lookback = ms_per_day * num_day
    edges = print_dependency(lookback=lookback)
    if DATASET == 'socialnetwork':
        start = 'social-graph-service'
    elif DATASET == 'onlineboutique':
        start = 'emailservice'
    elif DATASET == 'mediamicroservices':
        start = 'compose-review-service'
    elif DATASET == 'hotelreservation':
        start = 'search'
    print(f"The calling chain from {start} is: {bfs(edges, start)}")
    print(f"All services are: {get_all_service(edges)}")
