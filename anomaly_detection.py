from sklearn.neighbors import NearestNeighbors
import numpy as np
# from sklearn_som.som import SOM
from som import SOM
from kneed import KneeLocator
import matplotlib.pyplot as plt
from collections import Counter
from operator import itemgetter
import autoEncoderTrain as aeTrain
import autoEncoderGetThreshold as aeThresh
import autoEncoderTest as aeTest
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from setting import BUGID, ROOT_CAUSE_FUNCTION_ANALYSIS
import pandas as pd

#FIXME: Duplicate in calculate_correlation.py
MINUTE_BEFORE_CHANGE_POINT = 10  # 12 minutes
MINUTE_AFTER_CHANGE_POINT = 3  # 3 minutes
if BUGID == 5:  
    MINUTE_BEFORE_CHANGE_POINT = 13  # 12 minutes
    MINUTE_AFTER_CHANGE_POINT = 7  # 2 minutes


def knn_anomaly_detection(data, n_neighbors=2) -> list:
    threshold_choice = "percentile"  # "meanstd"
    data = data.reshape(-1, 1)
    # Fit the NearestNeighbors model to the data
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(data)

    # Calculate the distances and indices of the nearest neighbors
    distances, indices = neigh.kneighbors(data)

    # set the threshold to be 90 percentile of the distances
    if threshold_choice == "percentile":
        threshold = np.percentile(distances[:, 1], 90)
    elif threshold_choice == "meanstd":
        mean_distance = np.mean(distances[:, 1])
        std_distance = np.std(distances[:, 1])
        threshold = mean_distance + 2 * std_distance

    # Generate a list of binary labels based on the threshold
    labels = (distances[:, 1] > threshold).astype('int')
    # print(labels)

    # # Identify the anomalies as data points with a distance greater than the threshold
    # anomalies = data[distances[:,1] > threshold]

    # # Print the anomalies
    # print(anomalies)
    # convert labels to a list
    labels = labels.tolist()
    return labels


def percentile_anomaly_detection(data, high_percentile=90, low_percentile=0):
    high_threshold = np.percentile(data, high_percentile)  # 85 for no smoothing, 60 for smoothing
    low_threshold = np.percentile(data, low_percentile)  # 15 for no smoothing, 40 for smoothing
    anomalies = []
    for i, value in enumerate(data):
        if value > high_threshold or value < low_threshold:
            anomalies.append(1)
        else:
            anomalies.append(0)
    return anomalies


def meanstd_anomaly_detection(data):
    mean = np.mean(data)
    std = np.std(data)
    threshold = std
    anomalies = []

    for i, value in enumerate(data):
        if abs(value - mean) > threshold:
            anomalies.append(1)
        else:
            anomalies.append(0)

    return anomalies


def get_2d(one_d_index, m, n):
    r_idx = one_d_index // n 
    c_idx = one_d_index % n 
    return int(r_idx), int(c_idx)


def get_1d(r_idx, c_idx, m, n):
    one_d_index = (r_idx * n) + c_idx
    return int(one_d_index)


def get_neighbor_dist(som, bmu_indices, m, n):
    """
    for each data bmu,
    get the avg neighborhood distance of up to 4 neighbors near the bmu
    ---
    data is 1D
    """
    avg_neigh_dist_list = []
    # get neighbor indices and distance - aware of out of bounds
    for neuron_idx in bmu_indices:
        # get neighbors
        r_idx, c_idx = get_2d(neuron_idx, m, n)
        neigh_1d_indices_list = []
        dir = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        for dx, dy in dir:
            # skip if neighbor idx out of bounds
            nr_idx = r_idx + dx
            nc_idx = c_idx + dy
            if nr_idx < 0 or nr_idx >= m or nc_idx < 0 or nc_idx >= n:
                continue
            neigh_1d_indices_list.append(get_1d(nr_idx, nc_idx, m, n))
        # get avg distance

        # TODO: get avg distance
        # neigh_dist_list = neigh_1d_indices_list
        # neigh_values_list = []  # neigh_1d_indices_list
        neigh_val_list = []
        for i in neigh_1d_indices_list:
            neigh_val_list.extend(som.weights[i].tolist())
        # neigh_dist_list = [] 
        # Calculate distance between x and each weight, 1d
        x_row = [som.weights[int(neuron_idx)].item()] * len(neigh_val_list) 
        neigh_dist_list = np.abs(np.subtract(x_row, neigh_val_list))
        # x_stack = np.stack(som.weights[int(neuron_idx)].item() * len(neigh_val_list), axis=0)
        # neigh_dist_list_2 = np.linalg.norm(x_row - neigh_val_list, axis=1)  # 1D
        avg_neigh_dist = np.average(neigh_dist_list)
        avg_neigh_dist_list.append(avg_neigh_dist)
    return avg_neigh_dist_list


def som_anomaly_detection_ubl(data, high_percentile=95, low_percentile=5, metric_file=None):
    """
    uses UBL som algorithm
    ---
    data is 1D
    """
    som_percentile = 95  # 85 #FIXME: debug only, please pass from last call
    pre_train_index = MINUTE_BEFORE_CHANGE_POINT*2*3//4 # bug injection didn't happen here. use these data for pre_train.
    print(data)
    # Split the data
    train_data = data[:pre_train_index]  # First 10 elements as normal data
    test_data = data[pre_train_index:]  # Remaining data for anomaly detection
    
    # Reshape data for the SOM
    train_data = train_data.reshape(-1, 1)  # Reshape training data to 2D array
    test_data = test_data.reshape(-1, 1)    # Reshape test data to 2D array
    
    # Normalize the data
    scaler = MinMaxScaler()
    train_data_normalized = scaler.fit_transform(train_data)
    test_data_normalized = scaler.transform(test_data)  # Use same scaler to keep the scale consistent

    # Create and train SOM
    m_dim = 32
    n_dim = 32
    som = SOM(m=m_dim, n=n_dim, dim=1, max_iter=100)
    som.fit(train_data_normalized)

    # Finding the BMUs and their distances
    bmus = np.array([som.find_bmu(x) for x in test_data_normalized])  # Find BMU for each data point
    bmu_indices = bmus[:, 0]  # BMU index in the map
    bmu_distances = bmus[:, 1]  # Distance to the BMU
    bmu_distances = get_neighbor_dist(som, bmu_indices, m_dim, n_dim)

    # Detecting outliers
    threshold = np.percentile(bmu_distances, som_percentile)  # Threshold as 95th percentile of distances
    outliers = np.where(bmu_distances >= threshold)[0]  # Indices of outlier
    # convert to list
    outliers = outliers.tolist()
    anomalies = []
    for i in range(pre_train_index):
        anomalies.append(0)
    for i in range(data.shape[0]-pre_train_index):
        if i in outliers:  # and (high_threshold >= value >= low_threshold):
            anomalies.append(1)
        else:
            anomalies.append(0)

    return anomalies  # anomalies cluster_labels labels


def som_anomaly_detection_ubl_mean(data, high_percentile=95, low_percentile=5, metric_file=None):
    """
    uses UBL som algorithm
    ---
    data is 1D
    """
    som_percentile = 85  #FIXME: debug only, please pass from last call
    # pre_train_index = (MINUTE_BEFORE_CHANGE_POINT - 2) // 0.5  # 2 min before change increase buffer
    pre_train_index = MINUTE_BEFORE_CHANGE_POINT*2*3//4 # bug injection didn't happen here. use these data for pre_train.
    print(data)
    # Split the data
    # Convert to DataFrame for easier manipulation
    data_df = pd.DataFrame(data, columns=['Original'])

    # Applying a simple moving average with a window of 3 for smoothing
    data_df['Smoothed'] = data_df['Original'].rolling(window=3, min_periods=1).mean()
    # Normalize the smoothed data
    scaler = StandardScaler()
    data_df['Normalized'] = scaler.fit_transform(data_df[['Smoothed']])

    train_data = data_df['Normalized'].iloc[:pre_train_index].values.reshape(-1, 1) # First n elements as normal data
    test_data = data_df['Normalized'].iloc[pre_train_index:].values.reshape(-1, 1)  #   Remaining data for anomaly detection 

    # Create and train SOM
    m_dim = 32
    n_dim = 32
    som = SOM(m=m_dim, n=n_dim, dim=1, max_iter=300)  # , max_iter=500 200
    som.fit(train_data)

    # Finding the BMUs and their distances
    bmus = np.array([som.find_bmu(x) for x in test_data])  # Find BMU for each data point
    bmu_indices = bmus[:, 0]  # BMU index in the map
    bmu_distances = bmus[:, 1]  # Distance to the BMU
    bmu_distances = get_neighbor_dist(som, bmu_indices, m_dim, n_dim)

    # Detecting outliers
    threshold = np.percentile(bmu_distances, som_percentile)  # Threshold as 95th percentile of distances
    outliers = np.where(bmu_distances >= threshold)[0]  # Indices of outlier
    # convert to list
    outliers = outliers.tolist()
    
     # Apply additional rule: ignore anomalies too close to the mean
    data_mean = np.mean(data_df['Original'])
    data_median = np.median(data_df['Original'])

    # Identify anomalies considering closeness to mean
    anomalies = [0] * pre_train_index

    for i in range(data.shape[0]-pre_train_index):
        if i in outliers:  
            anomalies.append(1)
        else:
            anomalies.append(0)
            
    # Apply additional rule for service analysis: ignore anomalies too close to the mean
    if not ROOT_CAUSE_FUNCTION_ANALYSIS:
        mean_dist_thresh = 0.2 * data_mean  # data_mean data_median
        low_threshold = np.percentile(data, low_percentile)  # 15 for no smoothing, 40 for smoothing
        if metric_file == "CPU_percentage_pod" or metric_file == "memory": # or "faulty_avg_span":
            for i in range(len(anomalies)):
                if abs(data[i] - data_mean) < mean_dist_thresh and anomalies[i] == 1:
                    anomalies[i] = 0
                if data[i] <= low_threshold and anomalies[i] == 1:
                    anomalies[i] = 0

    return anomalies  # anomalies cluster_labels labels


def _som_anomaly_detection(data, high_percentile=90, low_percentile=0):
    print(data.shape)
    anomalies = []
    # run model
    data = np.reshape(data, (data.shape[0], 1))
    model_som = SOM(m=32, n=32, dim=len(data[0]),
                    random_state=1)  # , max_iter=1000, dim=data.shape[1], dim=len(data[0])
    # returns clusters, not distances
    cluster_labels = model_som.fit_predict(data)
    cluster_labels = cluster_labels.tolist()
    # cluster_labels = np.reshape(cluster_labels, (data.shape[0], )).
    # print(cluster_labels)

    # find minority cluster
    c = Counter(cluster_labels)
    # min_key, min_count = min(c.items(), key=itemgetter(1))
    cluster_data_indices = dict()
    for idx, value in enumerate(cluster_labels):
        cluster_data_indices.setdefault(value, []).append(idx)  # sorted idx list

    # find min cluster size with consecutive anomalies
    single_cluster_size = 1
    min_count_allowed = 2  # 2 3
    # three_count_found = False
    # filtered_items = {item for item, count in c.items() if count >= min_count_allowed}
    min_count_found = len(data) + 1

    consecutive = dict()
    for label, count in c.items():
        # get sorted index locations
        indices = cluster_data_indices[label]  # already sorted idx list
        # get difference array
        # check diff is all ones?
        diff_arr = np.diff(indices)
        if np.all(diff_arr == 1):  # len of one still considered consecutive
            consecutive[label] = True
            # if count == 3:
            #     three_count_found = True
            # get min count
            if min_count_allowed <= count < min_count_found:
                min_count_found = count
        else:
            consecutive[label] = False

    # # make cluster size of 3 the min if present
    # if three_count_found:
    #     # min_count_allowed = 3
    #     min_count_found = 3

    # filter anomalies
    for _, value in enumerate(cluster_labels):
        if c[value] == min_count_found and consecutive[value] is True:
            anomalies.append(1)
        else:
            anomalies.append(0)

    # add cluster size of one anomalies if consecutive
    # if min_count_found == len(data) + 1:  # if no consecutive anomalies found for size above one, add size of one
    for idx, value in enumerate(cluster_labels):
        if c[value] == single_cluster_size:
            anomalies[idx] = 1
    # remove anomaly if not consecutive (to left or right anomaly neighbors)
    for idx, sample in enumerate(anomalies):
        if sample == 1:
            if idx - 1 > 0:
                if anomalies[idx - 1] == 1:
                    # consecutive
                    continue
            if idx + 1 < len(data):
                if anomalies[idx + 1] == 1:
                    # consecutive
                    continue
            anomalies[idx] = 0

    # don't allow anomalies become the most frequent points
    num_anomalies = 0
    for sample in anomalies:
        if sample == 1:
            num_anomalies += 1
    if num_anomalies >= len(data) // 2:  # size above 1 majority, no anomalies
        anomalies = [0] * len(data)  # min_count_found = single_cluster_size  # min_count 1 # return [0] * len(data)
        # num_anomalies = 0

    # apply percentile rule afterward
    # drop anomalies that do not meet percentile threshold
    # if num_anomalies >= :
    high_threshold = np.percentile(data, high_percentile)  # 85 for no smoothing, 60 for smoothing
    low_threshold = np.percentile(data, low_percentile)  # 15 for no smoothing, 40 for smoothing
    for i, value in enumerate(data):
        # if value > high_threshold or value < low_threshold:
        if anomalies[i] == 1 and (high_threshold <= value <= low_threshold):
            anomalies[i] = 0

    return anomalies  # anomalies cluster_labels labels


def autoencoder_anomaly_detection(data):
    # anomalies = []
    save_model_loc = "./"
    save_model_name = "fca"  # no extension

    data_train = data = np.reshape(data, (data.shape[0], 1))

    aeTrain.run(data_train, save_model_loc, save_model_name)
    _ = aeThresh.get_threshold(data_train, save_model_loc, save_model_name)
    anomalies = aeTest.run(data_train, data, save_model_loc, save_model_name)  

    return anomalies


def anomaly_detection(data, model="percentile", high_percentile=90, low_percentile=0, metric_file=None):
    if model == "knn":
        return knn_anomaly_detection(data)
    elif model == "meanstd":
        return meanstd_anomaly_detection(data)
    elif model == "percentile":
        return percentile_anomaly_detection(data, high_percentile, low_percentile)
    elif model == "ae":
        return autoencoder_anomaly_detection(data)
    elif model == "som":
        return som_anomaly_detection_ubl_mean(data, high_percentile, low_percentile, metric_file)
    else:
        raise Exception("model not supported")


if __name__ == "__main__":
    function = np.array(
        [137, 135, 161, 152, 156, 144, 144, 137, 140, 142, 142, 148, 148, 154, 145, 141, 141, 142,
         140, 137, 141, 0, 0, 0, 173, 149, 141, 140, 142, 142])
    # checkout service avg span duration
    checkout = np.array(
        [0, 0, 1, 1, 1, 1, 0, 0, 3, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 39045, 0, 11893, 2059,
         0, 3])
    email = np.array([14569472, 14999552, 15601664, 15507456, 16064512, 16158720, 16662528, 16601088, 16908288,
                      16977920, 17182720, 17428480, 17309696, 17694720, 17858560, 17960960,
                      18288640, 18059264, 18083840, 18206720, 18235392, 18448384, 21299200, 26361856, 29851648,
                      31215616, 20729856, 16691200])
    currency = np.array(
        [3, 3, 2, 2, 3, 4, 2, 3, 3, 3, 2, 2, 3, 3, 3, 2, 3, 2, 2, 3, 3, 2, 2, 1, 2, 2, 3, 2])  # currency
    model = "som"  # "knn", "meanstd", "percentile", som, ae
    print(f"{anomaly_detection(checkout, model, 90, 0)}")
    print(f"{anomaly_detection(email, model, 90, 0)}")
    print(f"{anomaly_detection(function, model, 100, 10)}")
    print(f"{anomaly_detection(currency, model)}")
