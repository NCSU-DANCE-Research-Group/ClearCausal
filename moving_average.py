import numpy as np

def moving_average(data, window_size):
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')

def padding(data, window_size):
    padding = np.repeat(data[0], window_size-1)
    return np.append(padding, data)

def moving_average_with_padding(data, window_size):
    data = padding(data, window_size)
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')

if __name__ == "__main__":
    data = [1, 2, 3, 4, 5, 6, 7]
    window_size = 5

    result = moving_average(data, window_size)
    print("Moving average: ", result)
    result = moving_average_with_padding(data, window_size)
    print("Moving average with padding: ", result)
    print("After using the padding: ", padding(data, window_size))
    result = moving_average(padding(data, window_size), window_size)
    print("Moving average: ", result)
