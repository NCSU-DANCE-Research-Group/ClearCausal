import numpy as np
import matplotlib.pyplot as plt
from dtwalign import dtw

np.random.seed(1234)
# test data
x = np.sin(2*np.pi*3.1*np.linspace(0,1,101))
x += np.random.rand(x.size)
y = np.sin(2*np.pi*3*np.linspace(0,1,120))
y += np.random.rand(y.size)

def dtw_align(x, y, debug=False):
    # check if x is not a numpy array
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    res = dtw(x,y)
    x_path = res.path[:,0]
    y_path = res.path[:,1]
    x_aligned = x[x_path]
    y_aligned = y[y_path]
    if debug:
        print(res.distance)
        print(res.normalized_distance)
        print(res.path)
        print(len(x))
        print(len(y))
        print(x.shape)
        print(y.shape)
        print(len(x_aligned))
        print(len(y_aligned))
        plt.plot(x_aligned,label="aligned query")
        plt.plot(y_aligned,label="aligned reference")
        plt.legend()
        plt.ylim(0,50)
        plt.show()
    return x_aligned, y_aligned

x = [11.284375, 9.844875, 8.023, 10.4425, 11.13925, 17.4335, 10.803, 9.561666666666666, 8.0785, 12.5795, 9.4785, 9.773833333333334, 31.8066, 0.0, 9.97025, 10.291666666666666, 9.4265, 8.630166666666666, 8.698833333333335, 11.09875, 97094.87625, 76592.09325, 56988.353, 1021.971375, 11.582, 9.38425, 9.1475, 11.0455]
y = [0.70260431,0.57667098,0.68577083,0.66707298,0.73002442,0.57291392,0.71505525,0.73262922,0.72179103,0.71695558,0.70799453,0.66066883,0.81726784,0.54206065,0.89047595,0.76136023, 0.56974067, 0.67454079, 0.65530661,0.74890662,2.35105923, 44.94557021, 57.27886234, 57.29249295, 12.9946503, 0.3757001, 0.2806903, 0.45960116]
x_aligned, y_aligned = dtw_align(x, y)