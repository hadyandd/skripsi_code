path = '/Users/Hadyan/Desktop/Rizki Lampung'

from scipy.signal import filtfilt
from scipy import stats
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

plt.figure(figsize = (10,3), dpi=120)
def plot(channel):
    data = pd.read_csv(path + "/test.csv", header=None)
    data.columns = ["Ch_1", "Ch_2", "Ch_3", "Ch_4", "Ch_5", "Ch_6", "Ch_7", "Ch_8"]
    data_signal = data[[channel]]
    
    data_signal = np.array(data_signal)
    
    time = np.linspace(0, 0.0002, 1000)
    
    plt.plot(time, data_signal)
    plt.show()

plot('Ch_5')