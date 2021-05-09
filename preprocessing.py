from scipy.signal import filtfilt
from scipy import stats
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pywt
from skimage.restoration import denoise_wavelet
import os

def preprocess(path):
    os.chdir(path)
    x=0
    for f in os.listdir(path):
        if f.endswith('.csv') and f.startswith('Open'):
            data = pd.read_csv(f, header=None)
            data.columns = ["Ch_1", "Ch_2", "Ch_3", "Ch_4", "Ch_5", "Ch_6", "Ch_7", "Ch_8"]
            
            new_array = []
            x+=1
            print("\n----------------FILE: {} ----------------- " .format(x))
            print("------------{}---------" .format(f))
            
            i = 0
            for columns in data:
                raw_signal = data[[columns]]
                raw_signal = np.array(raw_signal)

                time = np.linspace(0, 0.0002, 1500)

                filtered_signal = bandPassFilter(raw_signal)
                
                denoised_signal = denoise(filtered_signal)
                
                
                print(i)
                if i==0:
                    new_array = denoised_signal

                if i>0:
                    new_array = np.concatenate((new_array, denoised_signal), axis=1)
                
                i+=1
                
                print(new_array.shape)
            
            np.savetxt("new_{}" .format(f), new_array, delimiter=",")
            print("\nSAVED : new_{}\n" .format(f))

def bandPassFilter(signal):
    
    fs = 1500.0
    lowcut = 20.0
    highcut = 50.0
    
    nyq = 0.2 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    order = 2
    
    b, a = scipy.signal.butter(order, [low, high], 'bandpass', analog=False)
    y = scipy.signal.filtfilt(b, a, signal, axis=0)
    
    return (y)
    
def denoise(signal):
    denoised = denoise_wavelet(signal, method='BayesShrink', mode='hard', wavelet_levels=3,
                            wavelet='sym8', rescale_sigma='True')
    return (denoised)       

                
path_1 = '/Users/Hadyan/Desktop/Rizki Lampung/Fist'
path_2 = '/Users/Hadyan/Desktop/Rizki Lampung/Normal'  
            
# preprocess(path_1)
preprocess(path_2)