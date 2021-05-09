import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import csv
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from time import time

start = time()

#calculate run time
def even(sample):
    even_list = []
    even_set = set(sample)
    
    for num in even_set:
        if num % 2 == 0:
            even_list.append(num)

    return even_list


path_1 = '/Users/Hadyan/Desktop/Rizki Lampung/Fist'
path_2 = '/Users/Hadyan/Desktop/Rizki Lampung/Normal'
# path_1 = input("\nEnter path 1 : ")
# path_2 = input("\nEnter path 2 : ")

'''delete header and spare 1500 rows, then convert to csv'''
def write_to_csv(path):
    filenames = []
    for f in os.listdir(path):
        if f.endswith('txt'):
            filenames.append(f)
    print("\nfilenames : ", filenames)
    
    for f in filenames:
        df = pd.read_csv(path + '/' + f, delimiter = ",", skiprows=4)
        df = df.drop(columns=['Sample Index',' Accel Channel 0',' Accel Channel 1', 
            ' Accel Channel 2',' Other', ' Other.1', ' Other.2', ' Other.3',
            ' Other.4', ' Other.5', ' Other.6', ' Analog Channel 0',
            ' Analog Channel 1', ' Analog Channel 2', ' Timestamp', ' Other.7',
            ' Timestamp (Formatted)'])
        df = df.tail(1500)  #Spare 1500 rows of data
        
        f = f[:-4] #delete ".txt" in file name
        df.to_csv(path + '/' + "{}.csv" .format(f), header = None, index = False)

# write_to_csv(path_1)
# write_to_csv(path_2)
print("\nWRITE CSV PASSSED.....\n")

'''Preprocess data : Bandpass filtering and wavelet denoising'''
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

# preprocess(path_1)
# preprocess(path_2)
print("\nPREPROCESSING PASSED......\n")

'''Write data from csv into an array'''
def write_to_array(path_1, path_2):
    global list
    list = []
    i = 0
    for file in os.listdir(path_1):
        if file.endswith('csv') and file.startswith('new'):
            i +=1
            globals()['array%s' % i] = np.genfromtxt(path_1 + '/' + file, delimiter=',')
            list.append(globals()['array%s' % i])

    for file in os.listdir(path_2):
        if file.endswith('csv') and file.startswith('new'):
            i +=1
            globals()['array%s' % i] = np.genfromtxt(path_2 + '/' + file, delimiter=',')
            list.append(globals()['array%s' % i])

    array_X = np.stack((list))
    # array_X = array_X/1500
    print("\nShape : ", array_X.shape)

    global Feature
    Feature = array_X

write_to_array(path_1, path_2) #output variable = Feature
print("\nWRITE TO ARRAY PASSED......\n")

'''Write 0 and 1 as label (Y)'''
def write_label(path_1, path_2):
    global label
    label = []

    for file in os.listdir(path_1):
        if file.endswith(".csv") and file.startswith('new'):           #exclude txt file
            label = np.append(label, [0])   #append label '0' for all file in this dir
        label_path_1 = np.count_nonzero(label == 0)
    
    for file in os.listdir(path_2):
        if file.endswith(".csv") and file.startswith('new'):
            label = np.append(label, [1]) #append label '1' for all file in this dir
        label_path_2 = np.count_nonzero(label == 1)

    print("\nClass 1 : {} \nClass 2 : {}" .format(label_path_1, label_path_2))   

write_label(path_1, path_2)        
print("\nWRITE LABEL PASSED.......\n")

"""dataset = Feature, Label"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    Feature, label, test_size = 0.3, random_state = 100)

print("\nSplitting train and test Data......") 
# print("\n train_label : {} \n test_label : {}" .format(y_train, y_test)


"""Create Sequential Model"""

from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense, LSTM, Bidirectional

LSTM = tf.keras.models.Sequential([
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

CNN = tf.keras.models.Sequential([
    Conv1D(16, (3), activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling1D(pool_size=(2)),

    Conv1D(32, (3), activation='relu'),
    MaxPooling1D(pool_size=(2)),

    Conv1D(64, (3), activation='relu'),
    MaxPooling1D(pool_size=(2)),

    Flatten(),

    Dense(512, activation='relu'),

    # Dropout(0.2),

    Dense(1, activation='sigmoid')
])

CNN.summary()

"""COMPILE and Train!!!"""

from tensorflow.keras.optimizers import RMSprop

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

def training_model(model):
    # model.summary()
    model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

    history = model.fit(X_train, y_train, 
                steps_per_epoch=20,
                epochs=10,
                validation_data=(X_test, y_test),
                verbose=1)

    """EVALUATE MODEL"""
    print("\nEvaluate Model ... \n")
    model.evaluate(X_test, y_test)

    '''plot accuracy & loss'''
    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")

# training_model(LSTM)
# training_model(CNN)



training_model(eval(input('\nModel Architecture? (CNN/LSTM)')))

# '''run time'''
# my_list = [i for i in range(10000000)]
# even(my_list)
# end = time()
# print(f"runtime : {end - start}")


