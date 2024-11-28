import logging

import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Dropout, Flatten, Input,
                                     MaxPooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

logging.getLogger('tensorflow').setLevel(logging.ERROR)


data_dir = '/Users/BAEK/Code/neurEx/data/N170/Data_Preprocessed'

subject_folders = [sub for sub in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, sub))]
'''
subjects = []
for sub in os.listdir(data_dir):
    if os.path.isdir(os.path.join(data_dir, sub)):
        subjects.append(sub)
'''

X_list = []
y_list = []

for subject in subject_folders:
    
    subject_data_dir = os.path.join(data_dir, subject)
    
    X_file = os.path.join(subject_data_dir, f'Epochs_{subject}.fif')
    
    if os.path.exists(X_file):
        
        print()
        print(f'***** Loading the processed data: {subject}')
        print()
        
        X_subject = mne.read_epochs(X_file, preload=True, verbose=False)
        y_subject = X_subject.events[:, 2]
        
        X_sub_data = X_subject.get_data()
        X_list.append(X_sub_data)
        y_list.append(y_subject)
        
    else:
        print()
        print(f'***** Data file Does Not Exist: {subject}')
        print()

if len(X_list) == 0:
    print()
    raise ValueError("***** No data was loaded. Please check your data directory and files.")

channels = X_subject.ch_names
coi = ['P8', 'PO8', 'O2', 'P10']
chan_idx = [channels.index(chan) for chan in coi] 

X = np.concatenate(X_list, axis = 0) 

X = X[:, chan_idx, :] 

y = np.concatenate(y_list, axis = 0)

print()
print(f'***** Combined data shape before filtering: X={X.shape}, y={y.shape}')
print()

stimulus_labels = [1, 2]

stimulus_mask = np.isin(y, stimulus_labels)
X = X[stimulus_mask] 
y = y[stimulus_mask]

y = y - 1

tmin, tmax = 0.10, 0.2 
times = X_subject.times  
time_mask = (times >= tmin) & (times <= tmax)

X_focused = X[:, :, time_mask]

print()
print(f'***** Combined data shape after filtering: X={X_focused.shape}, y={y.shape}')
print()

def normalize_per_channel(X_data):
    
    X_norm = np.zeros_like(X_data)
    
    for i in range(X_data.shape[1]):  
        
        scaler = StandardScaler()
        X_channel = X_data[:, i, :]
        X_norm[:, i, :] = scaler.fit_transform(X_channel)
        
    return X_norm

X_normalized = normalize_per_channel(X_focused)

X = X_normalized[..., np.newaxis]

def augment_data(X, y):
    
    X_augmented = []
    y_augmented = []
    
    for i in range(X.shape[0]):
        
        for shift in [-2, -1, 1, 2]:  
            X_shifted = np.roll(X[i], shift, axis = 2)
            X_augmented.append(X_shifted)
            y_augmented.append(y[i])
        
        noise = np.random.normal(0, 0.01, X[i].shape)
        X_noisy = X[i] + noise
        X_augmented.append(X_noisy)
        y_augmented.append(y[i])

    return np.array(X_augmented), np.array(y_augmented)

X_aug, y_aug = augment_data(X, y)

print()
print(f'***** Augmented data shape: X={X_aug.shape}, y={y_aug.shape}')
print()
