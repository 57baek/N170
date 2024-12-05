import logging
import os
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

def DeepConvNet(nb_classes, Chans, Samples):
    
    input_main = Input((Chans, Samples, 1))
    
    block1 = Conv2D(25, (1, 5), padding = 'same', use_bias = False) (input_main)
    block1 = Conv2D(25, (Chans, 1), use_bias = False) (block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D((1, 2))(block1)
    block1 = Dropout(0.2)(block1)
    
    block2 = Conv2D(50, (1, 5), padding='same', use_bias=False)(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D((1, 2))(block2)
    block2 = Dropout(0.3)(block2)
    
    block3 = Conv2D(100, (1, 5), padding='same', use_bias=False)(block2)
    block3 = BatchNormalization()(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D((1, 2))(block3)
    block3 = Dropout(0.4)(block3)
    
    block4 = Conv2D(200, (1, 5), padding='same', use_bias=False)(block3)
    block4 = BatchNormalization()(block4)
    block4 = Activation('elu')(block4)
    block4 = MaxPooling2D((1, 2))(block4)
    block4 = Dropout(0.5)(block4)
    
    
    flatten = Flatten()(block4)
    dense = Dense(nb_classes)(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs = input_main, outputs = softmax)

nb_classes = 2  
Chans = X.shape[1]
Samples = X.shape[2]

model = DeepConvNet(nb_classes = nb_classes, 
                    Chans = Chans, 
                    Samples = Samples)

model.compile(optimizer = Adam(learning_rate = 1e-3),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', 
                               patience=25, 
                               restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.5, 
                              patience=10,
                              min_lr=1e-6, 
                              verbose=1)

epochs = 500
batch_size = 16

X_train_val, X_test, y_train_val, y_test = train_test_split(X_aug, 
                                                            y_aug, 
                                                            test_size = 0.2, 
                                                            random_state = 42, 
                                                            stratify = y_aug)


history = model.fit(X_train_val, 
                    y_train_val,
                    epochs = epochs,
                    batch_size = batch_size,
                    validation_data = (X_test, y_test),
                    callbacks = [early_stopping, reduce_lr],
                    verbose = 1)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose = 1)

print()
print(f'***** Test Accuracy: {test_accuracy * 100:.2f}%')
print()

plt.figure(figsize = (12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label = 'Train Accuracy', color = 'blue')
plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy', color = 'orange')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label = 'Train Loss', color = 'blue')
plt.plot(history.history['val_loss'], label = 'Validation Loss', color = 'orange')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()

if not os.path.exists('figures'):
    os.makedirs('figures')
    
plt.savefig('figures/model_accuracy_loss.png')

y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, 
            annot = True, 
            fmt = 'd', 
            cmap = 'Blues',
            xticklabels = ['Face', 'Car'],
            yticklabels = ['Face', 'Car'])

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

plt.savefig('figures/confusion_matrix.png')

report = classification_report(y_test, y_pred, target_names=['Face', 'Car'])

with open('figures/classification_report.txt', 'w') as f:
    f.write(report)

model.save('figures/V2.tf')

print()
print('***** Batch Done. Have a nice day!')
print()