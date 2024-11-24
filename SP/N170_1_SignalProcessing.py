import os 
import mne
from mne.preprocessing import ICA
from autoreject import AutoReject
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

bids_root = "/Users/BAEK/Code/neurEx/data/N170/N170 Raw Data BIDS-Compatible V2"

subjects = [sub for sub in os.listdir(bids_root) if sub.startswith('sub-')]
task = 'N170'

#all_epochs = []
#all_labels = []




# TEST PURPOSE
subjects = subjects[:1] # TEST PURPOSE
# TEST PURPOSE




for subject in subjects:
    
    print()
    print(f'***** Processing subject: {subject}')
    print()
    
    eeg_path = os.path.join(bids_root, subject, 'eeg', f'{subject}_task-{task}_eeg.set')
    
    raw = mne.io.read_raw_eeglab(eeg_path, preload = True)
    
    eog_channels = ['HEOG_left', 'HEOG_right', 'VEOG_lower']
    for ch in eog_channels:
        if ch in raw.ch_names:
            raw.set_channel_types({ch: 'eog'})
    
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case = False, on_missing = 'ignore')
    
    raw.filter(0.1, 30., fir_design='firwin')
    raw.notch_filter(60., fir_design='firwin')
    
    raw.set_eeg_reference('average', projection=False)
    
    print()
    print(f'***** Processing ICA: {subject}')
    print()
    
    ica = ICA(n_components=15, method='fastica', random_state=97, max_iter='auto')
    ica.fit(raw)
    
    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=eog_channels)
    ica.exclude = eog_indices
    
    raw_corrected = ica.apply(raw.copy())
    
    annotations_mapping = {desc: int(desc) for desc in set(raw_corrected.annotations.description)}
    
    events, _ = mne.events_from_annotations(raw_corrected, event_id=annotations_mapping)
    
    event_code_to_condition = {}
    
    for code in range(1, 41):
        event_code_to_condition[code] = 'Stimulus/Face'
    for code in range(41, 81):
        event_code_to_condition[code] = 'Stimulus/Car'
    for code in range(101, 141):
        event_code_to_condition[code] = 'Stimulus/ScrambledFace'
    for code in range(141, 181):
        event_code_to_condition[code] = 'Stimulus/ScrambledCar'
    event_code_to_condition[201] = 'Response/Correct'
    event_code_to_condition[202] = 'Response/Error'
    
    condition_label_to_event_id = {
        'Stimulus/Face': 1,
        'Stimulus/Car': 2,
        'Stimulus/ScrambledFace': 3,
        'Stimulus/ScrambledCar': 4,
        'Response/Correct': 5,
        'Response/Error': 6
    }
    
    original_to_new_event_id = {}
    for code in event_code_to_condition:
        condition_label = event_code_to_condition[code]
        new_event_id = condition_label_to_event_id[condition_label]
        original_to_new_event_id[code] = new_event_id
    
    events[:, 2] = np.array([original_to_new_event_id.get(code, -1) for code in events[:, 2]])
    
    events = events[events[:, 2] != -1]
    
    event_id = condition_label_to_event_id
    
    tmin, tmax = -0.2, 0.6  
    baseline = (tmin, 0)
    
    epochs = mne.Epochs(raw = raw_corrected, 
                        events = events, 
                        event_id = event_id, 
                        tmin = tmin, 
                        tmax = tmax,
                        baseline = baseline, 
                        preload = True, 
                        detrend = 1)
    
    print()
    print(f'***** Processing AutoReject: {subject}')
    print()    
    
    ar = AutoReject()
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
    
    epochs_clean.apply_baseline(baseline=baseline)
    
    #x = epochs_clean.get_data()  
    #y = epochs_clean.events[:, 2]
    
    dataDir = '/Users/BAEK/Code/neurEx/data/N170'
    
    if not os.path.exists(dataDir + f'{subject}'):
        os.mkdir(dataDir + f'{subject}')
    
    #np.save(dataDir + f'{subject}/X_epochs_{subject}.npy', x)
    #np.save(dataDir + f'{subject}/Y_labels_{subject}.npy', y)
    
    epochs_clean.save(dataDir + f'{subject}/Epochs_{subject}.fif')
    
    #all_epochs.append(x)
    #all_labels.append(y)

#print()
#print('***** Processing X and Y')
#print()    

'''
x = np.vstack(all_epochs)
y = np.concatenate(all_labels)

n_epochs, n_channels, n_times = x.shape
x = x.reshape(n_epochs, -1)
scaler = StandardScaler()
x = scaler.fit_transform(x)
x = x.reshape(n_epochs, n_channels, n_times)

counter = Counter(y)
ros = RandomOverSampler(random_state = 42)
x_resampled, y_resampled = ros.fit_resample(x.reshape(n_epochs, -1), y)
x = x_resampled.reshape(-1, n_channels, n_times)
y = y_resampled

np.save('/Users/BAEK/Code/neurEx/N170/X_epochs_total.npy', x)
np.save('/Users/BAEK/Code/neurEx/N170/Y_labels_total.npy', y)
'''

print("Preprocessing complete. Data is optimized for deep learning models.")
