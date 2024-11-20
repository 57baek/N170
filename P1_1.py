import os
import mne
from mne.preprocessing import ICA
from autoreject import AutoReject
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import matplotlib.pyplot as plt

bids_root = "/Users/BAEK/Code/neurEx/data/N170/N170 Raw Data BIDS-Compatible V2"

subjects = [sub for sub in os.listdir(bids_root) if sub.startswith('sub')]

task = 'task-N170'

for subject in subjects:
    
    print(f'Processing subject: {subject}')

    eeg_path = os.path.join(bids_root, subject, 'eeg', f'{subject}_{task}_eeg.set')

    raw = mne.io.read_raw_eeglab(eeg_path, preload=True)
    
    annotations_mapping = {desc: int(desc) for desc in set(raw.annotations.description)}
    
    events, _ = mne.events_from_annotations(raw, event_id = annotations_mapping)

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
    
    eog_channels = ['HEOG_left', 'HEOG_right', 'VEOG_lower']
    for ch in eog_channels:
        if ch in raw.ch_names:
            raw.set_channel_types({ch: 'eog'})
            
    montage = mne.channels.make_standard_montage('standard_1020')

    raw.set_montage(montage, match_case=False)
    
    raw.filter(0.1, 30., fir_design='firwin')
    raw.notch_filter(60., fir_design='firwin')
    
    raw.set_eeg_reference(ref_channels='average')

    print(f'Processing ICA: {subject}')

    ica = ICA(n_components=15, method='fastica', random_state=97, max_iter='auto')
    ica.fit(raw)

    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name = eog_channels)
    ica.exclude = eog_indices

    raw_corrected = ica.apply(raw.copy())

    tmin, tmax = -0.2, 0.6
    baseline = (tmin, 0)

    epochs = mne.Epochs(
        raw=raw_corrected,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        detrend=1
    )
    
    print(f'Processing AutoReject: {subject}')

    ar = AutoReject()
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log = True)
    
    np.save(f'{subject}_reject_log.npy', reject_log)

    epochs_clean.apply_baseline(baseline = baseline)

    raw_corrected.save(f'{subject}_processed_raw.fif')
    epochs.save(f'{subject}_epochs-epo.fif')
    
    del raw, epochs, raw_corrected
    
    print(f'Processing Completion: {subject}')
