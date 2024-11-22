import os 
import mne
from mne.preprocessing import ICA
from autoreject import AutoReject
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import matplotlib.pyplot as plt

# Define the root directory where the data is stored
bids_root = '/Users/owenanderson/Documents/NeurEx/Projects/P1_N170/Data/N170 Raw Data BIDS-Compatible'

# List all subjects (adjust as needed)
subjects = [sub for sub in os.listdir(bids_root) if sub.startswith('sub-')]
task = 'N170'

# Initialize lists to hold preprocessed epochs and labels
all_epochs = []
all_labels = []

# For demonstration, processing only one subject
#subject = subjects[1]
for subject in subjects:
    
    print(f'Processing subject: {subject}')
    
    # Define the path to the EEG data in EEGLAB format (.set file)
    eeg_path = os.path.join(bids_root, subject, 'eeg', f'{subject}_task-{task}_eeg.set')
    
    # Read the raw EEG data using MNE's EEGLAB reader
    raw = mne.io.read_raw_eeglab(eeg_path, preload=True)
    
    # Set channel types for EOG channels
    eog_channels = ['HEOG_left', 'HEOG_right', 'VEOG_lower']
    for ch in eog_channels:
        if ch in raw.ch_names:
            raw.set_channel_types({ch: 'eog'})
    
    # Apply montage directly, ignoring missing channels
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False, on_missing='ignore')
    
    # Apply band-pass filter
    raw.filter(0.1, 30., fir_design='firwin')
    
    # Apply notch filter at 60 Hz
    raw.notch_filter(60., fir_design='firwin')
    
    # Re-reference to average
    raw.set_eeg_reference('average', projection=False)
    #raw.resample(sfreq=250, npad="auto")
    
    # Apply ICA to remove EOG artifacts
    ica = ICA(n_components=15, method='fastica', random_state=97, max_iter='auto')
    ica.fit(raw)
    
    # Find ICA components correlated with EOG artifacts
    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=eog_channels)
    ica.exclude = eog_indices
    
    # Apply the ICA solution to raw data
    raw_corrected = ica.apply(raw.copy())
    
    # Create a mapping from annotations descriptions to integer event codes
    annotations_mapping = {desc: int(desc) for desc in set(raw_corrected.annotations.description)}
    print("Annotations mapping:", annotations_mapping)
    
    # Extract events using the annotations mapping
    events, _ = mne.events_from_annotations(raw_corrected, event_id=annotations_mapping)
    
    # Create a mapping from original event codes to condition labels
    event_code_to_condition = {}
    
    # Map "1-40" to "Stimulus - faces"
    for code in range(1, 41):
        event_code_to_condition[code] = 'Stimulus/Face'
    
    # Map "41-80" to "Stimulus - cars"
    for code in range(41, 81):
        event_code_to_condition[code] = 'Stimulus/Car'
    
    # Map "101-140" to "Stimulus - scrambled faces"
    for code in range(101, 141):
        event_code_to_condition[code] = 'Stimulus/ScrambledFace'
    
    # Map "141-180" to "Stimulus - scrambled cars"
    for code in range(141, 181):
        event_code_to_condition[code] = 'Stimulus/ScrambledCar'
    
    # Map "201" to "Response - correct"
    event_code_to_condition[201] = 'Response/Correct'
    
    # Map "202" to "Response - error"
    event_code_to_condition[202] = 'Response/Error'
    
    # Assign new integer event IDs to each condition label
    condition_label_to_event_id = {
        'Stimulus/Face': 1,
        'Stimulus/Car': 2,
        'Stimulus/ScrambledFace': 3,
        'Stimulus/ScrambledCar': 4,
        'Response/Correct': 5,
        'Response/Error': 6
    }
    
    # Create a mapping from original event codes to new integer event IDs
    original_to_new_event_id = {}
    for code in event_code_to_condition:
        condition_label = event_code_to_condition[code]
        new_event_id = condition_label_to_event_id[condition_label]
        original_to_new_event_id[code] = new_event_id
    
    # Apply the event ID mapping to the events array
    events[:, 2] = np.array([original_to_new_event_id.get(code, -1) for code in events[:, 2]])
    
    # Remove events with code -1 (events not in our mapping)
    events = events[events[:, 2] != -1]
    
    # Define the event_id mapping for epoching
    event_id = condition_label_to_event_id
    
    # Epoch the data around the events of interest
    tmin, tmax = -0.2, 0.6  # Time window from -200 ms to 600 ms
    baseline = (None, 0)    # Baseline correction from -200 ms to 0 ms
    
    epochs = mne.Epochs(raw_corrected, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, detrend=1)
    
    # Apply Autoreject to automatically reject bad epochs
    ar = AutoReject()
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
    
    # Apply baseline correction after artifact rejection
    epochs_clean.apply_baseline(baseline=baseline)
    
    # **Added section: Compute and plot averaged responses**
    
    # Compute the average (evoked) response for each condition
    evoked_face = epochs_clean['Stimulus/Face'].average()
    evoked_Car = epochs_clean['Stimulus/Car'].average()
    
    # Get ylims for face
    ymin_face = evoked_face.data.min() * (10**6) - 1
    ymax_face = evoked_face.data.max() * (10**6) + 1
    ylim = {'eeg': [ymax_face, ymin_face]} 
    
    # Plot the averaged responses
    # Plotting the ERP for 'Stimulus/Face'
    faceplot = evoked_face.plot(titles={'eeg': f'Face Stimulus ERP, {subject}'}, spatial_colors=True, ylim=ylim)
    saveDirectory = f'/Users/owenanderson/Documents/NeurEx/Projects/P1_N170/Data/Plots/{subject}'
    if not os.path.exists(saveDirectory):
        os.mkdir(saveDirectory)
        os.mkdir(saveDirectory+'/Evoked_Face')
        os.mkdir(saveDirectory+'/Evoked_Car')
        os.mkdir(saveDirectory+'/Comparison')
    faceplot.savefig(saveDirectory + '/Evoked_Face/' + f'Average_ERP_Face_{subject}.png')
    
    # Get ylims for the car
    ymin_car = evoked_Car.data.min() * (10**6) - 1
    ymax_car = evoked_Car.data.max() * (10**6) + 1
    ylim = {'eeg': [ymax_car, ymin_car]} 
    
     
    # Plotting the ERP for 'Stimulus/Object'
    carplot = evoked_Car.plot(titles={'eeg': f'Car Stimulus ERP, {subject}'}, spatial_colors=True, ylim = ylim)
    carplot.savefig(saveDirectory + '/Evoked_Car/' + f'Average_ERP_Car_{subject}.png')
    
    # Getting the ylims for comparison plots
    #ylim = {'eeg': [max(ymax_car, ymax_face), min(ymin_face, ymin_car)]}
    
    # Optional: Plot both conditions on the same plot for comparison
    comparisonplot = mne.viz.plot_compare_evokeds({'Face': evoked_face, 'Object': evoked_Car}, 
                                 picks='P10', invert_y=True, title=f'Face vs Car: {subject}')
    comparisonplot[0].figure.savefig(saveDirectory + '/Comparison/' + f'Average_Comparison_{subject}.png')
    
    
    # Extract data and labels for deep learning
    X = epochs_clean.get_data()  # Shape: (n_epochs, n_channels, n_times)
    y = epochs_clean.events[:, 2]
    
    dataDir = '/Users/owenanderson/Documents/NeurEx/Projects/P1_N170/Data/Data/'
    if not os.path.exists(dataDir + f'{subject}'):
        os.mkdir(dataDir + f'{subject}')
    np.save(dataDir + f'{subject}/X_eeg_{subject}.npy', X)
    np.save(dataDir + f'{subject}/y_labels_{subject}.npy', y)
    #epochs_clean.save(dataDir + f'{subject}/Epochs_{subject}.fif')
    
    # Append to the list of all epochs and labels
    all_epochs.append(X)
    all_labels.append(y)

# Concatenate all data from all subjects (currently only one subject)
# Vertically stacks all the arrays that were appended to all_epochs
# X will be a single 3D array containing EEG data for all subjects, where each subject’s data is stacked along the first axis
X = np.vstack(all_epochs) 
# concatenates all the label arrays that were appended to all_labels. So, y will be a single 1D array containing all the event labels from all subjects, in the same order as the epochs in X.
y = np.concatenate(all_labels)

# Normalize the data (z-score normalization)
n_epochs, n_channels, n_times = X.shape
X = X.reshape(n_epochs, -1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X.reshape(n_epochs, n_channels, n_times)

# Resample to balance classes
counter = Counter(y)
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X.reshape(n_epochs, -1), y)
X = X_resampled.reshape(-1, n_channels, n_times)
y = y_resampled

# Save the preprocessed data
#np.save('/Users/owenanderson/Documents/NeurEx/Projects/P1_N170/Data/Data/X_eeg.npy', X)
#np.save('/Users/owenanderson/Documents/NeurEx/Projects/P1_N170/Data/Data/y_labels.npy', y)

print("Preprocessing complete. Data is optimized for deep learning models.")
