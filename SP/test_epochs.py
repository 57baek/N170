import mne

# Paths to the two FIF files
file1 = '/Users/BAEK/Code/neurEx/N170/SP/sub-021/Epochs_sub-021.fif'
file2 = '/Users/BAEK/Code/neurEx/N170/SP/sub-021/OWEN_Epochs_sub-021.fif'

# Load epochs
epochs1 = mne.read_epochs(file1, preload=True)
epochs2 = mne.read_epochs(file2, preload=True)

# Plot the average ERP for each
evoked1 = epochs1.average()
evoked2 = epochs2.average()
evoked1.plot()
evoked2.plot()