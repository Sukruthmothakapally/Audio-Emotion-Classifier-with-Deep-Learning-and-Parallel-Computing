import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from multiprocessing import Pool
from glob import glob
from skimage.transform import resize
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torchaudio
import torch.nn.functional as F
import gc
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
angry = glob(r'/home/mothakapally.s/Project/data/Emotions/Angry/*.wav')
angry = sorted(angry)
Disgusted = glob(r'/home/mothakapally.s/Project/data/Emotions/Disgusted/*.wav')
Disgusted = sorted(Disgusted)
Fearful = glob(r'/home/mothakapally.s/Project/data/Emotions/Fearful/*.wav')
Fearful = sorted(Fearful)
Happy = glob(r'/home/mothakapally.s/Project/data/Emotions/Happy/*.wav')
Happy = sorted(Happy)
Neutral = glob(r'/home/mothakapally.s/Project/data/Emotions/Neutral/*.wav')
Neutral = sorted(Neutral)
Sad = glob(r'/home/mothakapally.s/Project/data/Emotions/Sad/*.wav')
Sad = sorted(Sad)
Surprised = glob(r'/home/mothakapally.s/Project/data/Emotions/Surprised/*.wav')
Surprised = sorted(Surprised)

# Function to process each emotion
def process_emotion(emotion_files, label, emotion, target_shape=(128,128)):
    start_time = time.time()  # Start timer
    print(f"Processing {emotion}...")
    emotion_train = []
    label_emotion = []
    for file in emotion_files:
        waveform, sr = torchaudio.load(file)
        y_stretched = torchaudio.transforms.TimeStretch(hop_length=None, fixed_rate=1)(waveform)
        Mel_spectrogram = torchaudio.transforms.MelSpectrogram(sr)(y_stretched)
        Mel_spectrogram = resize(Mel_spectrogram.squeeze().numpy(), target_shape)
        emotion_train.append(Mel_spectrogram)
        label_emotion.append(label)
    end_time = time.time()  # End timer
    print(f'Execution time for {emotion}: {end_time-start_time} seconds')
    return emotion_train, label_emotion

# Start the timer
start_time = time.time()
target_shape = (128,128)
# Process each emotion
angry_train, label_angry = process_emotion(angry, 0, 'angry', target_shape)
Disgusted_train, label_Disgusted = process_emotion(Disgusted, 1, 'Disgusted', target_shape)
Fearful_train, label_Fearful = process_emotion(Fearful, 2, 'Fearful', target_shape)
Happy_train, label_happy = process_emotion(Happy, 3, 'Happy', target_shape)
Neutral_train, label_Neutral = process_emotion(Neutral, 4, 'Neutral', target_shape)
Sad_train, label_Sad = process_emotion(Sad, 5, 'Sad', target_shape)

# Stop the timer
end_time = time.time()

# Calculate the total execution time
execution_time = end_time - start_time

# Print the total execution time
print(f"Total execution time: {execution_time} seconds")

# main_script.py
import torch
import multiprocessing
from train_test import train_and_test
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from skimage.transform import resize

# Combine the preprocessed data
exp_train = angry_train + Disgusted_train + Fearful_train + Happy_train + Neutral_train + Sad_train
exp_label = label_angry + label_Disgusted + label_Fearful + label_happy + label_Neutral + label_Sad

# Check the number of unique classes
num_classes = len(set(exp_label))

# Resize all spectrograms to the same size
target_shape = (128, 128)
exp_train_resized = [resize(spec.squeeze(), target_shape) for spec in exp_train]

# Ensure all spectrograms have the same shape
exp_train_reshaped = []
exp_label_reshaped = []
for x, label in zip(exp_train_resized, exp_label):
    try:
        reshaped_x = x.reshape(1, 128, 128)
        exp_train_reshaped.append(reshaped_x)
        exp_label_reshaped.append(label)
    except ValueError:
        continue

# Convert lists to tensors
exp_train_tensor = torch.stack([torch.from_numpy(x) for x in exp_train_reshaped])
exp_label_tensor = torch.tensor(exp_label_reshaped)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(exp_train_tensor, exp_label_tensor, test_size=0.25, random_state=42)

# Define the number of CPUs to use in the loop
num_cpus_list = [1, 2, 4, 8]

# Dictionary to store results for each number of CPUs
results_dict = {}

# Loop over different numbers of CPUs
for num_cpus in num_cpus_list:
    results_dict[num_cpus] = train_and_test(X_train, y_train, X_test, y_test, num_cpus)

# Access the training_time_list from the results_dict and print it
for num_cpus, training_time_list in results_dict.items():
    print(f'Training times with {num_cpus} CPU/s:', training_time_list)
    
# Save the results dictionary for later use in Jupyter Notebook
import pickle

with open('results_dict.pkl', 'wb') as f:
    pickle.dump(results_dict, f)