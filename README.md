# Audio-Emotion-Classifier-with-Deep-Learning-and-Parallel-Computing

## Problem Statement : 
Emotion recognition from audio data is a challenging and important problem with applications in many fields such as human-computer interaction, music information retrieval, and mental health care. The complexity and variability of human emotions make this a difficult task.


## Background : 
Traditional methods for emotion recognition from audio often involve handcrafted features and shallow learning models. However, these methods may not capture the complex patterns associated with different emotions effectively.


## Motivation : 
With the advent of deep learning, there is an opportunity to automatically learn more expressive features from audio data for emotion recognition. Furthermore, the use of parallel computing can potentially speed up the training process and enable the handling of large-scale datasets.


## Goal : 
The goal of this project is to develop a deep learning-based system for audio emotion recognition that leverages parallel computing capabilities of PyTorch to improve efficiency.


## Methodology :
•	Data Preprocessing : The raw audio data will be preprocessed into a suitable format for deep learning. This may involve techniques such as Fourier Transform for converting time-domain signals into frequency-domain spectrograms, or Mel-frequency cepstral coefficients (MFCCs) extraction which are commonly used features in speech and audio processing.

•	Model Architecture : A suitable deep learning model will be chosen for the task. Convolutional Neural Networks (CNNs) or Recurrent Neural Networks (RNNs) could be potential choices given their success in audio and sequence data respectively. The model will be trained to classify different emotions based on the processed audio data.

•	Parallel Computing : PyTorch’s built-in capabilities for parallel computing will be used to speed up the training process. This could involve techniques such as model parallelism (splitting a single model across multiple GPUs) or data parallelism (splitting the data across multiple GPUs and training separate models on each).

•	Evaluation : The performance of the model will be evaluated using suitable metrics such as accuracy, precision, recall, or F1 score. A separate test set will be used for this purpose to ensure an unbiased estimate of the model’s performance.


## Description of Dataset
This dataset is a comprehensive collection of audio files from four different sources: RAVDESS, CREMA-D, SAVEE, and TESS. The recordings are sorted into seven categories based on the emotion expressed: Angry, Happy, Sad, Neutral, Fearful, Disgusted, and Surprised.

Here’s a brief breakdown:

•	Angry: Contains 2167 records, making up 16.7% of the dataset.
•	Happy: Contains 2167 records, making up 16.46% of the dataset.
•	Sad: Contains 2167 records, making up 16.35% of the dataset.
•	Neutral: Contains 1795 records, making up 14.26% of the dataset.
•	Fearful: Contains 2047 records, making up 16.46% of the dataset.
•	Disgusted: Contains 1863 records, making up 15.03% of the dataset.
•	Surprised: Contains 592 records, making up 4.74% of the dataset.

In terms of the source of these files:

•	CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset) contributes the most with 7442 files, which is about 58.15% of the total data.
•	TESS (Transiting Exoplanet Survey Satellite) provides 2800 files, approximately 21.88% of the total data.
•	RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) adds another 2076 files, around 16.22% of the total data.
•	SAVEE (Surrey Audio-Visual Expressed Emotion) contributes the least with 480 files, making up about 3.75% of the total data.


## Data Source
https://www.kaggle.com/datasets/uldisvalainis/audio-emotions
