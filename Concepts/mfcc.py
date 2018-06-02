import os
import warnings
warnings.filterwarnings("ignore")


import numpy as np
from scipy import signal
from scipy.io import wavfile
import librosa


import matplotlib.pyplot as plt
import librosa.display

from matplotlib.pyplot import *
from classifiers import *

def mfcc_compute(filenames, classes, root_path, n_mfcc=13, sample_length=16000, hop_length=512, n_mels=128):
    x_mfcc = np.zeros(shape=(len(filenames), n_mfcc, int(np.ceil(sample_length/hop_length))))
    y_mfcc = np.zeros(shape=(len(filenames), 1))
    print(x_mfcc.shape)
    
    # For each file
    for i, file in enumerate(filenames):
        print("Progress: {0:.04f}".format(i+1/len(filenames)), end="\r")
        
        # Save label
        file_class = file.split("/")[0]
        print(i, file_class, np.where(classes == file_class))
        y_mfcc[i] = np.where(classes == file_class)[0][0]
        
        # Read .wav file
        sample_rate, samples = wavfile.read(os.path.join(root_path, file))
        
        # Find mel spectrogram
        S = librosa.feature.melspectrogram(samples, sr=sample_rate, hop_length=hop_length, n_mels=n_mels)
        
        # Find log-power Mel spectrogram
        log_S = librosa.power_to_db(S, ref=np.max)
        
        # Find MFCC
        mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=n_mfcc)
        
        # Find 2nd order delta_mfcc
        delta_mfcc = librosa.feature.delta(mfcc, order=2)
        print(delta_mfcc.shape[1])
        
        x_mfcc[i] = delta_mfcc
        
    return x_mfcc, y_mfcc
    
# load from wave files into training, validation and test splits
train_filenames=[]
for i in unique_classes:
    temp_filenames=next(os.walk('../Datasets/audio/train/audio/'+i+'/'))[2]
    temp_filenames=[i+'/' + line for line in temp_filenames]
    train_filenames+=temp_filenames

val_filenames=[]
for i in unique_classes:
    temp_filenames=next(os.walk('../Datasets/audio/train/valid/'+i+'/'))[2]
    temp_filenames=[i+'/' + line for line in temp_filenames]
    val_filenames+=temp_filenames

temp_test_filenames=open('../Datasets/audio/testing_list.txt','r').readlines()
test_filenames=[]
for i in temp_test_filenames:
    test_filenames.append(i[:-1])
    
# Compute MFCC features for each split

mfcc_features_train, mfcc_labels_train = mfcc_compute(train_filenames, unique_classes, root_path='../Datasets/audio/train/audio/')
mfcc_features_val, mfcc_labels_val = mfcc_compute(val_filenames, unique_classes, root_path='../Datasets/audio/train/valid/')

mfcc_features_test, mfcc_labels_test = mfcc_compute(test_filenames, unique_classes, root_path='../Datasets/audio/')

# split into train, validation and test
x_train = np.reshape(xtrain, [len(xtrain), 13*32])
x_val = np.reshape(xval, [len(xval), 13*32])
x_test = np.reshape(xtest, [len(xtest), 13*32])
print(x_train.shape, x_val.shape, x_test.shape)

# Save file
import scipy.io as sio
sio.savemat('tf_speech_mfcc.mat', mdict={'unique_classes':unique_classes,'x_train':x_train,'ytrain':ytrain,'x_val':x_val,'yval':yval,'x_test':x_test, 'ytest':ytest})
