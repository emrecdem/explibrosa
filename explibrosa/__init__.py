import os
import sys
import elanwriter
import numpy as np
import pathlib
import pandas as pd
import warnings

import librosa

import matplotlib.pyplot as plt

__version__ = "0.0.0.dev1"

def get_info(audio_file_path):
    """


    """
    wave, fs = librosa.load(audio_file_path, sr=None)
    recordingDurationMinutes = (wave.size / fs) / 60
    recordingDurationMinutes = round(recordingDurationMinutes, 1)

    return {"#frames":wave.size, 
        "duration (min)": recordingDurationMinutes, 
        "Sample freq (kHz)": fs/1000}


def get_feature_time_series(audio_file_path, cut_start=None, cut_end=None):

    wave, fs = librosa.load(audio_file_path, sr=None)

    print("hier")
    if cut_start!=None and cut_end!=None:
        print("Cutting")
        #midpoint = round(wave.size / 2)
        print(fs, wave.size)
        i_start = ((fs * cut_start*60))
        i_end = ((fs * cut_end*60))
        print(i_start, i_end)
        wave = wave[i_start : i_end ]

    t = np.linspace(0, len(wave)/fs, len(wave))

    frame_len = int(20 * fs /1000) # 20ms
    frame_shift = int(10 * fs /1000) # 10ms

    print("RMS energy")
    # calculate RMS energy for each frame
    rmse = librosa.feature.rmse(wave, frame_length=frame_len, hop_length=frame_shift)
    print("rmse", len(rmse[0]), len(wave), frame_shift, (len(rmse[0])-1)*(frame_shift))
    rmse = rmse[0]
    rmse = librosa.util.normalize(rmse, axis=0) # normalize first axis to -1,1
    print(rmse)

    print("Zero crossing")
    # calculate zero-crossing rate
    zrc = librosa.feature.zero_crossing_rate(wave, frame_length=frame_len, hop_length=frame_shift, threshold=0)
    zrc = zrc[0]
    zrc = librosa.util.normalize(zrc, axis=0) # normalize first axis to -1,1

    # function needed for pitch detection
    def extract_max(pitches, shape):
        new_pitches = []
        for i in range(0, shape[1]):
            new_pitches.append(np.max(pitches[:,i]))
        return new_pitches

    def smooth(x,window_len=11,window='hanning'):
            if window_len<3:
                    return x
            if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                    raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
            s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
            if window == 'flat': #moving average
                    w=np.ones(window_len,'d')
            else:
                    w=eval('np.'+window+'(window_len)')
            y=np.convolve(w/w.sum(),s,mode='same')
            return y[window_len:-window_len+1]


    print("Pitches")
    # Slice a time series into overlapping
    frames = librosa.util.frame(wave, frame_length=frame_len, hop_length=frame_shift)
    # Pitch tracking on thresholded parabolically-interpolated STFT:
    pitches, magnitudes = librosa.core.piptrack(wave, sr=fs, hop_length=frame_shift, threshold=0.75)

    pitch_track = extract_max(pitches, pitches.shape)
    pitch_smoothtrack = smooth(pitch_track, window_len=10)

    # Plot RMSE and ZCR
    plt.figure(figsize=(12, 9))

    plt.subplot(4, 1, 2)
    plt.plot(rmse.T)
    plt.title('RMS Energy')
    #plt.xticks([])

    plt.subplot(4, 1, 3)
    plt.plot(pitch_smoothtrack)
    plt.title('Pitch (Hertz)')
    #plt.xticks([])

    plt.subplot(4, 1, 1)
    plt.plot(zrc.T)
    plt.title('zero-corssing rate')
    #plt.xticks([])

    plt.subplot(4, 1, 4)
    plt.plot(t, wave)
    plt.title('Waveform')
    plt.show()