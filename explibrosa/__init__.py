import os
import sys
import elanwriter
import numpy as np
import pathlib
import pandas as pd
import warnings
import time
import librosa

import exploface

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


def get_feature_time_series(audio_file_path, output_directory, cut_start=None, cut_end=None):
    start_time = time.time()
    
    wave, fs = librosa.load(audio_file_path, sr=None)

    if cut_start!=None and cut_end!=None:
        i_start = ((fs * cut_start*60))
        i_end = ((fs * cut_end*60))
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
    print("     %s seconds" % round(time.time() - start_time, 2))
    
    # THIS NEEDS CHECKING!
    t_frame = np.linspace(0, (len(rmse))*0.01, (len(rmse)))

    print("Zero crossing")
    # calculate zero-crossing rate
    zrc = librosa.feature.zero_crossing_rate(wave, frame_length=frame_len, hop_length=frame_shift, threshold=0)
    zrc = zrc[0]
    zrc = librosa.util.normalize(zrc, axis=0) # normalize first axis to -1,1
    print("     %s seconds" % round(time.time() - start_time, 2))

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
    print("     %s seconds" % round(time.time() - start_time,2))
    print("  Pitches smoothing")
    pitch_track = extract_max(pitches, pitches.shape)
    pitch_smoothtrack = smooth(pitch_track, window_len=10)
    print("     %s seconds" % round(time.time() - start_time, 2))

    features = pd.DataFrame({"timestamp": t_frame, "rmse":rmse, "zrc":zrc, "pitch":pitch_smoothtrack})
    #original_wave = pd.DataFrame({"t_wave": t, "wave": wave})

    # Outputting results
    input_filename = os.path.splitext(os.path.basename(audio_file_path))[0]
    output_path = os.path.join(output_directory, input_filename)
    feature_file_path = output_path+"_features.csv"
    features.to_csv(feature_file_path)
    #original_wave.to_csv(output_path+"_wave.csv")

    print("TOTAL execution time: %s min" % round((time.time() - start_time)/60, 2))

    return features, feature_file_path #original_wave


def get_detections(features):
    thresholds = {
        "rmse": 0.05,
        "zrc": 0.6,
        "pitch": 1500,
    }

    feature_name_list, start_list, end_list = [], [], []


    feature_names = features.columns
    for feature_name in feature_names:
        if feature_name != "timestamp":
            threshold = thresholds[feature_name]
            times = exploface.extraction.get_activation_times(
                        df=features,
                        emo_key=feature_name, 
                        intensity_threshold=threshold, 
                        method="threshold",
                        confidence_cut = None,
                        inverse_threshold = False, 
                        smooth_time_threshold = 0.7,
                        time_threshold = None,
                        )

            for t in times:
                #if t[0] < df["timestamp"].iloc[-1] - skip_seconds_at_end:
                feature_name_list.append(feature_name)
                start_list.append(t[0])
                end_list.append(t[1])

    return pd.DataFrame({"start": start_list, "end":end_list, "feature": feature_name_list})
    

    #for index,row in features.itterrows():
    #    if row["rmse"] 

def write_elan_file(features, video_file, output_path):

    detections = get_detections(features)

        ##
    # Make the Elan file
    ##
    if output_path:
        #output_directory = os.path.dirname(output_path)
        #output_filename = os.path.basename(output_path)

        #if not column_selection:
        column_selection = set(detections["feature"])

        ed = elanwriter.ElanDoc(video_file)
        
        ##
        # Start looping over the FEATURES
        for feat in column_selection:
            
            times = detections[detections["feature"] == feat]
            for i in range(len(times)):
                annotation_name = times.iloc[i]["feature"]
                #if "modifier" in times.columns:
                #    if times.iloc[i]["modifier"] and not np.isnan(times.iloc[i]["modifier"]):
                annotation_name = feat#"I="+str(times.iloc[i]["modifier"])

                ed.add_annotation((1000*times.iloc[i]["start"], 1000*times.iloc[i]["end"]), 
                                  annotation_name, tier_name=times.iloc[i]["feature"])
            if len(times) == 0:
                ed.add_annotation((0,0.1), 
                                  "", feat)

        ed.write(output_path)