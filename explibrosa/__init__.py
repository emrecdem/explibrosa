import os
import elanwriter
import numpy as np
import pathlib
import pandas as pd
import warnings

import librosa
import librosa.display

__version__ = "0.0.0.dev1"

def get_info(audio_file_path):
    """


    """
    wave, fs = librosa.load(audio_file_path, sr=None)
    recordingDurationMinutes = (wave.size / fs) / 60
    recordingDurationMinutes = round(recordingDurationMinutes, 1)

    return {"duration (min)": recordingDurationMinutes, "Sample freq (kHz)": fs/1000}