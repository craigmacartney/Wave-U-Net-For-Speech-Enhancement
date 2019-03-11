'''
Purpose of this file is to evaulate performance of a model in different noise environments. To run this file, please install pypesq hosted at:https://github.com/vBaiCai/python-pesq
'''
import librosa
from pypesq import pesq
import pandas as pd 
import numpy as np
import argparse
import os
from math import log10
parser = argparse.ArgumentParser()
parser.add_argument("test_folder",help="directory with files pertaining to testing")


def get_pesq(filepath_1, filepath_2):
    '''
    argument:
        filepath_1: original clean .wav file path
        filepath_2: generated clean .wav file path
    return:
        PESQ score
    '''
    clean_wave, clean_sampe_rate = librosa.load(filepath_1, sr=16000)
    cleaned_wave, cleaned_sampe_rate = librosa.load(filepath_2, sr=16000)

    signal_power = 20*log10(np.mean(np.square(clean_wave)))

    score = pesq(clean_wave, cleaned_wave, clean_sampe_rate)
    
    return score,signal_power

if __name__ =="__main__":
    args = parser.parse_args()
    assert os.path.exists(args.test_folder),"Please enter correct directory"

    folders = os.listdir(args.test_folder)
    data = {"id":[],"SNR":[],"PESQ":[]}
    for folder in folders:
        data["id"].append(folder)
        filepath1 = "{0}/{1}/clean.wav".format(args.test_folder,folder)
        filepath2 = "{0}/{1}/mixed_speech.wav".format(args.test_folder,folder)
        noise,s = librosa.load("{0}/{1}/noise.wav".format(args.test_folder,folder),sr=16000)
        noise_power = 20*log10(np.mean(np.square(noise)))
        score,signal_power = get_pesq(filepath1,filepath2)
        data["SNR"].append(signal_power-noise_power)
        data["PESQ"].append(score)

    data = pd.DataFrame(data)
    data.set_index("id")
    data.to_csv("analysis.csv")

