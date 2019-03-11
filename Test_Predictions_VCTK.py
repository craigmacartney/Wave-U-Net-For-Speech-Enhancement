# -*- coding: utf-8 -*-

from sacred import Experiment
from sacred.observers import FileStorageObserver
from Config import config_ingredient
import Estimate_Sources
import os


folders = os.listdir('test_data')
test_file_list = []

for folder in folders:
    test_file_list.append('test_data/{0}/mixed.wav'.format(folder))
    

for index,name in enumerate(folders):
    ex = Experiment('Waveunet Prediction', ingredients=[config_ingredient])
    ex.observers.append(FileStorageObserver.create('my_runs/Predictions/' + name))

    @ex.config
    def cfg():
        model_path = os.path.join("checkpoints", "866537", "866537-168084") # Load model from checkpoints folder. E.g. a particular model, "105373-450225" from "checkpoints/105373"
        input_path = test_file_list[index]# Which audio file to separate. In this example, within path "audio_examples/noisy_testset_wav/p*.wav"
        output_path = test_file_list[index][:-9] # Where to save results. Default: Same location as input.

    @ex.automain
    def main(cfg, model_path, input_path, output_path):
        model_config = cfg["model_config"]
        Estimate_Sources.produce_source_estimates(model_config, model_path, input_path, output_path)
