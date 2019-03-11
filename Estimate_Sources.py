import numpy as np
import tensorflow as tf
import librosa

import os
import glob

from Input import Input
import Models.UnetAudioSeparator

import Utils

def predict(track, model_config, load_model):
    '''
    Takes audio track and computes corresponding source estimates.
    :param track: Track object
    :return: Source estimates dictionary
    '''

    # Determine input and output shapes, if we use U-net as separator
    disc_input_shape = [model_config["batch_size"], model_config["num_frames"], 0]  # Shape of discriminator input
    if model_config["network"] == "unet":
        separator_class = Models.UnetAudioSeparator.UnetAudioSeparator(model_config["num_layers"], model_config["num_initial_filters"],
                                                                   output_type=model_config["output_type"],
                                                                   context=model_config["context"],
                                                                   mono=model_config["mono_downmix"],
                                                                   upsampling=model_config["upsampling"],
                                                                   num_sources=model_config["num_sources"],
                                                                   filter_size=model_config["filter_size"],
                                                                   merge_filter_size=model_config["merge_filter_size"])

    else:
        raise NotImplementedError

    sep_input_shape, sep_output_shape = separator_class.get_padding(np.array(disc_input_shape))
    separator_func = separator_class.get_output

    # Batch size of 1
    sep_input_shape[0] = 1
    sep_output_shape[0] = 1

    mix_context, sources = Input.get_multitrack_placeholders(sep_output_shape, model_config["num_sources"], sep_input_shape, "input")

    print("Testing...")

    # BUILD MODELS
    # Separator
    separator_sources = separator_func(mix_context, False, reuse=False)

    # Start session and queue input threads
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Load model
    # Load pretrained model to continue training, if we are supposed to
    restorer = tf.train.Saver(None, write_version=tf.train.SaverDef.V2)
    print("Num of variables" + str(len(tf.global_variables())))
    restorer.restore(sess, load_model)
    print('Pre-trained model restored for prediction')

    mix_audio, orig_sr, mix_channels = track.audio, track.rate, track.audio.shape[1] # Audio has (n_samples, n_channels) shape
    separator_preds = predict_track(model_config, sess, mix_audio, orig_sr, sep_input_shape, sep_output_shape, separator_sources, mix_context)

    # Upsample predicted source audio and convert to stereo
    pred_audio = [Utils.resample(pred, model_config["expected_sr"], orig_sr) for pred in separator_preds]

    if model_config["mono_downmix"] and mix_channels > 1: # Convert to multichannel if mixture input was multichannel by duplicating mono estimate
        pred_audio = [np.tile(pred, [1, mix_channels]) for pred in pred_audio]

    # Set estimates for source separation task for speech enhancement
    estimates = {  # [noise, speech] order
        'speech' : pred_audio[1],
        'noise' : pred_audio[0] #comment-out this line to only yield speech file
    }

    # Close session, clear computational graph
    sess.close()
    tf.reset_default_graph()

    return estimates

def predict_track(model_config, sess, mix_audio, mix_sr, sep_input_shape, sep_output_shape, separator_sources, mix_context):
    '''
    Outputs source estimates for a given input mixture signal mix_audio [n_frames, n_channels] and a given Tensorflow session and placeholders belonging to the prediction network.
    It iterates through the audio track, collecting segment-wise predictions to form the output.
    :param model_config: Model configuration dictionary
    :param sess: Tensorflow session used to run the network inference
    :param mix_audio: [n_frames, n_channels] audio signal (numpy array). Can have higher sampling rate or channels than the model supports, will be downsampled correspondingly.
    :param mix_sr: Sampling rate of mix_audio
    :param sep_input_shape: Input shape of separator ([batch_size, num_samples, num_channels])
    :param sep_output_shape: Input shape of separator ([batch_size, num_samples, num_channels])
    :param separator_sources: List of Tensorflow tensors that represent the output of the separator network
    :param mix_context: Input tensor of the network
    :return:
    '''
    # Load mixture, convert to mono and downsample then
    assert(len(mix_audio.shape) == 2)
    if model_config["mono_downmix"]:
        mix_audio = np.mean(mix_audio, axis=1, keepdims=True)
    else:
        if mix_audio.shape[1] == 1:# Duplicate channels if input is mono but model is stereo
            mix_audio = np.tile(mix_audio, [1, 2])
    mix_audio = Utils.resample(mix_audio, mix_sr, model_config["expected_sr"])

    # Preallocate source predictions (same shape as input mixture)
    source_time_frames = mix_audio.shape[0]
    source_preds = [np.zeros(mix_audio.shape, np.float32) for _ in range(model_config["num_sources"])]

    input_time_frames = sep_input_shape[1]
    output_time_frames = sep_output_shape[1]

    # Pad mixture across time at beginning and end so that neural network can make prediction at the beginning and end of signal
    pad_time_frames = (input_time_frames - output_time_frames) / 2
    mix_audio_padded = np.pad(mix_audio, [(pad_time_frames, pad_time_frames), (0,0)], mode="constant", constant_values=0.0)

    # Iterate over mixture magnitudes, fetch network prediction
    for source_pos in range(0, source_time_frames, output_time_frames):
        # If this output patch would reach over the end of the source spectrogram, set it so we predict the very end of the output, then stop
        if source_pos + output_time_frames > source_time_frames:
            source_pos = source_time_frames - output_time_frames

        # Prepare mixture excerpt by selecting time interval
        mix_part = mix_audio_padded[source_pos:source_pos + input_time_frames,:]
        mix_part = np.expand_dims(mix_part, axis=0)

        source_parts = sess.run(separator_sources, feed_dict={mix_context: mix_part})

        # Save predictions
        for i in range(model_config["num_sources"]):
            source_preds[i][source_pos:source_pos + output_time_frames] = source_parts[i][0, :, :]

    return source_preds

def produce_source_estimates(model_config, load_model, input_path, output_path=None):
    '''
    For a given input mixture file, saves source predictions made by a given model.
    :param model_config: Model configuration
    :param load_model: Model checkpoint path
    :param input_path: Path to input mixture audio file
    :param output_path: Output directory where estimated sources should be saved. Defaults to the same folder as the input file, if not given
    :return: Dictionary of source estimates containing the source signals as numpy arrays
    '''
    print("Producing source estimates for input mixture file " + input_path)
    # Prepare input audio for prediction function
    audio, sr = Utils.load(input_path, sr=None, mono=False)
    # Create something that looks sufficiently like a track object to our MUSDB function
    class TrackLike(object):
        def __init__(self, audio, rate, shape):
            self.audio = audio
            self.rate = rate
            self.shape = shape
    track = TrackLike(audio, sr, audio.shape)

    sources_pred = predict(track, model_config, load_model) # Input track to prediction function, get source estimates

    # Save source estimates as audio files into output dictionary
    input_folder, input_filename = input_path[:-9],input_path[-9:-4]
    if output_path is None:
        # By default, set it to the input_path folder
        output_path = input_folder
    if not os.path.exists(output_path):
        print("WARNING: Given output path " + output_path + " does not exist. Trying to create it...")
        os.makedirs(output_path)
    assert(os.path.exists(output_path))
    for source_name, source_audio in sources_pred.items():
        librosa.output.write_wav(os.path.join(output_path, input_filename) + "_" + source_name + ".wav", source_audio, sr)
        
        
        
        
        
        
        
        
        
