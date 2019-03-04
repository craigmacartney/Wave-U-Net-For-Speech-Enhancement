import os.path
import subprocess

import librosa
import numpy as np
import tensorflow as tf
from soundfile import SoundFile

import Utils
import Metadata

def get_multitrack_placeholders(shape, num_sources, input_shape=None, name=""):
    '''
    Creates Tensorflow placeholders for mixture, background noise and speech.
    :param shape: Shape of each individual sample
    :return: List of multitrack placeholders for mixture, noise and speech
    '''
    if input_shape is None:
        input_shape = shape
    mix = (tf.placeholder(dtype=tf.float32, shape=input_shape, name="mix_input" + name))

    sources = list()
    for i in range(num_sources):
        sources.append(tf.placeholder(dtype=tf.float32, shape=shape, name="source_" + str(i) + "_input" + name))
    return mix, sources


def random_amplify(magnitude):
    '''
    Randomly amplifies or attenuates the input magnitudes
    :param magnitude: SINGLE Magnitude spectrogram excerpt, or list of spectrogram excerpts that each have their own amplification factor
    :return: Amplified magnitude spectrogram
    '''
    if isinstance(magnitude, np.ndarray):
        return np.random.uniform(0.7, 1.0) * magnitude
    else:
        assert(isinstance(magnitude, list))
        factor = np.random.uniform(0.7, 1.0)
        for i in range(len(magnitude)):
            magnitude[i] = factor * magnitude[i]
        return magnitude

def readWave(audio_path, start_frame, end_frame, mono=True, sample_rate=None, clip=True):
    snd_file = SoundFile(audio_path, mode='r')
    inf = snd_file._info
    audio_sr = inf.samplerate

    start_read = max(start_frame, 0)
    pad_front = -min(start_frame, 0)
    end_read = min(end_frame, inf.frames)
    pad_back = max(end_frame - inf.frames, 0)

    snd_file.seek(start_read)
    audio = snd_file.read(end_read - start_read, dtype='float32', always_2d=True) # (num_frames, channels)
    snd_file.close()

    # Pad if necessary (start_frame or end_frame out of bounds)
    audio = np.pad(audio, [(pad_front, pad_back), (0, 0)], mode="constant", constant_values=0.0)

    # Convert to mono if desired
    if mono:
        audio = np.mean(audio, axis=1, keepdims=True)

    # Resample if needed
    if sample_rate is not None and sample_rate != audio_sr:
        res_length = int(np.ceil(float(audio.shape[0]) * float(sample_rate) / float(audio_sr)))
        audio = np.pad(audio, [(1, 1), (0,0)], mode="reflect")  # Pad audio first
        audio = Utils.resample(audio, audio_sr, sample_rate)
        skip = (audio.shape[0] - res_length) // 2
        audio = audio[skip:skip+res_length,:]

    # Clip to [-1,1] if desired
    if clip:
        audio = np.minimum(np.maximum(audio, -1.0), 1.0)

    return audio, audio_sr

def readAudio(audio_path, offset=0.0, duration=None, mono=True, sample_rate=None, clip=True, pad_frames=0, metadata=None):
    '''
    Reads an audio file wholly or partly, and optionally converts it to mono and changes sampling rate.
    By default, it loads the whole audio file. If the offset is set to None, the duration HAS to be not None,
    and the offset is then randomly determined so that a random section of the audio is selected with the desired duration.
    Optionally, the file can be zero-padded by a certain amount of seconds at the start and end before selecting this random section.

    :param audio_path: Path to audio file
    :param offset: Position in audio file (s) where to start reading. If None, duration has to be not None, and position will be randomly determined.
    :param duration: How many seconds of audio to read
    :param mono: Convert to mono after reading
    :param sample_rate: Convert to given sampling rate if given
    :param pad_frames: number of frames with wich to pad the audio at most if the samples at the borders are not available
    :param metadata: metadata about audio file, accelerates reading audio since duration does not need to be determined from file
    :return: Audio signal, Audio sample rate
    '''

    if os.path.splitext(audio_path)[1][1:].lower() == "mp3":  # If its an MP3, call ffmpeg with offset and duration parameters
        # Get mp3 metadata information and duration
        if metadata is None:
            audio_sr, audio_channels, audio_duration = Metadata.get_mp3_metadata(audio_path)
        else:
            audio_sr = metadata[0]
            audio_channels = metadata[1]
            audio_duration = metadata[2]
        print(audio_duration)

        pad_front_duration = 0.0
        pad_back_duration = 0.0

        ref_sr = sample_rate if sample_rate is not None else audio_sr
        padding_duration = float(pad_frames) / float(ref_sr)

        if offset is None:  # In this case, select random section of audio file
            assert (duration is not None)
            max_start_pos = audio_duration+2*padding_duration-duration
            if (max_start_pos <= 0.0):  # If audio file is longer than duration of desired section, take all of it, will be padded later
                print("WARNING: Audio file " + audio_path + " has length " + str(audio_duration) + " but is expected to be at least " + str(duration))
                return Utils.load(audio_path, sample_rate, mono)  # Return whole audio file
            start_pos = np.random.uniform(0.0,max_start_pos) # Otherwise randomly determine audio section, taking padding on both sides into account
            offset = max(start_pos - padding_duration, 0.0) # Read from this position in audio file
            pad_front_duration = max(padding_duration - start_pos, 0.0)
        assert (offset is not None)

        if duration is not None: # Adjust duration if it overlaps with end of track
            pad_back_duration = max(offset + duration - audio_duration, 0.0)
            duration = duration - pad_front_duration - pad_back_duration # Subtract padding from the amount we have to read from file
        else: # None duration: Read from offset to end of file
            duration = audio_duration - offset

        pad_front_frames = int(pad_front_duration * float(audio_sr))
        pad_back_frames = int(pad_back_duration * float(audio_sr))


        args = ['ffmpeg', '-noaccurate_seek',
                '-ss', str(offset),
                '-t', str(duration),
                '-i', audio_path,
                '-f', 's16le', '-']

        audio = []
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=open(os.devnull, 'wb'))
        num_reads = 0
        while True:
            output = process.stdout.read(4096)
            if output == '' and process.poll() is not None:
                break
            if output:
                audio.append(librosa.util.buf_to_float(output, dtype=np.float32))
                num_reads += 1

        audio = np.concatenate(audio)
        if audio_channels > 1:
            audio = audio.reshape((-1, audio_channels)).T

    else: #Not an MP3: Handle with PySoundFile
        # open audio file
        snd_file = SoundFile(audio_path, mode='r')
        inf = snd_file._info
        audio_sr = inf.samplerate

        pad_orig_frames = pad_frames if sample_rate is None else int(np.ceil(float(pad_frames) * (float(audio_sr) / float(sample_rate))))

        pad_front_frames = 0
        pad_back_frames = 0

        if offset is not None and duration is not None:
            start_frame = int(offset * float(audio_sr))
            read_frames = int(duration * float(audio_sr))
        elif offset is not None and duration is None:
            start_frame = int(offset * float(audio_sr))
            read_frames = inf.frames - start_frame
        else:  # In this case, select random section of audio file
            assert (offset is None)
            assert (duration is not None)
            num_frames = int(duration * float(audio_sr))
            max_start_pos = inf.frames - num_frames # Maximum start position when ignoring padding on both ends of the file
            if (max_start_pos <= 0):  # If audio file is longer than duration of desired section, take all of it, will be padded later
                print("WARNING: Audio file " + audio_path + " has frames  " + str(inf.frames) + " but is expected to be at least " + str(num_frames))
                raise Exception("Could not read minimum required amount of audio data")
                #return Utils.load(audio_path, sample_rate, mono)  # Return whole audio file
            start_pos = np.random.randint(0, max_start_pos)  # Otherwise randomly determine audio section, taking padding on both sides into account

            # Translate source position into mixture input positions (take into account padding)
            start_mix_pos = start_pos - pad_orig_frames
            num_mix_frames = num_frames + 2*pad_orig_frames
            end_mix_pos = start_mix_pos + num_mix_frames

            # Now see how much of the mixture is available, pad the rest with zeros

            start_frame = max(start_mix_pos, 0)
            end_frame = min(end_mix_pos, inf.frames)
            read_frames = end_frame - start_frame
            pad_front_frames = -min(start_mix_pos, 0)
            pad_back_frames = max(end_mix_pos - inf.frames, 0)

        assert(num_frames > 0)
        snd_file.seek(start_frame)
        audio = snd_file.read(read_frames, dtype='float32', always_2d=True)
        snd_file.close()

        centre_start_frame = start_pos
        centre_end_frame = start_pos + num_frames

    # Pad as indicated at beginning and end
    audio = np.pad(audio, [(pad_front_frames, pad_back_frames), (0,0)],mode="constant",constant_values=0.0)

    # Convert to mono if desired
    if mono:
        audio = np.mean(audio, axis=1, keepdims=True)

    # Resample if needed
    if sample_rate is not None and sample_rate != audio_sr:
        audio = Utils.resample(audio, audio_sr, sample_rate)

    # Clip to [-1,1] if desired
    if clip:
        audio = np.minimum(np.maximum(audio, -1.0), 1.0)

    if float(audio.shape[0])/float(sample_rate) < 1.0:
        raise IOError("Error while reading " + audio_path + " - ended up with audio shorter than one second!")

    if os.path.splitext(audio_path)[1][1:].lower() == "mp3":
        return audio, audio_sr, offset, offset+duration
    else:
        return audio, audio_sr, centre_start_frame, centre_end_frame, start_mix_pos, end_mix_pos
