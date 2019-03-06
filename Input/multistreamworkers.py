import numpy as np
import Input
from Sample import Sample

class MultistreamWorker_GetSpectrogram:
    @staticmethod
    def run(communication_queue, exit_flag, options):
        '''
        Worker method that reads audio from a given file list and appends the processed spectrograms to the cache queue.
        :param communication_queue: Queue of the cache from which examples are added to the cache
        :param exit_flag: Flag to indicate when to exit the process
        :param options: Audio processing parameters and file list
        '''

        filename_list = options["file_list"]
        num_files = len(filename_list)

        # Alternative method: Load data in completely into RAM. Use only when data is small enough to fit into RAM and num_workers=1 in this case!
        #n_fft = options['num_fft']
        #hop_length = options['num_hop']

        # Re-seed RNG for this process
        np.random.seed()

        while not exit_flag.is_set():
            # Decide which element to read next
            id_file_to_read = np.random.randint(num_files)
            item = filename_list[id_file_to_read]

            # Calculate the required amounts of padding
            duration_frames = int(options["duration"] * options["expected_sr"])
            padding_duration = options["padding_duration"]
            num_files = 0
            skip_files = 0
            try:
                if isinstance(item, Sample):  # Single audio file: Use metadata to read section from it
                    metadata = [item.sample_rate, item.channels, item.duration]
                    audio, _, _, _ = Input.readAudio(item.path, offset=None, duration=options["duration"],
                                                     sample_rate=options["expected_sr"], pad_frames=options["pad_frames"],
                                                     metadata=metadata, mono=options["mono_downmix"])

                    communication_queue.put(audio)

                elif isinstance(item,
                                float):  # This means the track is a (not as file existant) silence track so we insert a zero spectrogram

                    communication_queue.put(np.zeros(duration_frames, np.float32))
                else:
                    assert (hasattr(item, '__iter__')) # Supervised case: Item is a list of files to read, starting with the mixture
                    # We want to get the spectrogram of the mixture (first entry in list), and of the sources and store them in cache as one training sample
                    sample = list()
                    #N.B. under assumption that sources are additive we can only load sources in, then add them to form mixture!
                    file = item[0]
                    metadata = [file.sample_rate, file.channels, file.duration]
                    mix_audio, mix_sr, source_start_frame, source_end_frame, start_read, end_read = Input.readAudio(file.path, offset=None, duration=options[ "duration"],
                                                                                              sample_rate=options[ "expected_sr"],
                                                                                              pad_frames=options["pad_frames"],
                                                                                              metadata=metadata,
                                                                                              mono = options["mono_downmix"])
                    sample.append(mix_audio)

                    for file in item[1:]:
                        if isinstance(file, Sample):
                            if options["augmentation"]:
                                source_audio, _ = Input.readWave(file.path, start_read, end_read,
                                                                 sample_rate=options["expected_sr"],
                                                                 mono=options["mono_downmix"])
                            else:
                                source_audio, _ = Input.readWave(file.path, source_start_frame, source_end_frame, sample_rate=options["expected_sr"], mono=options["mono_downmix"])
                        else:
                            assert (isinstance(file, float))  # This source is silent in this track

                            source_audio = np.zeros(mix_audio.shape, np.float32)
                        sample.append(source_audio)

                    # Check correct number of output channels
                    try:
                        assert (sample[0].shape[1] == options["num_channels"]
                                and sample[1].shape[1] == options["num_channels"]
                                and sample[2].shape[1] == options["num_channels"])
                    except Exception as e:
                        print("WARNING: Song " + file.path + " seems to be mono, will duplicate channels to convert into stereo for training!")
                        print("Channels for mix and sources" + str([sample[i].shape[1] for i in range(len(sample))]))

                    if options["augmentation"]: # Random attenuation of source signals
                        mix_audio = np.zeros(sample[0].shape, np.float32)
                        for i in range(1, len(sample)):
                            amped_audio = Input.random_amplify(sample[i])
                            mix_audio += amped_audio
                            if options["pad_frames"] > 0:
                                amped_audio = amped_audio[options["pad_frames"]:-options["pad_frames"]]
                            sample[i] = amped_audio

                        sample[0] = mix_audio

                    communication_queue.put(sample)
                    num_files = num_files + 1
                    #print("No Error while computing spectrogram. Skipping file.",skip_files/(skip_files+num_files))
            except Exception as e:
                skip_files = skip_files + 1
                print(e)
                print("Error while computing spectrogram. Skipping file.",skip_files/(skip_files+num_files))

        #print("Number of files pushed to training :{}".format(num_files))
        #print("Number of files skipped to training :{}".format(skip_files))
        # This is necessary so that this process does not block. In particular, if there are elements still in the queue
        # from this process that were not yet 'picked up' somewhere else, join and terminate called on this process will
        # block
        communication_queue.cancel_join_thread()
