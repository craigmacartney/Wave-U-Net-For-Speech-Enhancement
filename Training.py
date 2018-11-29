from sacred import Experiment
from sacred.observers import FileStorageObserver #CMedit

from Config import config_ingredient
import tensorflow as tf
import numpy as np
import os

import Datasets
from Input import Input as Input
from Input import batchgenerators as batchgen
import Utils
import Models.UnetAudioSeparator
import cPickle as pickle
import Validation

ex = Experiment('Waveunet Training', ingredients=[config_ingredient])
ex.observers.append(FileStorageObserver.create('my_runs'))

@config_ingredient.capture
def train(model_config, experiment_id, sup_dataset, load_model=None):
    # Determine input and output shapes
    disc_input_shape = [model_config["batch_size"], model_config["num_frames"], 0]  # Shape of input
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

    # Creating the batch generators
    assert((sep_input_shape[1] - sep_output_shape[1]) % 2 == 0)
    pad_durations = np.array([float((sep_input_shape[1] - sep_output_shape[1])/2), 0, 0]) / float(model_config["expected_sr"])  # Input context that the input audio has to be padded ON EACH SIDE
    sup_batch_gen = batchgen.BatchGen_Paired(
        model_config,
        sup_dataset,
        sep_input_shape,
        sep_output_shape,
        pad_durations[0]
    )

    print("Starting worker")
    sup_batch_gen.start_workers()
    print("Started worker!")

    # Placeholders and input normalisation
    mix_context, sources = Input.get_multitrack_placeholders(sep_output_shape, model_config["num_sources"], sep_input_shape, "sup")
    #tf.summary.audio("mix", mix_context, 16000, collections=["sup"]) #Enable listening to source estimates via Tensorboard
    mix = Utils.crop(mix_context, sep_output_shape)

    print("Training...")

    # BUILD MODELS
    # Separator
    separator_sources = separator_func(mix_context, True, not model_config["raw_audio_loss"], reuse=False) # Sources are output in order [noise, speech] for speech enhancement

    # Supervised objective: MSE in log-normalized magnitude space
    separator_loss = 0
    for (real_source, sep_source) in zip(sources, separator_sources):
        separator_loss += tf.reduce_mean(tf.square(real_source - sep_source))
    separator_loss = separator_loss / float(len(sources)) # Normalise by number of sources

    # TRAINING CONTROL VARIABLES
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int64)
    increment_global_step = tf.assign(global_step, global_step + 1)

    # Set up optimizers
    separator_vars = Utils.getTrainableVariables("separator")
    print("Sep_Vars: " + str(Utils.getNumParams(separator_vars)))
    print("Num of variables" + str(len(tf.global_variables())))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        with tf.variable_scope("separator_solver"):
            separator_solver = tf.train.AdamOptimizer(learning_rate=model_config["init_sup_sep_lr"]).minimize(separator_loss, var_list=separator_vars)

    # SUMMARIES
    tf.summary.scalar("sep_loss", separator_loss, collections=["sup"])
    sup_summaries = tf.summary.merge_all(key='sup')

    # Start session and queue input threads
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(model_config["log_dir"] + os.path.sep + str(experiment_id),graph=sess.graph)

    # CHECKPOINTING
    # Load pretrained model to continue training, if we are supposed to
    if load_model != None:
        restorer = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)
        print("Num of variables" + str(len(tf.global_variables())))
        restorer.restore(sess, load_model)
        print('Pre-trained model restored from file ' + load_model)

    saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)

    # Start training loop
    run = True
    _global_step = sess.run(global_step)
    _init_step = _global_step
    it = 0
    while run:
        # TRAIN SEPARATOR
        sup_batch = sup_batch_gen.get_batch()
        feed = {i:d for i,d in zip(sources, sup_batch[1:])}
        feed.update({mix_context : sup_batch[0]})
        _, _sup_summaries = sess.run([separator_solver, sup_summaries], feed)
        writer.add_summary(_sup_summaries, global_step=_global_step)

        # Increment step counter, check if maximum iterations per epoch is achieved and stop in that case
        _global_step = sess.run(increment_global_step)

        if _global_step - _init_step > model_config["epoch_it"]:
            run = False
            print("Finished training phase, stopping batch generators")
            sup_batch_gen.stop_workers()

    # Epoch finished - Save model
    print("Finished epoch!")
    save_path = saver.save(sess, model_config["model_base_dir"] + os.path.sep + str(experiment_id) + os.path.sep + str(experiment_id), global_step=int(_global_step))

    # Close session, clear computational graph
    writer.flush()
    writer.close()
    sess.close()
    tf.reset_default_graph()

    return save_path

@config_ingredient.capture
def optimise(model_config, experiment_id, dataset):
    epoch = 0
    best_loss = 10000
    model_path = None
    best_model_path = None
    for i in range(2):
        worse_epochs = 0
        if i==1:
            print("Finished first round of training, now entering fine-tuning stage")
            model_config["batch_size"] *= 2
            model_config["cache_size"] *= 2
            model_config["min_replacement_rate"] *= 2
            model_config["init_sup_sep_lr"] = 1e-5
        while worse_epochs < model_config["worse_epochs"]: # Early stopping on validation set after a few epochs
            print("EPOCH: " + str(epoch))
            model_path = train(sup_dataset=dataset["train"], load_model=model_path)
            curr_loss = Validation.test(model_config, model_folder=str(experiment_id), audio_list=dataset["valid"], load_model=model_path)
            epoch += 1
            if curr_loss < best_loss:
                worse_epochs = 0
                print("Performance on validation set improved from " + str(best_loss) + " to " + str(curr_loss))
                best_model_path = model_path
                best_loss = curr_loss
            else:
                worse_epochs += 1
                print("Performance on validation set worsened to " + str(curr_loss))
    print("TRAINING FINISHED - TESTING NOW AVAILABLE WITH BEST MODEL " + best_model_path)

@ex.automain
def run(cfg):
    model_config = cfg["model_config"]
    print("SCRIPT START")
    # Create subfolders if they do not exist to save results
    for dir in [model_config["model_base_dir"], model_config["log_dir"]]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # Set up data input
    pickle_file = "dataset.pkl"
    if os.path.exists(pickle_file): # Check whether our dataset file is already there, then load it
        with open(pickle_file, 'r') as file:
            dataset = pickle.load(file)
        print("Loaded dataset from pickle!")
    else: # Otherwise create the dataset pickle

        print("Preparing dataset! This could take a while...")

        # Specify path to dataset, as a tracklist composed by an XML file parsed using etree in Datasets.getAudioData
        # Each track element, describing 3 sources [speech.wav, noise.wav, mix.wav] and their relevant metadata, is parsed using etree in Datasets.py
        dataset_train = Datasets.getAudioData("")

        # Pick 10 random songs for validation from train set (this is always the same selection each time since the random seed is fixed)
        val_idx = np.random.choice(len(dataset_train), size=10, replace=False)
        train_idx = [i for i in range(len(dataset_train)) if i not in val_idx]
        print("Validation with training items no. " + str(train_idx))

        # Draw randomly from datasets
        dataset = dict()
        dataset["train"] = dataset_train
        dataset["valid"] = [dataset_train[i] for i in val_idx]

        # Now create dataset, for source separation task for speech enhancement
        assert(model_config["task"] == "speech")

        for subset in ["train", "valid"]:
            for i in range(len(dataset[subset])):
                dataset[subset][i] = (dataset[subset][i][0], dataset[subset][i][1], dataset[subset][i][2])

        # Save dataset
        with open("dataset.pkl", 'wb') as file:
            pickle.dump(dataset, file)
        print("Wrote source separation for speech enhancement dataset!")

        print("LOADED DATASET")

    # The dataset structure is a dictionary with "train", "valid", "test" keys, whose entries are lists, where each element represents a noisy speech file.
    # Each noisy speech file is represented as a tuple of (mix, noise, speech) in the source separation task for speech enhancement.

    # Optimize in a supervised fashion until validation loss worsens
    sup_model_path, sup_loss = optimise(dataset=dataset)
    print("Supervised training finished! Saved model at " + sup_model_path + ". Performance: " + str(sup_loss))
