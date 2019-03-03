# Speech Enhancement with the Wave-U-Net

This repo is for personal research into existing wavenet architectures for audio denoising

The [Wave-U-Net applied to speech enhancement](http://arxiv.org/abs/1811.11307) [1], an adaptation of the [original implementation](https://github.com/f90/Wave-U-Net) for music source separation by [Stoller et al](https://arxiv.org/abs/1806.03185) [2].

The Wave-U-Net is a convolutional neural network applicable to audio source separation tasks, recently introduced by Stoller et al for the separation of music vocals and accompaniment [2]. A 1D convolutional time domain variant of the 2D convolutions within the U-Net [3], this end-to-end learning method for audio source separation operates directly in the time domain, permitting the integrated modelling of phase information and being able to take large temporal contexts into account.

Experiments on audio source separation for speech enhancement in [1] show that the proposed method rivals state-of-the-art architectures, improving upon various metrics, namely PESQ, CSIG, CBAK, COVL and SSNR, with respect to the single-channel speech enhancement task on the Voice Bank corpus (VCTK) dataset. Future experimentation will focus on increasing effectiveness and efficiency by further adapting the model size and other parameters, e.g. filter sizes, to the task and expanding to multi-channel audio and multi-source-separation.

<br>

## Architecture
The architecture is the same as that employed in [2] with the exception of the number of hidden layers and the validation set size. The number of hidden layers was experimented with and the results suggest the optimum size to be 9 layers.

See diagram below for a summary visual representation of the architecture:

![alt text](https://github.com/craigmacartney/Wave-U-Net-For-Speech-Enhancement/blob/master/Wave-U-Net_Diagram-v1.png)

## Initialisation
<b>Requirements</b>

Under our implementation, training took c.36 hours using GeForce GTX 1080 Ti GPU with 11178 MiB, on Linux Ubuntu 16.04, with Python 2.7. In a new virtual environment, required Python 2.7 packages can be installed using `pip install -r requirements.txt`. N.B. this presumes the installation of <b>ffmpeg</b> and <b>libsndfile</b>.

<br>

<b>Data Preparation</b>

Train and test datasets provided by the 28-speaker [Voice Bank Corpus (VCTK)](https://datashare.is.ed.ac.uk/handle/10283/2791) [4] (30 speakers in total - 28 intended for training and 2 reserved for testing). The noisy training data were generated by mixing the clean data with various noise datasets, as per the instructions provided in [4, 5, 6].

The training dataset should then be prepared for being parsed as an XML file (not provided) using the ElementTree XML API in <b>Datasets.getAudioData</b>.

<br>

## Training
Training can be executed by running the command `python Training.py`, modyfing the parameters in <b>Config.py</b> as desired.

## Testing
Testing experiments can be performed by running the command `python Test_Predictions_VCTK.py`.

Speech source estimates should then be evaluated against the clean speech file they are estimating. This can be done using <b>Evaluate.m</b>, which selects multiple files and performs the <b>composite.m</b> script [7] (available [here](https://ecs.utdallas.edu/loizou/speech/software.htm)) upon each one, calculating the PESQ, SSNR, CSIG, CBAK and COVL.

Audio examples of both speech and background noise estimates of the VCTK test set, alongside the noisy test files and clean speech for reference, are available for download in the <b>audio_examples</b> directory.

<br>

## References
[1] Craig Macartney and Tillman Weyde. Improved Speech Enhancement with the Wave-U-Net. 2018. URL http://arxiv.org/abs/1811.11307

[2] Daniel Stoller, Sebastian Ewert, and Simon Dixon. Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source Separation. 6 2018. URL https://arxiv.org/abs/1806.03185.

[3] Andreas Jansson, Eric J. Humphrey, Nicola Montecchio, Rachel M. Bittner, Aparna Kumar, and Tillman Weyde. Singing voice separation with deep u-net convolutional networks. In Proceedings of the 18th International Society for Music Information Retrieval Conference, ISMIR 2017, Suzhou, China, October 23-27, 2017, pages 745–751, 2017. URL https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf.

[4] Cassia Valentini-Botinhao. Noisy speech database for training speech enhancement algorithms and TTS models, 2016 [sound]. University of Edinburgh. School of Informatics. Centre for Speech Technology Research (CSTR), 2017. URL http://dx.doi.org/10.7488/ds/2117.

[5] Santiago Pascual, Antonio Bonafonte, and Joan Serrà. SEGAN: Speech Enhancement Generative Adversarial Network. doi: 10.7488/ds/1356. URL http://dx.doi.org/10.7488/ds/1356.

[6] Cassia Valentini-Botinhao, Xin Wang, Shinji Takaki, and Junichi Yamagishi. Investigating RNN-based speech enhancement methods for noise-robust Text-to-Speech. Technical report. URL https://www.research.ed.ac.uk/portal/files/26581510/SSW9_Cassia_1.pdf.

[7] Philipos C Loizou. Speech Enhancement: Theory and Practice. CRC Press, Inc., Boca Raton, FL, USA, 2nd edition, 2013. ISBN 1466504218, 9781466504219.
