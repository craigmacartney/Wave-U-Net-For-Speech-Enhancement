import os
import os.path
import librosa
from Sample import Sample

from lxml import etree


def create_sample(db_path, noise_node):
   path = db_path + os.path.sep + noise_node.xpath("./relativeFilepath")[0].text
   sample_rate = int(noise_node.xpath("./sampleRate")[0].text)
   channels = int(noise_node.xpath("./numChannels")[0].text)
   duration = float(noise_node.xpath("./length")[0].text)
   return Sample(path, sample_rate, channels, duration)

def getAudioData(source_folder):

    def _extract_sample(file_name):
        '''
        Extract wavfile properties to feed into the Sample class
        '''
        y,rate = librosa.load(file_name,16000)
        num_channels = 1
        duration = float(librosa.get_duration(y,rate))
        return Sample(file_name,rate,num_channels,duration)


    tracks = os.listdir(source_folder)
    samples = list()

    for track in tracks:
        speech = _extract_sample("{0}/{1}/clean.wav".format(source_folder,track))
        mix = _extract_sample("{0}/{1}/mixed.wav".format(source_folder,track))
        noise = _extract_sample("{0}/{1}/noise.wav".format(source_folder,track))
        samples.append((mix, noise, speech))

    return samples



