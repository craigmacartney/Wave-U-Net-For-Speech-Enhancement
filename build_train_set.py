'''
File to carry out combining the Voice Bank Corpus data for training the Wave-U-Net model.
'''
import os
import argparse
from scipy.io.wavfile import read,write
import shutil 

parser = argparse.ArgumentParser()
parser.add_argument("clean_source",help="Source directory containing clean audio files from the Voice Bank Corpus (VCTK)] data set")
parser.add_argument("noisy_source",help = "Source directory containing the contaminated audio files from the Voice Bank Corpus (VCTK)] data set")
parser.add_argument("storage_folder",help = "Destination folder in your working directory to store the data in a more organized fashion")

if __name__ =="__main__":
    args = parser.parse_args()

    if not os.path.exists(args.storage_folder):
        os.mkdir(args.storage_folder)
    
    files = os.listdir(args.clean_source)
    for file_name in files:
        if not os.path.exists("{0}/{1}".format(args.storage_folder,file_name[:-4])):
            os.mkdir("{0}/{1}".format(args.storage_folder,file_name[:-4]))
        shutil.copy2("{0}/{1}".format(args.clean_source,file_name),"{0}/{1}/clean.wav".format(args.storage_folder,file_name[:-4]))
        shutil.copy2("{0}/{1}".format(args.noisy_source,file_name),"{0}/{1}/mixed.wav".format(args.storage_folder,file_name[:-4]))
        
        fs,clean = read("{0}/{1}".format(args.clean_source,file_name))
        fs,mixed = read("{0}/{1}".format(args.noisy_source,file_name))
        write("{0}/{1}/noise.wav".format(args.storage_folder,file_name[:-4]),fs,mixed-clean)
