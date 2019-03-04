import os
import argparse
import wave
import shutil 

parser = argparse.ArgumentParser()
parser.add_argument("clean_source")
parser.add_argument("noisy_source")
parser.add_argument("storage_folder")

if __name__ =="__main__":
    args = parser.parse_args()
    if not os.path.exists(args.storage_folder):
        os.mkdir(args.storage_folder)
    
    files = os.listdir(args.clean_source)
    for file_name in files:
        if not os.path.exists("{0}/{1}".format(args.storage_folder,file_name[:4])):
            os.mkdir("{0}/{1}".format(args.storage_folder,file_name[:4]))
        shutil.copy2("{0}/{1}".format(args.clean_source,file_name),"{0}/{1}/clean.wav".format(args.storage_folder,file_name[:4]))
        shutil.copy2("{0}/{1}".format(args.noisy_source,file_name),"{0}/{1}/mixed.wav".format(args.storage_folder,file_name[:4]))
        
    
