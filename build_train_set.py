import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("clean_source")
parser.add_argument("noisy_source")

if __name__ =="__main__":
    args = parser.parse_args()
    files = os.listdir(args.clean_source)
    clean_files = set(["{0}/{1}".format(args.clean_source,file_name) for file_name in files])
    files = os.listdir(args.noisy_source)
    mixed_files = set(["{0}/{1}".format(args.noisy_source,file_name) for file_name in files])
    print(clean_files - mixed_files)
    print(mixed_files - clean_files)
