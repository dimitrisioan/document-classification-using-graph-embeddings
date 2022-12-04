import os
import shutil
import time
import glob
import gensim.utils
from nltk.corpus import stopwords
import re

start_time = time.time()

dataset_path = "../document-classification-using-graph-embeddings/newsgroups_dataset/"
parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_parsed/"
combined_parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_combined_parsed/"

# alternative solution for spliting text into sentences for word2vec model
# class SentenceIterator:
#     def __init__(self, filepath):
#         self.filepath = filepath
#
#     def __iter__(self):
#         for line in open(self.filepath):
#             yield line.split()
#
#
# sentences = SentenceIterator('all_text.txt')
# print(sentences)

if __name__ == '__main__':
    if os.path.exists('clean_text.txt'):
        os.remove('clean_text.txt')
    else:
        print("Can not delete the file as it doesn't exists")

    # combines all the text files from newsgroup_dataset directory into one txt file 'all_text.txt'
    with open('all_text.txt', 'w') as wfd:
        for f in glob.glob(r'../document-classification-using-graph-embeddings/newsgroups_dataset/*.txt'):
            with open(f, 'rt', errors="ignore") as fd:
                shutil.copyfileobj(fd, wfd)

    # read in lines from all_text.txt
    with open('all_text.txt', 'r') as f:
        lines = f.readlines()
    # exclude first 3 lines after reading the string "Newsgroup:" and save on clean.txt
    with open('clean_text.txt', 'w') as nf:
        i = 0
        while i < len(lines):
            line = lines[i]
            # skip 3 lines after finding string "Newsgroup:"
            if line.startswith("Newsgroup:"):
                i += 3
            else:
                line = line.lower()
                # print(line)
                nf.write(line)
                i += 1

    dirName = combined_parsed_path
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")

    for file in os.listdir(dataset_path):
        # print(file)
        dataset_filepath = dataset_path + file
        combined_parsed_filepath = combined_parsed_path + file
        file_name = dataset_filepath.replace('.txt', '')

        with open(dataset_filepath, 'rt', errors="ignore") as fd:
            lines = fd.readlines()
            # exclude first 3 lines after reading the string "Newsgroup:" and save on clean.txt
            with open(os.path.join(dirName, str(file) + ".txt"), 'w') as nf:
                i = 0
                while i < len(lines):
                    line = lines[i]
                    # skip 3 lines after finding string "Newsgroup:"
                    if line.startswith("Newsgroup:"):
                        i += 3
                    else:
                        line = line.lower()
                        # print(line)
                        nf.write(line)
                        i += 1

print("--- %s seconds ---" % (time.time() - start_time))
