import os
import shutil
import time
import glob
import string
import gensim.utils
from nltk.corpus import stopwords
import re

start_time = time.time()

dataset_path = "../document-classification-using-graph-embeddings/newsgroups_dataset/"
parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_parsed/"
combined_parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_combined_parsed/"


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



    # Delete all files on 'combined_parsed_path'
    for f in os.listdir(combined_parsed_path):
        file_path = os.path.join(combined_parsed_path, f)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    # Create directory 'combined_parsed_path'
    dirName = combined_parsed_path
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")

    # Iterate through all txts in 'dataset_path' parse them and save them under 'combined_parsed'
    for file in os.listdir(dataset_path):
        # print(file)
        dataset_filepath = dataset_path + file
        combined_parsed_filepath = combined_parsed_path + file
        file_name = dataset_filepath.replace('.txt', '')

        with open(dataset_filepath, 'rt', errors="ignore") as fd:
            lines = fd.readlines()
        # exclude first 3 lines after reading the string "Newsgroup:" and save on clean.txt
        with open(os.path.join(dirName, str(file)), 'w') as nf:
            i = 0
            while i < len(lines):
                line = lines[i]
                # skip 3 lines after finding string "Newsgroup:"
                if line.startswith("Newsgroup:"):
                    i += 3
                else:
                    line = line.lower()
                    list_of_words = line.split()
                    # print(list_of_words)
                    # print(line)
                    # Remove punctuation
                    table = str.maketrans('', '', string.punctuation)
                    stripped = [w.translate(table) for w in list_of_words]
                    # Remove numbers
                    list_of_words = [word for word in stripped if word.isalpha()]
                    list_of_words_size = len(list_of_words)
                    for word in list_of_words:
                        print(word)
                        nf.write(word)
                        nf.write("\n")
                    i += 1

    # for text_file in os.listdir(dirName):
    #     combined_parsed_filepath = combined_parsed_path + text_file
    #     with open(combined_parsed_filepath, 'r') as file:
    #         # splitting the file data into lines
    #         raw_sentences = [[item.rstrip('\n')] for item in file]
    #         # splitting the lines into [tokens]
    #         raw_sentences = [item[0].split(" ") for item in raw_sentences]
    #         sentences = []
    #         for sentence in raw_sentences:
    #             # removing digits and punctuation from sentences
    #             table = str.maketrans('', '', string.punctuation)
    #             stripped = [w.translate(table) for w in sentence]
    #             sentence = [word for word in stripped if word.isalpha()]
    #             sentences.append(sentence)
    #         # format of sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
    #         print(sentences)
    #
    #     # take all words from clean_text.txt
    #     words = []
    #     for sentence in sentences:
    #         for word in sentence:
    #             words.append(word)

    print("--- %s seconds ---" % (time.time() - start_time))
