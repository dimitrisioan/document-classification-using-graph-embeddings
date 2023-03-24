import gensim
from gensim.models.doc2vec import TaggedDocument
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from gensim.models import Word2Vec
import numpy as np
import time
import os
from sklearn.metrics import accuracy_score
import chardet
# from classification_functs import *
# from plot_functs import *

start_time = time.time()

parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_parsed/"

if __name__ == "__main__":
    # filecount = 0
    #
    # model = Word2Vec.load("../document-classification-using-graph-embeddings/word2vec_models/sentences_word2vec.model")
    #
    # # Loop through each directory in "newsgroups_dataset_parsed"
    # with open("test.txt", 'w') as output_file:
    #     for directory in os.listdir(parsed_path):
    #         if os.path.isdir(os.path.join(parsed_path, directory)):
    #
    #             # Loop through each file in the directory and read its contents
    #             for filename in os.listdir(os.path.join(parsed_path, directory)):
    #                 with open(os.path.join(parsed_path, directory, filename), 'r', encoding='utf-8', errors='ignore') as file:
    #                     filecount += 1
    #                     print(file)
    #                     # text = file.read().split()
    #                     # print(text)
    #                     # output_file.write(text)
    #
    # print("Text files are:", filecount)

    # count_txt_files = 0
    #
    # for root, dirs, files in os.walk(parsed_path):
    #     for file in files:
    #         if file.endswith(".txt"):
    #             file_path = os.path.join(root, file)
    #             with open(file_path, "rt", encoding='ascii') as f:
    #                 file_contents = f.read()
    #                 print(f)
    #                 # process the file contents as needed
    #             count_txt_files += 1
    #
    # print(f"Total txt files: {count_txt_files}")
    all_files = 0
    ascii_files = 0
    non_ascii_files_cnt = 0
    non_ascii_files = []
    for root, dirs, files in os.walk(parsed_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "rb") as f:
                    all_files += 1
                    raw_data = f.read()
                    result = chardet.detect(raw_data)
                    encoding = result["encoding"]
                    if encoding == 'ascii':
                        ascii_files += 1
                    else:
                        non_ascii_files.append(file_path)
                        print(file_path)
                        print(encoding)
                        non_ascii_files_cnt += 1
    print("All files are: ", all_files)
    print("Ascii encoding files are: ", ascii_files)
    print("Non ascii encoding files are: ", non_ascii_files_cnt)
    print("Non ascii files:", non_ascii_files)



    print("--- %s seconds ---" % (time.time() - start_time))
