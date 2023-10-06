from gensim.models import Word2Vec
import time
import gensim.downloader as api
import gensim
import os
import numpy as np

from gensim.models import KeyedVectors

start_time = time.time()

dataset_path = "../document-classification-using-graph-embeddings/newsgroups_dataset/"
parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_parsed/"

test = "../document-classification-using-graph-embeddings/newsgroups_dataset_parsed/comp.os.ms-windows.misc/10085.txt"

if __name__ == '__main__':
    model = Word2Vec.load("../document-classification-using-graph-embeddings/word2vec_models/sentences_word2vec.model")
    # print(f"Word 'could' appeared {model.wv.get_vecattr('could', 'count')} times in the training corpus.")
    # print(model.wv['could'])
    #
    # print(f"Word 'help' appeared {model.wv.get_vecattr('help', 'count')} times in the training corpus.")
    # print(model.wv['help'])
    # print("-------------------------")
    # result_addition = model.wv.get_vector("could") + model.wv.get_vector("help")
    # print(result_addition)
    with open("test_nums.txt", 'w') as output_file:
        with open(test, 'rt', errors="ignore") as f:
            # words = []
            words = f.read().split()

        words_found_vectors = [model.wv.get_vector(word) for word in words if word in model.wv.key_to_index]
        words_found = [word for word in words if word in model.wv.key_to_index]
        for word in words_found:
            print(f"Word '{word}'appeared {model.wv.get_vecattr(word, 'count')} times in the training corpus.")
            print(model.wv[word])
            print(model.wv[word].shape)
            # output_file.write(word)
            print(model.wv[word][299])
            output_file.write(str(model.wv[word][299]))
            output_file.write(" + ")
            # output_file.write(str(model.wv[word]))
            # output_file.write("+")

            # output_file.write("\n")

            print("\n----------------------------------------------------------------------\n")

        result_np_sum = np.sum(words_found_vectors, axis=0)
        print("result with np sum")
        print(result_np_sum)
