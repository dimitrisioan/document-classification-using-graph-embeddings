import time
import os
import nltk
import networkx as nx
import numpy
from node2vec import Node2Vec
import statistics
from gensim.models import Word2Vec
import string
import matplotlib.pyplot as plt
from graph_creation_scripts import *
from k_core_modules import *

import pickle

start_time = time.time()

dataset_path = "../document-classification-using-graph-embeddings/newsgroups_dataset/"
parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_parsed/"
combined_parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_combined_parsed/"

# this script is designed in order to make adjmatrix and graphs for every category txt file
# testing for talk.religion.misc

test_file_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_parsed/talk.religion.misc/82758.txt"
node2vec_models_path = "../document-classification-using-graph-embeddings/node2vec_models/"

if __name__ == '__main__':

    # Load my pretrained model
    model = Word2Vec.load("../document-classification-using-graph-embeddings/word2vec_models/sentences_word2vec.model")
    print("Top 10 most similar words for my model for word man")
    print(model.wv.most_similar('man', topn=10))

    words_talk_religion = []

    # # Loop through every subdirectory, read words from each category file
    # for file in os.listdir(test_dir):
    #     test_file_path = test_dir + '/' + file
    #     file = open(test_file_path, "rt")
    #     words = file.read().split()
    #     for word in words:
    #         words_talk_religion.append(word)
    with open(test_file_path, "r") as fd:
        words = fd.read().split()
        for word in words:
            words_talk_religion.append(word)

    print(words_talk_religion)
    print("all talk religion words are:", len(words_talk_religion))
    # unique words for talk religion
    unique_words_talk_religion = list(set(words_talk_religion))
    print("all unique talk religion words are:", len(unique_words_talk_religion))

    model_words = model.wv.index_to_key
    with open("dictionary_embed_talk_religion_82758.txt", "w+") as f:
        for unique_word in unique_words_talk_religion:
            # #alternative solution
            # if unique_word in words:
            for word in model_words:
                if unique_word == word:
                    # ------------------------dictionary text file creation for talk_religion---------------------------
                    # create dictionary in the appropriate form for adjmatrix func

                    f.write(unique_word)
                    f.write(" ")
                    for item in model.wv[word]:
                        f.write("%s" % item + " ")
                    f.write("\n")

    # ------------------------dictionary creation for talk_religion---------------------------
    final_list = []

    with open("dictionary_embed_talk_religion_82758.txt", "r") as f:
        for line in f:
            line = line.rstrip("\n")
            line_list = line.split(' ')
            final_list.append(line_list)

    mydict = {}

    for x in final_list:
        mydict[x[0]] = x[1:len(x) - 1]

    mydict = ({k: list(map(float, mydict[k])) for k in mydict})
    # print(mydict)

    # output = splitFileConstantWindow(all_text_file, 5, 0)
    adjmatrix = CreateAdjMatrixFromInvIndexWithWindow_embe(unique_words_talk_religion, test_file_path, 5, 0, 0, mydict)
    print(adjmatrix)
    G = graphUsingAdjMatrix(adjmatrix, unique_words_talk_religion)
    nx.draw(G)
    plt.show()
    print(mydict)
    # print(output)

    # initialization of the graph
    # G = nx.read_edgelist(filename)
    # G = nx.k_core(G, k=None, core_number=None)
    # # train Node2Vec model with G as input
    # node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=10, workers=1)
    # model = node2vec.fit(window=10, min_count=1, batch_words=4)
    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)  # Use temp_folder for big graphs

    # Embed nodes
    model = node2vec.fit(window=10, min_count=1,batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

    # Save embeddings for later use
    my_file = os.path.join(node2vec_models_path, "n2v_embed_talk_religion_82758" + ".txt")
    with open(my_file, "wb") as file:
        model.wv.save_word2vec_format(file)
    # Look for most similar nodes
    print(model.wv.most_similar('84'))  # Output node names are always strings
    # Save model for later use
    model.save(node2vec_models_path+"model_talk_religion_82758.model")

    # # ---------------------------------example from node2vec documentation---------------------------------
    # # Create a graph
    # graph = nx.fast_gnp_random_graph(n=100, p=0.5)
    #
    # # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    # node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)  # Use temp_folder for big graphs
    #
    # # Embed nodes
    # model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)
    #
    # # Look for most similar nodes
    # print(model.wv.most_similar('2'))  # Output node names are always strings
    #
    # # Save embeddings for later use
    # model.wv.save_word2vec_format(EMBEDDING_FILENAME)
    #
    # # Save model for later use
    # model.save(EMBEDDING_MODEL_FILENAME)
    #
    # # Embed edges using Hadamard method
    # from node2vec.edges import HadamardEmbedder
    #
    # edges_embs = HadamardEmbedder(keyed_vectors=model.wv)
    #
    # # Look for embeddings on the fly - here we pass normal tuples
    # edges_embs[('1', '2')]
    # ''' OUTPUT
    # array([ 5.75068220e-03, -1.10937878e-02,  3.76693785e-01,  2.69105062e-02,
    #        ... ... ....
    #        ..................................................................],
    #       dtype=float32)
    # '''
    #
    # # Get all edges in a separate KeyedVectors instance - use with caution could be huge for big networks
    # edges_kv = edges_embs.as_keyed_vectors()
    #
    # # Look for most similar edges - this time tuples must be sorted and as str
    # edges_kv.most_similar(str(('1', '2')))
    #
    # # Save embeddings for later use
    # edges_kv.save_word2vec_format(EDGES_EMBEDDING_FILENAME)
    # # bellow code write on script1 to load my_var on script2
    # pickle.dump(my_var, open('df.p', 'wb'))
    # # bellow code write on script2 to read my_var from script1
    # my_var = pickle.load(open('df.p', 'rb'))
    print("--- %s seconds ---" % (time.time() - start_time))
