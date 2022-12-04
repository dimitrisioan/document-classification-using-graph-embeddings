import nltk
import numpy
import sys
import networkx as nx
import os
from node2vec import Node2Vec
from gensim.models import Word2Vec
import time
import statistics
import string
from graph_creation_scripts import *
from k_core_modules import *

start_time = time.time()

dataset_path = "../document-classification-using-graph-embeddings/newsgroups_dataset/"
parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_parsed/"

file = "../document-classification-using-graph-embeddings/newsgroups_dataset_parsed/comp.os.ms-windows.misc/8514.txt"
all_text_file = 'all_words.txt'
node_embeddings_file = "../node2vec_model/n2v_emb.txt"

if __name__ == '__main__':
    # Load my pretrained model
    model = Word2Vec.load("../document-classification-using-graph-embeddings/word2vec_models/sentences_word2vec.model")
    words = model.wv.index_to_key
    # create dictionary in the appropriate form for adjmatrix func
    f = open("dictionary_embed.txt", "w+")

    for word in words:
        f.write(word)
        f.write(" ")
        cnt = 0
        for item in model.wv[word]:
            f.write("%s" % item + " ")
        f.write("\n")
        # f.write("%s" % model.wv[word])

    # ------------------------dictionary creation---------------------------
    final_list = []
    with open("dictionary_embed.txt", "r") as f:
        for line in f:
            line = line.rstrip("\n")
            line_list = line.split(' ')
            final_list.append(line_list)

    mydict = {}

    for x in final_list:
        mydict[x[0]] = x[1:len(x) - 1]

    mydict = ({k: list(map(float, mydict[k])) for k in mydict})
    # print(mydict)

    #get all words from whole text of dataset
    #difference between words and file_words is that words are the unique ones from word2vec
    with open(all_text_file, "r") as file:
        data = file.read()
        file_words = data.split()
        # unique words list
        unique_words = list(set(file_words))
        print(f"Words: {len(file_words)}")
        print(f"Unique words: {len(unique_words)}")

        # #output = splitFileConstantWindow(all_text_file, 5, 0)
        # adjmatrix = CreateAdjMatrixFromInvIndexWithWindow_embe(unique_words, all_text_file, 5, 0, 0, mydict)
        # G = graphUsingAdjMatrix(adjmatrix, unique_words)
        # nx.draw(G)
        # plt.show()

    # print(output)

    # initialization of the graph
    # G = nx.read_edgelist(filename)
    # G = nx.k_core(G, k=None, core_number=None)
    # # train Node2Vec model with G as input
    # node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=10, workers=1)
    # model = node2vec.fit(window=10, min_count=1, batch_words=4)

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


print("--- %s seconds ---" % (time.time() - start_time))
