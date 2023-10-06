import sys
from gensim.models import Word2Vec
import os
import time
from graph_creation_scripts import *
from k_core_modules import *
import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle

# from nodevectors import Node2Vec

start_time = time.time()

dataset_path = "../document-classification-using-graph-embeddings/newsgroups_dataset/"
parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_parsed/"

all_text_file = 'all_words.txt'

if __name__ == '__main__':
    # with open(all_text_file, "r") as file:
    #     data = file.read()
    #     file_words = data.split()
    #     # unique words list
    #     unique_words = list(set(file_words))
    #     print(f"Words: {len(file_words)}")
    #     print(f"Unique words: {len(unique_words)}")
    # filecount = 0
    # cnt = 0
    # data = []
    # files_list = []
    # for subdirectory in os.listdir(parsed_path):
    #     subdirectory_path = os.path.join(parsed_path, subdirectory)
    #     print(subdirectory_path)
    #     for file in os.listdir(subdirectory_path):
    #         cnt += 1
    #         filecount += 1
    #         doc_file_path = os.path.join(subdirectory_path, file)
    #         print(doc_file_path)

    # G = union_graph
    # nx.draw(G)
    # plt.show()
    # # Node2Vec
    # # node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)  # default
    # temp_folder = "../document-classification-using-graph-embeddings/temp_folder"
    # node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=10, workers=1, temp_folder=temp_folder)
    # model = node2vec.fit(window=10, min_count=1, batch_words=4)
    #
    # embeddings_list = [model.wv[term] for term in model.wv.index_to_key]
    # print(embeddings_list)
    # # data.append({'id': file, 'embedding': embeddings_list, 'category': subdirectory})
    # # print(embeddings_list)
    # # print(data)

    with open('union_graph_3.pkl', 'rb') as file:
        union_graph = pickle.load(file)

    G = union_graph
    print(nx.info(G))
    # graphToPng(G)
    # nx.draw(G)
    # plt.show()

    G = nx.k_core(G, k=None, core_number=None)
    print('core done')
    with open('core_union_graph_3.pkl', 'wb') as file:
        pickle.dump(G, file)
    print('Core ', nx.info(G))
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=10, workers=1)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    model.wv.save_word2vec_format('node_embed_3.txt')

    # # Create Dataframe and save data in a CSV file
    # df = pd.DataFrame(data)
    # df.to_csv('data_for_classifiers_node2vec.csv', index=False)

    # print("files are:", filecount)
    print("--- %s seconds ---" % (time.time() - start_time))
