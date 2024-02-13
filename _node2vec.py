import multiprocessing
import os
import time
import networkx as nx
import pickle
from fastnode2vec import Node2Vec, Graph
from useful_methods import *

parsed_path, prefix, choice = choose_dataset()
load_save_path = load_save_results(prefix, choice)

start_time = time.time()

if __name__ == '__main__':
    X = input(
        '1.create index using maincore \n 2.create index without considering maincore \n 3.Create index using Density '
        'method \n 4.Create index using CoreRank method\nX = ')
    if int(X) == 1:
        # Save the union_graph to a file inside the specified directory
        file_path = os.path.join(load_save_path, 'union_graph_1.pkl')
        with open(file_path, 'rb') as file:
            union_graph = pickle.load(file)
    elif int(X) == 2:
        file_path = os.path.join(load_save_path, 'union_graph_2.pkl')
        with open(file_path, 'rb') as file:
            union_graph = pickle.load(file)
    elif int(X) == 3:
        file_path = os.path.join(load_save_path, 'union_graph_3.pkl')
        with open(file_path, 'rb') as file:
            union_graph = pickle.load(file)

    # file_path = os.path.join(load_save_path, 'union_graph_1.pkl')
    # with open(file_path, 'rb') as file:
    #     union_graph = pickle.load(file)

    G = union_graph
    print(nx.info(G))

    unique_words = []
    for node in G.nodes():
        unique_words.append(node)

    # Get the number of available CPU cores
    num_cores = multiprocessing.cpu_count()

    # Convert G from networkx to Graph structure in order to run fastnode2vec
    edges = [(u, v) for u, v in G.edges()]
    graph = Graph(edges, directed=False, weighted=False)

    model = Node2Vec(graph, dim=64, walk_length=30, window=10, p=1.0, q=1.0, workers=num_cores)
    model.train(epochs=15)
    # model = Node2Vec(graph, dim=64, walk_length=30, window=10, p=2.0, q=0.5, workers=num_cores)
    # model = Node2Vec(graph, dim=64, walk_length=30, window=10, p=0.5, q=2.0, workers=num_cores)

    # print(model.wv["the"])

    # Save the Node2Vec model to the corresponding dataset directory
    model.save(os.path.join(load_save_path, f'{prefix}_node2vec'))

    # model.save("../document-classification-using-graph-embeddings/node2vec_models/node2vec.model")
    print(model.wv.most_similar('man', topn=10))

    print("--- %s seconds ---" % (time.time() - start_time))
