import time
import os
from graph_creation_scripts import *
from k_core_modules import *
from karateclub import Graph2Vec
from useful_methods import *
import pandas as pd
import numpy
import networkx as nx
import matplotlib.pyplot as plt

parsed_path, prefix, choice = choose_dataset()
load_save_path = load_save_results(prefix, choice)

start_time = time.time()
newsgroups_categories_included = ['comp.windows.x',
                                  'rec.sport.baseball',
                                  'sci.space',
                                  'talk.religion.misc']

if __name__ == '__main__':
    # # 1st approach: each category in a separate model
    # postingl = []
    # all_data = []
    # filecount = 0
    #
    # for category in os.listdir(parsed_path):
    #     category_path = os.path.join(parsed_path, category)
    #     data = []
    #     graphs = []
    #     # cnt = 0
    #     for file in os.listdir(category_path):
    #         file_path = os.path.join(category_path, file)
    #         print(file_path)
    #         with open(file_path, 'r') as f:
    #             text = f.read().split()
    #             # Skip file if it has less than 3 words
    #             if len(text) < 3:
    #                 continue
    #
    #         filecount += 1
    #         # Convert text document into Graph using GSB
    #         unique_terms, term_freq, postingl, count_of_text = createInvertedIndexFromFile(file_path, postingl)
    #         adjacency_matrix = CreateAdjMatrixFromInvIndex(unique_terms, term_freq)
    #         G = graphUsingAdjMatrix(adjacency_matrix, unique_terms)
    #         graphs.append(G)
    #
    #         document_id = file.replace('.txt', '')
    #         data.append({'document_id': document_id, 'embedding': [], 'category': category})
    #         filecount += 1
    #         # cnt += 1
    #         # if cnt == 20:
    #         #     break
    #
    #         # nx.draw(G, with_labels=True, node_color='skyblue', font_weight='bold')
    #         # plt.show()
    #         # print(unique_terms)
    #         # print(term_freq)
    #         # print(postingl)
    #         # print(count_of_text)
    #         # print(adjacency_matrix)
    #         # break
    #     # break
    #
    #     # Graph2Vec training
    #     model = Graph2Vec(epochs=10)
    #     model.fit(graphs)
    #     graph_embeddings = model.get_embedding()
    #
    #     # Add embeddings for each document in data after training
    #     for i, embedding in enumerate(graph_embeddings):
    #         data[i]['embedding'] = embedding.tolist()
    #
    #     all_data.extend(data)
    #
    #     print(f'Graph embeddings for {category} are: ', len(graph_embeddings))
    #
    #     # for _ in data:
    #     #     print(_)
    #
    # # Create Dataframe and save data in a CSV file
    # df = pd.DataFrame(all_data)
    #
    # # Save the CSV file for Graph2Vec to the corresponding dataset directory
    # df.to_csv(os.path.join(load_save_path, f'{prefix}_embeddings_graph2vec.csv'), index=False)

    # =============================================================================================

    # 2nd approach: all categories in 1 model
    postingl = []
    data = []
    graphs = []
    filecount = 0

    for category in os.listdir(parsed_path):
        # if category not in newsgroups_dataset_4_categories:
        #     continue
        category_path = os.path.join(parsed_path, category)
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            print(file_path)

            with open(file_path, 'r') as f:
                text = f.read().split()
                # Skip file if it has less than 3 words
                if len(text) < 3:
                    continue

            filecount += 1
            # Convert text document into Graph using GSB
            unique_terms, term_freq, postingl, count_of_text = createInvertedIndexFromFile(file_path, postingl)
            adjacency_matrix = CreateAdjMatrixFromInvIndex(unique_terms, term_freq)
            G = graphUsingAdjMatrix(adjacency_matrix, unique_terms)
            graphs.append(G)

            document_id = file.replace('.txt', '')
            data.append({'document_id': document_id, 'embedding': [], 'category': category})
            filecount += 1
            # break
        # break
    print(f"all items in list graphs are {len(graphs)}")
    # Graph2Vec training
    model = Graph2Vec(min_count=1)
    model.fit(graphs)
    graph_embeddings = model.get_embedding()

    # Add embeddings for each document in data after training
    for i, embedding in enumerate(graph_embeddings):
        data[i]['embedding'] = embedding.tolist()

    # Create Dataframe and save data in a CSV file
    df = pd.DataFrame(data)

    df.to_csv('all_categories.csv', index=False)

    print("Text files are:", filecount)
    print("--- %s seconds ---" % (time.time() - start_time))
