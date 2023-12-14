import time
import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
import networkx as nx
from graph_creation_scripts import *
from k_core_modules import *
from karateclub import Graph2Vec
from sklearn.manifold import TSNE
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from fastnode2vec import Node2Vec, Graph
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from useful_methods import *
from visualization import *

parsed_path, prefix, choice = choose_dataset()
load_save_path = load_save_results(prefix, choice)

start_time = time.time()

if __name__ == '__main__':
    # ----------------------------------graph2vec---------------------------------------

    tsne_visualization('data_for_classifiers_graph2vec.csv')
    tsne_visualization('4_categories_graph2vec.csv')

    # ---------------------------------------TSNE VISUALIZATION-----------------------------------------------

    # # -------------------------20NEWSGROUPS-----------------------------------
    #
    # # -------------------------Word2Vec------------------------------
    # file_path = 'results/20newsgroups_parsed_1/'
    # tsne_visualization(os.path.join(file_path, '20newsgroups_embeddings_word2vec.csv'))
    #
    # file_path = 'results/20newsgroups_parsed_2/'
    # tsne_visualization(os.path.join(file_path, '20newsgroups_embeddings_word2vec.csv'))
    #
    # file_path = 'results/20newsgroups_parsed_3/'
    # tsne_visualization(os.path.join(file_path, '20newsgroups_embeddings_word2vec.csv'))
    #
    # # -------------------------Doc2Vec------------------------------
    # file_path = 'results/20newsgroups_parsed_1/'
    # tsne_visualization(os.path.join(file_path, '20newsgroups_embeddings_doc2vec.csv'))
    #
    # file_path = 'results/20newsgroups_parsed_2/'
    # tsne_visualization(os.path.join(file_path, '20newsgroups_embeddings_doc2vec.csv'))
    #
    # file_path = 'results/20newsgroups_parsed_3/'
    # tsne_visualization(os.path.join(file_path, '20newsgroups_embeddings_doc2vec.csv'))
    #
    # # -------------------------Node2Vec------------------------------
    # file_path = 'results/20newsgroups_parsed_1/'
    # tsne_visualization(os.path.join(file_path, '20newsgroups_embeddings_node2vec.csv'))
    #
    # file_path = 'results/20newsgroups_parsed_2/'
    # tsne_visualization(os.path.join(file_path, '20newsgroups_embeddings_node2vec.csv'))
    #
    # file_path = 'results/20newsgroups_parsed_3/'
    # tsne_visualization(os.path.join(file_path, '20newsgroups_embeddings_node2vec.csv'))
    #
    # # -------------------------BBC-----------------------------------
    #
    # # -------------------------Word2Vec------------------------------
    # file_path = 'results/bbc_parsed_1/'
    # tsne_visualization(os.path.join(file_path, 'bbc_embeddings_word2vec.csv'))
    #
    # file_path = 'results/bbc_parsed_2/'
    # tsne_visualization(os.path.join(file_path, 'bbc_embeddings_word2vec.csv'))
    #
    # file_path = 'results/bbc_parsed_3/'
    # tsne_visualization(os.path.join(file_path, 'bbc_embeddings_word2vec.csv'))
    #
    # # -------------------------Doc2Vec------------------------------
    # file_path = 'results/bbc_parsed_1/'
    # tsne_visualization(os.path.join(file_path, 'bbc_embeddings_doc2vec.csv'))
    #
    # file_path = 'results/bbc_parsed_2/'
    # tsne_visualization(os.path.join(file_path, 'bbc_embeddings_doc2vec.csv'))
    #
    # file_path = 'results/bbc_parsed_3/'
    # tsne_visualization(os.path.join(file_path, 'bbc_embeddings_doc2vec.csv'))
    #
    # # -------------------------Node2Vec------------------------------
    # file_path = 'results/bbc_parsed_1/'
    # tsne_visualization(os.path.join(file_path, 'bbc_embeddings_node2vec.csv'))
    #
    # file_path = 'results/bbc_parsed_2/'
    # tsne_visualization(os.path.join(file_path, 'bbc_embeddings_node2vec.csv'))
    #
    # file_path = 'results/bbc_parsed_3/'
    # tsne_visualization(os.path.join(file_path, 'bbc_embeddings_node2vec.csv'))
    #
    # # -------------------------EMAILS-----------------------------------
    #
    # # -------------------------Word2Vec------------------------------
    # file_path = 'results/emails_parsed_1/'
    # tsne_visualization(os.path.join(file_path, 'emails_embeddings_word2vec.csv'))
    #
    # file_path = 'results/emails_parsed_2/'
    # tsne_visualization(os.path.join(file_path, 'emails_embeddings_word2vec.csv'))
    #
    # file_path = 'results/emails_parsed_3/'
    # tsne_visualization(os.path.join(file_path, 'emails_embeddings_word2vec.csv'))
    #
    # # -------------------------Doc2Vec------------------------------
    # file_path = 'results/emails_parsed_1/'
    # tsne_visualization(os.path.join(file_path, 'emails_embeddings_doc2vec.csv'))
    #
    # file_path = 'results/emails_parsed_2/'
    # tsne_visualization(os.path.join(file_path, 'emails_embeddings_doc2vec.csv'))
    #
    # file_path = 'results/emails_parsed_3/'
    # tsne_visualization(os.path.join(file_path, 'emails_embeddings_doc2vec.csv'))
    #
    # # -------------------------Node2Vec------------------------------
    # file_path = 'results/emails_parsed_1/'
    # tsne_visualization(os.path.join(file_path, 'emails_embeddings_node2vec.csv'))
    #
    # file_path = 'results/emails_parsed_2/'
    # tsne_visualization(os.path.join(file_path, 'emails_embeddings_node2vec.csv'))
    #
    # file_path = 'results/emails_parsed_3/'
    # tsne_visualization(os.path.join(file_path, 'emails_embeddings_node2vec.csv'))

    # ---------------------------------------PCA VISUALIZATION-----------------------------------------------

    # #-------------------------20NEWSGROUPS-----------------------------------

    # # -------------------------Word2Vec------------------------------
    # file_path = 'results/20newsgroups_parsed_1/'
    # pca_visualization(os.path.join(file_path, '20newsgroups_embeddings_word2vec.csv'))

    # file_path = 'results/20newsgroups_parsed_2/'
    # pca_visualization(os.path.join(file_path, '20newsgroups_embeddings_word2vec.csv'))

    # file_path = 'results/20newsgroups_parsed_3/'
    # pca_visualization(os.path.join(file_path, '20newsgroups_embeddings_word2vec.csv'))

    # # -------------------------Doc2Vec------------------------------
    # file_path = 'results/20newsgroups_parsed_1/'
    # pca_visualization(os.path.join(file_path, '20newsgroups_embeddings_doc2vec.csv'))

    # file_path = 'results/20newsgroups_parsed_2/'
    # pca_visualization(os.path.join(file_path, '20newsgroups_embeddings_doc2vec.csv'))

    # file_path = 'results/20newsgroups_parsed_3/'
    # pca_visualization(os.path.join(file_path, '20newsgroups_embeddings_doc2vec.csv'))

    # # -------------------------Node2Vec------------------------------
    # file_path = 'results/20newsgroups_parsed_1/'
    # pca_visualization(os.path.join(file_path, '20newsgroups_embeddings_node2vec.csv'))

    # file_path = 'results/20newsgroups_parsed_2/'
    # pca_visualization(os.path.join(file_path, '20newsgroups_embeddings_node2vec.csv'))

    # file_path = 'results/20newsgroups_parsed_3/'
    # pca_visualization(os.path.join(file_path, '20newsgroups_embeddings_node2vec.csv'))

    # # -------------------------BBC-----------------------------------

    # # -------------------------Word2Vec------------------------------
    # file_path = 'results/bbc_parsed_1/'
    # pca_visualization(os.path.join(file_path, 'bbc_embeddings_word2vec.csv'))

    # file_path = 'results/bbc_parsed_2/'
    # pca_visualization(os.path.join(file_path, 'bbc_embeddings_word2vec.csv'))

    # file_path = 'results/bbc_parsed_3/'
    # pca_visualization(os.path.join(file_path, 'bbc_embeddings_word2vec.csv'))

    # # -------------------------Doc2Vec------------------------------
    # file_path = 'results/bbc_parsed_1/'
    # pca_visualization(os.path.join(file_path, 'bbc_embeddings_doc2vec.csv'))

    # file_path = 'results/bbc_parsed_2/'
    # pca_visualization(os.path.join(file_path, 'bbc_embeddings_doc2vec.csv'))

    # file_path = 'results/bbc_parsed_3/'
    # pca_visualization(os.path.join(file_path, 'bbc_embeddings_doc2vec.csv'))

    # # -------------------------Node2Vec------------------------------
    # file_path = 'results/bbc_parsed_1/'
    # pca_visualization(os.path.join(file_path, 'bbc_embeddings_node2vec.csv'))

    # file_path = 'results/bbc_parsed_2/'
    # pca_visualization(os.path.join(file_path, 'bbc_embeddings_node2vec.csv'))

    # file_path = 'results/bbc_parsed_3/'
    # pca_visualization(os.path.join(file_path, 'bbc_embeddings_node2vec.csv'))

    # # -------------------------EMAILS-----------------------------------

    # # -------------------------Word2Vec------------------------------
    # file_path = 'results/emails_parsed_1/'
    # pca_visualization(os.path.join(file_path, 'emails_embeddings_word2vec.csv'))

    # file_path = 'results/emails_parsed_2/'
    # pca_visualization(os.path.join(file_path, 'emails_embeddings_word2vec.csv'))

    # file_path = 'results/emails_parsed_3/'
    # pca_visualization(os.path.join(file_path, 'emails_embeddings_word2vec.csv'))

    # # -------------------------Doc2Vec------------------------------
    # file_path = 'results/emails_parsed_1/'
    # pca_visualization(os.path.join(file_path, 'emails_embeddings_doc2vec.csv'))

    # file_path = 'results/emails_parsed_2/'
    # pca_visualization(os.path.join(file_path, 'emails_embeddings_doc2vec.csv'))

    # file_path = 'results/emails_parsed_3/'
    # pca_visualization(os.path.join(file_path, 'emails_embeddings_doc2vec.csv'))

    # # -------------------------Node2Vec------------------------------
    # file_path = 'results/emails_parsed_1/'
    # pca_visualization(os.path.join(file_path, 'emails_embeddings_node2vec.csv'))

    # file_path = 'results/emails_parsed_2/'
    # pca_visualization(os.path.join(file_path, 'emails_embeddings_node2vec.csv'))

    # file_path = 'results/emails_parsed_3/'
    # pca_visualization(os.path.join(file_path, 'emails_embeddings_node2vec.csv'))

    print("--- %s seconds ---" % (time.time() - start_time))
