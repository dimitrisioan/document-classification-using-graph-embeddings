from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import gensim.downloader as api
import time
import os
from useful_methods import *

parsed_path, prefix, choice = choose_dataset()
load_save_path = load_save_results(prefix, choice)

start_time = time.time()

if __name__ == '__main__':
    # # Load pretrained gensim model 'word2vec-google-news-300'
    # model_google_300 = api.load("word2vec-google-news-300")
    # print("Top 10 most similar words for Google News 300 model")
    # print(model_google_300.most_similar('man', topn=10))
    # # print(model_google_300.most_similar(positive=['woman','king'], negative=['man']))

    print("-------------------------------------------------------------")

    # Load the Word2Vec model from the corresponding dataset directory
    model = Word2Vec.load(os.path.join(load_save_path, f'{prefix}_word2vec'))

    # # Load my pretrained model
    # model = Word2Vec.load("../document-classification-using-graph-embeddings/word2vec_models/word2vec.model")

    print("Top 10 most similar words for my model")
    print(model.wv.most_similar('man', topn=10))
    # print(model.wv.most_similar(positive=['woman','king'], negative=['man']))
    print("\n")
    print(f"Word 'man' appeared {model.wv.get_vecattr('man', 'count')} times in the training corpus.")
    # print(model.wv['computer'])

    # # Write dictionary {word,embedding} into a txt file
    # words = model.wv.index_to_key
    # we_dict = {word: model.wv[word] for word in words}
    # print(we_dict)
    #
    # for word, embedding in we_dict.items():
    #     print(f"{word}: {embedding}")

    # with open('5_300_word_embeddings.txt', 'w') as f:
    #     print(we_dict, file=f)

    # # create w2v dictionary for create graphs input later
    # words = model.wv.index_to_key
    # f = open("w2v_dictionary.txt", "w+")
    # word_index = 0
    #
    # for word in words:
    #     word_index += 1
    #     f.write(str(word_index))
    #     f.write("\t")
    #     f.write(word)
    #     f.write("\t")
    #     f.write("%s" % model.wv[word])
    #     f.write("\n")

    print("--- %s seconds ---" % (time.time() - start_time))
