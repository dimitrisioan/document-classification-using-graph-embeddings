from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import gensim.downloader as api
import time
import os


start_time = time.time()

dataset_path = "../document-classification-using-graph-embeddings/newsgroups_dataset/"
parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_parsed/"

# That's a file for testing the Word2Vec model

# just to see what to do
# # define training data
# sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
# 			['this', 'is', 'the', 'second', 'sentence'],
# 			['yet', 'another', 'sentence'],
# 			['one', 'more', 'sentence'],
# 			['and', 'the', 'final', 'sentence']]
# # train model
# model = Word2Vec(sentences, min_count=1)
# # summarize the loaded model
# print(model)
# # summarize vocabulary
# words = list(model.wv.key_to_index)
# print(words)
#
# # access vector for one word
# print(model.wv['sentence'])
# # save model
# model.save('model.bin')
# # load model
# new_model = Word2Vec.load('model.bin')
# print(new_model)# new_model = Word2V

if __name__ == '__main__':
    # # Load pretrained gensim model 'word2vec-google-news-300'
    # model_google_300 = api.load("word2vec-google-news-300")
    # print("Top 10 most similar words for Google News 300 model")
    # print(model_google_300.most_similar('man', topn=10))
    # # print(model_google_300.most_similar(positive=['woman','king'], negative=['man']))

    print("-------------------------------------------------------------")

    # Load my pretrained model
    model = Word2Vec.load("../document-classification-using-graph-embeddings/word2vec_models/sentences_word2vec.model")
    print("Top 10 most similar words for my model")
    print(model.wv.most_similar('man', topn=10))
    # print(model.wv.most_similar(positive=['woman','king'], negative=['man']))
    print("\n")
    print(f"Word 'man' appeared {model.wv.get_vecattr('man', 'count')} times in the training corpus.")
    # print(model.wv['computer'])

    # Write dictionary {word,embedding} into a txt file
    words = model.wv.index_to_key
    we_dict = {word: model.wv[word] for word in words}
    print(we_dict)

    with open('5_300_word_embeddings.txt', 'w') as f:
        print(we_dict, file=f)

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
