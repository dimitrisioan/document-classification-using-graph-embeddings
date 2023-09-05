import os
from gensim.models import Word2Vec
import time
import string

start_time = time.time()

parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_parsed/"
directory_path = "../document-classification-using-graph-embeddings/"


if __name__ == '__main__':
    # words = []
    # for subdirectory in os.listdir(parsed_path):
    #     new_file_path = parsed_path + subdirectory + '/'
    #     for file in os.listdir(new_file_path):
    #         test_file_path = new_file_path + file
    #         file_content = read_file(test_file_path).split("\n")
    #         for word in file_content:
    #             words.append(word)
    #
    # print(len(words))

    # alternative solution for spliting text into sentences for word2vec model
    # class SentenceIterator:
    #     def __init__(self, filepath):
    #         self.filepath = filepath
    #
    #     def __iter__(self):
    #         for line in open(self.filepath):
    #             yield line.split()
    #
    #
    # sentences = SentenceIterator('all_text.txt')
    # print(sentences)

    # read data from clean_text.txt and make sentences for word2vec model
    with open('clean_text.txt', 'r') as file:
        # splitting the file data into lines
        raw_sentences = [[item.rstrip('\n')] for item in file]
        # splitting the lines into [tokens]
        raw_sentences = [item[0].split(" ") for item in raw_sentences]
        sentences = []
        for sentence in raw_sentences:
            # removing digits and punctuation from sentences
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in sentence]
            sentence = [word for word in stripped if word.isalpha()]
            # removing empty objects []
            if sentence:
                sentences.append(sentence)
        # format of sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
        print(sentences)

    # # take all words from clean_text.txt
    # words = []
    # for sentence in sentences:
    #     for word in sentence:
    #         words.append(word)
    # print("****************************************************")
    # print(words)
    # print(len(words))


    # TODO stemming, lemmatization, stopword filtering if needed later
    # TODO remove empty [] items from sentences
    # print(raw_sentences)

    print("Recommended values for window,vector_size")
    print("window = 5, 8, 10")
    print("vector_size = 64, 128, 300")
    # default are window=5 and vector_size=100
    window = int(input("window = "))
    vector_size = int(input("vector_size = "))
    # train word2vec model1 with sentences list
    model1 = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=1, workers=2)
    model1.save("../document-classification-using-graph-embeddings/word2vec_models/sentences_word2vec.model")
    print(model1.wv.most_similar('man', topn=10))

    # train word2vec model2 with sentences list by using train() function
    model2 = Word2Vec(min_count=1)
    model2.build_vocab(sentences)  # prepare the model vocabulary
    model2.train(sentences, total_examples=model2.corpus_count, epochs=model2.epochs)
    model2.save("../document-classification-using-graph-embeddings/word2vec_models/sentences2_word2vec.model")
    print(model2.wv.most_similar('man', topn=10))

    # # train word2vec model1 with words list
    # model2 = Word2Vec(words, vector_size=vector_size, window=window, min_count=1, workers=2)
    # model2.save("../document-classification-using-graph-embeddings/word2vec_models/words_word2vec.model")
    # print(model2.wv.most_similar('man', topn=10))

    # # train word2vec model1 with raw_sentences list
    # model2 = Word2Vec(words, vector_size=vector_size, window=window, min_count=1, workers=2)
    # model2.save("../document-classification-using-graph-embeddings/word2vec_models/my_word2vec.model")
    # print(model2.wv.most_similar('man', topn=10))

    # model = Word2Vec.load("word2vec.model")
    # model.train(words, total_examples=1, epochs=1)
    # model.wv.most_similar('computer', topn=10)

    # TODO
    #  1. check model.train() difference between the initialization of the model
    #  2. make a menu with prefixed choices for word2vec settings
    #   a. window = 5, vector-size = 64
    #   b. window = 10, vector-size = 64
    #   c. window = 5, vector-size = 128
    #   d. window = 10, vector-size = 128
    #  3. if possible vector-size = 300
    #  4. search what other parameters on word2vec are worth modifying
    #  5. create word2vec model and save embeddings on word2vec.model
    #  6. save every unique word on a dictionary |word| id| embedding value
    print("--- %s seconds ---" % (time.time() - start_time))
