import os
from gensim.models import Word2Vec
import time
import string

start_time = time.time()

parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_parsed/"
directory_path = "../document-classification-using-graph-embeddings/"

# TODO STEMMING, LEMMATIZATION, STOPWORD FILTERING IF NEEDED LATER.

if __name__ == '__main__':


    # Alternative solution for spliting text into sentences for Word2Vec model
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

    # Read data from clean_text.txt and make sentences for Word2Vec model
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
    # print(raw_sentences)

    print("Recommended values for window, vector_size")
    print("window = 5, 8, 10")
    print("vector_size = 64, 128, 300")

    # Default are window=5 and vector_size=100
    window = int(input("window = "))
    vector_size = int(input("vector_size = "))

    # Train Word2Vec model
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=1, workers=4)
    # model.build_vocab(sentences)  # prepare the model vocabulary
    # model.train(sentences, total_examples=model2.corpus_count, epochs=model2.epochs)
    model.save("../document-classification-using-graph-embeddings/word2vec_models/word2vec.model")
    print(model.wv.most_similar('man', topn=10))

    print("--- %s seconds ---" % (time.time() - start_time))
