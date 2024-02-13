import os
from gensim.models import Word2Vec
import time
import multiprocessing
from useful_methods import *

parsed_path, prefix, X = choose_dataset_for_word2vec()
load_save_path = load_save_results(prefix, X)

start_time = time.time()

if __name__ == '__main__':
    filecount = 0
    sentences = []
    if prefix == '20newsgroups':
        # Loop through every subdirectory, read each text
        for category in os.listdir(parsed_path):
            category_path = os.path.join(parsed_path, category)
            for file in os.listdir(category_path):
                file_path = os.path.join(category_path, file)
                print(file_path)
                filecount += 1
                if X == 1:
                    file_sentences = file_to_sentences(file_path, remove_headers=True)
                if X == 2:
                    file_sentences = file_to_sentences(file_path, remove_headers=True, remove_stopwords=True)
                if X == 3:
                    file_sentences = file_to_sentences(file_path, remove_headers=True, stemming=True,
                                                       lemmatization=True)
                sentences.extend(file_sentences)
                # print(sentences)
            #     break
            # break

    if prefix == 'bbc' or prefix == 'emails':
        # Loop through every subdirectory, read each text
        for category in os.listdir(parsed_path):
            category_path = os.path.join(parsed_path, category)
            for file in os.listdir(category_path):
                file_path = os.path.join(category_path, file)
                print(file_path)
                filecount += 1
                if X == 1:
                    file_sentences = file_to_sentences(file_path)
                if X == 2:
                    file_sentences = file_to_sentences(file_path, remove_stopwords=True)
                if X == 3:
                    file_sentences = file_to_sentences(file_path, stemming=True, lemmatization=True)
                sentences.extend(file_sentences)
                # print(sentences)
            #     break
            # break
    print(sentences)
    print(f'The total number of sentences are: {len(sentences)}')
    # exit(0)
    # Get the number of available CPU cores
    num_cores = multiprocessing.cpu_count()

    # Train Word2Vec model
    model = Word2Vec(vector_size=300, window=5, min_count=5, workers=num_cores, epochs=15, sg=1)
    model.build_vocab(sentences)  # prepare the model vocabulary
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

    # Save the Word2Vec model to the corresponding dataset directory
    model.save(os.path.join(load_save_path, f'{prefix}_word2vec'))
    # model.save("../document-classification-using-graph-embeddings/word2vec_models/word2vec.model")

    print(f'The total number of words are: {len(model.wv.key_to_index)}')

    print(model.wv.most_similar('man', topn=10))

    print("Text files are:", filecount)
    print("--- %s seconds ---" % (time.time() - start_time))
