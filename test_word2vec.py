import os
import multiprocessing
from gensim.models import Word2Vec
import time
import string
from useful_methods import *
import numpy as np
import pandas as pd

start_time = time.time()

parsed_path, prefix, X = choose_dataset_for_word2vec()
load_save_path = load_save_results(prefix, X)

if __name__ == '__main__':
    filecount = 0
    sentences = []
    data = []
    # Loop through every subdirectory, read each text
    for category in os.listdir(parsed_path):
        category_path = os.path.join(parsed_path, category)
        sentences = []
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            print(file_path)
            filecount += 1
            file_sentences = file_to_sentences(file_path)
            sentences.extend(file_sentences)

        # Get the number of available CPU cores
        num_cores = multiprocessing.cpu_count()

        # Train Word2Vec model
        model = Word2Vec(vector_size=300, window=5, min_count=1, workers=num_cores, epochs=10)
        model.build_vocab(sentences)  # prepare the model vocabulary
        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            print(file_path)

            # with open(file_path, "r", errors="ignore") as text_file:
            with open(file_path, "r") as text_file:
                words = text_file.read().split()
                # Skip file if it has less than 3 words
                if len(words) < 3:
                    continue
                # Compute the embedding for the document
                words_found_vectors = [model.wv.get_vector(word) for word in words if word in model.wv.key_to_index]
                result_embedding = np.sum(words_found_vectors, axis=0)

                # Normalize the embedding
                result_embedding = result_embedding / len(words_found_vectors)

                # Convert the embedding to a list
                result_embedding = np.array(result_embedding).tolist()
                # print(result_embedding)
                document_id = file.replace('.txt', '')

                data.append({'document_id': document_id, 'embedding': result_embedding, 'category': category})

    print(sentences)
    # Create Dataframe and save data in a CSV file
    df = pd.DataFrame(data)

    # Save the CSV file for Word2Vec to the corresponding dataset directory
    df.to_csv('all_categories_word2vec.csv', index=False)

    print("--- %s seconds ---" % (time.time() - start_time))
    exit(0)

    # # Save the Word2Vec model to the corresponding dataset directory
    # model.save(os.path.join(load_save_path, f'{prefix}_word2vec'))

    # model.save("../document-classification-using-graph-embeddings/word2vec_models/word2vec.model")
    print(model.wv.most_similar('man', topn=10))

    print("--- %s seconds ---" % (time.time() - start_time))
