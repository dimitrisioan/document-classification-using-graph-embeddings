from gensim.models import Word2Vec
import time
import os
from useful_methods import *

parsed_path, prefix, choice = choose_dataset()
load_save_path = load_save_results(prefix, choice)

start_time = time.time()

if __name__ == '__main__':
    # Load the Word2Vec model from the corresponding dataset directory
    model = Word2Vec.load(os.path.join(load_save_path, f'{prefix}_word2vec'))
    # model = Word2Vec.load("../document-classification-using-graph-embeddings/word2vec_models/word2vec.model")

    filecount = 0
    data = []

    # Loop through every subdirectory, read each word from every file
    for category in os.listdir(parsed_path):
        category_path = os.path.join(parsed_path, category)
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            print(file_path)

            # with open(file_path, "r", errors="ignore") as text_file:
            with open(file_path, "r") as text_file:
                words = text_file.read().split()
                # Skip file if it has less than 3 words
                if len(words) < 3:
                    continue
                filecount += 1
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

    # Create Dataframe and save data in a CSV file
    df = pd.DataFrame(data)

    # Save the CSV file for Word2Vec to the corresponding dataset directory
    df.to_csv(os.path.join(load_save_path, f'{prefix}_embeddings_word2vec.csv'), index=False)
    # df.to_csv("data_for_classifiers_word2vec.csv", index=False)

    print("Text files are:", filecount)
    print("--- %s seconds ---" % (time.time() - start_time))


