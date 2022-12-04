import os
import shutil
import string
import time
from nltk.corpus import stopwords

start_time = time.time()

# Create the following folders in your system!
dataset_path = "../document-classification-using-graph-embeddings/newsgroups_dataset/"
parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_parsed/"
combined_parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_combined_parsed/"

# split data into txts like I did in sentences "Subject:" included
if __name__ == '__main__':
    # Delete all files on 'newsgroup_dataset_parsed'
    for f in os.listdir(combined_parsed_path):
        file_path = os.path.join(combined_parsed_path, f)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    dirName = combined_parsed_path
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")

    for file in os.listdir(dataset_path):
        # print(file)
        dataset_filepath = dataset_path + file
        combined_parsed_filepath = combined_parsed_path + file
        file_name = dataset_filepath.replace('.txt', '')

        with open(dataset_filepath, 'rt', errors="ignore") as fd:
            lines = fd.readlines()
        # exclude first 3 lines after reading the string "Newsgroup:" and save on clean.txt
        with open(os.path.join(dirName, str(file)), 'w') as nf:
            i = 0
            while i < len(lines):
                line = lines[i]
                # skip 3 lines after finding string "Newsgroup:"
                if line.startswith("Newsgroup:"):
                    i += 3
                else:
                    line = line.lower()
                    list_of_words = line.split()
                    # print(list_of_words)
                    # print(line)
                    # Remove punctuation
                    table = str.maketrans('', '', string.punctuation)
                    stripped = [w.translate(table) for w in list_of_words]
                    # Remove numbers
                    list_of_words = [word for word in stripped if word.isalpha()]
                    list_of_words_size = len(list_of_words)
                    for word in list_of_words:
                        print(word)
                        nf.write(word)
                        nf.write("\n")
                    i += 1

    # for text_file in os.listdir(dirName):
    #     combined_parsed_filepath = combined_parsed_path + text_file
    #     with open(combined_parsed_filepath, 'r') as file:
    #         # splitting the file data into lines
    #         raw_sentences = [[item.rstrip('\n')] for item in file]
    #         # splitting the lines into [tokens]
    #         raw_sentences = [item[0].split(" ") for item in raw_sentences]
    #         sentences = []
    #         for sentence in raw_sentences:
    #             # removing digits and punctuation from sentences
    #             table = str.maketrans('', '', string.punctuation)
    #             stripped = [w.translate(table) for w in sentence]
    #             sentence = [word for word in stripped if word.isalpha()]
    #             sentences.append(sentence)
    #         # format of sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
    #         print(sentences)
    #
    #     # take all words from clean_text.txt
    #     words = []
    #     for sentence in sentences:
    #         for word in sentence:
    #             words.append(word)

    print("--- %s seconds ---" % (time.time() - start_time))
