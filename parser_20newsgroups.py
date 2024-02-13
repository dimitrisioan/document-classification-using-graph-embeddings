import os
import time
from useful_methods import *

# Create the following folders in your system!

# original dataset path
original_dataset_path = "original_datasets/20newsgroups/"
# dataset path after split
dataset_path = "datasets/20newsgroups/20newsgroups_dataset/"
# basic preprocessing
dataset_parsed_1 = "datasets/20newsgroups/20newsgroups_parsed_1/"
# basic preprocessing, stopwords removal
dataset_parsed_2 = "datasets/20newsgroups/20newsgroups_parsed_2/"
# basic preprocessing, stemming, lemmatization
dataset_parsed_3 = "datasets/20newsgroups/20newsgroups_parsed_3/"

newsgroups_categories_excluded = ['alt.atheism.txt', 'comp.graphics.txt', 'list.csv']

start_time = time.time()

if __name__ == '__main__':

    print('Which dataset preprocessing to do?')
    print('0 - Apply file splitting on original dataset')
    print('1 - Apply basic preprocessing on dataset')
    print('2 - Apply basic preprocessing & stopwords removal on dataset')
    print('3 - Apply basic preprocessing & stemming/lemmatization dataset')
    X = int(input("Enter a value for X: "))

    if X == 0:
        # Create subdirectories under 'newsgroup_dataset_parsed'
        for file in os.listdir(original_dataset_path):
            if file in newsgroups_categories_excluded:
                continue

            # print(file)
            original_dataset_file_path = original_dataset_path + file
            dataset_file_path = dataset_path + file

            # Create subdirectory to save all parsed txt files
            if os.path.isfile(original_dataset_file_path):
                # print(file)
                dirName = dataset_file_path.replace('.txt', '')
                try:
                    # Create target Directory
                    os.mkdir(dirName)
                    print("Directory ", dirName, " Created ")
                except FileExistsError:
                    print("Directory ", dirName, " already exists")

                # Open every file form 'newsgroup_dataset_parsed' and
                # ignore unicode character errors
                with open(original_dataset_file_path, 'r', errors="ignore") as fd:

                    # text = fd.readline()
                    # line = fd.readline()
                    # sline = line.split()

                    # Loop through every file on dataset and split it into
                    # different text files when "Newsgroup:" keyword appears
                    with fd as bigfile:
                        reader = bigfile.read()
                        for i, text_part in enumerate(reader.split("Newsgroup:")[1:]):
                            words = text_part.split()
                            # Get document_id and save every individual text as a
                            # separate txt file with its document_id as filename
                            document_id = words[2]
                            with open(os.path.join(dirName, str(document_id) + ".txt"), 'w') as new_file:
                                full_text = "Newsgroup:" + text_part
                                new_file.write(full_text)

    if X == 1:
        # save from dataset_path to parsed_1
        # apply basic preprocessing on dataset
        filecount = 0
        # Loop through every subdirectory, tokenize every txt file and re-save each file
        for category in os.listdir(dataset_path):
            category_path = os.path.join(dataset_path, category)
            new_category_path = os.path.join(dataset_parsed_1, category)
            # Create a directory in 'new_path' with the same name as 'category'
            if not os.path.exists(new_category_path):
                os.makedirs(new_category_path)

            for file in os.listdir(category_path):
                file_path = os.path.join(category_path, file)
                new_file_path = os.path.join(new_category_path, file)
                filecount += 1
                tokens = preprocess_file(file_path, remove_headers=True)
                write_file(new_file_path, tokens)
                # print(tokens)
        print("Text files are:", filecount)

    if X == 2:
        # save from dataset_path to parsed_2
        # apply basic preprocessing & stopwords removal on dataset
        filecount = 0
        # Loop through every subdirectory, tokenize every txt file and re-save each file
        for category in os.listdir(dataset_path):
            category_path = os.path.join(dataset_path, category)
            new_category_path = os.path.join(dataset_parsed_2, category)
            # Create a directory in 'new_path' with the same name as 'category'
            if not os.path.exists(new_category_path):
                os.makedirs(new_category_path)

            for file in os.listdir(category_path):
                file_path = os.path.join(category_path, file)
                new_file_path = os.path.join(new_category_path, file)
                filecount += 1
                tokens = preprocess_file(file_path, remove_headers=True, remove_stopwords=True)
                write_file(new_file_path, tokens)
                # print(tokens)
        print("Text files are:", filecount)

    if X == 3:
        # save from dataset_path to parsed_3
        # apply basic preprocessing & stemming/lemmatization on dataset
        filecount = 0
        # Loop through every subdirectory, tokenize every txt file and re-save each file
        for category in os.listdir(dataset_path):
            category_path = os.path.join(dataset_path, category)
            new_category_path = os.path.join(dataset_parsed_3, category)
            # Create a directory in 'new_path' with the same name as 'category'
            if not os.path.exists(new_category_path):
                os.makedirs(new_category_path)

            for file in os.listdir(category_path):
                file_path = os.path.join(category_path, file)
                new_file_path = os.path.join(new_category_path, file)
                filecount += 1
                tokens = preprocess_file(file_path, remove_headers=True, stemming=True, lemmatization=True)
                write_file(new_file_path, tokens)
                # print(tokens)
        print("Text files are:", filecount)

    print("--- %s seconds ---" % (time.time() - start_time))
