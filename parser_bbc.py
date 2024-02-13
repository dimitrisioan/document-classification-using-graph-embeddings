import os
import time
from useful_methods import *

# Create the following folders in your system!

# original dataset path
original_dataset_path = "original_datasets/bbc/"
# dataset path after split
dataset_path = "datasets/bbc/bbc_dataset/"
# basic preprocessing
dataset_parsed_1 = "datasets/bbc/bbc_parsed_1/"
# basic preprocessing, stopwords removal
dataset_parsed_2 = "datasets/bbc/bbc_parsed_2/"
# basic preprocessing, stemming, lemmatization
dataset_parsed_3 = "datasets/bbc/bbc_parsed_3/"

start_time = time.time()

if __name__ == '__main__':

    print('Which dataset preprocessing to do?')
    print('0 - Apply file splitting on original dataset')
    print('1 - Apply basic preprocessing on dataset')
    print('2 - Apply basic preprocessing & stopwords removal on dataset')
    print('3 - Apply basic preprocessing & stemming/lemmatization dataset')
    X = int(input("Enter a value for X: "))

    if X == 0:
        print("Already done")

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
                tokens = preprocess_file(file_path)
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
                tokens = preprocess_file(file_path, remove_stopwords=True)
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
                tokens = preprocess_file(file_path, stemming=True, lemmatization=True)
                write_file(new_file_path, tokens)
                # print(tokens)
        print("Text files are:", filecount)

    print("--- %s seconds ---" % (time.time() - start_time))
