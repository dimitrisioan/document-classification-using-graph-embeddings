import os
import shutil
import time
import pandas as pd
from useful_methods import *

# Create the following folders in your system!

# original dataset path
original_dataset_path = "original_datasets/emails/emails.csv"
# dataset path after split
dataset_path = "datasets/emails/emails_dataset/"
# basic preprocessing
dataset_parsed_1 = "datasets/emails/emails_parsed_1/"
# basic preprocessing, stopwords removal
dataset_parsed_2 = "datasets/emails/emails_parsed_2/"
# basic preprocessing, stemming, lemmatization
dataset_parsed_3 = "datasets/emails/emails_parsed_3/"

start_time = time.time()

if __name__ == '__main__':

    print('Which dataset preprocessing to do?')
    print('0 - Apply file splitting on original dataset')
    print('1 - Apply basic preprocessing on dataset')
    print('2 - Apply basic preprocessing & stopwords removal on dataset')
    print('3 - Apply basic preprocessing & stemming/lemmatization dataset')
    X = int(input("Enter a value for X: "))

    if X == 0:
        # Create the "spam" and "ham" directories if they don't exist
        spam_dir = os.path.join(dataset_path, 'spam')
        ham_dir = os.path.join(dataset_path, 'ham')
        os.makedirs(spam_dir, exist_ok=True)
        os.makedirs(ham_dir, exist_ok=True)
        print("Directory ", spam_dir, " Created ")
        print("Directory ", ham_dir, " Created ")

        # Read the dataset
        df = pd.read_csv(original_dataset_path)

        for index, row in df.iterrows():
            text = row['Text']
            is_spam = row['Spam']
            # Choose the right directory to save txts
            if is_spam == 1:
                output_path = os.path.join(spam_dir, f'{index}.txt')
            else:
                output_path = os.path.join(ham_dir, f'{index}.txt')

            # Write the text content to the corresponding text file
            with open(output_path, 'w') as new_file:
                print(output_path)
                new_file.write(text)

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
