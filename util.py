import os
import copy

DATA_PATH_CLEAN = "data/clean_large"
DATA_PATH_SPAM = "data/spam_large"

def get_all_words(filename):
    """ Returns list of all words in the file at filename."""
    with open(filename, 'r', encoding = "ISO-8859-1") as f:
        words = f.read().split()
    return words

def get_files_in_folder(folder):
    """ Returns a list of file paths in folder"""
    filenames = os.listdir(folder)
    
    full_filenames = [os.path.join(folder, filename) for filename in filenames]
    return full_filenames

def get_count_features(file_list,word_dict):
    """
    Given a list of files, counts total number of occurences of each word per file and 
    outputs a list containing every feature dictionary.
    """
    output = []
    for file in file_list:    
        words = get_all_words(file)
        nextFeature = word_dict.copy()
        
        for word in words:
            if word in nextFeature:
                nextFeature[word]+=1
        output.append(nextFeature)
    
    return output

def get_words_features_clean(word_dict):
    """
    Given the template dictionary, creates a feature list for all files in the clean folder
    """
    fileList = get_files_in_folder(DATA_PATH_CLEAN)
    return get_count_features(fileList,word_dict)

def get_words_features_spam(word_dict):
    """
    Given the template dictionary, creates a feature list for all files in the spam folder
    """
    fileList = get_files_in_folder(DATA_PATH_SPAM)
    return get_count_features(fileList,word_dict)

def get_dictionary():
    """
    Creates empty dictionary of all words in the training set, initialized to 0.
    """
    fileList1 = get_files_in_folder(DATA_PATH_CLEAN)
    fileList2 = get_files_in_folder(DATA_PATH_SPAM)
    output = dict()
    for file in fileList1:
        words = set(get_all_words(file))
        for word in words:
            output[word] = 0
    for file in fileList2:
        words = set(get_all_words(file))
        for word in words:
            output[word] = 0
    return output