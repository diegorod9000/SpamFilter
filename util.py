import os

DATA_PATH_CLEAN = "/data/clean"
DATA_PATH_SPAM = "/data/spam"

class Dict_Default(dict):
    """
    Identical to dict, except returns default instead of None if no key found.
    """
    def __init__(self,default_val):
        self.default_val = default_val
        
    def __missing__(self, key):
        return self.default_val
    

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

def get_counts(file_list):
    """
    Given a list of files, counts total number of occurences of each word and 
    outputs a dict containing the total for all files.
    """
    output = Dict_Default(0)
    for file in file_list:
        
        words = set(get_all_words(file))
        
        for word in words:
            output[word] +=1
    
    return output

def get_words_clean():
    fileList = get_files_in_folder(DATA_PATH_CLEAN)
    return get_counts(fileList)

def get_words_spam():
    fileList = get_files_in_folder(DATA_PATH_SPAM)
    return get_counts(fileList)