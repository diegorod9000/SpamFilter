import numpy as np
import tensorflow as tf
import util
import sklearn
from sklearn.neural_network import MLPClassifier

DATA_PATH_TEST = "data/testing"  

def create_training_features():
    """
    Create training set of features and dictionary from clean and spam folders
    """
    
    dictionary = util.get_dictionary()
    print("creating data dicts")
    train_clean = util.get_words_features_clean(dictionary)
    train_spam = util.get_words_features_spam(dictionary)
    
    train_features = []
    train_labels = []
    print("creating clean data features")
    for feature in train_clean:
        train_features.append(list(feature.values()))
        train_labels.append(0)
    print("creating spam data features")
    for feature in train_spam:
        train_features.append(list(feature.values()))
        train_labels.append(1)
        
    features_array = np.array(train_features)
    labels_array = np.array(train_labels)
    
    
    dataset = [features_array,labels_array]
    
    
    return dataset


def train_model(dataset):
    model = MLPClassifier()
    print("running training")
    model.fit(dataset[0],dataset[1])
    print("training complete")
    return model

def eval_model(model):
    files = util.get_files_in_folder(DATA_PATH_TEST)
    
    
    
if __name__ == "__main__":
    dataset = create_training_features()
    model = train_model(dataset)