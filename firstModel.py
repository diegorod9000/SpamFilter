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
    
    
    return [dataset,dictionary]


def train_model(dataset):
    model = MLPClassifier()
    print("running training")
    model.fit(dataset[0],dataset[1])
    print("training complete")
    return model

def eval_model(model,dictionary):
    
    files = util.get_files_in_folder(DATA_PATH_TEST)
    true_vals = []
    test_features = []
    
    totalSpam = 0
    totalClean = 0
    print("creating evaluation set")
    for filename in files:    
        words = util.get_all_words(filename)
        nextFeature = dictionary.copy()
        
        if filename.split(".")[-2]=="spam":
            true_vals.append(1)
            totalSpam+=1
        else:
            true_vals.append(0)
            totalClean+=1
            
        for word in words:
            if word in nextFeature:
                nextFeature[word]+=1
        test_features.append(list(nextFeature.values()))
        
    print("predicting")
    predictions = model.predict(test_features)
    
    correctSpam = 0
    correctClean =0
    for i in range(len(predictions)):
        if predictions[i]==true_vals[i]:
            if predictions[i] == 0:
                correctClean+=1
            else:
                correctSpam+=1
                
    print("correctly identified %d out of %d clean emails" % (correctClean,totalClean))
    print("correctly identified %d out of %d spam emails" % (correctSpam,totalSpam))
    
    
if __name__ == "__main__":
    dataset,dictionary = create_training_features()
    model = train_model(dataset)
    eval_model(model,dictionary)