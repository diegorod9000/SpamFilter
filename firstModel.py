import numpy as np
import tensorflow as tf
import util




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
        
    features_array = np.array(train_features,dtype=np.float32)
    labels_array = np.array(train_labels,dtype=np.float32)
    
    
    dataset = tf.data.Dataset.from_tensor_slices((features_array,labels_array))
    
    print(len(train_features))
    print(len(train_labels))

if __name__ == "__main__":
    create_training_features()