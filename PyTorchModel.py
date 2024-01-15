import numpy as np
import util
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_PATH_TEST = "data/testing_large"
WEIGHT_CLEAN = 2
WEIGHT_SPAM = 1
LEARNING_RATE = 0.03
REGULARIZE = 0.1
NUM_EPOCHS = 30

class Net(nn.Module):
    def __init__(self, length):
        super(Net, self).__init__()
        length1 = length>>3
        length2 = length>>5
        self.fc1 = nn.Linear(length, length1) 
        self.fc2 = nn.Linear(length1, length2)
        self.fc3 = nn.Linear(length2, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
        for i in range(WEIGHT_CLEAN):
            train_features.append(list(feature.values()))
            train_labels.append(0)
    print("creating spam data features")
    for feature in train_spam:
        for i in range(WEIGHT_SPAM):
            train_features.append(list(feature.values()))
            train_labels.append(1)
        
    features = torch.tensor(train_features, dtype=torch.float32)
    labels = torch.tensor(train_labels)
    if(torch.cuda.is_available()):
        features = features.to(device="cuda")
        labels = labels.to(device="cuda")
    
    
    return features, labels, dictionary


def train_network(features,labels):
    
    net = Net(len(features[0]))
    print("running training")
    
    optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZE)
    criterion = nn.CrossEntropyLoss()
    for i in range(NUM_EPOCHS):
        optimizer.zero_grad()
        output = net(features)
        loss = criterion(output, labels)
        print(loss)
        loss.backward()
        optimizer.step()
    
    return net

def eval_network(net,dictionary):
    
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
    test_features = torch.Tensor(test_features)
    
    if(torch.cuda.is_available()):
        test_features = test_features.to(device="cuda")
    
    print("predicting")
    predictions = net(test_features)
    
    correctSpam = 0
    correctClean = 0
    for i in range(len(predictions)):
        prediction = 1 if predictions[i][0]<predictions[i][1] else 0
        if prediction==true_vals[i]:
            if prediction == 0:
                correctClean+=1
            else:
                correctSpam+=1
                
    print("correctly identified %d out of %d clean emails" % (correctClean,totalClean))
    print("correctly identified %d out of %d spam emails" % (correctSpam,totalSpam))
    
    
if __name__ == "__main__":
    features, labels, dictionary = create_training_features()
    model = train_network(features,labels)
    eval_network(model,dictionary)
    
