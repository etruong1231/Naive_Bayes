

# Import numpy and make floating point division the default for Python 2.x
import numpy as np
from scipy.io import loadmat

def load_data():
    data = loadmat('enron.mat')
    trainFeat = np.array(data['trainFeat'], dtype=bool)
    trainLabels = np.squeeze(data['trainLabels'])
    testFeat = np.array(data['testFeat'], dtype=bool)
    testLabels = np.squeeze(data['testLabels'])
    vocab = np.squeeze(data['vocab'])
    vocab = [vocab[i][0].encode('ascii', 'ignore') for i in range(len(vocab))]
    data = dict(trainFeat=trainFeat, trainLabels=trainLabels,
                testFeat=testFeat, testLabels=testLabels, vocab=vocab)
    return data

# Load data
data = load_data()
trainFeat = data['trainFeat']
trainLabels = data['trainLabels']
testFeat = data['testFeat']
testLabels = data['testLabels']
vocab = data['vocab']
W = len(vocab)

vocabInds = np.arange(W)  # 

# Separate "ham" and "spam" classes, subsample selected vocabulary words
trainHam  = trainFeat[trainLabels == 0][:, vocabInds]
trainSpam = trainFeat[trainLabels == 1][:, vocabInds]

# Number of training examples of each class
numHam = len(trainHam)
numSpam = len(trainSpam)

# Count number of times each word occurs in each class
countsHam = np.sum(trainHam, axis=0)
# P(X_ij=1 | Y_i=H) can be computed from countsHam and numHam
countsSpam = np.sum(trainSpam, axis=0)
# P(X_ij=1 | Y_i=S) can be computed from countsSpam and numSpam

# Display words that are common in one class, but rare in the other
ind = np.argsort(countsHam-countsSpam)


# need to find how many spam Emails there are
# 0 for Ham 1 for Spam
from collections import defaultdict



# get the spam probability of each word 
word_spam_prob = defaultdict(float) # all the probablilites spam words 



for i,x in enumerate (countsSpam):
    word_spam_prob[vocab[i]] = x / numSpam


# now need to get all Ham probabilities
# creates a dictionary of key and values of vocab, reccurences

# get the spam probability of each word 
word_ham_prob = defaultdict(float) # all the probablilites ham words 

# to get the opposite probability 1- P(Xij = 1 | Y = H) : get each word and 1 - word_ham_prob[word]

for i, x in enumerate (countsHam):
    word_ham_prob[vocab[i]] = x / numHam





def predict(word,index):
    ''' returns a list of 0 or 1 accordingly to prediction '''
    predict_list = []
    for emails in testFeat:
        if emails[index]: # does bayes if money is present
            if word_spam_prob[word]* (numSpam/len(trainLabels)) > word_ham_prob[word] * (numHam/len(trainLabels)):
                predict_list.append(1) # it is spam
            else:
                predict_list.append(0) # it is ham
        else: # else not present
            if (1 - word_spam_prob[word])* (numSpam/len(trainLabels)) > (1-word_ham_prob[word]) * (numHam/len(trainLabels)):
                predict_list.append(1)
            else:
                predict_list.append(0)

    return predict_list


def calc_accuracy_(predict_list):
    correct_tup = zip(predict_list, testLabels)

    total_error = 0

    for x,y in correct_tup:
        total_error += abs(x-y)

    accuracy = 1 - (total_error / len(testLabels))

    return accuracy





from math import log

predict_list = np.zeros(trainFeat.shape[0])

for x in range(trainFeat.shape[0]):
    log_spam = 0
    log_ham = 0
    all_occur = np.where(trainFeat[x] == True)
    non_occur = np.where(trainFeat[x] == False)
    
    log_spam += np.sum(np.log10(countsSpam[all_occur]/numSpam))
    log_ham += np.sum(np.log10(countsHam[all_occur]/numHam))
    
    log_spam += np.sum([np.log10((1 - (countsSpam[non_occur]/numSpam)))])
    log_ham += np.sum([np.log10((1 - (countsHam[non_occur]/numHam)))])
    
    if log_spam > log_ham:
        predict_list[x] = 1
    else:
        predict_list[x] = 0
        

accuracy = calc_accuracy(predict_list)

print(accuracy)
print("\nThe Bayes Classifier Accuracy on all words on testFeat: ", accuracy)






      
