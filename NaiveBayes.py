import sys
import getopt
import os
import math
import operator
import random
from collections import defaultdict

class NaiveBayes:
    class Splitter:
        """Represents a set of training/testing data. self.train is a list of Examples, as is self.test. 
        """
        def __init__(self):
            self.train = []
            self.test = []
            self.dev = []

    class Example:
        """Represents a document with a label. klass is 'pos' or 'neg' by convention.
            words is a list of strings.
        """
        def __init__(self):
            self.klass = ''
            self.words = []


    def __init__(self):
        """Naive Bayes initialization"""
        self.USE_BIGRAMS = False
        self.BOOLEAN_NB = False
        self.stopList = set(self.readFile(os.path.join('data', 'english.stop')))

        #TODO: add other data structures needed in classify() and/or addExample() below
        self.counts_pos = defaultdict(int)
        self.counts_neg = defaultdict(int)
        self.num_docs = defaultdict(int)
        self.V = -1
        self.T_pos = -1
        self.T_neg = -1


    #############################################################################
    # TODO TODO TODO TODO TODO 
    # Implement the the Naive Bayes Classifier with Boolean (Binarized) features.
    #
    # We have provided you with a Naive Bayes Classfier. You can use it as is or replace
    # it with your solution from PA2.
    #
    # If the BOOLEAN_NB flag is true, your methods must implement Boolean (Binarized)
    # Naive Bayes (that relies on feature presence/absence) instead of the usual algorithm
    # that relies on feature counts.
    #
    # If the USE_BIGRAMS flag is true, your methods must use bigram features instead of the usual 
    # bag-of-words (unigrams)
    #
    # If any one of the BOOLEAN_NB or USE_BIGRAMS flags is on, the 
    # other is meant to be off.
    
    def classify(self, words):
        total_docs = sum(self.num_docs.values())
        logPriorPos = math.log(float(self.num_docs['pos']) / total_docs)
        logPriorNeg = math.log(float(self.num_docs['neg']) / total_docs)

        if self.T_pos == -1:
            self.T_pos = sum(self.counts_pos.values())
        if self.T_neg == -1:
            self.T_neg = sum(self.counts_neg.values())
        if self.V == -1:
            self.V = len(set(list(self.counts_pos.keys()) + list(self.counts_neg.keys())))
       

        logProbPos = logPriorPos 
        logProbNeg = logPriorNeg

        seen = set()
        if self.USE_BIGRAMS:
            words = ['<s>'] + words + ['</s>']
            for ix in range(0,len(words)-1):
                w = (words[ix],words[ix+1])
                if w not in seen:
                    seen.add(w)
                    if w in self.counts_pos or w in self.counts_neg:
                        loglikelihoodPos = math.log(float(self.counts_pos[w] + 1) / (self.T_pos + self.V))
                        loglikelihoodNeg = math.log(float(self.counts_neg[w] + 1) / (self.T_neg + self.V))
                        logProbPos += loglikelihoodPos
                        logProbNeg += loglikelihoodNeg

        else:
            for w in words:
                if w not in seen:
                    seen.add(w)
                    if w in self.counts_pos or w in self.counts_neg:
                        loglikelihoodPos = math.log(float(self.counts_pos[w] + 1) / (self.T_pos + self.V))
                        loglikelihoodNeg = math.log(float(self.counts_neg[w] + 1) / (self.T_neg + self.V))
                        logProbPos += loglikelihoodPos
                        logProbNeg += loglikelihoodNeg


        if logProbPos >= logProbNeg:
            return 'pos'
        else:
            return 'neg'

    def addExample(self, klass, words):
        self.num_docs[klass] += 1

        seen = set()
        if self.USE_BIGRAMS:
            words = ['<s>'] + words +  ['</s>']

            for ix in range(0,len(words)-1):
                w = (words[ix],words[ix+1])
                if w not in seen:
                    seen.add(w)
                    if klass == 'pos':
                        self.counts_pos[w] += 1
                    else:
                        self.counts_neg[w] += 1
        else:
            for word in words:
                if word not in seen:
                    seen.add(word)
                    if klass == 'pos':
                        self.counts_pos[word] += 1
                    else:
                        self.counts_neg[word] += 1


# END TODO(Modify code beyond here with caution)#######################################################################################
   
    def readFile(self, fileName):
        contents = []
        f = open(fileName, encoding='latin-1')
        contents = f.read()
        f.close()
        return contents

    def buildSplit(self,include_test=True):

        split = self.Splitter()
        datasets = ['train','dev']
        if include_test:
            datasets.append('test')
        for dataset in datasets:
            for klass in ['pos', 'neg']:
                filePath = os.path.join('data', dataset, klass)
                dataFiles = os.listdir(filePath)
                for dataFile in dataFiles:
                    words = self.readFile(os.path.join(filePath, dataFile)).replace('\n',' ')
                    example = self.Example()
                    example.words = words.split()
                    example.words = self.filterStopWords(example.words)
                    example.klass = klass
                    if dataset == 'train':
                        split.train.append(example)
                    elif dataset == 'dev':
                        split.dev.append(example)
                    else :
                        split.test.append(example)
        return split


    def filterStopWords(self, words):
        """Filters stop words."""
        filtered = []
        for word in words:
            if not word in self.stopList and word.strip() != '':
                filtered.append(word)
        return filtered

def evaluate(BOOLEAN_NB, USE_BIGRAMS):
    classifier = NaiveBayes()
    classifier.BOOLEAN_NB = BOOLEAN_NB
    classifier.USE_BIGRAMS = USE_BIGRAMS
    split = classifier.buildSplit(include_test=False)

    for example in split.train:
        classifier.addExample(example.klass, example.words)

    train_accuracy = calculate_accuracy(split.train, classifier)
    dev_accuracy = calculate_accuracy(split.dev, classifier)

    print('Train Accuracy: {}'.format(train_accuracy))
    print('Dev Accuracy: {}'.format(dev_accuracy))


def calculate_accuracy(dataset, classifier):
    acc = 0.0
    for example in dataset:
        guess = classifier.classify(example.words)    
        if example.klass == str(guess):
            acc += 1.0
    return acc / len(dataset)


def main():
    BOOLEAN_NB = False
    USE_BIGRAMS = False
    (options, args) = getopt.getopt(sys.argv[1: ], 'bu')
    if ('-b', '') in options:
        BOOLEAN_NB = True

    elif ('-u', '') in options:
        USE_BIGRAMS= True

    evaluate(BOOLEAN_NB, USE_BIGRAMS)

if __name__ == "__main__":
        main()