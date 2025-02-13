import pickle
import sys
import math
from collections import Counter
import numpy as np
from collections import defaultdict

POSITIVE = "en"
NEGATIVE = "nl"

class DecisionTreeNode:
    def __init__(self, attribute=None, is_leaf=False, classification=None):
        self.attribute = attribute
        self.is_leaf = is_leaf
        self.classification = classification
        self.children = {}

    def predict(self, sample):
        if self.is_leaf:
            return self.classification
        return self.children[sample[self.attribute]].predict(sample)

def PLURALITY_VALUE(examples):
    count = Counter(example[-1] for example in examples)
    return count.most_common(1)[0][0]

def entropy(examples, weights):
    labels = [example[-1] for example in examples]
    label_counts = Counter()
    total_weight = sum(weights)
    
    for label, weight in zip(labels, weights):
        label_counts[label] += weight
    
    res = 0
    for count in label_counts.values():    
        res -= (count / total_weight) * math.log2(count / total_weight)
    return res
    

def IMPORTANCE(attribute, examples, weights):
    total_entropy = entropy(examples, weights)
    values = set(example[attribute] for example in examples)
    subset_entropy = 0
    total_weight = sum(weights)
    
    for value in values:
        subset = [example for example in examples if example[attribute] == value]
        subset_weights = [weights[i] for i in range(len(examples)) if examples[i][attribute] == value]
        probability = sum(subset_weights) / total_weight
        subset_entropy += probability * entropy(subset, subset_weights)
    
    return total_entropy - subset_entropy

def LEARN_DECISION_TREE(examples, attributes, parent_examples=None, depth=0, isAda=False, weights=None):
    if weights is None:
        weights = [1] * len(examples)
    
    if isAda and depth == 1:
        return DecisionTreeNode(is_leaf=True, classification=PLURALITY_VALUE(examples))

    if not examples:
        return DecisionTreeNode(is_leaf=True, classification=PLURALITY_VALUE(parent_examples))
    elif all(example[-1] == examples[0][-1] for example in examples):
        return DecisionTreeNode(is_leaf=True, classification=examples[0][-1])
    elif not attributes:
        return DecisionTreeNode(is_leaf=True, classification=PLURALITY_VALUE(examples))
    else:
        max_importance = float('-inf')
        best_attribute = None
        for attribute in attributes:
            importance = IMPORTANCE(attribute, examples, weights)
            if importance > max_importance:
                max_importance = importance
                best_attribute = attribute
        A = best_attribute

        tree = DecisionTreeNode(attribute=A)
        for value in set(example[A] for example in examples):
            exs = [example for example in examples if example[A] == value]
            exs_weights = [weights[i] for i in range(len(examples)) if examples[i][A] == value]
            subtree = LEARN_DECISION_TREE(exs, [attr for attr in attributes if attr != A], examples, depth+1, isAda, exs_weights)
            tree.children[value] = subtree
        return tree




class AdaBoost:
    def __init__(self, dataset):
        self.dataset = dataset
        self.w = [1/len(self.dataset)] * len(self.dataset)
        self.h = []  # a vector of K hypotheses
        self.z = []  # a vector of K hypothesis weights

    def train(self, L, K):
        N = len(self.dataset)
        self.w = [1/len(self.dataset)] * len(self.dataset)
        epsilon = 0.00000000000001
        for _ in range(K):
            hypothesis = L(self.dataset, list(range(len(self.dataset[0]) - 1)), isAda=True, weights=self.w)
            self.h.append(hypothesis)
            error = 0
            for i in range(N):
                if hypothesis.predict(self.dataset[i]) != self.dataset[i][-1]:
                    error += self.w[i]
            if error > 0.5:
                break
            error = max(error, epsilon)
            for i in range(N):
                if hypothesis.predict(self.dataset[i]) == self.dataset[i][-1]:
                    self.w[i] = self.w[i] * (error / (1 - error))

            total_weight = sum(self.w)
            self.w = [w / total_weight for w in self.w]
            self.z.append(0.5 * math.log((1 - error) / error))
        
        return WeightedMajority(self.h, self.z)

class WeightedMajority:
    def __init__(self, h, z):
        self.h = h
        self.z = z

    def predict(self, example):
        return self.weighted_majority([predictor.predict(example) for predictor in self.h], self.z)

    def weighted_majority(self, h, z):
        totals = defaultdict(int)
        for v, w in zip(h, z):
            totals[v] += w
        return max(totals, key=totals.__getitem__)
    

def read_data(features, filename):
    dataset = []
    features_lst = []
    with open(features, 'r') as f:
        for line in f:
            features_lst.append(line.strip())
    with open(filename, 'r') as f:
        for line in f:
            label, sentence = line.strip().split("|")
            sample = [feature in sentence for feature in features_lst]
            sample.append(label)
            dataset.append(sample)
    return dataset

def read_data_prediction(features, filename):

    features_lst = []
    with open(features, 'r') as f:
        for line in f:
            features_lst.append(line.strip())
    dataset = []
    with open(filename, 'r') as f:
        for line in f:
            sentence = line.strip()
            sample = [feature in sentence for feature in features_lst]
            dataset.append(sample)
    return dataset

def main():
    method = sys.argv[1]
    if method == "train":
        examples, features, hypOut, learnType = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
        dataset = read_data(features, examples)
        if learnType == "dt":
            attributes = list(range(len(dataset[0]) - 1))
            model = LEARN_DECISION_TREE(dataset, attributes)
            with open(hypOut, 'wb') as f:
                pickle.dump(model, f)
        elif learnType == "ada":
            ada = AdaBoost(dataset)
            model = ada.train(LEARN_DECISION_TREE, 8)
            with open(hypOut, 'wb') as f:
                pickle.dump(model, f)
        else:
            raise Exception("Invalid learning type")
    elif method == "predict":
        file, features, hypothesis = sys.argv[2], sys.argv[3], sys.argv[4]
        with open(hypothesis, 'rb') as f:
            model = pickle.load(f)
        sample = read_data_prediction(features, file)
        prediction = [model.predict(s) for s in sample]
        for pred in prediction:
            print(pred)
        return prediction
    else:
        raise Exception("Invalid method")

if __name__ == "__main__":
    main()
