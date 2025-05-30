import re
import numpy as np
from sys import argv
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.optim as optim

# Claire Gillaspie
# 4/30/25
# CMSC 437
# Word Sense Disambiguation with GloVe Embeddings

### DESCRIPTION & HOW TO RUN ###
# This program determines the sense of ambiguous words
# Using GloVe word embeddings and a specified model (Neural Network or SVM)
# Or defaults to Logistic Regression
# Run the program on the command line: python3 wsd-embeddings.py line-train.txt line-test.txt <embedding file> [OPTIONAL: NN|SVM] > my-line-answers.txt
# line-train.txt - contains labelled training data
# line-test.txt - contains ambiguous words to be identified by the model
# <embedding-file> - the pre-trained GloVe embeddings
# my-line-answers.txt - contains answer tags for each sentence

### ALGORITHM ###
# Cleans input from line-train.txt, finds a window of context surrounding the target word, and stores the given sense
# Then, pre-trained GloVe vectors are loaded from the given embedding file
# And the average GloVe vector for each instance is computed by taking the mean of all vectors for each word in the instance
# Then, the chosen model is trained (SVM, NN, or LR)
# Once trained, the model reads in line-test.txt and parses for the context of each instance
# And then classifies the sense for each instance

### SAMPLE INPUT & OUTPUT ###
# input:
# python3 wsd-embeddings.py line-data/line-train.txt line-data/line-test.txt glove.6B.100d.txt NN  > my-line-answers.txt
#
# output (first 10 lines):
# <answer instance="line-n.w8_059:8174:" senseid="phone"/>
# <answer instance="line-n.w7_098:12684:" senseid="product"/>
# <answer instance="line-n.w8_106:13309:" senseid="phone"/>
# <answer instance="line-n.w9_40:10187:" senseid="phone"/>
# <answer instance="line-n.w9_16:217:" senseid="product"/>
# <answer instance="line-n.w8_119:16927:" senseid="product"/>
# <answer instance="line-n.w8_008:13756:" senseid="phone"/>
# <answer instance="line-n.w8_041:15186:" senseid="phone"/>
# <answer instance="line-n.art7} aphb 05601797:" senseid="phone"/>
# <answer instance="line-n.w8_119:2964:" senseid="product"/>

### MODEL DESCRIPTIONS ###
# Support Vector Machine (SVM):
    # Classifier separates data points by finding the hyperplane
    # that best separates the two classes of data points
# Neural Network (NN):
    # A two layer neural network built using PyTorch
    # Uses a Linear layer to reduce input dimensions by half
    # Then uses ReLU activation to introduce non-linearity
    # Uses a second Linear layer to reduce the output to a single neuron
    # Finally, Sigmoid activation ensures the output is between 0 and 1
    # The output represents the probability for either sense
    # The model uses Binary Cross Entropy Loss and the Adam optimizer
# Logistic Regression:
    # Used for predicting binary classes
    # Type of linear regression for categorical target variables
    # Using log of odds as the dependent variable

### ACCURACY & CONFUSION MATRIX ###
## PA5 ##
# NN:
# Accuracy: 0.9524
#
# Confusion Matrix:
#         phone   product
# phone   69      3
# product 3       51
#
# SVM:
# Accuracy: 0.9762
#
# Confusion Matrix:
#         phone   product
# phone   70      2
# product 1       53
#
## PA4 ##
# SVM:
# Accuracy: 0.8968
#
# Confusion Matrix:
# 	phone	product
# phone	63	9
# product	4	50
#
# SGD:
# Accuracy: 0.8968
#
# Confusion Matrix:
#         phone   product
# phone   63      9
# product 4       50
#
# NB:
# Accuracy: 0.8968
#
# Confusion Matrix:
#         phone   product
# phone   63      9
# product 4       50
#
## PA3: ##
# Accuracy: 0.9127
#
# Confusion Matrix:
# 	phone	product
# phone	68	4
# product	7	47
#
## Most Frequent Sense: product
# MFS Baseline Accuracy: 0.5241


# find all instances, contexts, and senses in training file
def parse_instances(input_file):
    with open(input_file) as file:
        text = file.read().lower()
        # remove newlines
        text = text.replace('\n', '')
        # remove unnecessary delimiters and punctuation
        text = re.sub(r"(<s>|</s>|<@>|<p>|</p>|[,'!?])", "", text)
        # remove stop words
        text = re.sub(r'\b(the|an?|and|of|is|that|to|in|with|no|didnt|but|have|as|its?|was)\b', '', text)
        # clean spaces
        text = re.sub(r'\s+', ' ', text)

        # store tuples of (instance id, context, sense)
        instances = []
        instance_ids = []
        senses = []

        instance_matches = re.findall(r'<(.*?)</instance>', text)

        # find window of context (training & testing) and given sense (training only)
        for curr_instance in instance_matches:
            find_current_instance_id = re.search(r'instance id="(line.*:)">', curr_instance)
            current_instance_id = find_current_instance_id.group(1)

            find_current_context = re.search(r'<context>(.*)</context>', curr_instance)
            current_context = find_current_context.group(1)
            # clean context of symbols that cannot be removed from entire instance
            current_context = re.sub(r'["-.][0-9]*', '', current_context)

            find_target = re.search(r'(<head>.*?</head>)', current_context)
            target = find_target.group(1)

            words = current_context.split()
            target_position = -1
            # number of words of context (features) to consider before & after target
            # determined using trial & error based on accuracy score
            window_size = 8

            # find ambiguous word index
            for p, word in enumerate(words):
                if word == target:
                    target_position = p
                    break

            # find window of context surrounding target & keep within length of total context
            start = max(0, target_position - window_size)
            end = min(len(words), target_position + window_size + 1)
            windowed_context = ' '.join(words[start:end])

            # assign sense
            if 'test' in input_file:  # test
                assigned_sense = None
            else:  # training
                find_senseid = re.search(r'senseid="(\w+)"', curr_instance)
                assigned_sense = find_senseid.group(1)

            instances.append(windowed_context)
            senses.append(assigned_sense)
            instance_ids.append(current_instance_id)

        return instances, instance_ids, senses


# create GloVe embeddings
def process_vectors(GloVe_vectors):
    embeddings = {}
    with open(GloVe_vectors, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            # get word at start
            word = values[0]
            # get vector numbers
            vector = np.array([float(x) for x in values[1:]], dtype=np.float32)
            # dictionary with word as key, vector as value
            embeddings[word] = vector
    return embeddings


# average all word vectors for a given instance
def instance_vector(current_instance, embeddings):
    # get each word in the instance context
    words = current_instance.split()
    # get each words vector
    vectors = [embeddings[word] for word in words if word in embeddings]
    # take the mean of all vectors
    return np.mean(vectors, axis=0)


def extract_features(instances, senses, glove_embeddings):
    # get mean vector for words in each instance
    X_train = [instance_vector(s, glove_embeddings) for s in instances]
    # associated instance sense
    y_train = [0 if s == 'phone' else 1 for s in senses]
    return np.array(X_train), np.array(y_train)


class NN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, int(input_dim/2)),
            nn.ReLU(),
            nn.Linear(int(input_dim / 2), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def train_model(model, X_train, y_train):
    # Support Vector Machine
    if model == 'SVM':
        sv = svm.SVC()
        model = sv.fit(X_train, y_train)

    # Neural Network
    elif model == 'NN':
        # convert X_train & y_train to PyTorch tensors
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

        model = NN(input_dim=X_train.shape[1])
        # using Binary Cross Entropy loss function
        loss_fn = nn.BCELoss()
        # using Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # train the model for 100 epochs
        for epoch in range(100):
            model.train()
            y_pred = model(X_tensor)
            loss = loss_fn(y_pred, y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model = model.eval()

    # default to Logistic Regression
    else:
        logreg = LogisticRegression()
        model = logreg.fit(X_train, y_train)

    return model


# use trained model to classify a given word
def classify(model, current_instance, glove_embeddings):
    vector_instance = instance_vector(current_instance, glove_embeddings).reshape(1, -1)

    # NN
    if isinstance(model, nn.Module):
        with torch.no_grad():
            tensor_input = torch.tensor(vector_instance, dtype=torch.float32)
            output = model(tensor_input)
            # round down to 0 or up to 1
            predicted_num = int(output.round().item())
    # SVM/ LR
    else:
        predicted_num = model.predict(vector_instance)[0]

    return 'phone' if predicted_num == 0 else 'product'


if __name__ == "__main__":
    training_data = argv[1]
    testing_data = argv[2]
    embedding_file = argv[3]
    given_model = None
    if len(argv) >= 5:
        given_model = argv[4]

    # retrieve GloVe embeddings
    glove = process_vectors(embedding_file)

    # training
    train_instances, _, train_senses = parse_instances(training_data)
    X_train, y_train = extract_features(train_instances, train_senses, glove)
    trained_model = train_model(given_model, X_train, y_train)

    # testing
    test_instances, instance_id, _ = parse_instances(testing_data)

    for i, instance in enumerate(test_instances):
        predicted_sense = classify(trained_model, instance, glove)
        print(f'<answer instance="{instance_id[i]}" senseid="{predicted_sense}"/>')