import re
import numpy as np
from sys import argv
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# Claire Gillaspie
# 3/31/25
# CMSC 437
# Word Sense Disambiguation Machine Learning Project

### DESCRIPTION & HOW TO RUN ###
# This program determines the sense of ambiguous words
# Uses the bag of words feature representation
# Uses a specified model (Support Vector Machine, Stochastic Gradient Descent, Naive Bayes)
# Or defaults to Logistic Regression
# Run the program on the command line: python3 wsd-ml.py line-train.txt line-test.txt [OPTIONAL MODEL] > my-line-answers.txt
# line-train.txt - contains labelled training data
# line-test.txt - contains ambiguous words to be identified by the model
# my-line-answers.txt - contains answer tags for each sentence

### ALGORITHM ###
# Cleans input from line-train.txt, finds a window of context surrounding the target word, and stores the given sense
# Input contexts are vectorized and used as input to train the chosen model
# Once trained, the model reads in line-test.txt and parses for the context of each instance
# And then classifies the sense for each instance

### SAMPLE INPUT & OUTPUT ###
# input:
# python wsd-ml.py line-data/line-train.txt line-data/line-test.txt SVM > line-data/my-line-answers.txt
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
# Stochastic Gradient Descent (SGD):
    # A linear model that updates model parameters iteratively
    # Rather than using the full dataset, SGD uses one random data point
    # to compute the gradient at each iteration
# Naive Bayes (NB):
    # Applies Bayes' theorem with the "naive" assumption that the
    # effect of a particular feature is independent of other features
# Logistic Regression:
    # Used for predicting binary classes
    # Type of linear regression for categorical target variables
    # Using log of odds as the dependent variable

### ACCURACY & CONFUSION MATRIX ###
## SVM:
# Accuracy: 0.8968
#
# Confusion Matrix:
# 	phone	product
# phone	63	9
# product	4	50
#
## Decision List:
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
        text = re.sub(r'\b(the|an?|and|of|is|that|to|in|with|no|didnt|but|have|as|its?|was|by|at|were|only|been|any|are)\b', '', text)
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
            current_context = re.sub(r'["-.]', '', current_context)

            find_target = re.search(r'(<head>.*?</head>)', current_context)
            target = find_target.group(1)

            words = current_context.split()
            target_position = -1
            # number of words of context (features) to consider before & after target
            # determined using trial & error based on accuracy score
            window_size = 8

            # find ambiguous word index
            for i, word in enumerate(words):
                if word == target:
                    target_position = i
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


# vectorize instances & features
def extract_features(instances, senses):
    # use CountVectorizer to build instance vector
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(instances)
    num_senses = len(senses)

    # vectorize senses, X_train[i] -> y_train[i]
    # default to 0 to represent phone
    y_train = np.zeros(num_senses)
    for i, sense in enumerate(senses):
        if sense == 'product':
            y_train[i] = 1

    return X_train, y_train, vectorizer


def train_model(model, X_train, y_train):
    # Support Vector Machine
    if model == 'SVM':
        sv = svm.SVC()
        model = sv.fit(X_train, y_train)

    # Stochastic Gradient Descent
    elif model == 'SGD':
        sgd = SGDClassifier()
        model = sgd.fit(X_train, y_train)

    # Naive Bayes
    elif model == 'NB':
        gnb = GaussianNB()
        model = gnb.fit(X_train, y_train)

    # default to Logistic Regression
    else:
        logreg = LogisticRegression()
        model = logreg.fit(X_train, y_train)

    return model


# use trained model to classify a given word
def classify(model, v, current_instance):
    vector_instance = v.transform([current_instance])
    predicted_num = model.predict(vector_instance)
    if predicted_num[0] == 0:
        return 'phone'
    else:
        return 'product'


if __name__ == "__main__":
    training_data = argv[1]
    testing_data = argv[2]
    given_model = None
    if len(argv) == 6:
        given_model = argv[3]

    # training
    train_instances, _, train_senses = parse_instances(training_data)
    X, y, vect = extract_features(train_instances, train_senses)
    trained_model = train_model(given_model, X, y)

    # testing
    test_instances, test_instance_ids, _ = parse_instances(testing_data)

    for i, instance in enumerate(test_instances):
        instance_id = test_instance_ids[i]
        predicted_sense = classify(trained_model, vect, instance)
        print(f'<answer instance="{instance_id}" senseid="{predicted_sense}"/>')
