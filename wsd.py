import re
import numpy as np
from sys import argv
from collections import defaultdict, Counter

# Claire Gillaspie
# 3/18/25
# CMSC 437
# Word Sense Disambiguation Project

### DESCRIPTION & HOW TO RUN ###
# This program determines the sense of ambiguous words
# Using the decision list algorithm and bag of words feature representation
# Run the program on the command line: python3 wsd.py line-train.txt line-test.txt my-model.txt > my-line-answers.txt
# line-train.txt - contains labelled training data
# line-test.txt - contains ambiguous words to be identified by the model
# my-model.txt - contains the learned model
    # shows each feature, associated log-likelihood, predicted sense, and feature count
# my-line-answers.txt - contains answer tags for each sentence

### ALGORITHM ###
# Cleans input from line-train.txt, finds a window of context surrounding the target word, and stores the given sense
# Then, counts the number of times each word occurs for each sense and the total occurrences of each sense
# Finally, the log-likelihood score for each feature/sense pair is calculated to determine the strength of the relationship
# This log-likelihood score is used to rank the relevance of each feature, which determines the order of tests
# For tests with equivalent log-likelihood scores, feature count is used for additional sorting
# Zero-probabilities are smoothed to 0.0000000001 to avoid zero-division errors
# Once trained, the model reads in line-test.txt and parses for the context of each instance
# And then classifies the sense for each instance based on the learned tests

### SAMPLE INPUT & OUTPUT ###
# Input: python3 wsd.py line-data/line-train.txt line-data/line-test.txt line-data/my-model.txt > line-data/my-line-answers.txt
#
# First 10 lines of output (my-line-answers.txt):
# <answer instance="line-n.w8_059:8174:" senseid="phone"/>
# <answer instance="line-n.w7_098:12684:" senseid="phone"/>
# <answer instance="line-n.w8_106:13309:" senseid="phone"/>
# <answer instance="line-n.w9_40:10187:" senseid="phone"/>
# <answer instance="line-n.w9_16:217:" senseid="phone"/>
# <answer instance="line-n.w8_119:16927:" senseid="product"/>
# <answer instance="line-n.w8_008:13756:" senseid="phone"/>
# <answer instance="line-n.w8_041:15186:" senseid="phone"/>
# <answer instance="line-n.art7} aphb 05601797:" senseid="phone"/>
# <answer instance="line-n.w8_119:2964:" senseid="product"/>
#
# First 10 lines of output (my-model.txt):
# Total instances: 374
# Phone sense instances: 178
# Product sense instances: 196
#
# Feature		Log-Likelihood		Predicted Sense		Feature Count
# telephone		23.0259				phone				52
# access		23.0259				phone				16
# phone		23.0259				phone				12
# car		23.0259				product				12
# dead		23.0259				phone				12

### ACCURACY & CONFUSION MATRIX ###
# Accuracy: 0.9127
#
# Confusion Matrix:
# 	phone	product
# phone	68	4
# product	7	47
#
# Most Frequent Sense: product
# MFS Baseline Accuracy: 196/374 = 0.5241


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

            instances.append((current_instance_id, windowed_context, assigned_sense))

        return instances


# bag of words feature extraction from a window of context
def extract_features(instances):
    # store counts of features for both senses
    f_counts = {
        "phone": defaultdict(int),
        "product": defaultdict(int)
    }

    # count instances of each sense
    s_counts = Counter([s for _, _, s in instances if s])

    for _, c, s in instances:
        # parse cleaned context
        words = re.findall(r'\b[a-z]+\b', c.lower())

        # add count for sense+feature pair
        for feat in words:
            f_counts[s][feat] += 1

    return f_counts, s_counts


# calculate log likelihood scores & rank tests accordingly
def log_likelihood(fc):
    # store sorted tests
    tests = []

    all_features = set()
    # combine phone & product features to get complete list
    for curr_sense in fc:
        all_features.update(fc[curr_sense].keys())

    # perform log-likelihood calculation per feature
    for curr_feat in all_features:
        # count feature occurrence for each sense
        phone_count = fc["phone"][curr_feat]
        product_count = fc["product"][curr_feat]

        # calculate sense probabilities
        p_phone = phone_count / (phone_count+product_count)
        p_product = product_count / (phone_count+product_count)

        # smoothing 0 probabilities
            # smoothing number adjusted by trial and error based on accuracy score
        if p_product == 0:
            p_product = 0.0000000001
        if p_phone == 0:
            p_phone = 0.0000000001

        # calculate log-likelihood, higher score = stronger test
        log_score = abs(np.log(p_phone / p_product))

        # choose prediction
        if p_phone > p_product:
            prediction = "phone"
            feature_count = phone_count
        else:
            prediction = "product"
            feature_count = product_count

        tests.append((curr_feat, log_score, prediction, feature_count))

    # sort by highest log-likelihood score
    # sort same log-likelihood scores by highest sense count
    tests.sort(key=lambda x: (x[1], x[3]), reverse=True)

    return tests


# use sorted test list & features from context to classify a given word
def classify(current_instance, tests):
    # parse context
    _, context, _ = current_instance
    words = set(re.findall(r'\b[a-z]+\b', context.lower()))

    # decision list: find sense based on sorted feature tests
    for current_feature, _, current_sense, _ in tests:
        if current_feature in words:
            return current_sense

    # default sense based on training data MFS
    return "product"


if __name__ == "__main__":
    training_data = argv[1]
    testing_data = argv[2]
    output_model = argv[3]

    # training
    train_instances = parse_instances(training_data)
    feature_counts, sense_counts = extract_features(train_instances)
    sorted_tests = log_likelihood(feature_counts)

    # write model to output file
    with open(output_model, 'w') as f:
        f.write(f"Total instances: {sum(sense_counts.values())}\n")
        f.write(f"Phone sense instances: {sense_counts['phone']}\n")
        f.write(f"Product sense instances: {sense_counts['product']}\n\n")
        f.write("Feature\t\tLog-Likelihood\t\tPredicted Sense\t\tFeature Count\n")

        for feature, score, sense, count in sorted_tests:
            f.write(f"{feature}\t\t{score:.4f}\t\t\t\t{sense}\t\t\t\t{count}\n")

    # testing
    test_instances = parse_instances(testing_data)
    for instance in test_instances:
        instance_id, _, _ = instance
        predicted_sense = classify(instance, sorted_tests)
        print(f'<answer instance="{instance_id}" senseid="{predicted_sense}"/>')

