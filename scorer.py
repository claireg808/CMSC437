import sys
import re
from collections import defaultdict

# Claire Gillaspie
# 3/18/25
# CMSC 437
# Word Sense Disambiguation Project

### DESCRIPTION & HOW TO RUN ###
# This program analyzes the accuracy of a word sense disambiguation model
# Run the program on the command line: python3 scorer.py line-data/my-line-answers.txt line-data/line-key.txt
# my-line-answers.txt - model answers
# line-key.txt - gold standard answers

### ALGORITHM ###
# Parses both model and key answers to build dictionaries associating each instance with its given sense
# Confusion matrix is built by flagging the [model_sense][key_sense] index for each instance
# Accuracy is calculated by summing all instances where model and key sense are equal, divided by total instances
# Most Frequent Sense (MFS) baseline is calculated by summing sense counts for the most common sense, divided by total sense count

### SAMPLE INPUT & OUTPUT ###
# Input: python3 scorer.py line-data/my-line-answers.txt line-data/line-key.txt
# Output:
# Accuracy: 0.9127
#
# Confusion Matrix:
# 	phone	product
# phone	68	4
# product	7	47
#
# Most Frequent Sense: product
# MFS Baseline Accuracy: 0.5241


# read files into dictionaries indexed by instance_id
def load_answers(filename):
    answers = {}

    with open(filename) as file:
        text = file.read()
        instances = re.findall(r'<answer instance="(.*?)".*?senseid="(.*?)"', text, re.DOTALL)

        for instance_id, sense_id in instances:
            answers[instance_id] = sense_id

    return answers


# calculate accuracy and build confusion matrix
def accuracy_cm(model_answers, gold_answers):
    # unique instance_ids
    unique_ids = list(gold_answers.keys())
    cm = defaultdict(lambda: defaultdict(int))
    correct = 0

    # label cm for each instance_id
    for instance_id in unique_ids:
        # find predicted and actual senses
        system_sense = model_answers[instance_id]
        gold_sense = gold_answers[instance_id]

        # mark outcome in confusion matrix
        cm[gold_sense][system_sense] += 1

        # count correct answers for accuracy
        if system_sense == gold_sense:
            correct += 1

    # calculate accuracy: num correct / total possible
    accuracy_score = correct / len(unique_ids)

    return accuracy_score, cm


if __name__ == "__main__":
    model_file = sys.argv[1]
    gold_file = sys.argv[2]

    mine = load_answers(model_file)
    key = load_answers(gold_file)

    senses = ('phone', 'product')

    # get accuracy score and confusion matrix
    accuracy, confusion_matrix = accuracy_cm(mine, key)

    print(f"Accuracy: {accuracy:.4f}\n")
    print("Confusion Matrix:")
    print("\tphone\tproduct")

    # print confusion matrix
    for key_sense in senses:
        row = [key_sense]
        for model_sense in senses:
            row.append(str(confusion_matrix.get(key_sense).get(model_sense)))
        print("\t".join(row))