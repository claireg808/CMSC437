from sys import argv
from collections import defaultdict
import random
import re


# Claire Gillaspie
# 2/24/25
# CMSC 437
# N-gram Project

### DESCRIPTION & HOW TO RUN ###
# This program uses an n-gram model to generate original sentences based on a given corpus
# Run the program on the command line: python ngram.py n m file(s)
# n - size of the n-grams used
# m - number of desired output sentences
# file(s) - one or more files to be used as training material

### ALGORITHM ###
# The program parses each input file to find all n-grams & count their frequency
# Then, it finds the probability for every word that follows each n-1-gram
# For example, if we see the bigram 'her name' twice: 'her name is' and 'her name was'
# The probability for choosing a word to build a new tri-gram
# Would be 50% 'is' and 50% 'was'
# Using these calculated probabilities, we can generate new, logical sentences


### SAMPLE INPUT & OUTPUT ###
# Input: python ngram.py 4 5 Poe1.txt Poe2.txt
# Output:
# This program generates random sentences based on an N-gram model.
# Created by Claire Gillaspie
# Command line settings: ngram.py 4 5
#
# I looked upon the bed.
#
# Israfel in heaven a spirit doth dwell whose heart-strings are a lut, and who has the sweetest voice of all god’s creatures could hope to utter.
#
# It is evident that we were the only ones who made a regular business of going out to the islands, as I tell you is truth I began to murmur, to hesitate, to resist.
#
# The depth lies in the creation of novel forms of beauty.
#
# Quit the bust above my door!


# find all ngrams & count the frequency
def find_ngrams(n):
    ng_freq = defaultdict(int)
    # find all ngrams for each file
    for i in range(3, len(argv)):
        with open(argv[i]) as current_file:

            # text modifications
            text = current_file.read().lower()  # lowercase
            text = re.sub(r"[\n—\s]+", " ", text)  # replace newlines, dashes, and extra spaces
            text = re.sub(r"[*)_(:;™\"“”]", "", text)  # remove extra punctuation/symbols
            text = re.sub(r"([.?!])", r" \1", text)  # add space before punctuation
            text = re.sub(r"([.?!)]) ", r"\1", text)  # remove space after punctuation
            parsed_sentences = re.findall(r"[^.?!]*[.?!]", text)  # split into sentences

            # generate  ngrams & update frequency counts
            for s in parsed_sentences:
                if len(s) >= n:
                    # add n-1 start tags
                    for k in range(0, n - 1):
                        s = '<s> ' + s
                    sentence = s.split(" ")
                    # find and count ngram occurrences
                    for start in range(len(sentence) - n + 1):
                        current_ng = tuple(sentence[start:start + n])
                        ng_freq[current_ng] += 1

    return ng_freq


# calculate relative frequency
def rel_freq(ngram_list):
    # initialize nested dict of relative frequencies
    relative_frequencies = defaultdict(lambda: defaultdict(float))

    # calculate relative frequency of nth word for each n-1-gram
    for ngram, c in ngram_list.items():
        start = ngram[:-1]  # everything but last word
        end = ngram[-1]  # last word
        relative_frequencies[start][end] += c

    # totals for all possible words following each n-1-gram will sum to 1
    for start, end in relative_frequencies.items():
        total = sum(end.values())
        for word in end:
            # divide by the total to normalize
            relative_frequencies[start][word] /= total

    return relative_frequencies


# generate new sentences
def ngram_model(rf, n, m):
    # loop for each sentence
    for i in range(0, m):
        sentence = []
        # add n-1 start tags
        for k in range(0, n - 1):
            sentence.append('<s>')
        # build sentence
        while True:
            # get most recent n-1 words
            current_history = tuple(sentence[-(n-1):])

            # stop sentence generation if there is no next word
            if current_history not in rf:
                final_sentence = ' '.join(sentence[n-1:]) + '.\n'
                final_sentence = re.sub(r"\si[\s.!?]", " I ", final_sentence.capitalize())
                print(final_sentence)
                break

            # pick the next word, factoring in probability
            next_word = random.choices(list(rf[current_history].keys()),
                                       list(rf[current_history].values()))[0]

            # stop sentence generation if next word is punctuation
            if next_word == '.' or next_word == '?' or next_word == '!':
                final_sentence = ' '.join(sentence[n-1:]) + next_word + '\n'
                final_sentence = re.sub(r"\si[\s.!?]", " I ", final_sentence.capitalize())
                print(final_sentence)
                break

            # update sentence
            sentence.append(next_word)


if __name__ == "__main__":
    ngram_freq = defaultdict(int)
    ngram_length = argv[1]
    num_sentences = argv[2]

    print('This program generates random sentences based on an Ngram model.\n'
          'Created by Claire Gillaspie\n'
          'Command line settings: ' + argv[0] + ' ' + ngram_length + ' ' + num_sentences + '\n')

    # find and count all ngrams
    ngrams = find_ngrams(int(ngram_length))

    # for each n-1-gram, calculate relative frequency of nth word
    relative_frequency = rel_freq(ngrams)

    # generate new sentences
    ngram_model(relative_frequency, int(ngram_length), int(num_sentences))
