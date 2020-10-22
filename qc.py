# awk '{print $1}' DEV.txt > DEV_labels.txt
# awk {'$1=""; print $0;}' DEV.txt > DEV_questions.txt

# python qc.py -coarse TRAIN.txt DEV-questions.txt > predicted-labels.txt # runs the coarse model
# python qc.py -fine TRAIN.txt DEV-questions.txt > predicted-labels.txt # runs the fine model
# python ./evaluate.py DEV-labels.txt predicted-labels.txt

import sys
import nltk
import random
import pickle
from string import punctuation
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from nltk.classify.scikitlearn import SklearnClassifier


stop_words = set(["´´", "``", "what", "the", "in", "of", "is", "a", "s", "'s", "d", "ll", "m", "o", "re", "ve", "y", "n't"] + list(punctuation))


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN


def process_words(filename):
    with open(filename, "r") as r:
        all_words = []
        for line in r.readlines():
            words = nltk.word_tokenize(line)
            classif = words[0] + '_' + words[2]
            words = words[3:]
            pos = nltk.pos_tag(words)
            for word in pos:
                if word[0].lower() not in stop_words:
                    w = WordNetLemmatizer().lemmatize(word[0], get_wordnet_pos(word[1]))
                    all_words.append((w.lower(), classif))
    return all_words


args = sys.argv
flag = args[1]
filename = args[2]
#testfile = args[3]

all_words = process_words(filename)

train_set, test_set = train_test_split(all_words, random_state=42, test_size=0.2, shuffle=True)