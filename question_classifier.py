# awk '{print $1}' DEV.txt > DEV_labels.txt
# awk {'$1=""; print $0;}' DEV.txt > DEV_questions.txt

# python qc.py -coarse TRAIN.txt DEV-questions.txt > predicted-labels.txt # runs the coarse model
# python qc.py -fine TRAIN.txt DEV-questions.txt > predicted-labels.txt # runs the fine model
# python ./evaluate.py DEV-labels.txt predicted-labels.txt

import sys
import nltk
import random
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from string import punctuation
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn import model_selection, preprocessing, linear_model, metrics, svm
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB


def main():

    args = sys.argv
    flag = args[1]
    filename = args[2]
    #testfile = args[3]

    questions = extract_questions(filename)
    print("Number of Questions",np.shape(questions))
    # print([question for question in questions if 'funnel' in question])

    words = extract_allwords(filename)

    stop_words = set(["´´", "``", "the", "in", "of", "is", "a", "s", "'s", "d", "ll", "m", "o", "re", "ve", "y", "n't"] + list(punctuation))

    question_categories = set(["how", "who", "when", "where", "what", "which"])

    ### Maybe because of set() the order of the vocabulary is always reordered when running the code
    vocabulary, coarse_labels, fine_labels = pre_processing(filename, stop_words)

    labels = coarse_labels
    if flag == 'fine':
        labels = fine_labels
        
    # grammatical_pattern_features(questions_processed, question_categories)

    tf_idf_matrix = tf_idf(questions, words)
    tf_idf_matrix_processed = tf_idf(questions, vocabulary)
    
    words_df = create_dataframe(words, tf_idf_matrix)
    vocabulary_df = create_dataframe(vocabulary, tf_idf_matrix_processed)
    print(vocabulary_df.shape)

    # Remove less common words that doesn't give too much information
    common_words = [column for column in vocabulary_df.columns if vocabulary_df[column].sum() > 0.2]
    vocabulary_df = vocabulary_df[common_words]

    print(vocabulary_df.shape)

    vocabularies = [words_df, vocabulary_df]
    classifiers = {'Gaussian':GaussianNB(), 'Multinomial':MultinomialNB(), 'SVM':svm.SVC()}

    accuracies={'Gaussian':[], 'Multinomial':[], 'SVM':[]}
    for classifier in classifiers.keys():
        for vocabulary in vocabularies:
            accuracies[classifier].append(classification_results(classifiers[classifier], vocabulary, labels))
        
    plot_classification_results(accuracies.keys(), accuracies)

    ### TO DO
        # - Add n-grams to features
        # - Add headwords to features
        # - Add question category to features
        # - Compare results between classifiers


def extract_questions(filename):

    data = open('data/{}'.format(filename)).read()
    
    labels, questions = [], []
    for line in data.split('\n'):

        line_corpus = line.split()
        label = line_corpus[0:1]
        question = line[len(label[0]):]

        labels.append(label)
        questions.append(question)

    return questions

def extract_allwords(filename):

    with open('data/{}'.format(filename), "r") as r:
        vocabulary = []
        for line in r.readlines():
            words = nltk.word_tokenize(line)
            for word in words[3:]:
                vocabulary.append(word.lower())
    return set(vocabulary)


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN
            

    return set(vocabulary)
def pre_processing(filename, stop_words):
    with open('data/{}'.format(filename), "r") as r:
        vocabulary = []
        coarse_labels = []
        fine_labels = []
        for line in r.readlines():
            words = nltk.word_tokenize(line)
            coarse_label = words[0]
            fine_label = words[0] + '_' + words[2]
            words = words[3:]
            pos = nltk.pos_tag(words)
            for word in pos:
                if word[0].lower() not in stop_words:
                    w = WordNetLemmatizer().lemmatize(word[0], get_wordnet_pos(word[1]))
                    vocabulary.append(w.lower())
            coarse_labels.append(coarse_label)
            fine_labels.append(fine_label)

    return set(vocabulary), coarse_labels, fine_labels

def grammatical_pattern_features(questions, q_categories):
    poses = []
    for question in questions:
        pos = nltk.pos_tag(question)
        poses.append(pos)

    print(poses[0:2])

def label_encoding(labels):
    encoder = preprocessing.LabelEncoder()
    return encoder.fit_transform(labels)

def tf_idf(corpus, vocabulary):
    # # ngram level tf-idf 
    # tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    # tfidf_vect_ngram.fit(data)

    pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),
                 ('tfid', TfidfTransformer())]).fit(corpus)
    return pipe.transform(corpus).toarray() #tfidf_vect_ngram.transform(vocabulary)

def create_dataframe(vocabulary, tf_idf_vocabulary):

    df = pd.DataFrame(tf_idf_vocabulary)
    df.columns = vocabulary
    return df

def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    return metrics.accuracy_score(predictions, valid_y)

def classification_results(classifier, X, y):

    train_x, valid_x, train_y, valid_y = train_test_split(X, y, random_state=42, test_size=0.2, shuffle=True)
    # SVM on Ngram Level TF IDF Vectors
    accuracy = train_model(classifier, train_x, train_y, valid_x, valid_y)

    return accuracy

def set_axes(xvalues: list, ax: plt.Axes = None, title: str = '', xlabel: str = '', ylabel: str = '', xrotation: int = 0,percentage=False):
    if ax is None:
        ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(xvalues, rotation=xrotation, fontsize='small', ha='center')

    return ax

def plot_classification_results(classifiers, accuracies):

    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    titles = ['Classify Dev Questions (without pre-processing)', 'Classify Dev Questions (with pre-processing)']
    for i in range(2):
        axs[i] = set_axes(classifiers, ax=axs[i], title=titles[i], xlabel='Classifiers', ylabel='Accuracies', percentage=True)

        x = 1  # the label locations
        colors = {'Gaussian':'blue', 'Multinomial':'orange', 'SVM':'green'}
        for classifier in classifiers:
            axs[i].bar(x, accuracies[classifier][i], width=0.4, color=colors[classifier],  align='center', label=classifier)
            x += 1
        axs[i].set_xticks(np.arange(len(classifiers))+1)
        axs[i].legend(fontsize='x-small', title_fontsize='small')
    
    plt.show()


if __name__ == '__main__':

    main()