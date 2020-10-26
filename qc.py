# awk '{print $1}' DEV.txt > DEV_labels.txt
# awk {'$1=""; print $0;}' DEV.txt > DEV_questions.txt

# python3 qc.py -coarse TRAIN.txt DEV-questions.txt > predicted-labels.txt # runs the coarse model
# python3 qc.py -fine TRAIN.txt DEV-questions.txt > predicted-labels.txt # runs the fine model
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
    testfile = args[3]

    stop_words = set(["´´", "``", "the", "in", "of", "is", "a", "s", "'s", "d", "ll", "m", "o", "re", "ve", "y", "n't"] + list(punctuation))

    question_categories = set(["how", "who", "when", "where", "what", "which"])

    words_df, vocabulary_df, labels = trainset_process(filename, flag, stop_words)

    features = vocabulary_df.columns

    # vocabularies = [words_df, vocabulary_df]
    # classifiers = {'LogisticRegression':linear_model.LogisticRegression(), 'Multinomial':MultinomialNB(), 'SVM':svm.SVC()}

    # accuracies={'LogisticRegression':[], 'Multinomial':[], 'SVM':[]}
    # for classifier in classifiers.keys():
    #     for vocabulary in vocabularies:
    #         accuracies[classifier].append(classification_results(classifiers[classifier], vocabulary, labels))
    
    # print('Accuracies', accuracies)
    # plot_classification_results(accuracies.keys(), accuracies)

    ### Test Pre-Processing
    test_vocabulary_df = testset_process(testfile, features, flag, stop_words)

    predictions = test_model(linear_model.LogisticRegression(), vocabulary_df, labels, test_vocabulary_df)
     
    print(predictions)
    # print(len(predictions))

    ### TO DO
        # - Add headwords to features
        # - Add question category to features


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
            

def pre_processing(filename, stop_words):
    with open('data/{}'.format(filename), "r") as r:
        corpus = r.readlines()
        vocabulary = []
        coarse_labels = []
        fine_labels = []
        for i, line in enumerate(corpus):
            words = nltk.word_tokenize(line)
            coarse_label = words[0]
            fine_label = words[0] + '_' + words[2]
            words = words[3:]
            pos = nltk.pos_tag(words)
            processed_line = []
            for word in pos:
                if word[0].lower() not in stop_words:
                    w = WordNetLemmatizer().lemmatize(word[0], get_wordnet_pos(word[1]))
                    vocabulary.append(w.lower())
                    processed_line.append(w.lower())
            corpus[i] = ' '.join(processed_line)
            coarse_labels.append(coarse_label)
            fine_labels.append(fine_label)

    return set(vocabulary), corpus, coarse_labels, fine_labels

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

    pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),
                 ('tfid', TfidfTransformer())]).fit(corpus)
    return pipe.transform(corpus).toarray() 

def tf_idf_ngrams(corpus):

    pipe = Pipeline([('count', CountVectorizer(ngram_range=(2,2))),
                 ('tfid', TfidfTransformer())]).fit(corpus)
    return pipe.transform(corpus).toarray(), pipe['count'].get_feature_names()

def top_rank_tfidf_features(matrix, features):
    # Getting top ranking features 
    sums = matrix.sum(axis = 0)
    print(len(features)) 
    print(np.unique(features))
    data1 = [] 
    for col, term in enumerate(features): 
        data1.append( (term, sums[col] )) 
    ranking = pd.DataFrame(data1, columns = ['term', 'rank']) 
    words = (ranking.sort_values('rank', ascending = False)) 
    print ("\n\nWords : \n", words.head(10)) 

def create_dataframe(vocabulary, tf_idf_vocabulary):

    df = pd.DataFrame(tf_idf_vocabulary)
    df.columns = vocabulary
    return df

def create_dataframe2(vocabulary, tf_idf_vocabulary, n_grams, tf_idf_ngrams):

    df1 = pd.DataFrame(tf_idf_vocabulary)
    df1.columns = vocabulary

    df2 = pd.DataFrame(tf_idf_ngrams)
    df2.columns = n_grams

    df = pd.concat([df1, df2], axis=1)
    return df


def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    return metrics.accuracy_score(predictions, valid_y)

def test_model(classifier, train_set, train_labels, test_set):
    classifier.fit(train_set, train_labels)
    return classifier.predict(test_set)



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
        colors = {'LogisticRegression':'blue', 'Multinomial':'orange', 'SVM':'green'}
        for classifier in classifiers:
            axs[i].bar(x, accuracies[classifier][i], width=0.4, color=colors[classifier],  align='center', label=classifier)
            x += 1
        axs[i].set_xticks(np.arange(len(classifiers))+1)
        axs[i].legend(fontsize='x-small', title_fontsize='small')
    
    plt.show()

def trainset_process(filename, flag, stop_words):

    questions = extract_questions(filename)
    # print("Number of Questions",np.shape(questions))
    # print([question for question in questions if 'California' in question])

    words = extract_allwords(filename)

    ### Maybe because of set() the order of the vocabulary is always reordered when running the code
    vocabulary, corpus, coarse_labels, fine_labels = pre_processing(filename, stop_words)
    # print(corpus)

    labels = coarse_labels
    if flag == '-fine':
        labels = fine_labels
    # print(flag)
    # print(labels)
        
    # grammatical_pattern_features(questions_processed, question_categories)

    tf_idf_matrix = tf_idf(questions, words)
    tf_idf_matrix_processed = tf_idf(corpus, vocabulary)

    tf_idf_ngrams_matrix, feature_names = tf_idf_ngrams(corpus)
    # top_rank_tfidf_features(tf_idf_ngrams_matrix, feature_names)
    
    words_df = create_dataframe2(words, tf_idf_matrix, feature_names, tf_idf_ngrams_matrix)
    vocabulary_df = create_dataframe2(vocabulary, tf_idf_matrix_processed, feature_names, tf_idf_ngrams_matrix)
    # words_df = create_dataframe(feature_names, tf_idf_ngrams_matrix)
    # vocabulary_df = create_dataframe(feature_names, tf_idf_ngrams_matrix)
    # print("Words Shape",len(words))
    # print("Pre Processing Vocabulary Shape",vocabulary_df.shape)

    # print("Questions without information", vocabulary_df.loc[vocabulary_df.sum(axis=1) != 0].shape)

    # Remove less common words that doesn't give too much information
    # relevant_words = [column for column in vocabulary_df.columns if vocabulary_df[column].sum() > 0.2]
    # print("Not Common Words", [column for column in vocabulary_df.columns if vocabulary_df[column].sum() < 0.2])
    # vocabulary_df = vocabulary_df[relevant_words]
    # print("Pre Processing Vocabulary with relevant words only Shape",vocabulary_df.shape)

    return words_df, vocabulary_df, labels

def testset_process(testfile, test_vocabulary, flag, stop_words):

    data = open('data/{}'.format(testfile)).read()
    
    questions = []
    for line in data.split('\n'):
        question = line
        if question != '':
            questions.append(question)

    tf_idf_matrix_processed = tf_idf(questions, test_vocabulary)
    
    vocabulary_df = create_dataframe(test_vocabulary, tf_idf_matrix_processed)

    return vocabulary_df

if __name__ == '__main__':

    main()