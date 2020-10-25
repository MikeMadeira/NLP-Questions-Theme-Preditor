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
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm


def main():

    args = sys.argv
    flag = args[1]
    filename = args[2]
    #testfile = args[3]

    questions = load_dataset(filename)
    # df = create_dataframe(labels, questions)
    # df.sort_values(by=['label'], inplace=True)

    # print(df[:20])

    stop_words = set(["´´", "``", "the", "in", "of", "is", "a", "s", "'s", "d", "ll", "m", "o", "re", "ve", "y", "n't"] + list(punctuation))

    question_categories = set(["how", "who", "when", "where", "what", "which"])

    vocabulary, coarse_labels, fine_labels = pre_processing(filename, stop_words)

    # grammatical_pattern_features(questions_processed, question_categories)

    tf_idf_matrix = tf_idf(questions, vocabulary)

    train_x, valid_x, train_y, valid_y = train_test_split(tf_idf_matrix, coarse_labels, random_state=42, test_size=0.2, shuffle=True)

    # SVM on Ngram Level TF IDF Vectors
    accuracy = train_model(svm.SVC(), train_x, train_y, valid_x, valid_y)
    print("SVM, N-Gram Vectors: ", accuracy)

    

    


def load_dataset(filename):

    data = open('data/{}'.format(filename)).read()
    
    labels, questions = [], []
    for line in data.split('\n'):

        line_corpus = line.split()
        label = line_corpus[0:1]
        question = line[len(label[0]):]

        labels.append(label)
        questions.append(question)

    return questions


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

def create_dataframe(questions, labels):

    df = pd.DataFrame()
    df['question'] = questions
    df['label'] = label_encoding(labels)
    return df

def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)


if __name__ == '__main__':

    main()