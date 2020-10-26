import sys
import re

from sklearn import metrics

def main():

    args = sys.argv
    flag = args[1]
    true_labels_file = args[2]
    predicted_labels_file = args[3]

    true_labels = []
    true_fine_labels = []
    predictions = []
    with open('data/{}'.format(true_labels_file), "r") as r:
        data = r.readlines()
        for i, line in enumerate(data):
            label = line.split("\n")
            true_fine_labels.append(label[0])

    coarse_labels = []
    for label in true_fine_labels:
        types = label.split(":")
        coarse_labels.append(types[0])

    for i in range (len(true_fine_labels)):
        true_fine_labels[i] = true_fine_labels[i].replace(":", "_")
    
    if flag == '-coarse':
        true_labels = coarse_labels
    else:
        true_labels = true_fine_labels
        

    with open('data/{}'.format(predicted_labels_file), "r") as r:
        data = r.readlines()
        for i, line in enumerate(data):
            ascii_data = re.sub("[^a-z\s_]", "", line, 0, re.IGNORECASE | re.MULTILINE)
            # print(i,ascii_data)
            labels_line = ascii_data.split("\n")
            print(i,labels_line)
            labels = labels_line[0].split(" ")
            # print(i, labels)
            for label in labels:
                if label != "":
                    predictions.append(label)

    # print(true_labels)
    print(predictions)
    print(metrics.accuracy_score(predictions, true_labels))

if __name__ == '__main__':

    main()