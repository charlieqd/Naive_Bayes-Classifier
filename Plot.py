import inline as inline
import matplotlib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
from pylab import figure, axes, pie, title, show


def plot(e_type, range_list, classifier, k, info, p=1):
    recall_list, precision_list, threshold_list, recall_c, precision_c = [], [], [], -1, -1
    for threshold in range_list:
        if e_type == "tf":
            pred = classifier.predict_with_threshold(classifier.test_data, threshold, "tf")
            recall_c, precision_c ,c,d= classifier.estimation(pred)
        elif e_type == "tfidf":
            pred = classifier.predict_with_threshold(classifier.test_data, threshold, "tfidf")
            recall_c, precision_c = classifier.estimation(pred)
        elif e_type == "0-1":
            pred = classifier.predict_with_threshold(classifier.test_data, threshold, "0-1")
            recall_c, precision_c = classifier.estimation(pred)
        elif e_type == "bernoulli":
            pred = classifier.predict_with_threshold(classifier.test_data, threshold, "bernoulli")
            recall_c, precision_c,c,d = classifier.estimation(pred)
        elif e_type == "kn":
            pred = classifier.predict_kn(classifier.test_data, threshold, k)
            recall_c, precision_c, c, d = classifier.estimation(pred)

        recall_list.append(recall_c)
        precision_list.append(precision_c)
        threshold_list.append(threshold)

    print("recall list is ", recall_list)
    print("precision list is ", precision_list)
    print(threshold_list)
    if p == 1:
        plt.plot(recall_list, precision_list)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve' + info)
        plt.grid()
        show()

        plot_name = classifier.name + ".png"
        # plt.savefig(plot_name)


def plot_compare(range_list, classifier, classifier_2, info):
    recall_list, precision_list, threshold_list, recall_c, precision_c = [], [], [], -1, -1
    recall_list_2, precision_list_2, recall_c_2, precision_c_2 = [], [], -1, -1

    for threshold in range_list:

        pred = classifier.predict_with_threshold(classifier.test_data, threshold)
        recall_c, precision_c = classifier.estimation(pred)

        pred_2 = classifier.predict_with_threshold_tfidf(classifier_2.test_data, threshold)
        recall_c_2, precision_c_2 = classifier_2.estimation(pred_2)

        recall_list.append(recall_c)
        precision_list.append(precision_c)
        recall_list_2.append(recall_c_2)
        precision_list_2.append(precision_c_2)

    plt.plot(recall_list, precision_list, label="TF")
    plt.plot(recall_list_2, precision_list_2, label="TFIDF")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve' + info)
    plt.legend(loc="right")
    plt.grid()
    show()
