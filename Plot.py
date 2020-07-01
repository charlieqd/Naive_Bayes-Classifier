import matplotlib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
from pylab import figure, axes, pie, title, show
import BasicClassifier


def plot(e_type, range_list, classifier, k, info, p=0, recall_r=None, precision_r=None, alpha=1):
    recall_list, precision_list, threshold_list, recall_c, precision_c = [], [], [], -1, -1
    for threshold in range_list:
        if e_type == "tf":
            pred = classifier.predict_with_threshold(classifier.test_data, threshold, "tf", alpha)
            recall_c, precision_c, c, d = classifier.estimation(pred)
        elif e_type == "tfidf":
            pred = classifier.predict_with_threshold(classifier.test_data, threshold, "tfidf",  alpha)
            recall_c, precision_c, c, d = classifier.estimation(pred)
        elif e_type == "0-1":
            pred = classifier.predict_with_threshold(classifier.test_data, threshold, "0-1",  alpha)
            recall_c, precision_c, c, d = classifier.estimation(pred)
        elif e_type == "bernoulli":
            pred = classifier.predict_with_threshold(classifier.test_data, threshold, "bernoulli",  alpha)
            recall_c, precision_c, c, d = classifier.estimation(pred)
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
        plt.plot(recall_list, precision_list, label=k)
        # plt.plot(recall_r, precision_r, label="Base Model", linestyle='dashed')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve' + info)
        plt.grid()
        # plt.legend(loc="right")
        plt.legend(bbox_to_anchor=(1.5, 1))
        show()

        plot_name = classifier.name + ".png"
        # plt.savefig(plot_name)
    return recall_list, precision_list


def plot_with_diff_alpha(data, range_list, e_type, feature_num, alpha_list, class_c):
    vocab_list_n = data.vocab_feature[0:feature_num]
    recall_matrix, precision_matrix = [], []
    classifier = BasicClassifier.BasicClassifier(data.train_data.data, data.train_data.target,
                                                 vocab_list_n, class_c, data.test_data.data,
                                                 data.test_data.target, "basic")
    classifier.fit()
    if e_type == "bernoulli":
        classifier.bernoulli_fit()

    for alpha in alpha_list:
        recall_list_n, precision_list_n = plot(e_type, range_list, classifier, 0, "info", 0, None, None, alpha)
        recall_matrix.append(recall_list_n)
        precision_matrix.append(precision_list_n)

    for i in range(len(alpha_list)):
        plt.plot(recall_matrix[i], precision_matrix[i], label=alpha_list[i])

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(e_type + ' Precision-Recall curve with different Alpha')
    plt.legend(bbox_to_anchor=(1.5, 1))
    plt.grid()
    show()


def plot_with_diff_features(data, range_list, e_type, feature_number_list, class_c):
    recall_matrix, precision_matrix = [], []
    for n in feature_number_list:
        vocab_list_n = data.vocab_feature[0:n]
        print(vocab_list_n)
        classifier = BasicClassifier.BasicClassifier(data.train_data.data, data.train_data.target,
                                                     vocab_list_n, class_c, data.test_data.data,
                                                     data.test_data.target, "basic")
        classifier.fit()
        if e_type == "bernoulli":
            classifier.bernoulli_fit()

        predict_result_n = classifier.predict_with_threshold(classifier.test_data, 0.5, e_type)
        print("Acc of feature ", n, " is ", classifier.accuracy())
        print(predict_result_n)

        recall_list_n, precision_list_n = plot(e_type, range_list, classifier, 0, "info", 0)
        recall_matrix.append(recall_list_n)
        precision_matrix.append(precision_list_n)

    for i in range(len(feature_number_list)):
        plt.plot(recall_matrix[i], precision_matrix[i], label=feature_number_list[i])

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(e_type + ' Precision-Recall curve with different number of features')
    plt.legend(bbox_to_anchor=(1.5, 1))
    plt.grid()
    show()
