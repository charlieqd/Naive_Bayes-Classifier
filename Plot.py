import matplotlib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
from pylab import figure, axes, pie, title, show
import BasicClassifier


def plot(e_type, range_list, classifier, k, info, p=1, recall_r=None, precision_r=None, alpha=1):
    recall_list, precision_list, threshold_list, recall_c, precision_c = [], [], [], -1, -1
    for threshold in range_list:
        if e_type == "tf":
            pred = classifier.predict_with_threshold(classifier.test_data, threshold, "tf", alpha)
            recall_c, precision_c, c, d = classifier.estimation(pred)
        elif e_type == "tfidf":
            pred = classifier.predict_with_threshold(classifier.test_data, threshold, "tfidf",  alpha)
            recall_c, precision_c = classifier.estimation(pred)
        elif e_type == "0-1":
            pred = classifier.predict_with_threshold(classifier.test_data, threshold, "0-1",  alpha)
            recall_c, precision_c = classifier.estimation(pred)
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
        plt.plot(recall_list, precision_list, label=k + 1)
        plt.plot(recall_r, precision_r, label="Basic Bernoulli Model", linestyle='dashed')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve' + info)
        plt.grid()
        plt.legend(loc="right")
        show()

        plot_name = classifier.name + ".png"
        # plt.savefig(plot_name)
    return recall_list, precision_list


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


def plot_with_diff_features(data, range_list, e_type, feature_number_list):
    class_c = 1
    recall_matrix, precision_matrix = [], []
    for n in feature_number_list:
        vocab_list_n = data.vocab_feature[0:n]
        print(vocab_list_n)
        classifier = BasicClassifier.BasicClassifier(data.train_data.data, data.train_data.target,
                                                     vocab_list_n, class_c, data.test_data.data,
                                                     data.test_data.target, "basic")
        classifier.fit()
        if e_type == "bernoulli":
            classifier.build_df_dict_train()

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
    plt.title(e_type + 'Precision-Recall curve with different number of features')
    plt.legend(loc="right")
    plt.grid()
    show()
