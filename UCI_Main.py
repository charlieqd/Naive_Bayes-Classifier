import bisect

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import UCI_BaseClassifier
import UCI_KnClassifier
import matplotlib.pyplot as plt
from pylab import figure, axes, pie, title, show
from random import seed
from random import randint


def run_classifier(classifier, x, y, p_base=None, r_base=None):
    kf = KFold(n_splits=5)
    kf.get_n_splits(x)
    y_real, y_prob, y_real_kn, y_prob_kn = [], [], [], {}
    flag, k_max = 0, 0
    for train_index, test_index in kf.split(x):
        x_train = x.sort_index().loc[train_index]
        x_test = x.sort_index().loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        x_train = x_train.reset_index(drop=True)
        x_test = x_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        if classifier == "multi":
            multi_classifier = UCI_BaseClassifier.UCIBaseClassifier(x_train, x_test, y_train, y_test, class_c)
            multi_classifier.fit()
            pred_result, pred_prob = multi_classifier.predict(0.5)
            precision, recall, accuracy = multi_classifier.estimation()
            y_real.append(multi_classifier.y_test)
            y_prob.append(multi_classifier.pred_prob_list)
        elif classifier == "kn":
            kn_classifier = UCI_KnClassifier.KnClassifier(x_train, x_test, y_train, y_test, class_c)
            kn_classifier.fit()
            if flag == 0:
                flag = 1
                k_max = kn_classifier.k_max
                for k in range(1, kn_classifier.k_max + 1):
                    y_prob_kn[k] = []
            pred_prob_matrix = kn_classifier.predict_prob_kn()
            y_real_kn.append(kn_classifier.y_test)
            for k in range(1, kn_classifier.k_max + 1):
                prediction, prob_list = kn_classifier.predict_kn(0.5, k)
                y_prob_kn[k].append(prob_list)

    if classifier == "multi":
        y_real = np.concatenate(y_real)
        y_prob = np.concatenate(y_prob)
        precision, recall, threshold = precision_recall_curve(y_real, y_prob)
        # eleven-point
        recall_list, precision_list = eleven_point_ave_precision_plot(precision, recall)
        pr_plot(recall_list, precision_list)
        # pr_plot(recall, precision)
        return precision_list, recall_list

    elif classifier == "kn":
        y_real_kn = np.concatenate(y_real_kn)
        for k in range(1, k_max + 1):
            y_prob_kn[k] = np.concatenate(y_prob_kn[k])
        for k in range(1, k_max + 1):
            precision, recall, threshold = precision_recall_curve(y_real_kn, y_prob_kn[k])
            recall, precision = eleven_point_ave_precision_plot(precision, recall)
            plt.plot(recall, precision, label=k)
        if r_base is not None:
            plt.plot(r_base, p_base, label="Base Model", linestyle='dashed')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.legend(loc="right", title="K Value")
        plt.grid()
        show()
        return


def pr_plot(recall, precision):
    plt.plot(recall, precision, label="Base Multinomial Model")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="right")
    plt.grid()
    show()
    return


def eleven_point_ave_precision_plot(precision, recall):
    recall_list = np.linspace(0, 1, 11)
    print(recall_list)
    precision_list = []
    print(len(precision))
    for r in recall_list:
        print(r)
        index = 0
        for i in range(len(recall)):
            if recall[i] > r:
                index = i
            else:
                break
        print(index)
        if index == 0:
            p_max = 0
        else:
            p_max = np.max(precision[:index])
        print(p_max)
        precision_list.append(p_max)
    return recall_list, precision_list


if __name__ == '__main__':
    # ------glass.data-----------
    df_data = pd.read_csv('uci_data/glass.data.csv', header=None)
    df_data = df_data.sample(frac=1).reset_index(drop=True)
    n = df_data.shape[1] - 1
    y = df_data[n]
    x = df_data.drop(n, axis=1)

    # Choose a random class as class_c
    seed(1)
    class_c = y[randint(0, len(y))]
    print("class c is ", class_c)
    x.columns = range(x.shape[1])
    # Base Multinomial Classifier
    precision_base, recall_base = run_classifier("multi", x, y)
    # K-N Voting Classifier
    run_classifier("kn", x, y, precision_base, recall_base)


