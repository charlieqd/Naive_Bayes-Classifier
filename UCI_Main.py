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


def test_plot():
    r, p = multi_classifier.estimation_test(pred_result)
    print(p, r)
    t = np.arange(0.01, 1.02, 0.02)
    recall_list, precision_list, threshold_list, recall_c, precision_c = [], [], [], -1, -1
    for threshold in t:
        pred, pb = multi_classifier.predict(threshold)
        recall_c, precision_c = multi_classifier.estimation_test(pred)

        recall_list.append(recall_c)
        precision_list.append(precision_c)
        threshold_list.append(threshold)

    plt.plot(recall_list, precision_list)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve1')
    plt.grid()
    show()

    print(recall_list, precision_list)


if __name__ == '__main__':
    # ------haberman.data-----------
    # df_data = pd.read_csv('haberman.data.csv', header=None)
    # df_data = df_data.sample(frac=1).reset_index(drop=True)
    # n = df_data.shape[1] - 1
    # y = df_data[n]
    # x = df_data.drop(n, axis=1)

    # ------housevotes.data-----------
    # df_data = pd.read_csv('housevotes.data.csv', header=None)
    # n = df_data.shape[1] - 1
    # y = df_data[0]
    # x = df_data.drop(0, axis=1)

    # ------zoo.data-----------
    # df_data = pd.read_csv('zoo.data.csv', header=None)
    # n = df_data.shape[1] - 1
    # y = df_data[n]
    # x = df_data.drop(n, axis=1)

    # ------lymphography.data-----------
    # df_data = pd.read_csv('lymphography.data.csv', header=None)
    # df_data = df_data.sample(frac=1).reset_index(drop=True)
    # n = df_data.shape[1] - 1
    # y = df_data[0]
    # x = df_data.drop(0, axis=1)

    # ------soybean-large.data-----------
    # df_data = pd.read_csv('soybean-large.data.csv', header=None)
    # df_data = df_data.sample(frac=1).reset_index(drop=True)  #Shuffle the data
    # n = df_data.shape[1] - 1
    # y = df_data[0]
    # x = df_data.drop(0, axis=1)
    # class_c = "frog-eye-leaf-spot"

    # ------audiology.standardized.data-----------
    # df_data = pd.read_csv('audiology.standardized.data.csv', header=None)
    # n = df_data.shape[1] - 1
    # y = df_data[n]
    # x = df_data.drop(n, axis=1)
    # class_c = "cochlear_unknown"

    # ------glass.data-----------
    df_data = pd.read_csv('glass.data.csv', header=None)
    df_data = df_data.sample(frac=1).reset_index(drop=True)
    n = df_data.shape[1] - 1
    y = df_data[n]
    x = df_data.drop(n, axis=1)

    # ------hepatitis.data-----------
    # df_data = pd.read_csv('hepatitis.data.csv', header=None)
    # df_data = df_data.sample(frac=1).reset_index(drop=True)
    # n = df_data.shape[1] - 1
    # y = df_data[0]
    # x = df_data.drop(0, axis=1)

    # ------balance-scale.data-----------
    # df_data = pd.read_csv('balance-scale.data.csv', header=None)
    # df_data = df_data.sample(frac=1).reset_index(drop=True)
    # n = df_data.shape[1] - 1
    # y = df_data[0]
    # x = df_data.drop(0, axis=1)

    # ------agaricus_mushroom.data-----------
    # df_data = pd.read_csv('agaricus_mushroom.data.csv', header=None)
    # df_data = df_data.sample(frac=1).reset_index(drop=True)
    # n = df_data.shape[1] - 1
    # y = df_data[0]
    # x = df_data.drop(0, axis=1)



    # Choose a random class_c
    seed(1)
    class_c = y[randint(0, len(y))]
    print("class c is ", class_c)
    x.columns = range(x.shape[1])
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
    # print(x_train.shape)
    # Naive Bayes Cassifier + K-Fold CV
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    kf = KFold(n_splits=5)
    kf.get_n_splits(x)
    y_real, y_prob = [], []
    for train_index, test_index in kf.split(x):
        print("train: ", train_index, "Test: ", test_index)
        print(x.shape)
        print("folder + 1")
        x_train = x.sort_index().loc[train_index]
        x_test = x.sort_index().loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        x_train = x_train.reset_index(drop=True)
        x_test = x_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
        print("n is ", x_train.shape[1])

        multi_classifier = UCI_BaseClassifier.UCIBaseClassifier(x_train, x_test, y_train, y_test, class_c)
        multi_classifier.fit()
        pred_result, pred_prob = multi_classifier.predict(0.5)
        precision, recall, accuracy = multi_classifier.estimation()
        print("here")
        print(precision, recall, accuracy)

        precision_list, recall_list, threshold_list = multi_classifier.plot(0)
        y_real.append(multi_classifier.y_test)
        y_prob.append(multi_classifier.pred_prob_list)

    print("out of loop")
    y_real = np.concatenate(y_real)
    y_prob = np.concatenate(y_prob)
    print(len(y_real), len(y_prob))
    precision, recall, threshold = precision_recall_curve(y_real, y_prob)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.grid()
    show()





    # multi_classifier = UCI_BaseClassifier.UCIBaseClassifier(x_train, x_test, y_train, y_test, class_c)
    # multi_classifier.fit()
    # pred_result, pred_prob = multi_classifier.predict(0.5)
    # precision, recall, accuracy = multi_classifier.estimation()
    # print("here")
    # print(precision, recall, accuracy)
    #
    # precision_list, recall_list, threshold_list = multi_classifier.plot(0)
    #
    #
    # K-N Voting Classifier
    y_prob_kn = {}
    y_real_kn = []
    flag = 0
    k_max = 0
    for train_index, test_index in kf.split(x):
        print("train: ", train_index, "Test: ", test_index)
        print(x.shape)
        print("folder + 1")
        x_train = x.sort_index().loc[train_index]
        x_test = x.sort_index().loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        x_train = x_train.reset_index(drop=True)
        x_test = x_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
        print("n is ", x_train.shape[1])

        kn_classifier = UCI_KnClassifier.KnClassifier(x_train, x_test, y_train, y_test, class_c)
        print("0")
        kn_classifier.fit()
        print("1")
        if flag == 0:
            flag = 1
            k_max = kn_classifier.k_max
            for k in range(1, kn_classifier.k_max + 1):
                y_prob_kn[k] = []

        pred_prob_matrix = kn_classifier.predict_prob_kn()
        print("2")
        y_real_kn.append(kn_classifier.y_test)
        for k in range(1, kn_classifier.k_max + 1):
            prediction, prob_list = kn_classifier.predict_kn(0.5, k)
            y_prob_kn[k].append(prob_list)

        print("3")
        # kn_classifier.plot_kn(1, recall_list, precision_list)

    print("out of loop")
    y_real_kn = np.concatenate(y_real_kn)
    for k in range(1, k_max + 1):
        y_prob_kn[k] = np.concatenate(y_prob_kn[k])
    print(len(y_real_kn), len(y_prob_kn[1]))
    for k in range(1, k_max + 1):
        precision, recall, threshold = precision_recall_curve(y_real_kn, y_prob_kn[k])
        plt.plot(recall, precision, label=k)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.grid()
    show()





    # kn_classifier = UCI_KnClassifier.KnClassifier(x_train, x_test, y_train, y_test, class_c)
    # print("0")
    # kn_classifier.fit()
    # print("1")
    # pred_prob_matrix = kn_classifier.predict_prob_kn()
    # print("2")
    # prediction, prob_list = kn_classifier.predict_kn(0.5, 2)
    # print("3")
    # print(prediction)
    # print(kn_classifier.y_test)
    # print(prob_list)
    # kn_classifier.plot_kn(1, recall_list, precision_list)







    # precisio, recal, threshol = precision_recall_curve(kn_classifier.y_test, prob_list)
    # print(precisio)
    # print(recal)
    # print(threshol)
    # plt.plot(recal, precisio)
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall curve')
    # plt.grid()
    # show()
