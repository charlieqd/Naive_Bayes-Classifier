import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
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
    # n = df_data.shape[1] - 1
    # y = df_data[0]
    # x = df_data.drop(0, axis=1)

    # ------soybean-large.data-----------
    df_data = pd.read_csv('soybean-large.data.csv', header=None)
    df_data = df_data.sample(frac=1).reset_index(drop=True)  #Shuffle the data
    n = df_data.shape[1] - 1
    y = df_data[0]
    x = df_data.drop(0, axis=1)

    # ------audiology.standardized.data-----------
    # df_data = pd.read_csv('audiology.standardized.data.csv', header=None)
    # n = df_data.shape[1] - 1
    # y = df_data[n]
    # x = df_data.drop(n, axis=1)
    # class_c = "cochlear_unknown"

    # ------glass.data-----------
    # df_data = pd.read_csv('glass.data.csv', header=None)
    # df_data = df_data.sample(frac=1).reset_index(drop=True)
    # n = df_data.shape[1] - 1
    # y = df_data[n]
    # x = df_data.drop(n, axis=1)

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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
    print(x_train.shape)
    # Naive Bayes Cassifier

    multi_classifier = UCI_BaseClassifier.UCIBaseClassifier(x_train, x_test, y_train, y_test, class_c)
    multi_classifier.fit()
    pred_result, pred_prob = multi_classifier.predict(0.5)
    precision, recall, accuracy = multi_classifier.estimation()
    print("here")
    print(precision, recall, accuracy)

    precision_list, recall_list, threshold_list = multi_classifier.plot(0)


    # K-N Voting Classifier
    kn_classifier = UCI_KnClassifier.KnClassifier(x_train, x_test, y_train, y_test, class_c)
    print("0")
    kn_classifier.fit()
    print("1")
    pred_prob_matrix = kn_classifier.predict_prob_kn()
    print("2")
    prediction, prob_list = kn_classifier.predict_kn(0.41, 1)
    print(prediction)
    print(kn_classifier.y_test)
    print(prob_list)
    kn_classifier.plot_kn(1, recall_list, precision_list)







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
