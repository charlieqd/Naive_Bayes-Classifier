import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
import UCI_BaseClassifier
import UCI_KnClassifier
import matplotlib.pyplot as plt
from pylab import figure, axes, pie, title, show

if __name__ == '__main__':
    class_c = 2
    df_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data', header=None)
    n = df_data.shape[1] - 1
    y = df_data[n]
    x = df_data.drop(n, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=False)
    # Naive Bayes Cassifier

    # multi_classifier = UCI_BaseClassifier.UCIBaseClassifier(x_train, x_test, y_train, y_test, class_c)
    # multi_classifier.fit()
    # pred = multi_classifier.predict(0.1)
    # precision, recall, accuracy = multi_classifier.estimation()
    # print(precision, recall, accuracy)
    #
    # y_true = multi_classifier.y_test
    # precisio, recal, threshol = precision_recall_curve(y_true, pred)
    # print(precisio)
    # print(recal)
    # print(threshol)
    # plt.plot(recal, precisio)
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall curve')
    # plt.grid()
    # show()

    # K-N Voting Classifier
    kn_classifier = UCI_KnClassifier.KnClassifier(x_train, x_test, y_train, y_test, class_c)
    kn_classifier.fit()
    print("1")
    pred_prob_matrix = kn_classifier.predict_prob_kn()
    print("2")
    prediction, prob_list = kn_classifier.predict_kn(0.41, 2)
    print(prediction)
    print(kn_classifier.y_test)
    print(prob_list)
