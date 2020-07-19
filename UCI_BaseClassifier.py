import numpy as np
import nltk
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from pylab import figure, axes, pie, title, show


class UCIBaseClassifier:
    def __init__(self, train_data, test_data, train_target, test_target, class_c):
        self.x_train = train_data
        self.x_test = test_data
        self.y_train = train_target
        self.test_target = test_target
        self.y_test = []
        self.class_c = class_c
        self.result = {}  # tf result
        self.y_test = []
        self.pred_result = []
        self.n = self.x_train.shape[1] # number of features
        self.num_c = 0
        self.num_nc = 0
        self.pred_prob_list = []

    def truth_build(self):
        for i in range(len(self.test_target)):
            if self.test_target[i] == self.class_c:
                self.y_test.append(1)
            else:
                self.y_test.append(0)

    def fit(self):
        self.truth_build()
        result = {0: {}, 1: {}}
        for _ in range(self.n):
            result[0][_] = {}
            result[1][_] = {}
        for i in range(len(self.x_train)):
            if self.y_train[i] == self.class_c:
                self.num_c += 1
                for m in range(self.n):
                    if self.x_train[m][i] in result[1][m].keys():
                        result[1][m][self.x_train[m][i]] += 1
                    else:
                        result[1][m][self.x_train[m][i]] = 1
                        result[0][m][self.x_train[m][i]] = 0
            else:
                self.num_nc += 1
                for m in range(self.n):
                    if self.x_train[m][i] in result[0][m].keys():
                        result[0][m][self.x_train[m][i]] += 1
                    else:
                        result[0][m][self.x_train[m][i]] = 1
                        result[1][m][self.x_train[m][i]] = 0
        self.result = result

    def log_prob(self, class_s, alpha=1):
        prob_list = []
        for m in range(len(self.x_test)):
            prob, total_count = 0, 0
            if class_s == 1:
                prob += np.log(self.num_c) - np.log(self.num_c + self.num_nc)
                total_count = self.num_c
            else:
                prob += np.log(self.num_nc) - np.log(self.num_c + self.num_nc)
                total_count = self.num_nc
            for f in range(self.n):
                test_feature = self.x_test[f][m]
                if test_feature in self.result[class_s][f].keys():
                    prob += np.log(self.result[class_s][f][test_feature] + alpha) - np.log(
                        total_count + len(self.result[class_s][f].keys())*alpha)
            prob_list.append(prob)

        return prob_list

    def predict(self, threshold):
        p_not_c = self.log_prob(0)
        p_c = self.log_prob(1)
        output, pred_prob = [], []
        for i in range(len(p_c)):

            if p_c[i] - p_not_c[i] > np.log(threshold) - np.log(1-threshold):
                output.append(1)
            else:
                output.append(0)

            a = np.exp(p_c[i] - p_not_c[i])
            p1 = a/(a+1)
            pred_prob.append(p1)

        self.pred_result = output
        self.pred_prob_list = pred_prob

        return output, pred_prob

    def estimation(self):
        acc = accuracy_score(self.y_test, self.pred_result)
        precision, recall, c, d = precision_recall_fscore_support(self.y_test, self.pred_result, average='binary')
        return precision, recall, acc

    def plot(self, plot=0):
        precision_list, recall_list, threshold_list = precision_recall_curve(self.y_test, self.pred_prob_list)
        if plot == 1:
            plt.plot(recall_list, precision_list)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall curve')
            plt.grid()
            show()
        return precision_list, recall_list, threshold_list
