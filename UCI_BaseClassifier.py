import numpy as np
import nltk
import operator
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


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
        self.n = self.x_train.shape[1]
        self.num_c = 0
        self.num_nc = 0

    def truth_build(self):
        for ite in range(len(self.test_target)):
            i = len(self.x_train) + ite
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
        print("n is ", self.n)
        for i in range(len(self.x_train)):
            # print(i)
            # print(self.y_train[i])
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
        for ite in range(len(self.x_test)):
            m = len(self. x_train) + ite
            prob = 0
            if class_s == self.class_c:
                prob += np.log(self.num_c) - np.log(self.num_c + self.num_nc)
            else:
                prob += np.log(self.num_nc) - np.log(self.num_c + self.num_nc)
            for f in range(self.n):
                test_feature = self.x_test[f][m]
                if test_feature in self.result[class_s][f].keys():
                    prob += np.log(self.result[class_s][f][test_feature] + alpha) - np.log(
                        self.num_c + len(self.result[class_s][f].keys())*alpha)
            prob_list.append(prob)

        return prob_list

    def predict(self, threshold):
        p_c = self.log_prob(1)
        p_not_c = self.log_prob(0)
        print(p_c)
        print(p_not_c)
        output = []
        output1 = []
        pred_prob = []
        for i in range(len(p_c)):

            # if p_c[i] / p_not_c[i] < np.log(threshold) / np.log(1-threshold):
            #     output1.append(1)
            # else:
            #     output1.append(0)

            if p_c[i] - p_not_c[i] > np.log(threshold) - np.log(1-threshold):
                output1.append(1)
            else:
                output1.append(0)

            a = np.exp(p_c[i] - p_not_c[i])
            p1 = a/(a+1)
            pred_prob.append(p1)

            pn = np.exp(p_c[i])/(np.exp(p_c[i]) + np.exp(p_not_c[i]))

            if pn > threshold:
                output.append(1)
            else:
                output.append(0)


            print("Comparison: ")
            print(pn, p1)
            # if p_c[i] / p_not_c[i] < np.log(threshold) / np.log(1-threshold):
            #     output.append(1)
            # else:
            #     output.append(0)
            # if p1[i] > p0[i]:
            #     output.append(1)
            # else:
            #     output.append(0)
        print("11111")
        print(output)
        print("222222")
        print(output1)
        print("333333")
        print(self.y_test)
        print(pred_prob)

        self.pred_result = output
        return pred_prob

    def estimation(self):
        acc = accuracy_score(self.y_test, self.pred_result)
        precision, recall, c, d = precision_recall_fscore_support(self.y_test, self.pred_result, average='binary')
        return precision, recall, acc
