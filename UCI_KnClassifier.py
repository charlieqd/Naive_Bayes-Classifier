from UCI_BaseClassifier import UCIBaseClassifier
import numpy as np
import nltk
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from pylab import figure, axes, pie, title, show
from nltk import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve


class KnClassifier(UCIBaseClassifier):
    def __init__(self, train_data, test_data, train_target, test_target, class_c):
        super().__init__(train_data, test_data, train_target, test_target, class_c)
        self.bottom_up_table = []
        self.k_max = self.n
        self.pred_prob_matrix = []

    def kn_fit(self):
        self.fit()

    def build_bu(self, p_c_list, p_not_c_list):
        bottom_up = np.zeros((self.n+1, self.n+1))

        for i in range(self.n + 1):
            bottom_up[0][i] = 0

        pred, pred_not = 0, 0
        for j in range(1, self.n+1):
            pred_not += p_not_c_list[j]
            pred += p_c_list[j]
            bottom_up[j][0] = pred_not
            bottom_up[j][j] = pred

        for i in range(2, self.n+1):
            for j in range(1, i):
                p1 = p_c_list[i] + bottom_up[i - 1][j - 1]
                p2 = p_not_c_list[i] + bottom_up[i - 1][j]
                a = max(p1, p2)
                bottom_up[i][j] = a + np.log(np.exp(p1 - a) + np.exp(p2 - a))

        self.bottom_up_table = bottom_up
        # print(bottom_up)
        # print("table size is", bottom_up.shape)

    def log_prob_kn(self, d_index, class_s, class_not_s, alpha=1):
        feature_prob_list = []
        prior = 0
        if class_s == 1:
            prior += np.log(self.num_c) - np.log(self.num_c + self.num_nc)
        else:
            prior += np.log(self.num_nc) - np.log(self.num_c + self.num_nc)

        for f in range(self.n):
            test_feature = self.x_test[f][d_index]
            if test_feature in self.result[class_s][f].keys():
                prob = np.log(self.result[class_s][f][test_feature] + alpha) - np.log(
                    self.result[class_s][f][test_feature] + self.result[class_not_s][f][test_feature] + 2 * alpha)
                feature_prob_list.append(prob)
                # a = (self.result[class_s][f][test_feature] + alpha)/(self.result[class_s][f][test_feature] + alpha + self.result[class_not_s][f][test_feature] + alpha)
                # print("a is ", a)
                # print("in ", class_s, "prob is ", prob)
                # print("data is ", self.result[class_s][f][test_feature], self.result[class_not_s][f][test_feature])
            else:
                print("No corresponding feature: ", f, test_feature)
                feature_prob_list.append(prior)
                continue

        # if not feature_prob_list:
        #     feature_prob_list.append(prior)

        return feature_prob_list

    def predict_prob_kn(self):
        pred_prob_matrix = np.zeros((self.k_max, len(self.x_test)))
        counter = 0
        for ite in range(len(self.x_test)):
            test_index = len(self. x_train) + ite
            p_c_list = self.log_prob_kn(test_index, 1, 0)
            p_nc_list = self.log_prob_kn(test_index, 0, 1)
            p_c_list.insert(0, -10000)
            p_nc_list.insert(0, -10000)
            self.build_bu(p_c_list, p_nc_list)

            for k in range(1, self.k_max + 1):
                prob_k = 0
                k_n_list = self.bottom_up_table[self.n][k:]
                a = max(k_n_list)
                for element in k_n_list:
                    prob_k += np.exp(element - a)
                prob_k = np.log(prob_k) + a
                pred_prob_matrix[k - 1][counter] = prob_k
            counter += 1

        self.pred_prob_matrix = pred_prob_matrix

        return pred_prob_matrix

    def predict_kn(self, threshold, k):
        prediction = []
        pred_prob = []
        for i in range(len(self.x_test)):
            pred = 0
            pred_prob.append(np.exp(self.pred_prob_matrix[k-1][i]))
            if self.pred_prob_matrix[k-1][i] > np.log(threshold):
                pred = 1
            prediction.append(pred)
        return prediction, pred_prob

    def plot_kn(self, plot=1, r_base=None, p_base=None):
        precision_matrix, recall_matrix = [], []
        for k in range(1, self.k_max + 1):
            prediction, prob_list = self.predict_kn(0.5, k)
            precision_list, recall_list, threshold_list = precision_recall_curve(self.y_test, prob_list)
            # print("p", precision_list)
            # print("r", recall_list)
            # print("t", threshold_list)
            precision_matrix.append(precision_list)
            recall_matrix.append(recall_list)
        t_list = [13]
        if plot == 1:
            for i in range(len(precision_matrix)):
            # for i in t_list:
                plt.plot(recall_matrix[i], precision_matrix[i], label=i+1)
        if r_base is not None:
            plt.plot(r_base, p_base, label="Base Model", linestyle='dashed')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('K-N Voting Classifier Precision-Recall curve')
        plt.grid()
        # plt.legend(loc="right")
        plt.legend(bbox_to_anchor=(1.5, 1))
        show()

        return precision_matrix, recall_matrix
