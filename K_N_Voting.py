from BasicClassifier import BasicClassifier
import numpy as np
import nltk
import matplotlib.pyplot as plt
from pylab import figure, axes, pie, title, show
from nltk import word_tokenize
from sklearn.metrics import accuracy_score


class KnClassifier(BasicClassifier):
    def __init__(self, train_data, train_target, vocab_feature, class_c, test_data, name, k):
        super().__init__(train_data, train_target, vocab_feature, class_c, test_data, name)
        self.bottom_up_table = []
        self.k_max = k
        self.data_k_pred_prob_matrix = []

    def kn_fit(self):
        self.fit()

    def log_prob_voting_kn(self, test_data, class_selected, class_not_selected):
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        prior = np.log(self.result[class_selected]["DOC_OF_CLASS"]) - np.log(self.result["TOTAL_DOC"])
        feature_prob = []
        for word in tokenizer.tokenize(test_data):
            word = word.lower()
            if word in self.result[class_selected].keys():
                word_occurrence_in_class = self.result[class_selected][word] + 1  # Tct + 1
                word_occurrence = self.result[class_selected][word] + 1 + self.result[class_not_selected][word] + 1
                output = np.log(word_occurrence_in_class) - np.log(word_occurrence)
                feature_prob.append(output)
            else:
                continue  # we don't care about the words not in the feature list
        if not feature_prob:
            feature_prob.append(prior)

        return feature_prob

    def log_prob_voting_kn_tfidf(self, test_data, class_selected, class_not_selected):
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        prior = np.log(self.result[class_selected]["DOC_OF_CLASS"]) - np.log(self.result["TOTAL_DOC"])
        feature_prob = []
        word_list = []
        for word in tokenizer.tokenize(test_data):
            word = word.lower()
            if word not in word_list:
                word_list.append(word)
            else:
                continue
            if word in self.result[class_selected].keys():
                word_occurrence_in_class = self.tfidf_result[class_selected][word] + 1  # Tct + 1
                word_occurrence = self.tfidf_result[class_selected][word] + 1 + self.tfidf_result[class_not_selected][word] + 1
                output = np.log(word_occurrence_in_class) - np.log(word_occurrence)
                feature_prob.append(output)
            else:
                continue  # we don't care about the words not in the feature list
        if not feature_prob:
            feature_prob.append(prior)

        return feature_prob

    def build_bu(self, n, p, p_not):
        bottom_up = np.zeros((n + 1, n + 1))

        for i in range(n + 1):
            bottom_up[0][i] = 0

        pred, pred_not = 0, 0
        for j in range(1, n + 1):
            pred_not += p_not[j]
            pred += p[j]
            bottom_up[j][0] = pred_not
            bottom_up[j][j] = pred

        for i in range(2, n + 1):
            for j in range(1, i):
                p1 = p[i] + bottom_up[i - 1][j - 1]
                p2 = p_not[i] + bottom_up[i - 1][j]
                a = max(p1, p2)
                bottom_up[i][j] = a + np.log(np.exp(p1 - a) + np.exp(p2 - a))
                # bottom_up[i][j] = p[i] * bottom_up[i-1][j-1] + (1 - p[i])*bottom_up[i-1][j]

        self.bottom_up_table = bottom_up

    def kn_voting(self, test_data, v_type):

        target_pred = np.zeros((self.k_max, len(test_data)))
        counter = 0

        for data in test_data:
            if v_type == "regular":
                p_c_list = self.log_prob_voting_kn(data, "CLASS_C", "NOT_CLASS_C")
                p_not_c_list = self.log_prob_voting_kn(data, "NOT_CLASS_C", "CLASS_C")
            elif v_type == "tfidf":
                p_c_list = self.log_prob_voting_kn_tfidf(data, "CLASS_C", "NOT_CLASS_C")
                p_not_c_list = self.log_prob_voting_kn_tfidf(data, "NOT_CLASS_C", "CLASS_C")

            # build the bottom up
            n = len(p_c_list)
            p_c_list.insert(0, -1)
            p_not_c_list.insert(0, -1)
            self.build_bu(n, p_c_list, p_not_c_list)

            for k in range(1, self.k_max):
                k_c = k
                if k > n:
                    k = n
                prob_k = 0
                k_n_list = self.bottom_up_table[n][k:]
                a = max(k_n_list)
                for element in k_n_list:
                    prob_k += np.exp(element - a)
                prob_k = np.log(prob_k) + a
                target_pred[k_c - 1][counter] = prob_k
            counter += 1

        self.data_k_pred_prob_matrix = target_pred
        return target_pred

    def predict_kn(self, test_data, threshold, k):
        # need to run kn_voting first
        prediction = []
        for i in range(len(test_data)):
            pred = 0
            if self.data_k_pred_prob_matrix[k - 1][i] > np.log(threshold):
                pred = 1
            prediction.append(pred)
        return prediction

    def estimation_kn(self, predict_result):

        recall_list, precision_list = [], []
        for predict_result_k in predict_result:
            unique, counts = np.unique(predict_result_k, return_counts=True)
            true_positive = 0
            total_actual_positive = 0
            if 1 not in unique:
                true_positive = 1
                total_pred_positive = 1
            elif 0 not in unique:
                total_pred_positive = counts[0]
            else:
                total_pred_positive = counts[1]

            for i in range(len(self.true_pred)):
                if self.true_pred[i] == 1:
                    total_actual_positive += 1
                    if predict_result_k[i] == 1:
                        true_positive += 1

            recall = true_positive / total_actual_positive
            precision = true_positive / total_pred_positive

            recall_list.append(recall)
            precision_list.append(precision)

        return recall_list, precision_list

    def plot_kn(self, range_list):
        pred_prob = self.data_k_pred_prob_matrix
        k = self.k_max
        n = len(range_list)
        recall_matrix = np.zeros((k, n))
        precision_matrix = np.zeros((k, n))
        threshold_list = []
        counter = 0
        for threshold in range_list:
            predict_result = np.zeros((len(pred_prob), len(pred_prob[1])), dtype=int)
            for j in range(len(pred_prob)):
                pred_prob_k = pred_prob[j]
                for i in range(len(pred_prob_k)):
                    pred = 0
                    p = pred_prob_k[i]
                    if p > np.log(threshold):
                        pred = 1
                    predict_result[j][i] = pred

            recall_list, precision_list = self.estimation_kn(predict_result)
            threshold_list.append(threshold)
            for i in range(k):
                recall_matrix[i][counter] = recall_list[i]
                precision_matrix[i][counter] = precision_list[i]

            counter += 1

        for i in range(self.k_max):
            plt.plot(recall_matrix[i], precision_matrix[i], label=i)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall curve of kn classifier')
            plt.legend(loc="right")
            plt.grid()
        plt.show()