from BasicClassifier import BasicClassifier
import numpy as np
import nltk
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from pylab import figure, axes, pie, title, show
from nltk import word_tokenize
from sklearn.metrics import accuracy_score


class KnClassifier(BasicClassifier):
    def __init__(self, train_data, train_target, vocab_feature, class_c, test_data, test_target, name, k):
        super().__init__(train_data, train_target, vocab_feature, class_c, test_data, test_target, name)
        self.bottom_up_table = []
        self.k_max = k
        self.data_k_pred_prob_matrix = []

    def kn_fit(self):
        self.fit()

    def log_prob_voting_kn_multi(self, index, class_selected, class_not_selected, alpha=1):
        ps = PorterStemmer()
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        prior = np.log(self.result[class_selected]["DOC_OF_CLASS"]) - np.log(self.result["TOTAL_DOC"])
        feature_prob = []
        test_data = self.test_data[index]
        for word in tokenizer.tokenize(test_data):
            word = ps.stem(word)
            word = word.lower()
            if word in self.result[class_selected].keys():
                word_occurrence_in_class = self.result[class_selected][word] + alpha  # Tct + 1
                word_occurrence = self.result[class_selected][word] + self.result[class_not_selected][word] + alpha * 2
                output = np.log(word_occurrence_in_class) - np.log(word_occurrence)
                feature_prob.append(output)
            else:
                continue  # we don't care about the words not in the feature list
        if not feature_prob:
            feature_prob.append(prior)

        return feature_prob

    def log_prob_voting_kn_test1(self, index, class_selected, class_not_selected, alpha=1):
        # 0 vs 1
        prior = np.log(self.result[class_selected]["DOC_OF_CLASS"]) - np.log(self.result["TOTAL_DOC"])
        feature_prob = []
        for word in self.tf_dict_test[index].keys():
            word_occurrence_in_class = self.result[class_selected][word] + alpha  # Tct + 1
            word_occurrence = self.result[class_selected][word] + self.result[class_not_selected][word] + alpha * 2
            output = np.log(word_occurrence_in_class) - np.log(word_occurrence)
            feature_prob.append(output)
        if not feature_prob:
            feature_prob.append(prior)

        return feature_prob

    #
    # def log_prob_voting_kn_tfidf(self, test_data, class_selected, class_not_selected):
    #     tokenizer = nltk.RegexpTokenizer(r"\w+")
    #     prior = np.log(self.result[class_selected]["DOC_OF_CLASS"]) - np.log(self.result["TOTAL_DOC"])
    #     feature_prob = []
    #     word_list = []
    #     for word in tokenizer.tokenize(test_data):
    #         word = word.lower()
    #         if word not in word_list:
    #             word_list.append(word)
    #         else:
    #             continue
    #         if word in self.result[class_selected].keys():
    #             word_occurrence_in_class = self.tfidf_result[class_selected][word] + 1  # Tct + 1
    #             word_occurrence = self.tfidf_result[class_selected][word] + 1 + self.tfidf_result[class_not_selected][word] + 1
    #             output = np.log(word_occurrence_in_class) - np.log(word_occurrence)
    #             feature_prob.append(output)
    #         else:
    #             continue  # we don't care about the words not in the feature list
    #     if not feature_prob:
    #         feature_prob.append(prior)
    #
    #     return feature_prob

    def log_prob_voting_kn_bernoulli(self, index, class_selected, class_not_selected, alpha=1):
        feature_prob = []
        for word in self.vocab_feature:
            total_doc = self.result["TOTAL_DOC"]
            # doc_with_word = self.df_dict_train[class_selected][word] + self.df_dict_train[class_not_selected][
            #     word] + alpha * 2
            doc_with_word = self.bernoulli_result[class_selected][word] + self.bernoulli_result[class_not_selected][
                word] + alpha * 2
            doc_without_word = total_doc - doc_with_word
            doc_with_word_in_class = self.bernoulli_result[class_selected][word] + alpha
            doc_without_word_in_class = self.result[class_selected]["DOC_OF_CLASS"] - doc_with_word_in_class
            if word in self.tf_dict_test[index].keys():
                acond_prob = doc_with_word_in_class / doc_with_word
                cond_prob = np.log(doc_with_word_in_class) - np.log(doc_with_word)
                feature_prob.append(cond_prob)
            else:
                acond_prob = doc_without_word_in_class / doc_without_word
                bcond_prob = doc_with_word_in_class / doc_with_word
                cond_prob = np.log(doc_without_word_in_class) - np.log(doc_without_word)
                feature_prob.append(cond_prob)

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

    def log_test(self, index, alpha=1):
        ps = PorterStemmer()
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        prior = np.log(self.result["CLASS_C"]["DOC_OF_CLASS"]) - np.log(self.result["TOTAL_DOC"])
        prior_n = np.log(self.result["NOT_CLASS_C"]["DOC_OF_CLASS"]) - np.log(self.result["TOTAL_DOC"])
        c = np.log(self.result["CLASS_C"]["DOC_OF_CLASS"])
        nc = np.log(self.result["NOT_CLASS_C"]["DOC_OF_CLASS"])
        feature_prob = []
        feature_n_prob = []
        test_data = self.test_data[index]
        total_word_c = len(self.vocab_feature) * alpha
        total_word_nc = len(self.vocab_feature) * alpha
        for element in self.result["CLASS_C"].keys():
            total_word_c += self.result["CLASS_C"][element]
        for e in self.result["NOT_CLASS_C"].keys():
            total_word_nc += self.result["NOT_CLASS_C"][e]
        for word in tokenizer.tokenize(test_data):
            word = ps.stem(word)
            word = word.lower()
            if word in self.result["CLASS_C"].keys():
                word_occurrence_in_class = self.result["CLASS_C"][word] + alpha
                word_occurrence_in_not_class = self.result["NOT_CLASS_C"][word] + alpha
                log_a = np.log(word_occurrence_in_class) - np.log(total_word_c) + np.log(c)
                log_b = np.log(word_occurrence_in_not_class) - np.log(total_word_nc) + np.log(nc)
                log_k = log_a - log_b
                a = 0
                if log_k > 0:
                    a = log_k
                output = log_k - (a + np.log(np.exp(0 - a) + np.exp(log_k - a)))
                feature_prob.append(output)
                feature_n_prob.append(output - log_k)
            else:
                continue  # we don't care about the words not in the feature list
        if not feature_prob:
            feature_prob.append(prior)
        if not feature_n_prob:
            feature_n_prob.append(prior_n)

        if len(feature_prob) != len(feature_n_prob):
            print("Isuueuhdsfjhdsfhbhg")

        return feature_prob, feature_n_prob

    def kn_voting(self, test_data, v_type, alpha=1):

        target_pred = np.zeros((self.k_max, len(test_data)))
        counter = 0

        for i in range(len(test_data)):
            p_c_list, p_not_c_list = [], []
            data = test_data[i]
            if v_type == "multi":
                p_c_list = self.log_prob_voting_kn_multi(i, "CLASS_C", "NOT_CLASS_C", alpha)
                p_not_c_list = self.log_prob_voting_kn_multi(i, "NOT_CLASS_C", "CLASS_C", alpha)
            # elif v_type == "tfidf":
            #     p_c_list = self.log_prob_voting_kn_tfidf(data, "CLASS_C", "NOT_CLASS_C")
            #     p_not_c_list = self.log_prob_voting_kn_tfidf(data, "NOT_CLASS_C", "CLASS_C")

            # build the bottom up
            elif v_type == "bernoulli":
                p_c_list = self.log_prob_voting_kn_bernoulli(i, "CLASS_C", "NOT_CLASS_C", alpha)
                p_not_c_list = self.log_prob_voting_kn_bernoulli(i, "NOT_CLASS_C", "CLASS_C", alpha)
                # print("pclist is ")
                # print(p_c_list)
                # print(p_not_c_list)
            elif v_type == "test":  # another way for multi
                p_c_list, p_not_c_list = self.log_test(i)
            elif v_type == "test1":  # 0 - 1
                p_c_list = self.log_prob_voting_kn_test1(i, "CLASS_C", "NOT_CLASS_C", alpha)
                p_not_c_list = self.log_prob_voting_kn_test1(i, "NOT_CLASS_C", "CLASS_C", alpha)
                if len(p_c_list) != len(p_not_c_list):
                    print(len(p_c_list), len(p_not_c_list))
                    print("probibfuigfs")

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
        # print("Target matrix is ")
        # print(target_pred)
        return target_pred

    def predict_kn(self, test_data, threshold, k):
        # need to run kn_voting first
        prediction = []
        # print("shape is ", self.data_k_pred_prob_matrix.shape)
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

            for i in range(len(self.y_test)):
                if self.y_test[i] == 1:
                    total_actual_positive += 1
                    if predict_result_k[i] == 1:
                        true_positive += 1

            recall = true_positive / total_actual_positive
            precision = true_positive / total_pred_positive

            recall_list.append(recall)
            precision_list.append(precision)

        return recall_list, precision_list

    def plot_kn(self, range_list, plot=1, precision_base=None, recall_base=None, k_range=[]):
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

        if plot == 1:
            if not k_range:
                for i in range(self.k_max):
                    plt.plot(recall_matrix[i], precision_matrix[i], label=i + 1)
            else:
                for i in k_range:
                    plt.plot(recall_matrix[i - 1], precision_matrix[i], label=i)
            if precision_base is not None:
                plt.plot(recall_base, precision_base, label="Base Model", linestyle='dashed')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall curve of kn classifier')
            plt.legend(bbox_to_anchor=(1.5, 1), title="K Value")
            plt.grid()
            plt.show()

        return recall_matrix, precision_matrix
