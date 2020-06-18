import numpy as np
import nltk
import operator
from nltk import word_tokenize
from sklearn.metrics import accuracy_score


class BasicClassifier:
    def __init__(self, train_data, train_target, vocab_feature, class_c, test_data, name):
        self.name = name
        self.doc_to_vocab = {}
        self.train_data = train_data
        self.test_data = test_data.data
        self.train_target = train_target
        self.test_target = test_data.target
        self.vocab_feature = vocab_feature
        self.class_c = class_c
        self.result = {}
        self.df = {}
        self.doc_to_vocab_tfidf = {}
        self.tfidf_result = {}
        self.true_pred = []
        self.pred_result = []
        self.pred_prob = []

    def truth_build(self):
        for i in range(len(self.test_target)):
            if self.test_target[i] == self.class_c:
                self.true_pred.append(1)
            else:
                self.true_pred.append(0)

    def dict_build(self):
        self.truth_build()
        nltk.download("punkt")
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        for i in range(len(self.train_data)):
            total_word = 0
            self.doc_to_vocab[i] = {}
            cur_doc = self.train_data[i]
            for word in tokenizer.tokenize(cur_doc):
                if word.lower() in self.vocab_feature:
                    total_word += 1
                    if word.lower() in self.doc_to_vocab[i].keys():
                        self.doc_to_vocab[i][word.lower()] += 1
                    else:
                        self.doc_to_vocab[i][word.lower()] = 1
            self.doc_to_vocab[i]["TOTAL_WORDS"] = total_word

    def tfidf_result_build(self):
        self.tfidf_result["CLASS_C"] = {}
        self.tfidf_result["NOT_CLASS_C"] = {}
        for word in self.vocab_feature:
            self.tfidf_result["CLASS_C"][word] = 0
            self.tfidf_result["NOT_CLASS_C"][word] = 0

        for i in range(len(self.doc_to_vocab_tfidf)):
            for word in self.doc_to_vocab_tfidf[i].keys():
                if self.train_target[i] == self.class_c:
                    self.tfidf_result["CLASS_C"][word] += self.doc_to_vocab_tfidf[i][word]
                else:
                    self.tfidf_result["NOT_CLASS_C"][word] += self.doc_to_vocab_tfidf[i][word]

    def fit(self):
        self.dict_build()
        total_doc = len(self.train_data)
        self.result["TOTAL_DOC"] = total_doc
        self.result["CLASS_C"] = {}
        self.result["NOT_CLASS_C"] = {}
        classes, doc_counters = np.unique(self.train_target, return_counts=True)

        self.result["CLASS_C"]["DOC_OF_CLASS"] = doc_counters[self.class_c]
        self.result["NOT_CLASS_C"]["DOC_OF_CLASS"] = total_doc - doc_counters[self.class_c]

        for word in self.vocab_feature:
            self.result["CLASS_C"][word] = 0
            self.result["NOT_CLASS_C"][word] = 0
            # self.tfidf_result["CLASS_C"][word] = 0
            # self.tfidf_result["NOT_CLASS_C"][word] = 0
            self.df[word] = 0

        for i in range(len(self.train_data)):
            if self.train_target[i] == self.class_c:
                for word in self.doc_to_vocab[i].keys():
                    if word == "TOTAL_WORDS":
                        continue
                    self.result["CLASS_C"][word] += self.doc_to_vocab[i][word]
                    self.df[word] += 1
            else:
                for word in self.doc_to_vocab[i].keys():
                    if word == "TOTAL_WORDS":
                        continue
                    self.result["NOT_CLASS_C"][word] += self.doc_to_vocab[i][word]
                    self.df[word] += 1

        total_word_class_c, total_word_not_class = 0, 0
        for feature_word in self.vocab_feature:
            total_word_class_c += self.result["CLASS_C"][feature_word]
            total_word_not_class += self.result["NOT_CLASS_C"][feature_word]

        for i in range(len(self.train_data)):
            self.doc_to_vocab_tfidf[i] = {}
            for word in self.doc_to_vocab[i].keys():
                if word == "TOTAL_WORDS":
                    continue
                term_freq = self.doc_to_vocab[i][word] / self.doc_to_vocab[i]["TOTAL_WORDS"]
                tf_idf = np.log(total_doc + 1) - np.log(self.df[word] + 1) * term_freq
                self.doc_to_vocab_tfidf[i][word] = tf_idf

        self.tfidf_result_build()

    def log_prob(self, test_data, class_selected):
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        count = 0
        output = np.log(self.result[class_selected]["DOC_OF_CLASS"]) - np.log(
            self.result["TOTAL_DOC"])  # log(pc) = log(Nc) - log(N)
        total_word_class = 0
        for feature_word in self.vocab_feature:
            total_word_class += self.result[class_selected][feature_word]  # sum of Tct'

        for word in tokenizer.tokenize(test_data):
            word = word.lower()
            if word in self.result[class_selected].keys():
                count += 1
                word_occurrence_in_class = self.result[class_selected][word] + 1  # Tct + 1
                curr_word_prob = np.log(word_occurrence_in_class) - np.log(
                    total_word_class + len(self.vocab_feature))  # log P(d|C) = log(Tct + 1) - log(Tct' + |V|)
                output += curr_word_prob
            else:
                continue

        return output, count

    def log_prob_tfidf(self, test_data, class_selected):
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        count = 0
        output = np.log(self.result[class_selected]["DOC_OF_CLASS"]) - np.log(
            self.result["TOTAL_DOC"])  # log(pc) = log(Nc) - log(N)
        total_word_in_class = 0
        for word in self.tfidf_result[class_selected].keys():
            total_word_in_class += self.tfidf_result[class_selected][word]

        for word in tokenizer.tokenize(test_data):
            word = word.lower()
            if word in self.result[class_selected].keys():
                count += 1
                word_count_in_class = self.tfidf_result[class_selected][word] + 1  # Tct + 1
                curr_word_prob = np.log(word_count_in_class) - np.log(
                    total_word_in_class + len(self.vocab_feature))  # log P(d|C) = log(Tct + 1) - log(Tct' + |V|)
                output += curr_word_prob
            else:
                continue

        return output, count

    def predict(self):

        target_pred = []
        pred_prob = []  # list of probability that belongs to class_c
        for data in self.test_data:
            best_pred = 1
            p_c, c0 = self.log_prob(data, "CLASS_C")
            p_not_c, c1 = self.log_prob(data, "NOT_CLASS_C")
            if p_not_c > p_c:
                best_pred = 0

            target_pred.append(best_pred)
            pred_prob.append(p_c / (p_c + p_not_c))

        return target_pred, pred_prob

    def predict_tfidf(self):
        target_pred = []
        for data in self.test_data:
            best_pred = 1
            p_c, c0 = self.log_prob_tfidf(data, "CLASS_C")
            p_not_c, c1 = self.log_prob_tfidf(data, "NOT_CLASS_C")
            if p_not_c > p_c:
                best_pred = 0
            target_pred.append(best_pred)

        return target_pred

    def predict_with_threshold(self, test_data, threshold):
        target_pred = []
        pred_prob = []  # list of probability that belongs to class_c
        for data in test_data:
            best_pred = 0
            p_c, c0 = self.log_prob(data, "CLASS_C")
            p_not_c, c1 = self.log_prob(data, "NOT_CLASS_C")
            t = 1 - threshold
            if p_c / p_not_c < np.log(threshold) / np.log(t):
                best_pred = 1
            # if t == 0:
            #     best_pred = 0
            # else:
            #     if p_c - p_not_c > np.log(threshold) - np.log(t):
            #         best_pred = 1
            target_pred.append(best_pred)
            pred_prob.append(p_c)

        self.pred_result = target_pred
        self.pred_prob = pred_prob
        return target_pred

    def predict_with_threshold_tfidf(self, test_data, threshold):
        target_pred = []
        pred_prob = []
        for data in test_data:
            best_pred = 0
            p_c, c0 = self.log_prob_tfidf(data, "CLASS_C")
            p_not_c, c1 = self.log_prob_tfidf(data, "NOT_CLASS_C")
            t = 1 - threshold
            # if t == 0:
            #     best_pred = 0
            # else:
            #     if p_c - p_not_c > np.log(threshold) - np.log(t):
            #         best_pred = 1
            if p_c / p_not_c < np.log(threshold) / np.log(t):
                best_pred = 1

            target_pred.append(best_pred)
            pred_prob.append(p_c)

        self.pred_result = target_pred
        self.pred_prob = pred_prob
        return target_pred

    def accuracy(self):
        return accuracy_score(self.true_pred, self.pred_result)

    def estimation(self, pred_result):

        unique, counts = np.unique(pred_result, return_counts=True)
        # print(unique)
        # print(counts)
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
                if pred_result[i] == 1:
                    true_positive += 1

        recall = true_positive / total_actual_positive
        precision = true_positive / total_pred_positive
        # print("total_pred_positive is ", total_pred_positive)
        # print(counts)
        # print("true positive is ", true_positive)
        #
        # print("false negative is ", total_actual_positive - true_positive)
        # print("false postive is ", total_pred_positive - true_positive)
        # print("Threshold is ", threshold)
        # print("Recall is ",recall)
        # print("Precision is ",precision)
        # print("Accuracy is ",accuracy)
        return recall, precision
