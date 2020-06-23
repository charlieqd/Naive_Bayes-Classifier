import numpy as np
import nltk
import operator
from nltk import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


class BasicClassifier:
    def __init__(self, train_data, train_target, vocab_feature, class_c, test_data, test_target, name):
        self.name = name
        self.doc_to_vocab = {}
        self.train_data = train_data
        self.test_data = test_data
        self.train_target = train_target
        self.test_target = test_target
        self.vocab_feature = vocab_feature
        self.class_c = class_c
        self.result = {}
        self.df_dict = {}
        self.df_dict_train = {}
        self.tf_dict = {}
        self.tfidf_dict = {}
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
            # self.df[word] = 0

        for i in range(len(self.train_data)):
            if self.train_target[i] == self.class_c:
                for word in self.doc_to_vocab[i].keys():
                    if word == "TOTAL_WORDS":
                        continue
                    self.result["CLASS_C"][word] += self.doc_to_vocab[i][word]
                    # self.df[word] += 1
            else:
                for word in self.doc_to_vocab[i].keys():
                    if word == "TOTAL_WORDS":
                        continue
                    self.result["NOT_CLASS_C"][word] += self.doc_to_vocab[i][word]
                    # self.df[word] += 1
        self.tf_dict = self.build_tf_dict_test()
        self.df_dict = self.build_df_dict()
        self.tfidf_dict = self.build_tfidf_dict()

        # total_word_class_c, total_word_not_class = 0, 0
        # for feature_word in self.vocab_feature:
        #     total_word_class_c += self.result["CLASS_C"][feature_word]
        #     total_word_not_class += self.result["NOT_CLASS_C"][feature_word]

        # for i in range(len(self.train_data)):
        #     self.doc_to_vocab_tfidf[i] = {}
        #     for word in self.doc_to_vocab[i].keys():
        #         if word == "TOTAL_WORDS":
        #             continue
        #         term_freq = self.doc_to_vocab[i][word] / self.doc_to_vocab[i]["TOTAL_WORDS"]
        #         tf_idf = np.log(total_doc + 1) - np.log(self.df[word] + 1) * term_freq
        #         self.doc_to_vocab_tfidf[i][word] = tf_idf

        # self.tfidf_result_build()

    def build_df_dict_train(self):
        df_dict = {"CLASS_C": {}, "NOT_CLASS_C": {}}

        for word in self.vocab_feature:
            df_dict["CLASS_C"][word] = 0
            df_dict["NOT_CLASS_C"][word] = 0
        for i in range(len(self.train_data)):
            for word in self.vocab_feature:
                if word in self.doc_to_vocab[i].keys():
                    if self.train_target[i] == 1:
                        df_dict["CLASS_C"][word] += 1
                    else:
                        df_dict["NOT_CLASS_C"][word] += 1
        self.df_dict_train = df_dict

    def build_tf_dict_test(self):
        # only contain the word in vocabulary feature list
        tf_dict = {}
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        for i in range(len(self.test_data)):
            tf_dict[i] = {}
            for word in tokenizer.tokenize(self.test_data[i]):
                word = word.lower()
                if word in self.vocab_feature:
                    if word in tf_dict[i].keys():
                        tf_dict[i][word] += 1
                    else:
                        tf_dict[i][word] = 1
        return tf_dict

    def build_df_dict(self):
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        df_dict = {}
        for word in self.vocab_feature:
            df_dict[word] = 0
        for data in self.test_data:
            for word_1 in df_dict.keys():
                for word_2 in tokenizer.tokenize(data):
                    if word_1.lower() == word_2.lower():
                        df_dict[word_1] += 1
                        break

        return df_dict

    def build_tfidf_dict(self):
        tfidf_dict = {}
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        n = len(self.test_data)  # number of document
        for i in range(len(self.test_data)):
            tfidf_dict[i] = {}
            total_words = 0
            for element in self.tf_dict[i].keys():
                total_words += self.tf_dict[i][element]
            for word in tokenizer.tokenize(self.test_data[i]):
                word = word.lower()
                if word in self.vocab_feature:
                    tf = self.tf_dict[i][word] / total_words
                    idf = np.log((n + 1)/(self.df_dict[word] + 1)) + 1
                    tf_idf = tf * idf
                    tfidf_dict[i][word] = tf_idf

        self.tfidf_dict = tfidf_dict

        return tfidf_dict

    def log_prob(self, class_selected, index, t_type, alpha=1):
        count = 0
        # alpha = 1
        output = np.log(self.result[class_selected]["DOC_OF_CLASS"]) - np.log(
             self.result["TOTAL_DOC"])  # log(pc) = log(Nc) - log(N)
        # output_1 = self.result[class_selected]["DOC_OF_CLASS"]/self.result["TOTAL_DOC"]
        # print("output 1 is ", output_1)
        total_word_class = 0
        for feature_word in self.vocab_feature:
            total_word_class += self.result[class_selected][feature_word]  # sum of Tct'

        # for word in tokenizer.tokenize(test_data):
        #     word = word.lower()
        #     if word in self.result[class_selected].keys():
        #         count += 1
        #         word_occurrence_in_class = self.result[class_selected][word] + 1  # Tct + 1
        #         curr_word_prob = np.log(word_occurrence_in_class) - np.log(
        #             total_word_class + len(self.vocab_feature))  # log P(d|C) = log(Tct + 1) - log(Tct' + |V|)
        #         output += curr_word_prob
        #     else:
        #         continue

        if t_type == "bernoulli":
            for feature_word in self.vocab_feature:
                cond_prob = (self.df_dict_train[class_selected][feature_word] + alpha) / (
                            self.result[class_selected]["DOC_OF_CLASS"] + alpha*2)
                # print(self.df_dict_train[class_selected][feature_word], self.result[class_selected]["DOC_OF_CLASS"])
                # print(feature_word, " cond is ", cond_prob)
                cond_prob_log = np.log(self.df_dict_train[class_selected][feature_word] + alpha) - np.log(
                    self.result[class_selected]["DOC_OF_CLASS"] + alpha * 2)
                if feature_word in self.tf_dict[index].keys():
                    output += cond_prob_log
                    # output_1 = cond_prob * output_1
                else:
                    output += np.log(1 - cond_prob)
                    # output_1 = output_1 * (1 - cond_prob)
        else:
            for word in self.tf_dict[index].keys():
                count += 1
                word_occurrence_in_class = self.result[class_selected][word] + alpha  # Tct + 1
                curr_word_prob = np.log(word_occurrence_in_class) - np.log(
                    total_word_class + len(self.vocab_feature) * alpha)  # log P(d|C) = log(Tct + 1) - log(Tct' + |V|)
                if t_type == "tf":
                    output += curr_word_prob * self.tf_dict[index][word]
                elif t_type == "tfidf":
                    output += curr_word_prob * self.tfidf_dict[index][word]
                elif t_type == "0-1":
                    if self.tf_dict[index][word] > 0:
                        output += curr_word_prob

        return output, count

    def predict(self, test_data, alpha=1):
        target_pred = []
        pred_prob = []  # list of probability that belongs to class_c
        for i in range(len(test_data)):
            data = test_data[i]
            best_pred = 1
            p_c, c0 = self.log_prob("CLASS_C", i, alpha)
            p_not_c, c1 = self.log_prob("NOT_CLASS_C", i, alpha)
            if p_not_c > p_c:
                best_pred = 0

            target_pred.append(best_pred)
            pred_prob.append(p_c / (p_c + p_not_c))

        return target_pred, pred_prob

    def predict_with_threshold(self, test_data, threshold, t_type, alpha=1):
        target_pred = []
        pred_prob = []  # list of probability that belongs to class_c
        for i in range(len(test_data)):
            best_pred = 0
            p_c, c0 = self.log_prob("CLASS_C", i, t_type, alpha)
            p_not_c, c1 = self.log_prob("NOT_CLASS_C", i, t_type, alpha)
            # print(p_c)
            # print(p_not_c)
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
        prec, reca, c, d = precision_recall_fscore_support(self.true_pred, pred_result, average='binary')
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
        return recall, precision, prec, reca
