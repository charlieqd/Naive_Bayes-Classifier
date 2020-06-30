import numpy as np
import nltk
import operator
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


class BasicClassifier:
    def __init__(self, train_data, train_target, vocab_feature, class_c, test_data, test_target, name):
        self.name = name
        self.doc_to_vocab = {}  # tf_dict_train
        self.train_data = train_data
        self.test_data = test_data
        self.train_target = train_target
        self.test_target = test_target
        self.vocab_feature = vocab_feature
        self.class_c = class_c
        self.result = {}  # tf result
        self.tfidf_result = {}  # tfidf result
        self.df_dict_train = {}
        self.tfidf_dict_train = {}
        self.df_dict_test = {}
        self.tf_dict_test = {}
        self.tfidf_dict_test = {}
        self.bernoulli_result = {}
        self.y_test = []
        self.pred_result = []
        self.pred_prob = []

    def truth_build(self):
        for i in range(len(self.test_target)):
            if self.test_target[i] == self.class_c:
                self.y_test.append(1)
            else:
                self.y_test.append(0)

    # def truth_build(self):
    #     for i in range(len(self.test_target)):
    #         if self.test_target[i] == self.class_c:
    #             self.y_test.append(0)
    #         else:
    #             self.y_test.append(1)

    def dict_build(self):
        self.truth_build()
        nltk.download("punkt")
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        ps = PorterStemmer()
        for i in range(len(self.train_data)):
            total_word = 0
            self.doc_to_vocab[i] = {}
            cur_doc = self.train_data[i]
            for word in tokenizer.tokenize(cur_doc):
                word = ps.stem(word)
                if word.lower() in self.vocab_feature:
                    total_word += 1
                    if word.lower() in self.doc_to_vocab[i].keys():
                        self.doc_to_vocab[i][word.lower()] += 1
                    else:
                        self.doc_to_vocab[i][word.lower()] = 1
            self.doc_to_vocab[i]["TOTAL_WORDS"] = total_word

    def fit(self):
        self.dict_build()
        self.build_df_dict_train()
        self.build_tfidf_dict_train()
        total_doc = len(self.train_data)
        self.result = {"TOTAL_DOC": total_doc, "CLASS_C": {}, "NOT_CLASS_C": {}}
        self.tfidf_result = {"CLASS_C": {}, "NOT_CLASS_C": {}}
        classes, doc_counters = np.unique(self.train_target, return_counts=True)

        self.result["CLASS_C"]["DOC_OF_CLASS"] = doc_counters[self.class_c]
        self.result["NOT_CLASS_C"]["DOC_OF_CLASS"] = total_doc - doc_counters[self.class_c]

        for word in self.vocab_feature:
            self.result["CLASS_C"][word] = 0
            self.result["NOT_CLASS_C"][word] = 0
            self.tfidf_result["CLASS_C"][word] = 0
            self.tfidf_result["NOT_CLASS_C"][word] = 0
        for i in range(len(self.train_data)):
            if self.train_target[i] == self.class_c:
                for word in self.doc_to_vocab[i].keys():
                    if word == "TOTAL_WORDS":
                        continue
                    self.result["CLASS_C"][word] += self.doc_to_vocab[i][word]
                    self.tfidf_result["CLASS_C"][word] += self.tfidf_dict_train[i][word]
            else:
                for word in self.doc_to_vocab[i].keys():
                    if word == "TOTAL_WORDS":
                        continue
                    self.result["NOT_CLASS_C"][word] += self.doc_to_vocab[i][word]
                    self.tfidf_result["NOT_CLASS_C"][word] += self.tfidf_dict_train[i][word]

        self.tf_dict_test = self.build_tf_dict_test()
        self.df_dict_test = self.build_df_dict_test()
        self.tfidf_dict_test = self.build_tfidf_dict_test()
        self.build_bernoulli_result()

    def build_df_dict_train(self):
        df_dict = {}
        for word in self.vocab_feature:
            df_dict[word] = 0
        for word_1 in self.vocab_feature:
            for i in range(len(self.doc_to_vocab)):
                if word_1 in self.doc_to_vocab[i].keys():
                    df_dict[word_1] += 1

        self.df_dict_train = df_dict

    def build_tfidf_dict_train(self):
        tfidf_dict = {}
        n = len(self.train_data)  # number of document
        for i in range(len(self.train_data)):
            tfidf_dict[i] = {}
            # total_words = 0
            # for element in self.doc_to_vocab[i].keys():
            #     if element == "TOTAL_WORDS":
            #         continue
            #     total_words += self.doc_to_vocab[i][element]
            for word in self.doc_to_vocab[i].keys():
                if word == "TOTAL_WORDS":
                    continue
                tf = self.doc_to_vocab[i][word]
                idf = np.log((n + 1) / (self.df_dict_train[word] + 1)) + 1
                tf_idf = tf * idf
                tfidf_dict[i][word] = tf_idf
        self.tfidf_dict_train = tfidf_dict
        return tfidf_dict

    def build_tf_dict_test(self):
        # only contain the word in vocabulary feature list
        tf_dict = {}
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        ps = PorterStemmer()
        for i in range(len(self.test_data)):
            tf_dict[i] = {}
            for word in tokenizer.tokenize(self.test_data[i]):
                word = ps.stem(word)
                word = word.lower()
                if word in self.vocab_feature:
                    if word in tf_dict[i].keys():
                        tf_dict[i][word] += 1
                    else:
                        tf_dict[i][word] = 1
        return tf_dict

    def build_df_dict_test(self):
        df_dict = {}
        for element in self.vocab_feature:
            df_dict[element] = 0
        for word in df_dict.keys():
            for i in range(len(self.tf_dict_test)):
                if word in self.tf_dict_test[i].keys():
                    df_dict[word] += 1
        return df_dict

    def build_tfidf_dict_test(self):
        tfidf_dict = {}
        n = len(self.test_data)  # number of document
        for i in range(len(self.test_data)):
            tfidf_dict[i] = {}
            # total_words = 0
            # for element in self.tf_dict_test[i].keys():
            #     total_words += self.tf_dict_test[i][element]
            for word in self.tf_dict_test[i].keys():
                # tf = self.tf_dict_test[i][word] / total_words
                tf = self.tf_dict_test[i][word]
                idf = np.log((n + 1) / (self.df_dict_test[word] + 1)) + 1
                tf_idf = tf * idf
                tfidf_dict[i][word] = tf_idf

        return tfidf_dict

    def build_bernoulli_result(self):
        bernoulli_result = {"CLASS_C": {}, "NOT_CLASS_C": {}}
        for element in self.vocab_feature:
            bernoulli_result["CLASS_C"][element] = 0
            bernoulli_result["NOT_CLASS_C"][element] = 0
        for i in range(len(self.doc_to_vocab)):
            for word in self.doc_to_vocab[i].keys():
                if word == "TOTAL_WORDS":
                    continue
                if self.train_target[i] == self.class_c:
                    bernoulli_result["CLASS_C"][word] += 1
                else:
                    bernoulli_result["NOT_CLASS_C"][word] += 1

        self.bernoulli_result = bernoulli_result

    def log_prob(self, class_selected, index, t_type, alpha=1):
        count = 0
        output = np.log(self.result[class_selected]["DOC_OF_CLASS"]) - np.log(
            self.result["TOTAL_DOC"])  # log(pc) = log(Nc) - log(N)
        total_word_class_tf = 0
        total_word_class_tfidf = 0
        # print("Total doc is ", self.result[class_selected]["DOC_OF_CLASS"])
        for feature_word in self.vocab_feature:
            total_word_class_tf += self.result[class_selected][feature_word]  # sum of Tct'
            total_word_class_tfidf += self.tfidf_result[class_selected][feature_word]
        if t_type == "bernoulli":
            # i = 0
            # print(self.test_data[index])
            for feature_word in self.vocab_feature:
                # print("feature word is ", feature_word)
                cond_prob = (self.bernoulli_result[class_selected][feature_word] + alpha) / (
                        self.result[class_selected]["DOC_OF_CLASS"] + alpha * 2)
                # print("in class that;s many document has that word", self.bernoulli_result[class_selected][feature_word])
                # print(self.df_dict_train[class_selected][feature_word], self.result[class_selected]["DOC_OF_CLASS"])
                # print(feature_word, " cond is ", cond_prob)
                cond_prob_log = np.log(self.bernoulli_result[class_selected][feature_word] + alpha) - np.log(
                    self.result[class_selected]["DOC_OF_CLASS"] + alpha * 2)
                if feature_word in self.tf_dict_test[index].keys():
                    # i += 1
                    output += cond_prob_log
                    # print("in ", cond_prob)
                    # output_1 = cond_prob * output_1
                else:
                    output += np.log(1 - cond_prob)
                    # print("not in ", 1 - cond_prob)
                    # output_1 = output_1 * (1 - cond_prob)
            # print("how many words ", i)
        else:
            for word in self.tf_dict_test[index].keys():
                count += 1
                if t_type == "tf":
                    word_occurrence_in_class = self.result[class_selected][word] + alpha  # Tct + 1
                    curr_word_prob = np.log(word_occurrence_in_class) - np.log(total_word_class_tf + len(
                        self.vocab_feature) * alpha)  # log P(d|C) = log(Tct + 1) - log(Tct' + |V|)
                    output += curr_word_prob * self.tf_dict_test[index][word]
                elif t_type == "tfidf":
                    word_occurrence_in_class = self.tfidf_result[class_selected][word] + alpha  # Tct + 1
                    curr_word_prob = np.log(word_occurrence_in_class) - np.log(total_word_class_tfidf + len(
                        self.vocab_feature) * alpha)  # log P(d|C) = log(Tct + 1) - log(Tct' + |V|)
                    output += curr_word_prob * self.tfidf_dict_test[index][word]
                elif t_type == "0-1":
                    if self.tf_dict_test[index][word] > 0:
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

            p_c, c0 = self.log_prob("CLASS_C", i, t_type, alpha)
            p_not_c, c1 = self.log_prob("NOT_CLASS_C", i, t_type, alpha)
            # print(p_c)
            # print(p_not_c)
            best_pred = 0
            t = 1 - threshold
            if p_c / p_not_c < np.log(threshold) / np.log(t):
                best_pred = 1

            # best_pred = 0
            # if p_c > p_not_c:
            #     best_pred = 1


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
        return accuracy_score(self.y_test, self.pred_result)

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

        for i in range(len(self.y_test)):
            if self.y_test[i] == 1:
                total_actual_positive += 1
                if pred_result[i] == 1:
                    true_positive += 1

        recall = true_positive / total_actual_positive
        precision = true_positive / total_pred_positive
        prec, reca, c, d = precision_recall_fscore_support(self.y_test, pred_result, average='binary')
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
