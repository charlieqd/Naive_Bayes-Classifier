import nltk
import operator
import numpy as np
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.datasets import fetch_20newsgroups


class Newsgroups_data:
    def __init__(self, class_c):
        self.train_cat = ['alt.atheism', 'comp.graphics']
        # self.train_data = fetch_20newsgroups(subset='train',  categories=self.train_cat, remove=('headers', 'footers', 'quotes'))
        self.train_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        self.test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
        self.class_c = class_c
        self.vocab = []
        self.doc_to_word = {}
        self.vocab_to_mi = {}
        self.vocab_feature = []
        print(len(self.train_data.data), " ", len(self.test_data.data))
        print(self.train_data.target_names)

    def data_process(self):
        stopWords = [
            "max>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'", 'subject:', 'from:', 'date:',
            'newsgroups:', 'message-id:', 'lines:', 'path:', 'organization:',
            'would', 'writes:', 'references:', 'article', 'sender:', 'nntp-posting-host:', 'people',
            'university', 'think', 'xref:', 'cantaloupe.srv.cs.cmu.edu', 'could', 'distribution:', 'first',
            'anyone', 'world', 'really', 'since', 'right', 'believe', 'still']
        nltk.download('stopwords')
        puntuations = list(punctuation)
        stopWords += puntuations + stopwords.words("english")
        nltk.download("punkt")
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        ### b. Build a dict : class to its docs
        class_to_docs = {}
        classes = self.train_data.target_names
        for i in range(len(classes)):
            class_to_docs[i] = []

        for i in range(len(self.train_data.data)):
            class_to_docs[self.train_data.target[i]].append(self.train_data.data[i])

        ### c. Build vocabulary list which has all the words and a dict doc -> word
        vocab = []
        ps = PorterStemmer()
        for doc in self.train_data.data:
            for word in tokenizer.tokenize(doc):
                word = ps.stem(word)
                if word.lower() not in stopWords:
                    if word.lower() not in vocab:
                        vocab.append(word.lower())

        print("total words in train data is ", len(vocab))

        doc_to_word = {}
        for i in range(len(self.train_data.data)):
            doc_to_word[i] = []
            doc = self.train_data.data[i]
            for word in tokenizer.tokenize(doc):
                word = ps.stem(word)
                if word.lower() not in stopWords and word.lower() not in doc_to_word[i]:
                    doc_to_word[i].append(word.lower())
        print(len(doc_to_word))

        self.vocab = vocab
        self.doc_to_word = doc_to_word

    def build_mi_dict(self):
        index = 0
        for word in self.vocab:
            N_11, N_10, N_00, N_01, mutual_info = 0, 0, 0, 0, 0
            for i in range(len(self.doc_to_word)):
                if word in self.doc_to_word[i]:
                    if self.train_data.target[i] == self.class_c:
                        N_11 += 1
                    else:
                        N_10 += 1
                else:
                    if self.train_data.target[i] == self.class_c:
                        N_01 += 1
                    else:
                        N_00 += 1

            N = N_11 + N_10 + N_01 + N_00
            if N_11 != 0:
                mutual_info += np.log2(N * N_11 / ((N_11 + N_10) * (N_01 + N_11))) * N_11 / (N)
            if N_01 != 0:
                mutual_info += np.log2(N * N_01 / ((N_01 + N_00) * (N_01 + N_11))) * N_01 / (N)
            if N_10 != 0:
                mutual_info += np.log2(N * N_10 / ((N_11 + N_10) * (N_00 + N_10))) * N_10 / (N)
            if N_00 != 0:
                mutual_info += np.log2(N * N_00 / ((N_00 + N_01) * (N_10 + N_00))) * N_00 / (N)
            self.vocab_to_mi[word] = mutual_info

            index += 1

    def build_vocab_feature_list(self, n):
        vocab_sorted = sorted(self.vocab_to_mi.items(), key=operator.itemgetter(1), reverse=True)
        vocab_feature = []
        for word in vocab_sorted[0:n]:
            vocab_feature.append(word[0])

        print(len(vocab_feature))
        print(vocab_feature)

        self.vocab_feature = vocab_feature
