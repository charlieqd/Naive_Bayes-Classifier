import nltk
import pickle
import numpy as np
import NewsgroupData
import BasicClassifier
import K_N_Voting
from sklearn.metrics import accuracy_score


def build_df_dict(data_1, vocab):
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    df_dict = {}
    for word in vocab:
        df_dict[word] = 0
    for data in data_1:
        print("1")
        for word_1 in vocab:
            print("word1 is ", word_1)
            for word_2 in tokenizer.tokenize(data):
                print("word2 is", word_2)
                if word_1.lower() == word_2.lower():
                    print("match")
                    df_dict[word_1] += 1
                    break
    print(df_dict)


# data = ["Hi this is john, who is u", "hi, nice to meet you"]
# vocab = ["hi", "this", "is", "who", "nice", "meet", "sad"]
# build_df_dict(data, vocab)
#
# class_c = 1
# train_data = ["China, Beijing,China", "China,China, Shanghai", "China, Macao", "Tokyo,Japan,China"]
# train_target = [1, 1, 1, 0]
# test_data = ["Tokyo, Japan, China,China,China"]
# test_target = [1]
# vocab_list = ["china", "beijing", "shanghai", "macao", "tokyo", "japan"]
# basic_classifier = BasicClassifier.BasicClassifier(train_data, train_target,
#                                                    vocab_list, class_c, test_data, test_target, "basic")
# basic_classifier.fit()
# basic_classifier.build_df_dict_train()
# predict_result = basic_classifier.predict_with_threshold(basic_classifier.test_data, 0.5, "bernoulli")
# print("result is")
# print(predict_result)

# data = pickle.load(open(file_name, "rb"))
    # vectorizer = TfidfVectorizer()
    # X = vectorizer.fit_transform(data.doc_)
    # print(data.train_data.data[0])
    # print(X.shape)
    # print(vectorizer.get_feature_names()[3])

# corpus = [
    #          'The gas.',
    #          'The.'
    #     ]
    # n = len(corpus)
    # print(n)
    # vocab = ["The", "gas"]
    # dict_tf = {0: {}, 1:{}}
    # dict_tf[0]["The"] = 1/2
    # dict_tf[0]["gas"] = 1/2
    # dict_tf[1]["The"] = 1
    # dict_tf[1]["gas"] = 0
    #
    # dict_df = {"The": 2, "gas": 1}
    # dict_tfidf = {}
    #
    # for i in range(len(corpus)):
    #     dict_tfidf[i] = {}
    #     for word in vocab:
    #         print(word)
    #         tf = dict_tf[i][word]
    #         df = dict_df[word]
    #         ans = tf * (np.log(3/(df+1))+1)
    #         ans = sklearn.preprocessing.normalize(ans)
    #         dict_tfidf[i][word] = ans
    # print("the answer for me is ")
    # print(dict_tfidf)
    #
    # vectorizer = TfidfVectorizer()
    # X = vectorizer.fit_transform(corpus)
    # print(vectorizer.get_feature_names())
    # print(X.shape)
    # print(X)

class_c = 1
num_feature = 200
k_max = 5
k = 1
threshold = 0.5
file_name = "data_" + str(num_feature) + ".pickle"
data = pickle.load(open(file_name, "rb"))
print("test is ", len(data.vocab_feature))
vocab_list = data.vocab_feature[0:5]  # data.vocab_feature
print(vocab_list)
test_data = data.test_data.data[9:10]
test_target = data.test_data.target[9:10]
kn_classifier = K_N_Voting.KnClassifier(data.train_data.data, data.train_data.target,
                                        vocab_list, class_c, test_data,
                                        test_target, "kn", k_max)
kn_classifier.kn_fit()
kn_classifier.build_df_dict_train()
kn_classifier.kn_voting(kn_classifier.test_data, "bernoulli")
prediction = kn_classifier.predict_kn(kn_classifier.test_data, threshold, k)
print("Truth is ")
print(kn_classifier.true_pred)
print("Predict result is ")
print(prediction)
unique, counts = np.unique(prediction, return_counts=True)
print(unique)
print(counts)
print("acc is ", accuracy_score(kn_classifier.true_pred, prediction))
recall, precision, c, d = kn_classifier.estimation(prediction)
print("precision and recall is ", precision, recall)