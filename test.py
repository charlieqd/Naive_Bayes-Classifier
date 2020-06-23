import nltk

import BasicClassifier


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

class_c = 1
train_data = ["China, Beijing,China", "China,China, Shanghai", "China, Macao", "Tokyo,Japan,China"]
train_target = [1, 1, 1, 0]
test_data = ["Tokyo, Japan, China,China,China"]
test_target = [1]
vocab_list = ["china", "beijing", "shanghai", "macao", "tokyo", "japan"]
basic_classifier = BasicClassifier.BasicClassifier(train_data, train_target,
                                                   vocab_list, class_c, test_data, test_target, "basic")
basic_classifier.fit()
basic_classifier.build_df_dict_train()
predict_result = basic_classifier.predict_with_threshold(basic_classifier.test_data, 0.5, "bernoulli")
print("result is")
print(predict_result)

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