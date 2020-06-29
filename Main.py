import sklearn
import matplotlib.pyplot as plt
import Plot
import pickle
import numpy as np
import NewsgroupData
import BasicClassifier
import K_N_Voting
from sklearn.metrics import accuracy_score
from pylab import figure, axes, pie, title, show
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import zeros


def load_data(class_c, num_of_feature, filename):
    news_data = NewsgroupData.Newsgroups_data(class_c)
    print("1")
    news_data.data_process()
    print("2")
    news_data.build_mi_dict()
    print("3")
    news_data.build_vocab_feature_list(num_of_feature)
    print("4")
    pickle.dump(news_data, file=open(filename, "wb"))


def run_base_classifier(data_name, class_c, threshold, plot, t_list, t_type, alpha):
    data = pickle.load(open(data_name, "rb"))
    print("data vocab is ", len(data.vocab_feature))
    vocab_list = data.vocab_feature[0:100]  # data.vocab_feature
    print(vocab_list)
    basic_classifier = BasicClassifier.BasicClassifier(data.train_data.data, data.train_data.target,
                                                       vocab_list, class_c, data.test_data.data, data.test_data.target,
                                                       "basic")
    basic_classifier.fit()

    if t_type == "tf":
        predict_result = basic_classifier.predict_with_threshold(basic_classifier.test_data, threshold, "tf", alpha)
    elif t_type == "tfidf":
        predict_result = basic_classifier.predict_with_threshold(basic_classifier.test_data, threshold, "tfidf", alpha)
    elif t_type == "0-1":
        predict_result = basic_classifier.predict_with_threshold(basic_classifier.test_data, threshold, "0-1", alpha)
    elif t_type == "bernoulli":
        predict_result = basic_classifier.predict_with_threshold(basic_classifier.test_data, threshold, "bernoulli",
                                                                 alpha)

    predict_result = basic_classifier.pred_result
    print("Truth is ")
    print(basic_classifier.y_test)
    print("Predict result is ")
    print(predict_result)
    print("acc is ", basic_classifier.accuracy())
    recall, precision, c, d = basic_classifier.estimation(basic_classifier.pred_result)
    print("recall and precisioon is ", recall, precision)
    print(d, c)
    recall_list, precision_list = [], []
    if plot == 1:
        print(t_list)
        recall_list, precision_list = Plot.plot(t_type, t_list, basic_classifier, 0, ' Multinomial NB with ' + t_type, 0)

    plt.plot(recall_list, precision_list)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.grid()
    plt.legend(bbox_to_anchor=(1.5, 1))
    show()

    return recall_list, precision_list


def run_kn_classifier(data_name, class_c, threshold, plot, t_list, c_type, k_max, k, alpha=1, recall_ref=None, precision_ref=None):
    data = pickle.load(open(data_name, "rb"))
    print("test is ", len(data.vocab_feature))
    vocab_list = data.vocab_feature[0:5]  # data.vocab_feature
    print(vocab_list)
    # kn_classifier = K_N_Voting.KnClassifier(data.train_data.data, data.train_data.target,
    #                                         data.vocab_feature, class_c, data.test_data.data,
    #                                         data.test_data.target, "kn", k_max)
    kn_classifier = K_N_Voting.KnClassifier(data.train_data.data, data.train_data.target,
                                            vocab_list, class_c, data.test_data.data,
                                            data.test_data.target, "kn", k_max)
    kn_classifier.kn_fit()
    if c_type == "multi":
        kn_classifier.kn_voting(kn_classifier.test_data, "multi", alpha)
    elif c_type == "bernoulli":
        kn_classifier.build_df_dict_train()
        kn_classifier.kn_voting(kn_classifier.test_data, "bernoulli", alpha)
    # elif type == "tfidf":
    #     kn_classifier.build_df_dict_train()
    #     kn_classifier.kn_voting(kn_classifier.test_data, "tfidf")
    prediction = kn_classifier.predict_kn(kn_classifier.test_data, threshold, k)
    print("Truth is ")
    print(kn_classifier.true_pred)
    print("Predict result is ")
    print(prediction)
    print("acc is ", accuracy_score(kn_classifier.true_pred, prediction))
    recall, precision, c, d = kn_classifier.estimation(prediction)
    print("precision and recall is ", precision, recall)
    if plot == "plot_one":
        print(t_list)
        Plot.plot("kn", t_list, kn_classifier, k, 'K-N Voting Classifier with ' + c_type, 1, recall_ref, precision_ref)
    elif plot == "plot_all":
        kn_classifier.plot_kn(t_list)


if __name__ == '__main__':
    # class_c = 7
    class_c = 1
    num_feature = 500
    file_name = "new_data_" + str(num_feature) + ".pickle"  # add stemming
    # data_1 = pickle.load(open(file_name, "rb"))
    # file_name = "data_" + str(num_feature) + " " + class_c + ".pickle"
    # load_data(class_c, num_feature, file_name)
    threshold = 0.5
    alpha = 1
    # recall_r, precision_r = run_base_classifier(file_name, class_c, threshold, 1, np.arange(0.01, 1, 0.01), "bernoulli", alpha)
    # print(recall_r)
    # print(precision_r)
    # run_base_classifier(file_name, class_c, threshold, 1, np.arange(0.01, 1, 0.01), "tf", alpha)
    # run_base_classifier(file_name, class_c, threshold, 1, np.arange(0.01, 1, 0.01), "tfidf", alpha)
    run_base_classifier(file_name, class_c, threshold, 1, np.arange(0.01, 1, 0.01), "bernoulli", alpha)
    # k_max = 10
    # k = 1
    # run_kn_classifier(file_name, class_c, threshold, "no_plot", np.arange(0.01, 1, 0.01), "bernoulli", k_max, k)
    # feature_number_list = [5, 10, 20, 50, 100, 200]
    # Plot.plot_with_diff_features(data_1, np.arange(0.01, 1, 0.01), "bernoulli", feature_number_list)
