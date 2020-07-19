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


def load_data(class_c, num_of_feature, filename):
    news_data = NewsgroupData.Newsgroups_data(class_c)
    news_data.data_process()
    news_data.build_mi_dict()
    news_data.build_vocab_feature_list(num_of_feature)
    pickle.dump(news_data, file=open(filename, "wb"))


def run_base_classifier(data_name, class_c, threshold, plot, t_list, t_type, alpha, num_feature=100):
    data = pickle.load(open(data_name, "rb"))
    vocab_list = data.vocab_feature[0:num_feature]
    basic_classifier = BasicClassifier.BasicClassifier(data.train_data.data, data.train_data.target,
                                                       vocab_list, class_c, data.test_data.data, data.test_data.target,
                                                       "basic")
    basic_classifier.fit()
    if t_type == "tf":
        predict_result = basic_classifier.predict_with_threshold(basic_classifier.test_data, threshold, "tf", alpha)
    elif t_type == "tfidf":
        basic_classifier.tfidf_fit()
        predict_result = basic_classifier.predict_with_threshold(basic_classifier.test_data, threshold, "tfidf", alpha)
    elif t_type == "0-1":
        predict_result = basic_classifier.predict_with_threshold(basic_classifier.test_data, threshold, "0-1", alpha)
    elif t_type == "bernoulli":
        basic_classifier.bernoulli_fit()
        predict_result = basic_classifier.predict_with_threshold(basic_classifier.test_data, threshold, "bernoulli",
                                                                 alpha)
    predict_result = basic_classifier.pred_result
    print("Y_Test is ")
    print(basic_classifier.y_test)
    print("Predict result is ")
    print(predict_result)
    print("Acc is ", basic_classifier.accuracy())
    recall, precision = basic_classifier.estimation(basic_classifier.pred_result)
    print("Recall and Precision is ", recall, precision)
    basic_classifier.base_plot(plot)


def run_kn_classifier(data_name, class_c, threshold, plot, t_list, c_type, k_max, k, alpha=1, recall_ref=None,
                      precision_ref=None, num_feature=100, k_range=[], c_name=None):
    data = pickle.load(open(data_name, "rb"))
    vocab_list = data.vocab_feature[0:num_feature]
    kn_classifier = K_N_Voting.KnClassifier(data.train_data.data, data.train_data.target,
                                            vocab_list, class_c, data.test_data.data,
                                            data.test_data.target, "kn", k_max)
    kn_classifier.kn_fit()
    if c_type == "multi":
        kn_classifier.kn_voting(kn_classifier.test_data, "multi", alpha)
    elif c_type == "multi_2":
        kn_classifier.kn_voting(kn_classifier.test_data, "multi_2", alpha)
    elif c_type == "bernoulli":
        kn_classifier.bernoulli_fit()
        kn_classifier.kn_voting(kn_classifier.test_data, "bernoulli", alpha)
    elif c_type == "0-1":
        kn_classifier.kn_voting(kn_classifier.test_data, "0-1", alpha)
    prediction = kn_classifier.predict_kn(kn_classifier.test_data, threshold, k)
    print("Y_Test is ")
    print(kn_classifier.y_test)
    print("Predict result is ")
    print(prediction)
    print("acc is ", accuracy_score(kn_classifier.y_test, prediction))
    recall, precision = kn_classifier.estimation(prediction)
    print("precision and recall is ", precision, recall)
    if c_name is not None:
        pickle.dump(kn_classifier, file=open(c_name, "wb"))
    if plot == "plot_one":
        # Plot P-R curve for one K
        Plot.plot("kn", t_list, kn_classifier, k, 'K-N Voting Classifier with ' + c_type, 1, recall_ref, precision_ref)
    elif plot == "plot_all":
        # Plot P-R curve for all K
        kn_classifier.plot_kn(t_list, 1, None, None, k_range)


def run_saved_model(classifier_name):
    cla = pickle.load(open(classifier_name, "rb"))
    prediction = cla.predict_kn(cla.test_data, threshold, k)
    print("Y_test is ")
    print(cla.y_test)
    print("Prediction is ")
    print(prediction)
    print("Acc is ", accuracy_score(cla.y_test, prediction))
    recall, precision = cla.estimation(prediction)
    print("precision and recall is ", precision, recall)


if __name__ == '__main__':
    class_c = 7
    num_feature = 2000
    file_name = "data/new_data_" + str(num_feature) + ".pickle"  # add stemming

    threshold = 0.5
    alpha = 1
    data = pickle.load(open(file_name, "rb"))

    # run different base classifier
    run_base_classifier(file_name, class_c, threshold, 1, np.arange(0.01, 1.02, 0.02), "tf", alpha, 800)
    run_base_classifier(file_name, class_c, threshold, 1, np.arange(0.01, 1.02, 0.02), "tfidf", alpha, 800)
    run_base_classifier(file_name, class_c, threshold, 1, np.arange(0.01, 1.02, 0.02), "0-1", alpha, 1600)
    run_base_classifier(file_name, class_c, threshold, 1, np.arange(0.01, 1.02, 0.02), "bernoulli", alpha, 20)

    # Test for diff alpha
    Plot.plot_with_diff_features(data, np.arange(0.01, 1.02, 0.02), "bernoulli", feature_list, class_c)

    # run K-N Voting Classifier
    k_max, k, num_f = 100, 10, 200
    run_kn_classifier(file_name, class_c, threshold, "plot_one", np.arange(0.01, 1.02, 0.02), "multi", k_max, k,
                      alpha, None, None, num_f)

    # Or Load Saved Model
    classifier_name = "data/bernoulli_classifier_800"
    run_saved_model(classifier_name)
