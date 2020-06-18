import Plot
import pickle
import numpy as np
import NewsgroupData
import BasicClassifier
import K_N_Voting
from sklearn.metrics import accuracy_score
from numpy import zeros


def load_data(class_c, num_of_feature, filename):
    news_data = NewsgroupData.Newsgroups_data(class_c)
    news_data.data_process()
    news_data.build_mi_dict()
    news_data.build_vocab_feature_list(num_of_feature)
    pickle.dump(news_data, file=open(filename, "wb"))


def run_base_classifier(data_name, class_c, threshold, plot, t_list, type):
    data = pickle.load(open(data_name, "rb"))
    print("test is ", len(data.vocab_feature))
    basic_classifier = BasicClassifier.BasicClassifier(data.train_data.data, data.train_data.target,
                                                       data.vocab_feature, class_c, data.test_data, "basic")
    basic_classifier.fit()

    if type == "tf":
        predict_result = basic_classifier.predict_with_threshold(basic_classifier.test_data, threshold)
    elif type == "tfidf":
        predict_result = basic_classifier.predict_with_threshold_tfidf(basic_classifier.test_data, threshold)

    # predict_result = basic_classifier.pred_result
    print("Truth is ")
    print(basic_classifier.true_pred)
    print("Predict result is ")
    print(predict_result)
    print("acc is ", basic_classifier.accuracy())
    recall, precision = basic_classifier.estimation(basic_classifier.pred_result)
    print("recall and precisioon is ", recall, precision)

    if plot == 1:
        print(t_list)
        Plot.plot(type, t_list, basic_classifier, 0, ' Multinomial NB with ' + type)


def run_kn_classifier(data_name, class_c, threshold, plot, t_list, type, k_max, k):
    data = pickle.load(open(data_name, "rb"))
    print("test is ", len(data.vocab_feature))
    kn_classifier = K_N_Voting.KnClassifier(data.train_data.data, data.train_data.target,
                                            data.vocab_feature, class_c, data.test_data, "kn", k_max)
    kn_classifier.kn_fit()
    if type == "tf":
        kn_classifier.kn_voting(kn_classifier.test_data, "regular")
    elif type == "tfidf":
        kn_classifier.kn_voting(kn_classifier.test_data, "tfidf")
    prediction = kn_classifier.predict_kn(kn_classifier.test_data, threshold, k)
    print("Truth is ")
    print(kn_classifier.true_pred)
    print("Predict result is ")
    print(prediction)
    print("acc is ", accuracy_score(kn_classifier.true_pred, prediction))
    recall, precision = kn_classifier.estimation(prediction)
    print("precision and recall is ", precision, recall)
    if plot == "plot_one":
        print(t_list)
        Plot.plot("kn", t_list, kn_classifier, k, 'K-N Voting Classifier with ' + type)
    elif plot == "plot_all":
        kn_classifier.plot_kn(t_list)


if __name__ == '__main__':
    class_c = 1
    num_feature = 100
    file_name = "data_" + str(num_feature) + ".pickle"
    # load_data(class_c, num_feature, file_name)
    threshold = 0.5
    run_base_classifier(file_name, class_c, threshold, 0, np.arange(0.01, 1, 0.01), "tf")
    k_max = 5
    k = 1
    run_kn_classifier(file_name, class_c, threshold, "no_plot", np.arange(0.01, 1, 0.01), "tf", k_max, k)
