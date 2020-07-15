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


def run_base_classifier(data_name, class_c, threshold, plot, t_list, t_type, alpha, num_feature=100):
    data = pickle.load(open(data_name, "rb"))
    print("data vocab is ", len(data.vocab_feature))
    vocab_list = data.vocab_feature[0:num_feature]  # data.vocab_feature
    print(vocab_list)
    basic_classifier = BasicClassifier.BasicClassifier(data.train_data.data, data.train_data.target,
                                                       vocab_list, class_c, data.test_data.data, data.test_data.target,
                                                       "basic")
    basic_classifier.fit()
    # predict_result = basic_classifier.predict_with_threshold(basic_classifier.test_data, threshold, t_type, alpha)
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
        recall_list, precision_list = Plot.plot(t_type, t_list, basic_classifier, 0, t_type, 0)
    if plot == 1:
        plt.plot(recall_list, precision_list)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.grid()
        plt.legend(bbox_to_anchor=(1.5, 1))
        show()

    basic_classifier.base_plot(1)

    return recall_list, precision_list


def run_kn_classifier(data_name, class_c, threshold, plot, t_list, c_type, k_max, k, alpha=1, recall_ref=None,
                      precision_ref=None, num_feature=100, k_range=[], c_name="t"):
    data = pickle.load(open(data_name, "rb"))
    print("test is ", len(data.vocab_feature))
    vocab_list = data.vocab_feature[0:num_feature]
    print("Word using ", len(vocab_list))
    print(vocab_list)
    # kn_classifier = K_N_Voting.KnClassifier(data.train_data.data, data.train_data.target,
    #                                         data.vocab_feature, class_c, data.test_data.data,
    #                                         data.test_data.target, "kn", k_max)
    kn_classifier = K_N_Voting.KnClassifier(data.train_data.data, data.train_data.target,
                                            vocab_list, class_c, data.test_data.data,
                                            data.test_data.target, "kn", k_max)
    kn_classifier.kn_fit()
    print("finish fit")
    if c_type == "multi":
        kn_classifier.kn_voting(kn_classifier.test_data, "multi", alpha)
    elif c_type == "bernoulli":
        kn_classifier.bernoulli_fit()
        kn_classifier.kn_voting(kn_classifier.test_data, "bernoulli", alpha)
    elif c_type == "test":
        kn_classifier.kn_voting(kn_classifier.test_data, "test", alpha)
    elif c_type == "test1":
        kn_classifier.kn_voting(kn_classifier.test_data, "test1", alpha)
    # elif type == "tfidf":
    #     kn_classifier.build_df_dict_train()
    #     kn_classifier.kn_voting(kn_classifier.test_data, "tfidf")
    prediction = kn_classifier.predict_kn(kn_classifier.test_data, threshold, k)
    print("Truth is ")
    print(kn_classifier.y_test)
    print("Predict result is ")
    print(prediction)
    print("acc is ", accuracy_score(kn_classifier.y_test, prediction))
    recall, precision, c, d = kn_classifier.estimation(prediction)
    print("precision and recall is ", precision, recall)

    pickle.dump(kn_classifier, file=open(c_name, "wb"))
    if plot == "plot_one":
        print(t_list)
        Plot.plot("kn", t_list, kn_classifier, k, 'K-N Voting Classifier with ' + c_type, 1, recall_ref, precision_ref)
    elif plot == "plot_all":
        kn_classifier.plot_kn(t_list, 1, None, None, k_range)


if __name__ == '__main__':
    class_c = 7
    # class_c = 1
    num_feature = 2000
    # num_feature = 500
    file_name = "new_data_" + str(num_feature) + ".pickle"  # add stemming
    # data_1 = pickle.load(open(file_name, "rb"))
    # file_name = "data_" + str(num_feature) + " " + class_c + ".pickle"
    # load_data(class_c, num_feature, file_name)

    threshold = 0.5
    alpha = 1
    # feature_list = [5, 50, 100]
    data = pickle.load(open(file_name, "rb"))
    # print(data.class_c)
    # Plot.plot_with_diff_features(data, np.arange(0.01, 1.02, 0.02), "bernoulli", feature_list, class_c)
    # recall_r, precision_r = run_base_classifier(file_name, class_c, threshold, 1, np.arange(0.01, 1, 0.01), "bernoulli", alpha)
    # print(recall_r)
    # print(precision_r)
    # run_base_classifier(file_name, class_c, threshold, 1, np.arange(0.01, 1.02, 0.02), "tf", alpha, 800)
    # run_base_classifier(file_name, class_c, threshold, 1, np.arange(0.01, 1.02, 0.02), "tfidf", alpha, 1600)
    # run_base_classifier(file_name, class_c, threshold, 1, np.arange(0.01, 1.02, 0.02), "0-1", alpha, 1600)
    run_base_classifier(file_name, class_c, threshold, 1, np.arange(0.01, 1.02, 0.02), "bernoulli", alpha, 20)

    # k_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    # k_range = [1,2,5,10,20,30,40,50,80,100,150]
    # k_max = 180
    # k = 80
    # num_f = 200
    # run_kn_classifier(file_name, class_c, threshold, "plot_one", np.arange(0.01, 1.02, 0.02), "multi", k_max, k,
    #                   alpha, None, None, 1600, k_range)
    # filename_t = "bernoulli_classifier_800"
    # filename_t = "bernoulli_classifier_200"
    # run_kn_classifier(file_name, class_c, threshold, "no", np.arange(0.01, 1.02, 0.02), "bernoulli", k_max, k, alpha, None, None, num_f, k_range, filenamej_t)
    # cla = pickle.load(open(filename_t, "rb"))
    # prediction = cla.predict_kn(cla.test_data, threshold, k)
    # print("Truth is ")
    # print(cla.y_test)
    # print("Predict result is ")
    # print(prediction)
    # print("acc is ", accuracy_score(cla.y_test, prediction))
    # recall, precision, c, d = cla.estimation(prediction)
    # print("precision and recall is ", precision, recall)
    # print(cla.data_k_pred_prob_matrix)
    # recall_b=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8484848484848485,
    #          0.8484848484848485, 0.8409090909090909, 0.8358585858585859, 0.8181818181818182, 0.8055555555555556,
    #          0.7474747474747475, 0.6868686868686869, 0.6212121212121212, 0.5303030303030303, 0.4065656565656566,
    #          0.2474747474747475, 0.09090909090909091, 0.012626262626262626, 0.0025252525252525255]
    # precision_b=[0.0525756771109931, 0.0525756771109931, 0.0525756771109931, 0.0525756771109931, 0.0525756771109931,
    #          0.0525756771109931, 0.0525756771109931, 0.0525756771109931, 0.0525756771109931, 0.0525756771109931,
    #          0.0525756771109931, 0.0525756771109931, 0.0525756771109931, 0.0525756771109931, 0.0525756771109931,
    #          0.0525756771109931, 0.06244192529269652, 0.06342015855039637, 0.08157765801077903, 0.11114842175957018,
    #          0.16, 0.2484423676012461, 0.37948717948717947, 0.5074626865671642, 0.6180904522613065, 0.711864406779661,
    #          0.7931034482758621, 0.9074074074074074, 0.9473684210526315, 0.8333333333333334, 1.0]
    #
    # cla.plot_kn(np.arange(0.01, 1.02, 0.02), 1, recall_b, precision_b, k_range)

    # alpha_list = [0.000001]
    # Plot.plot_with_diff_alpha(data, np.arange(0.01, 1.02, 0.02), "tf", 10, alpha_list, class_c)
