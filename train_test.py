import sys
import os
import numpy as np
import librosa
from feature_summary import *
#from feature_summary_CODEBOOK import *

from write_test_result import *

from sklearn.mixture import GaussianMixture as gmm
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier


def train_model(train_X, train_Y, valid_X, valid_Y, hyper_param1):
    # Choose a classifier
    clf = SGDClassifier(verbose=0, loss="hinge", alpha=hyper_param1, max_iter=1000, penalty="l2", random_state=0)
    #clf = SGDClassifier(verbose=0, loss="perceptron", alpha=hyper_param1, max_iter=1000, penalty="l2", random_state=0)
    #clf = KNeighborsClassifier(n_neighbors=2)

    # train
    clf.fit(train_X, train_Y)

    # validation
    valid_Y_hat = clf.predict(valid_X)

    accuracy = np.sum((valid_Y_hat == valid_Y)) / 200.0 * 100.0
    #print 'validation accuracy = ' + str(accuracy) + ' %'

    return clf, accuracy

def train_model_gmm(train_X):#, valid_X, valid_Y, f):
    n_components = 1

    models = [gmm(n_components=n_components),  # 0
              gmm(n_components=n_components),  # 1
              gmm(n_components=n_components),  # 2
              gmm(n_components=n_components),  # 3
              gmm(n_components=n_components),  # 4
              gmm(n_components=n_components),  # 5
              gmm(n_components=n_components),  # 6
              gmm(n_components=n_components),  # 7
              gmm(n_components=n_components),  # 8
              gmm(n_components=n_components)]  # 9

    for i in range(0, 10):
        models[i].fit(train_X[i * 100:(i + 1) * 100])


    return models

if __name__ == '__main__':
    f = open('find_cluster.txt', 'w')

    # main for simple GMM

    # load data
    train_X = load_data('train')
    valid_X = load_data('valid')
    test_X = load_data('test')
    #print(train_X.shape)

    # label generation
    cls = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    train_Y = np.repeat(cls, 100)
    valid_Y = np.repeat(cls, 20)
    test_Y = np.repeat(cls, 20)

    # feature normalizaiton
    train_X = train_X.T
    train_X_mean = np.mean(train_X, axis=0)
    train_X = train_X - train_X_mean
    train_X_std = np.std(train_X, axis=0)
    train_X = train_X / (train_X_std + 1e-5)

    valid_X = valid_X.T
    valid_X = valid_X - train_X_mean
    valid_X = valid_X / (train_X_std + 1e-5)

    train_X = np.append(train_X, valid_X, axis=0)

    # training model
    final_model = train_model_gmm(train_X)

    test_X = test_X.T
    test_X = test_X - train_X_mean
    test_X = test_X / (train_X_std + 1e-5)

    result = []
    for i in range(0, 10):
        result.append(final_model[i].score_samples(test_X))
    result = np.array(result)

    test_Y_hat = np.argmax(result, axis=0) + 1

    accuracy = np.sum((test_Y_hat == test_Y)) / 200.0 * 100.0
    print('test accuracy = ' + str(accuracy) + ' %' + '\n')

    inst = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'vocal']
    print('~class accuracy~' + '\n')
    class_acc = []
    for i in range(0, 10):
        acc = np.sum(test_Y_hat[20 * i:20 * (i + 1)] == test_Y[20 * i:20 * (i + 1)]) / 20.0 * 100.0
        class_acc.append(acc)
        print(inst[i] + '    ' + str(acc) + '%' + '\n')

        print(str(test_Y[20 * i:20 * (i + 1)]) + '\n')
        print(str(test_Y_hat[20 * i:20 * (i + 1)]) + '\n')
    """
    
    # main for gmm + codebook
    
    n = [5,20,23,37]
    gmm_nums = [1]
    # load data
    train_data = load_data('train')
    for n_cluster in n:
        f.write("============================================" + '\n')
        f.write("n_cluster: " + str(n_cluster) + '\n')

        kmeans = generate_codebook(train_data, n_cluster)
        train_X = generate_new_data('train', train_data, kmeans, n_cluster)
        f.write(str(train_X.shape) + '\n')

        valid_data = load_data('valid')
        valid_X = generate_new_data('valid', valid_data, kmeans, n_cluster)
        f.write(str(valid_X.shape) + '\n')


        test_data = load_data('test')
        test_X = generate_new_data('test', test_data, kmeans, n_cluster)
        f.write(str(test_X.shape) + '\n')

        # label generation
        cls = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        train_Y = np.repeat(cls, 100)
        valid_Y = np.repeat(cls, 20)
        test_Y = np.repeat(cls, 20)

        # feature normalizaiton
        train_X_mean = np.mean(train_X, axis=0)
        train_X = train_X - train_X_mean
        train_X_std = np.std(train_X, axis=0)
        train_X = train_X / (train_X_std + 1e-5)

        valid_X = valid_X - train_X_mean
        valid_X = valid_X / (train_X_std + 1e-5)



        # training model
        nn = [1,2,3,4,5,6,7,8,9,10]
        alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]

        model = []
        valid_acc = []
        for k in nn:
            f.write("knn: " + str(k) + '\n')
            for a in alphas:
                models, acc = train_model(train_X, train_Y, valid_X, valid_Y, a, k)
                model.append(models)
                valid_acc.append(acc)
            # choose the model that achieve the best validation accuracy
            final_model = model[np.argmax(valid_acc)]

            # now, evaluate the model with the test set
            test_X = test_X - train_X_mean
            test_X = test_X / (train_X_std + 1e-5)

            test_Y_hat = final_model.predict(test_X)

            accuracy = np.sum((test_Y_hat == test_Y)) / 200.0 * 100.0
            f.write('test accuracy = ' + str(accuracy) + ' %' + '\n')

            inst = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'vocal']
            f.write('~class accuracy~' + '\n')
            class_acc = []
            for i in range(0, 10):
                acc = np.sum(test_Y_hat[20 * i:20 * (i + 1)] == test_Y[20 * i:20 * (i + 1)]) / 20.0 * 100.0
                class_acc.append(acc)
                f.write(inst[i] + '    ' + str(acc) + '%' + '\n')
    """
    """
    # main for simple SVM or kNN
    
    # load data
    train_X = load_data('train')
    valid_X = load_data('valid')
    test_X = load_data('test')
    #print(train_X.shape)

    # label generation
    cls = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    train_Y = np.repeat(cls, 100)
    valid_Y = np.repeat(cls, 20)
    test_Y = np.repeat(cls, 20)

    # feature normalizaiton
    train_X = train_X.T
    train_X_mean = np.mean(train_X, axis=0)
    train_X = train_X - train_X_mean
    train_X_std = np.std(train_X, axis=0)
    train_X = train_X / (train_X_std + 1e-5)

    valid_X = valid_X.T
    valid_X = valid_X - train_X_mean
    valid_X = valid_X / (train_X_std + 1e-5)

    # training model
    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]

    model = []
    valid_acc = []
    for a in alphas:
        clf, acc = train_model(train_X, train_Y, valid_X, valid_Y, a)
        # clf, acc = train_model(train_X, train_Y, train_X, train_Y, a)
        model.append(clf)
        valid_acc.append(acc)

    # choose the model that achieve the best validation accuracy
    final_model = model[np.argmax(valid_acc)]

    # now, evaluate the model with the test set
    test_X = test_X.T
    test_X = test_X - train_X_mean
    test_X = test_X / (train_X_std + 1e-5)
    test_Y_hat = final_model.predict(test_X)

    accuracy = np.sum((test_Y_hat == test_Y)) / 200.0 * 100.0
    print 'test accuracy = ' + str(accuracy) + ' %'

    inst = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'vocal']
    print '~class accuracy~'
    class_acc = []
    for i in range(0,10):
        acc = np.sum(test_Y_hat[20*i:20*(i+1)] == test_Y[20*i:20*(i+1)])/20.0 * 100.0
        class_acc.append(acc)
        print inst[i] + '    ' + str(acc) + '%'
    """
"""
    write_result(6,
                 'KNeighborsClassifier',
                 'MFCC_DIM * 3 + ZCR_DIM + RMS_DIM + SPECTRAL_CENTROID_DIM*5 + SPECTRAL_FLATNESS_DIM  + SPECTRAL_BANDWIDTH_DIM + ATTACK_TIME_PATH',
                 accuracy,
                 class_acc,
                 'knn=2\n')
"""
# test results

# test 1
# clf = SGDClassifier(verbose=0, loss="hinge", alpha=hyper_param1, max_iter=1000, penalty="l2", random_state=0)
# mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
# accuracy = 43.0%

# test 2
# clf = SGDClassifier(verbose=0, loss="hinge", alpha=hyper_param1, max_iter=1000, penalty="l2", random_state=0)
# mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
# accuracy = 42.0%

# test 3
# clf = SGDClassifier(verbose=0, loss="hinge", alpha=hyper_param1, max_iter=1000, penalty="l2", random_state=0)
# mfcc = given method 2
# accuracy = 47.5%

# test 4
# clf = SGDClassifier(verbose=0, loss="hinge", alpha=hyper_param1, max_iter=1000, penalty="l2", random_state=0)
# mfcc = given method 2
# normalize audio its 
# accuracy = 47.5%

# test 5
# add mfcc_delta and mfcc_delta2 in test 3
# accuracy = 57.49%

# test 6
# add zcr in test 5
# accuracy = 57.99%

# test 7
# add rms in test 6
# accuracy = 60.5%

# test 8
# add spectral centroid in test 6
# accuracy = 62.0%

# test 9
# add spectral flatness in test 6
# accuracy = 66.5%

# test 10
# add spectral rolloff in test 6
# accuracy = 66.0%

# test 11
# use knn with n=3 in test 9
# accuracy = 69%

# test 11
# use knn with n=2 in test 9
# accuracy = 70.5%

# test 12
# mmm...
# accuracy = 73%