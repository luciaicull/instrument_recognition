import sys
import os
import numpy as np
import librosa
from feature_summary_CODEBOOK import *
from write_test_result import *

from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC as svc
from sklearn.mixture import GaussianMixture as gmm
from sklearn.cluster import KMeans


def train_model_gmm(train_X, train_Y, valid_X, valid_Y, hyper_param1, gmm_num):
    n_components = gmm_num
    """
    models = []
    for i in range(0,10):
        models.append(gmm(n_components=))
    """
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
        # print(train_X[i*100:(i+1)*100].shape)
    result = []
    for i in range(0, 10):
        result.append(models[i].score_samples(valid_X))
    result = np.array(result)
    valid_Y_hat = np.argmax(result, axis=0) + 1

    accuracy = np.sum((valid_Y_hat == valid_Y)) / 200.0 * 100.0
    f.write('validation accuracy = ' + str(accuracy) + ' %')

    return models, accuracy


if __name__ == '__main__':

    f = open('find_cluster.txt','w')

    """
    main for gmm + codebook
    """
    n = [37]
    gmm_nums = [1]
    # load data
    train_data = load_data('train')
    for n_cluster in n:
        f.write("============================================"+'\n')
        f.write("n_cluster: " + str(n_cluster) + '\n')

        kmeans = generate_codebook(train_data, n_cluster)
        train_X = generate_new_data('train', train_data, kmeans, n_cluster)
        f.write(str(train_X.shape)+'\n')

        valid_data = load_data('valid')
        valid_X = generate_new_data('valid', valid_data, kmeans, n_cluster)
        f.write(str(valid_X.shape)+'\n')

        train_X = np.append(train_X, valid_X, axis=0)
        f.write(str(train_X.shape)+'\n')

        test_data = load_data('test')
        test_X = generate_new_data('test', test_data, kmeans, n_cluster)
        f.write(str(test_X.shape)+'\n')

        # label generation
        cls = np.array([1,2,3,4,5,6,7,8,9,10])
        train_Y = np.repeat(cls, 100)
        valid_Y = np.repeat(cls, 20)
        test_Y = np.repeat(cls, 20)


        # feature normalizaiton
        train_X_mean = np.mean(train_X, axis=0)
        train_X = train_X - train_X_mean
        train_X_std = np.std(train_X, axis=0)
        train_X = train_X / (train_X_std + 1e-5)

        valid_X = valid_X - train_X_mean
        valid_X = valid_X/(train_X_std + 1e-5)

        for gmm_num in gmm_nums:

            f.write("gmm_num: " + str(gmm_num) + '\n')
            # training model
            final_model, acc = train_model_gmm(train_X, train_Y, valid_X, valid_Y, 0.1, gmm_num)
            """
            alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    
            model = []
            valid_acc = []
            
            for a in alphas:
                models, acc = train_model_gmm(train_X, train_Y, valid_X, valid_Y, a, gmm_num)
                model.append(models)
                valid_acc.append(acc)
    
            # choose the model that achieve the best validation accuracy
            final_model = model[np.argmax(valid_acc)]
            """

            # now, evaluate the model with the test set
            """
            test_X = test_X - train_X_mean
            test_X = test_X/(train_X_std + 1e-5)
            """
            test_X_mean = np.mean(test_X, axis=0)
            test_X = test_X - test_X_mean
            test_X_std = np.std(test_X, axis=0)
            test_X = test_X/(test_X_std + 1e-5)

            result = []
            for i in range(0, 10):
                result.append(final_model[i].score_samples(test_X))
            result = np.array(result)

            test_Y_hat = np.argmax(result, axis=0)+1

            accuracy = np.sum((test_Y_hat == test_Y))/200.0*100.0
            f.write('test accuracy = ' + str(accuracy) + ' %'+'\n')

            inst = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'vocal']
            f.write('~class accuracy~'+'\n')
            class_acc = []
            for i in range(0,10):
                acc = np.sum(test_Y_hat[20*i:20*(i+1)] == test_Y[20*i:20*(i+1)])/20.0 * 100.0
                class_acc.append(acc)
                f.write(inst[i] + '    ' + str(acc) + '%'+'\n')
"""
    write_result(6,
                 'GMM + CODEBOOK',
                 'MFCC_DIM * 3 + ZCR_DIM + RMS_DIM + SPECTRAL_CENTROID_DIM*5 + SPECTRAL_FLATNESS_DIM  + SPECTRAL_BANDWIDTH_DIM + ATTACK_TIME_DIM',
                 accuracy,
                 class_acc,
                 'gmm_component=1\nkmean_codebook_cluster=37\n')
"""