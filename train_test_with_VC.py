from feature_summary_various_codebook import *

from sklearn.metrics import confusion_matrix

from sklearn.mixture import GaussianMixture as gmm
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier

group = ['mfcc', 'zcr, rms, cent, flat, rolloff, bandwidth, attack, mean_cent_norm, max_cent_norm, cent_std, cent_norm_std']

def train_model_gmm(train_X):
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

def train_model(train_X, train_Y, valid_X, valid_Y, hyper_param1):
    # Choose a classifier
    #clf = SGDClassifier(verbose=0, loss="hinge", alpha=hyper_param1, max_iter=1000, penalty="l2", random_state=0)
    #clf = SGDClassifier(verbose=0, loss="perceptron", alpha=hyper_param1, max_iter=1000, penalty="l2", random_state=0)
    clf = KNeighborsClassifier(n_neighbors=2)

    # train
    clf.fit(train_X, train_Y)

    # validation
    valid_Y_hat = clf.predict(valid_X)

    accuracy = np.sum((valid_Y_hat == valid_Y)) / 200.0 * 100.0
    #print 'validation accuracy = ' + str(accuracy) + ' %'

    return clf, accuracy

if __name__ == '__main__':
    f = open('find_cluster.txt', 'w')

    mfcc_cluster = 16
    others_cluster = 5

    train_data = load_data('train', group)
    valid_data = load_data('valid',group)
    test_data = load_data('test', group)

    print(train_data[0].shape)
    print(train_data[1].shape)

    print(str(mfcc_cluster) + ' ' + str(others_cluster))

    f.write("============================================" + '\n')
    f.write("mfcc_cluster: " + str(mfcc_cluster) + '\n')
    f.write("others_cluster: " + str(others_cluster) + '\n')

    kmeans_mfcc = generate_codebook(train_data[0], mfcc_cluster)
    kmeans_others = generate_codebook(train_data[1], others_cluster)

    # main for codebook + GMM
    # (datanum, n_cluster)
    train_X = np.append(generate_new_data(1000, train_data[0], kmeans_mfcc, mfcc_cluster), generate_new_data(1000, train_data[1], kmeans_others, others_cluster), axis=1)
    f.write(str(train_X.shape) + '\n')

    valid_X = np.append(generate_new_data(200, valid_data[0], kmeans_mfcc, mfcc_cluster), generate_new_data(200, valid_data[1], kmeans_others, others_cluster), axis=1)
    f.write(str(valid_X.shape) + '\n')


    train_X = np.append(train_X, valid_X, axis=0)
    f.write(str(train_X.shape) + '\n')

    test_X = np.append(generate_new_data(200, test_data[0], kmeans_mfcc, mfcc_cluster), generate_new_data(200, test_data[1], kmeans_others, others_cluster), axis=1)
    f.write(str(test_X.shape) + '\n')

    # label generation
    cls = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    test_Y = np.repeat(cls, 20)

    # feature normalizaiton
    train_X_mean = np.mean(train_X, axis=0)
    train_X = train_X - train_X_mean
    train_X_std = np.std(train_X, axis=0)
    train_X = train_X / (train_X_std + 1e-5)

    # training model
    f.write("gmm_num: " + str(1) + '\n')
    final_model = train_model_gmm(train_X)

    # now, evaluate the model with the test set
    test_X = test_X - train_X_mean
    test_X = test_X / (train_X_std + 1e-5)

    result = []
    for i in range(0, 10):
        result.append(final_model[i].score_samples(test_X))
    result = np.array(result)

    test_Y_hat = np.argmax(result, axis=0) + 1

    accuracy = np.sum((test_Y_hat == test_Y)) / 200.0 * 100.0
    f.write('test accuracy = ' + str(accuracy) + ' %' + '\n')
    print('test accuracy = ' + str(accuracy) + ' %')

    inst = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'vocal']
    f.write('~class accuracy~' + '\n')
    class_acc = []
    for i in range(0, 10):
        acc = np.sum(test_Y_hat[20 * i:20 * (i + 1)] == test_Y[20 * i:20 * (i + 1)]) / 20.0 * 100.0
        class_acc.append(acc)
        f.write(inst[i] + '    ' + str(acc) + '%' + '\n')

        cm = confusion_matrix(test_Y[20 * i:20 * (i + 1)], test_Y_hat[20 * i:20 * (i + 1)])
        f.write(str(test_Y[20 * i:20 * (i + 1)]) + '\n')
        f.write(str(test_Y_hat[20 * i:20 * (i + 1)]) + '\n')


    '''
    # main for codebook + other classifiers
    # (datanum, n_cluster)
    train_X = np.append(generate_new_data(1000, train_data[0], kmeans_mfcc, mfcc_cluster),
                        generate_new_data(1000, train_data[1], kmeans_others, others_cluster), axis=1)
    f.write(str(train_X.shape) + '\n')

    valid_X = np.append(generate_new_data(200, valid_data[0], kmeans_mfcc, mfcc_cluster),
                        generate_new_data(200, valid_data[1], kmeans_others, others_cluster), axis=1)
    f.write(str(valid_X.shape) + '\n')

    test_X = np.append(generate_new_data(200, test_data[0], kmeans_mfcc, mfcc_cluster),
                       generate_new_data(200, test_data[1], kmeans_others, others_cluster), axis=1)
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

    test_X = test_X - train_X_mean
    test_X = test_X / (train_X_std + 1e-5)

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

    test_Y_hat = final_model.predict(test_X)

    accuracy = np.sum((test_Y_hat == test_Y)) / 200.0 * 100.0
    print 'test accuracy = ' + str(accuracy) + ' %'

    inst = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'vocal']
    print '~class accuracy~'
    class_acc = []
    for i in range(0, 10):
        acc = np.sum(test_Y_hat[20 * i:20 * (i + 1)] == test_Y[20 * i:20 * (i + 1)]) / 20.0 * 100.0
        class_acc.append(acc)
        print inst[i] + '    ' + str(acc) + '%'
    '''