import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

data_path = './dataset/'
mfcc_path = './features/mfcc/'
zcr_path = './features/zcr/'
rms_path = './features/rms/'
spectral_centroid_path = './features/spectral_centroid/'
spectral_flatness_path = './features/spectral_flatness/'
spectral_rolloff_path = './features/spectral_rolloff/'
spectral_bandwidth_path = './features/spectral_bandwidth/'
spectrogram_path = './features/spectrogram/'
attack_time_path = './features/attack_time/'

MFCC_DIM = 13
ZCR_DIM = 1
RMS_DIM = 1
SPECTRAL_CENTROID_DIM = 1
SPECTRAL_FLATNESS_DIM = 1
SPECTRAL_ROLLOFF_DIM = 1
SPECTRAL_BANDWIDTH_DIM = 1
ATTACK_TIME_DIM = 1


DIM = MFCC_DIM * 2 + ZCR_DIM + RMS_DIM + SPECTRAL_CENTROID_DIM*5 + SPECTRAL_FLATNESS_DIM  + SPECTRAL_BANDWIDTH_DIM + ATTACK_TIME_DIM

N_CLUSTERS = 23


def make_data(file_name):
    file_name = file_name.rstrip('\n')
    file_name = file_name.replace('.wav', '.npy')

    # load mfcc file
    mfcc_file = mfcc_path + file_name
    mfcc = np.load(mfcc_file)
    data = mfcc
    print(data.shape)

    # load zcr file
    zcr_file = zcr_path + file_name
    zcr = np.load(zcr_file)

    data = np.append(data, zcr, axis=0)
    #print(data.shape)

    # load rms file
    rms_file = rms_path + file_name
    rms = np.load(rms_file)

    data = np.append(data, rms, axis=0)
    #print(data.shape)

    # load spectral centroid file
    spectral_centroid_file = spectral_centroid_path + file_name
    cent = np.load(spectral_centroid_file)

    data = np.append(data, cent, axis=0)
    #print(data.shape)

    # load spectral flatness file
    spectral_flatness_file = spectral_flatness_path + file_name
    flat = np.load(spectral_flatness_file)

    data = np.append(data, flat, axis=0)
    #print(data.shape)


    # load spectral bandwidth file
    spectral_bandwidth_file = spectral_bandwidth_path + file_name
    bandwidth = np.load(spectral_bandwidth_file)

    data = np.append(data, bandwidth, axis=0)
    #print(data.shape)

    cent_mean = np.mean(cent)
    cent_std = np.std(cent)
    cent_norm = (cent - cent_mean) / (cent_std + 1e-5)

    # add mean of normalized spectral centroid
    tmp = np.empty(173)
    tmp.fill(np.mean(cent_norm))
    mean_cent_norm = np.array([tmp])

    data = np.append(data, mean_cent_norm, axis=0)


    # add max of normalized spectral centroid
    tmp.fill(np.max(cent_norm))
    max_cent_norm = np.array([tmp])

    data = np.append(data, max_cent_norm, axis=0)

    # add std of spectral centroid
    tmp.fill(cent_std)
    cent_std = np.array([tmp])

    data = np.append(data, cent_std, axis=0)

    # add std of normalized centroid
    tmp.fill(np.std(cent_norm))
    cent_norm_std = np.array([tmp])

    data = np.append(data, cent_norm_std, axis=0)

    # add attack time : onset ~ highest
    attack_time_file = attack_time_path + file_name
    attack_time = np.load(attack_time_file)

    data = np.append(data, attack_time, axis=0)
    # print(data.shape)

    return data


def load_data(dataset='train', n=N_CLUSTERS):
    f = open(data_path + dataset + '_list.txt', 'r')


    print("load " + dataset + " data..")
    data = np.zeros(shape=(DIM, 173))
    i = 0

    # get data
    for file_name in f:
        cur_data = make_data(file_name)  ## cur_data = (42,173)
        if i == 0:
            data = cur_data
        else:
            data = np.concatenate((data, cur_data), axis=1)
        i = i + 1
    f.close();
    ## data = (42, 173*(number of data))

    print("finish to load " + dataset + " data")

    # normalize data
    data_mean = np.mean(data, axis=1)  # (42,)
    data = data - data_mean.reshape(DIM, 1)  # (42, 173*(number of data))
    data_std = np.std(data, axis=1)  # (42,)
    data = data / (data_std.reshape(DIM, 1) + 1e-5)  # (42, 173*(number of data))

    data = data.T

    # return (173*(number of data), feature_num)
    return data


def generate_codebook(data, n):
    kmeans = KMeans(n_clusters=n, random_state=0)

    print("generating codebook..")
    # data = data.T #(173*(number of data), feature_num)
    # print(data.shape)

    kmeans.fit(data)

    return kmeans


def generate_new_data(dataset, data, kmeans, n):
    # generate new data
    print("generating new " + dataset + " data..")

    if dataset == "train":
        new_data = np.zeros(shape=(1000, n))
        labels = kmeans.predict(data).reshape(1000, 173)
    else:
        new_data = np.zeros(shape=(200, n))
        labels = kmeans.predict(data).reshape(200, 173)

    k = 0
    for d in labels:
        new_d = np.zeros(n)
        for i in range(0, n):
            new_d[i] = np.count_nonzero(d == i)
        new_data[k] = new_d
        k = k + 1

    # return (data_num , N_CLUSTERS)
    return new_data


if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    train_data = load_data('train')
    kmeans = generate_codebook(train_data)
    train_data = generate_new_data('train', train_data, kmeans)
    # print(train_data)
    for i in range(len(train_data)):
        if i % 100 == 0:
            print(str(i) + "==============================")
        print(train_data[i])

    valid_data = load_data('valid')
    valid_data = generate_new_data('valid', valid_data, kmeans)
    # print(valid_data)
    for i in range(len(valid_data)):
        if i % 100 == 0:
            print(str(i) + "==============================")
        print(valid_data[i])

    test_data = load_data('test')
    test_data = generate_new_data('test', test_data, kmeans)
    # print(test_data)
    for i in range(len(test_data)):
        if i % 100 == 0:
            print(str(i) + "==============================")
        print(test_data[i])