import sys
import os
import numpy as np
import matplotlib.pyplot as plt



data_path = './dataset/'
mfcc_path = './features/mfcc/'
zcr_path = './features/zcr/'
rms_path = './features/rms/'
spectral_centroid_path = './features/spectral_centroid/'
spectral_flatness_path = './features/spectral_flatness/'
spectral_rolloff_path = './features/spectral_rolloff/'
spectral_bandwidth_path = './features/spectral_bandwidth/'
attack_time_path = './features/attack_time/'


MFCC_DIM = 13
ZCR_DIM = 1
RMS_DIM = 1
SPECTRAL_CENTROID_DIM = 1
SPECTRAL_FLATNESS_DIM = 1
SPECTRAL_ROLLOFF_DIM = 1
SPECTRAL_BANDWIDTH_DIM = 1
ATTACK_TIME_DIM = 1

DIM = MFCC_DIM * 3 + ZCR_DIM + RMS_DIM + SPECTRAL_CENTROID_DIM * 5 + SPECTRAL_FLATNESS_DIM  + SPECTRAL_ROLLOFF_DIM + SPECTRAL_BANDWIDTH_DIM + ATTACK_TIME_DIM

def load_data(dataset='train'):
    f = open(data_path + dataset + '_list.txt', 'r')

    if dataset == 'train':
        data_mat = np.zeros(
            shape=(DIM, 1000))
    else:
        data_mat = np.zeros(
            shape=(DIM, 200))

    i = 0
    for file_name in f:
        file_name = file_name.rstrip('\n')
        file_name = file_name.replace('.wav', '.npy')

        # load mfcc file
        mfcc_file = mfcc_path + file_name
        mfcc = np.load(mfcc_file)

        # load zcr file
        zcr_file = zcr_path + file_name
        zcr = np.load(zcr_file)

        data = np.append(mfcc, zcr, axis=0)
        # print(data.shape)

        # load rms file
        rms_file = rms_path + file_name
        rms = np.load(rms_file)

        data = np.append(data, rms, axis=0)
        # print(data.shape)

        # load spectral centroid file
        spectral_centroid_file = spectral_centroid_path + file_name
        cent = np.load(spectral_centroid_file)

        data = np.append(data, cent, axis=0)
        # print(data.shape)

        # load spectral flatness file
        spectral_flatness_file = spectral_flatness_path + file_name
        flat = np.load(spectral_flatness_file)

        data = np.append(data, flat, axis=0)
        # print(data.shape)


        # load spectral rolloff file
        spectral_rolloff_file = spectral_rolloff_path + file_name
        rolloff = np.load(spectral_rolloff_file)

        data = np.append(data, rolloff, axis=0)
        #print(data.shape)


        # load spectral bandwidth file
        spectral_bandwidth_file = spectral_bandwidth_path + file_name
        bandwidth = np.load(spectral_bandwidth_file)

        data = np.append(data, bandwidth, axis=0)
        #print(data.shape)

        # load attack time file
        attack_time_file = attack_time_path + file_name
        attack_time = np.load(attack_time_file)

        data = np.append(data, attack_time, axis=0)

        # mean pooling
        # temp = np.mean(data, axis=1)
        pooled_data = np.mean(data, axis=1)

        # normalize spectral centroid
        cent_mean = np.mean(cent)
        cent_std = np.std(cent)
        cent_norm = (cent - cent_mean) / (cent_std + 1e-5)

        # add mean of normalized spectral centroid
        pooled_data = np.append(pooled_data, np.mean(cent_norm))

        # add max of normalized spectral centroid
        pooled_data = np.append(pooled_data, np.max(cent_norm))

        pooled_data = np.append(pooled_data, cent_std)

        pooled_data = np.append(pooled_data, np.std(cent_norm))

        data_mat[:, i] = pooled_data

        i = i + 1

    f.close();

    return data_mat




if __name__ == '__main__':
    # np.set_printoptions(threshold=sys.maxsize)
    train_data = load_data('train')
    # print(train_data.shape)
    # print(train_data)
    valid_data = load_data('valid')
    test_data = load_data('test')

    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.imshow(train_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(3, 1, 2)
    plt.imshow(valid_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(3, 1, 3)
    plt.imshow(test_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')

    plt.show()
