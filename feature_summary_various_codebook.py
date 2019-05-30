import numpy as np

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


def generate_codebook(data, n):
    kmeans = KMeans(n_clusters=n, random_state=0)

    print("generating codebook..")

    kmeans.fit(data)

    return kmeans

def generate_new_data(datanum, data, kmeans, n):
    print("generating new data..")

    new_data = np.zeros(shape=(datanum, n))
    labels = kmeans.predict(data).reshape(datanum, 173)

    k=0
    for d in labels:
        new_d = np.zeros(n)
        for i in range(0,n):
            new_d[i] = np.count_nonzero(d==i)
        new_data[k] = new_d
        k = k+1

    return new_data

def add_data(data, new_data):
    if data.size == 0:
        data = new_data
    else:
        data = np.append(data, new_data, axis=0)
    return data

def make_data(file_name, list):
    file_name = file_name.rstrip('\n')
    file_name = file_name.replace('.wav', '.npy')

    data = np.array([])
    for name in list:
        if 'mfcc' in name:
            mfcc_file = mfcc_path + file_name
            mfcc = np.load(mfcc_file)
            if name =='mfcc' :
                data = add_data(data, mfcc)

        elif 'zcr' in name:
            zcr_file = zcr_path + file_name
            zcr = np.load(zcr_file)

            tmp = np.empty(173)

            if name == 'zcr':
                data = add_data(data, zcr)

            elif name == 'zcr_mean':
                zcr_mean = np.mean(zcr)
                tmp.fill(zcr_mean)
                zcr_mean = np.array([tmp])

                data = add_data(data, zcr_mean)

            elif name == 'zcr_std':
                zcr_std = np.std(zcr)
                tmp.fill(zcr_std)
                zcr_std = np.array([tmp])

                data = add_data(data, zcr_std)

        elif 'rms' in name:
            rms_file = rms_path + file_name
            rms = np.load(rms_file)
            tmp = np.empty(173)

            if name == 'rms':
                data = add_data(data, rms)

            elif name == 'rms_mean':
                rms_mean = np.mean(rms)
                tmp.fill(rms_mean)
                rms_mean = np.array([tmp])

                data = add_data(data, rms_mean)

            elif name == 'rms_std':
                rms_std = np.std(rms)
                tmp.fill(rms_std)
                rms_std = np.array([tmp])

                data = add_data(data, rms_std)

        elif 'cent' in name:
            spectral_centroid_file = spectral_centroid_path + file_name
            cent = np.load(spectral_centroid_file)

            cent_mean = np.mean(cent)
            cent_std = np.std(cent)
            cent_norm = (cent - cent_mean) / (cent_std + 1e-5)

            tmp = np.empty(173)

            if name == 'cent':
                data = add_data(data, cent)

            elif name == 'mean_cent_norm':
                # add mean of normalized spectral centroid
                tmp.fill(np.mean(cent_norm))
                mean_cent_norm = np.array([tmp])

                data = add_data(data, mean_cent_norm)

            elif name == 'max_cent_norm':
                # add max of normalized spectral centroid
                tmp.fill(np.max(cent_norm))
                max_cent_norm = np.array([tmp])

                data = add_data(data, max_cent_norm)

            elif name == 'cent_std':
                # add std of spectral centroid
                tmp.fill(cent_std)
                cent_std = np.array([tmp])

                data = add_data(data, cent_std)

            elif name == 'cent_norm_std':
                # add std of normalized centroid
                tmp.fill(np.std(cent_norm))
                cent_norm_std = np.array([tmp])

                data = add_data(data, cent_norm_std)

            elif name == 'cent_mean':
                tmp.fill(cent_mean)
                cent_mean = np.array([tmp])

                data = add_data(data, cent_mean)


        elif 'flat' in name:
            spectral_flatness_file = spectral_flatness_path + file_name
            flat = np.load(spectral_flatness_file)
            tmp = np.empty(173)

            if name == 'flat':
                data = add_data(data, flat)

            elif name == 'flat_mean':
                flat_mean = np.mean(flat)
                tmp.fill(flat_mean)
                flat_mean = np.array([tmp])

                data = add_data(data, flat_mean)

            elif name == 'flat_std':
                flat_std = np.std(flat)
                tmp.fill(flat_std)
                flat_std = np.array([tmp])

                data = add_data(data, flat_std)

        elif 'bandwidth' in name:
            spectral_bandwidth_file = spectral_bandwidth_path + file_name
            bandwidth = np.load(spectral_bandwidth_file)
            tmp = np.empty(173)

            if name == 'bandwidth':
                data = add_data(data, bandwidth)

            elif name == 'bandwidth_mean':
                bandwidth_mean = np.mean(bandwidth)
                tmp.fill(bandwidth_mean)
                bandwidth_mean = np.array([tmp])

                data = add_data(data, bandwidth_mean)

            elif name == 'bandwidth_std':
                bandwidth_std = np.std(bandwidth)
                tmp.fill(bandwidth_std)
                bandwidth_std = np.array([tmp])

                data = add_data(data, bandwidth_std)

        elif 'attack' in name:
            attack_time_file = attack_time_path + file_name
            attack_time = np.load(attack_time_file)

            if name == 'attack':
                data = add_data(data, attack_time)

    return data

def normalize_data(data):
    # data = (feature_num, 173 * data_num)

    data_mean = np.mean(data, axis=1)   # (feature_num,)
    data = data - data_mean.reshape(len(data_mean), 1)  # (feature_num, 173 * data_num)
    data_std = np.std(data, axis=1)     # (feature_num,)
    data = data / (data_std.reshape(len(data_std), 1) + 1e-5)   # (feature_num, 173 * data_num)

    data = data.T  #(173 * data_num, feature_num)

    return data


def load_data(dataset = 'train', group=[]):
    # group = ['name of features for codebook 1', 'name of features for codebook 2', ...]

    # load train data
    total_data = []
    for g in group:
        f = open(data_path + dataset + '_list.txt', 'r')

        i = 0
        for file_name in f:
            cur_data = make_data(file_name, g.split(', ')) #(n_features, 173)
            if i == 0:
                data = cur_data
            else:
                data = np.concatenate((data, cur_data), axis=1)
            i = i + 1
        data = np.array(data)   #(feature_num, 173*data_num)

        data = normalize_data(data) # normalized (173*data_num, feature_num)

        total_data.append(data)
    return total_data
