import os
import numpy as np
import librosa
import librosa.display


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


def extract_features(dataset='train'):
    f = open(data_path + dataset + '_list.txt', 'r')

    i = 0
    for file_name in f:
        # progress check
        i = i + 1
        if not (i % 10):
            print(i)

        # load audio file
        file_name = file_name.rstrip('\n')
        file_path = data_path + file_name
        y, sr = librosa.load(file_path, sr=22050)

        # extract features
        # extract mfcc
        extract_mfcc(file_name, y, sr)

        # extract zcr
        extract_zcr(file_name, y)

        # extract rms
        extract_rms(file_name, y)

        # extract spectral centroid
        extract_spectral_centroid(file_name, y, sr)

        # extract spectral flatness
        extract_spectral_flatness(file_name, y)

        # extract spectral rolloff
        extract_spectral_rolloff(file_name, y, sr)

        # extract spectral bandwidth
        extract_spectral_bandwidth(file_name, y, sr)

        # extract attack time
        extract_attack_time(file_name, y, sr)

    f.close();


def extract_rms(file_name, audio):
    # get rms

    # method 1. from audio input
    # rms = librosa.feature.rms(y=audio)

    # method 2. from spectrogram input
    S, phase = librosa.magphase(librosa.stft(audio))
    rms = librosa.feature.rms(S=S)

    # save rms as a file
    file_name = file_name.replace('.wav', '.npy')
    save_file = rms_path + file_name

    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    np.save(save_file, rms)


def extract_attack_time(file_name, y, sr):
    # get rms

    # method 1. from audio input
    # rms = librosa.feature.rms(y=audio)

    # method 2. from spectrogram input
    S, phase = librosa.magphase(librosa.stft(y))
    rms = librosa.feature.rms(S=S)

    # get onset frames
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)

    # calculate attack time
    highest = np.argmax(rms)
    start = onset_frames[0]

    time = highest - start

    attack_time = np.empty(173)
    attack_time.fill(time)
    attack_time = np.array([attack_time])

    # save attack time as a file
    file_name = file_name.replace('.wav', '.npy')
    save_file = attack_time_path + file_name

    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    np.save(save_file, attack_time)


def extract_spectral_centroid(file_name, audio, sr):
    # get spectral centroid

    # method 1. from time-series input
    # cent = librosa.feature.spectral_centroid(y=audio, sr=sr)

    # method 2. from spectrogram input
    S, phase = librosa.magphase(librosa.stft(y=audio))
    cent = librosa.feature.spectral_centroid(S=S)

    # method 3. from variable bin center frequencies
    # if_gram, D = librosa.ifgram(audio)
    # cent = librosa.feature.spectral_centroid(S=np.abs(D), freq=if_gram)

    # save spectral centroid as a file
    file_name = file_name.replace('.wav', '.npy')
    save_file = spectral_centroid_path + file_name

    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    np.save(save_file, cent)


def extract_spectral_flatness(file_name, audio):
    # get spectral flatness

    # method 1. from time-series input
    # flatness = librosa.feature.spectral_flatness(y=audio)

    # method 2. from spectrogram input
    S, phase = librosa.magphase(librosa.stft(y=audio))
    flatness = librosa.feature.spectral_flatness(S=S)

    # method 3. from power spectrogram input
    # S, phase = librosa.magphase(librosa.stft(y=audio))
    # S_power = S ** 2
    # flatness = librosa.feature.spectral_flatness(S=S_power, power=1.0)

    # save spectral flatness as a file
    file_name = file_name.replace('.wav', '.npy')
    save_file = spectral_flatness_path + file_name

    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    np.save(save_file, flatness)



def extract_spectral_rolloff(file_name, audio, sr):
    # get spectral rolloff

    # method 1. from time-series input
    # rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    # Approximate minimum frequencies with roll_percent=0.1
    # rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.1)

    # method 2. from spectrogram input
    S, phase = librosa.magphase(librosa.stft(y=audio))
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)

    # save spectral rolloff as a file
    file_name = file_name.replace('.wav', '.npy')
    save_file = spectral_rolloff_path + file_name

    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    np.save(save_file, rolloff)


def extract_spectral_bandwidth(file_name, y, sr):
    # get spectral bandwidth

    # method 1. from time-series input
    # spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    # method 2. from spectrogram input
    S, phase = librosa.magphase(librosa.stft(y=y))
    bandwidth = librosa.feature.spectral_bandwidth(S=S)

    # save spectral bandwidth as a file
    file_name = file_name.replace('.wav', '.npy')
    save_file = spectral_bandwidth_path + file_name

    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    np.save(save_file, bandwidth)



def extract_mfcc(file_name, audio, sr):
    # STFT
    S = librosa.core.stft(audio, n_fft=1024, hop_length=512, win_length=1024)
    # power spectrum
    D = np.abs(S) ** 2
    # mel spectrogram (512 --> 40)
    mel_basis = librosa.filters.mel(sr, 1024, n_mels=40)
    mel_S = np.dot(mel_basis, D)
    # log compression
    log_mel_S = librosa.power_to_db(mel_S)
    # mfcc (DCT)
    mfcc = librosa.feature.mfcc(S=log_mel_S, n_mfcc=13)
    mfcc = mfcc.astype(np.float32)  # to save the memory (64 to 32 bits)

    # mfcc-delta
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta = mfcc_delta.astype(np.float32)
    mfcc = np.append(mfcc, mfcc_delta, axis=0)

    mfcc_delta2 = librosa.feature.delta(mfcc_delta)
    mfcc_delta2 = mfcc_delta2.astype(np.float32)
    mfcc = np.append(mfcc, mfcc_delta2, axis=0)


    # save mfcc as a file
    file_name = file_name.replace('.wav', '.npy')
    save_file = mfcc_path + file_name

    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    np.save(save_file, mfcc)



def extract_zcr(file_name, audio):
    # get zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    # print(zcr)
    # print(zcr.shape)
    # print(type(zcr))

    # save zcr as a file
    file_name = file_name.replace('.wav', '.npy')
    save_file = zcr_path + file_name

    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    np.save(save_file, zcr)



if __name__ == '__main__':
    extract_features(dataset='train')
    extract_features(dataset='valid')
    extract_features(dataset='test')



