import librosa
import scipy
import json
import numpy as np
from tqdm import tqdm
import os

def generate_speaker_list():
    x=set()
    basepath = r'C:\Users\Dell\Desktop\New_code'
    for entry in os.listdir(basepath):
        if os.path.isfile(os.path.join(basepath, entry)):
            x.add(entry[0:4])
    print(len(x))
    with open('speaker_list.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(x)

def preemphasis(x, preemph):
    return scipy.signal.lfilter([1, -preemph], [1], x)


def mulaw_encode(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def mulaw_decode(y, mu):
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


def process_wav(wav_path, out_path, sr=16000, preemph=0.97, n_fft=2048, n_mels=80, hop_length=160,
                win_length=400, fmin=50, top_db=80, bits=8, offset=0.0, duration=None):
    wav, _ = librosa.load(wav_path, sr=sr,
                          offset=offset, duration=duration)
    wav = wav / np.abs(wav).max() * 0.999

    mel = librosa.feature.melspectrogram(preemphasis(wav, preemph),
                                         sr=sr,
                                         n_fft=n_fft,
                                         n_mels=n_mels,
                                         hop_length=hop_length,
                                         win_length=win_length,
                                         fmin=fmin,
                                         power=1)
    logmel = librosa.amplitude_to_db(mel, top_db=top_db)
    logmel = logmel / top_db + 1

    wav = mulaw_encode(wav, mu=2**bits)

    np.save(out_path + 'wav', wav)
    np.save(out_path + 'mel', logmel)
    return out_path, logmel.shape[-1]


def preprocess_dataset(data_path, out_path_init):
    i = 1

    with os.scandir(data_path) as entries:
        for file in entries:
            out_path_final = out_path_init + f'{i}'
            out_path, logmel_shape = process_wav(
                file, out_path_final, duration=5)
            print("outpath, logmel_shape")
            print(out_path, logmel_shape)
            i = i+1


if __name__ == "__main__":
    # preprocess_dataset(r"C:\Users\Dell\Desktop\New_code\speech_data",
    #                    r"C:\Users\Dell\Desktop\New_code\input_data")
    a=np.load(r'C:\Users\Dell\Desktop\New_code\input_data1wav.npy')
    b=mulaw_encode(a,256)
    for i in range(len(b)):
        if(isinstance(b[i], float)):
            print("haha")
    # print(b.shape)

