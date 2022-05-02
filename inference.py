import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fftshift
from scipy.io import wavfile
from scipy.signal import windows as wnd
from torch.utils.data import DataLoader
import CoughDataset as CD
import CoughNet as CN
import torch
import time


def spectrogram(audio_arr, fs, show=False, offset = 1e-7):
    desc_freqs = 200
    seg_len = (desc_freqs - 1) * 2 # we want 200 descrite freq from F=0 to F = fs/2
    noverlap = int(seg_len * 4 / 5)
    window = wnd.get_window("tukey", seg_len)
    f, t, Sxx = signal.spectrogram(
        x=audio_arr,
        fs=fs,
        window=window,
        nperseg=seg_len,
        noverlap=noverlap
    )
    

    (row, col) = Sxx.shape
    fps = col * fs / len(audio_arr)
    
    # give all the data a slight offset
    Sxx = np.add(Sxx, offset)

    # log scale the data
    Sxx = np.log10(Sxx)
    if show:
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.savefig('temp.png')
    return f, t, Sxx.T, fps


def get_audio_arr(filename:str="./audio/jp.wav"):
    samplerate, data = wavfile.read(filename)
    length = data.shape[0] / samplerate
    return samplerate, data


if __name__ == "__main__":
    net = CN.CoughNet_Lite()
    net.load_state_dict(torch.load("./ChNet_Lite.mod"))

    while True:
        fs, audio_arr = get_audio_arr("./cur.wav")
        f, t, gram, fps = spectrogram(audio_arr, fs, show=False)
        cd = CD.CoughDataset(f, t, gram, fps)
        dataloader = DataLoader(dataset=cd, batch_size=4, shuffle=True)
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data[0], data[1]
            predict = net(inputs)
            for i in range (4):
                good = 1
                if round(predict[i].item()) == 1:
                    good = 0
                    break
            if good:
                print("ur good")
            else:
                print("stay at home")
            break
        time.sleep(0.5)
        