import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fftshift
from scipy.io import wavfile
from scipy.signal import windows as wnd


def get_audio_arr(filename:str="./audio/jp.wav"):
    print("opening file: ", filename)
    samplerate, data = wavfile.read(filename)
    length = data.shape[0] / samplerate
    print(f"length = {length}s")
    return samplerate, data


def spectrogram(audio_arr, fs, show=False):
    desc_freqs = 200
    seg_len = (desc_freqs - 1) * 2
    seg_len = 398 # we want 200 descrite freq from F=0 to F = fs/2
    noverlap = int(seg_len * 2 / 3)
    window = wnd.get_window("tukey", seg_len)
    f, t, Sxx = signal.spectrogram(
        x=audio_arr,
        fs=fs,
        window=window,
        nperseg=seg_len,
        noverlap=noverlap
    )
    print (f.shape, t.shape, Sxx.shape)
    # print (type(f[0]),type(t[0]),type(Sxx[0][0]))
    (row, col) = Sxx.shape
    print (row, col)
    # Sxx = Sxx.T
    f, t, Sxx = signal.spectrogram(audio_arr, fs)
    Sxx = np.log10(Sxx)
    if show:
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.savefig('temp.png')
    

if __name__ == "__main__":
    fs, audio_arr = get_audio_arr("./audio/mix.wav")
    print ("audio sampling freq: ", fs)
    print ("audio length in sample: ", audio_arr.shape[0])
    spectrogram(audio_arr, fs, show=True)
