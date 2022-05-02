import pyaudio
import wave
import numpy as np

import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fftshift
from scipy.io import wavfile
import scipy.io
from scipy.signal import windows as wnd

def list_device():
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        print(p.get_device_info_by_index(i).get('name'))


# USB mic should be index 1
def record_audio(time:float=1, name:str="temp.wav"):
    # audio setting
    form_1 = pyaudio.paInt16 # 16-bit resolution
    chans = 1 # 1 channel
    samp_rate = 44100 # 44.1kHz sampling rate
    chunk = 4096 # 2^12 samples for buffer
    record_secs = time # seconds to record
    dev_index = 1 # device index found by p.get_device_info_by_index(ii)
    wav_output_filename = name # name of .wav file

    audio = pyaudio.PyAudio()

    stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
        input_device_index = dev_index,input = True, \
        frames_per_buffer=chunk)
    
    print("recording")
    frames = []

    for ii in range(0,int((samp_rate/chunk)*record_secs)):
        data = stream.read(chunk)
        frames.append(data)

    print("finished recording")
 
    # stop the stream, close it, and terminate the pyaudio instantiation
    stream.stop_stream()
    stream.close()
    audio.terminate()
 
    wavefile = wave.open(wav_output_filename,'wb')
    wavefile.setnchannels(chans)
    wavefile.setsampwidth(audio.get_sample_size(form_1))
    wavefile.setframerate(samp_rate)
    wavefile.writeframes(b''.join(frames))
    wavefile.close()


def get_audio_arr_old(filename:str="./audio/jp.wav"):
    ifile = wave.open(filename)
    samples = ifile.getnframes()
    audio = ifile.readframes(samples)
    audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
    return audio_as_np_int16


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
    fs, audio_arr = get_audio_arr("./audio/jp.wav")
    print ("audio sampling freq: ", fs)
    print ("audio length in sample: ", audio_arr.shape[0])
    spectrogram(audio_arr, fs, show=True)
