import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fftshift
from scipy.io import wavfile
from scipy.signal import windows as wnd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import CoughDataset as CD

import CoughNet as CN


def get_audio_arr(filename:str="./audio/jp.wav"):
    print("opening file: ", filename)
    samplerate, data = wavfile.read(filename)
    length = data.shape[0] / samplerate
    print(f"length = {length}s")
    return samplerate, data


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
    

    # print (type(f[0]),type(t[0]),type(Sxx[0][0]))
    (row, col) = Sxx.shape
    print ("desc freq num: ", row, ", time frames: ", col)
    fps = col * fs / len(audio_arr)
    print ("frame per second: ", fps)
    
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


def linear_scale(arr, max=10):
    gram_min = np.min(arr)
    arr = np.add(arr, -gram_min)
    gram_max = np.max(arr)
    scale_factor = max / gram_max
    arr = np.dot(arr, scale_factor)
    return arr


if __name__ == "__main__":
    fs, audio_arr = get_audio_arr("./audio/mix.wav")
    print ("audio sampling freq: ", fs)
    print ("audio length in sample: ", audio_arr.shape[0])
    f, t, gram, fps = spectrogram(audio_arr, fs, show=False)
    
    cd = CD.CoughDataset(f, t, gram, fps)
    dataloader = DataLoader(dataset=cd, batch_size=4, shuffle=True)
    
    cn = CN.CoughNet_Lite()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(cn.parameters(), lr=0.001, momentum=0.9)
    
    useGPU = True
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    
    if useGPU:
        cn.to(device)
        print("traning on GPU")
    else:
        print("traning on CPU")
    

    for epoch in range(20):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            if useGPU:
                inputs, labels = data[0].to(device), data[1].to(device)
            else:
                inputs, labels = data[0], data[1]
                
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = cn(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        if (useGPU):
            torch.save(cn.cpu().state_dict(), "./temp.mod") # saving model
            cn.to(device) # moving model to GPU for further training
        else:
            torch.save(cn.state_dict(), "./temp.mod") # saving model

    print('Finished Training')
    torch.save(cn.state_dict(), "cn_lite.mod")
    