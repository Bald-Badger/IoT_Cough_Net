import CoughNet as CN
import CoughDataset as CD
import torch
import spectrogram as sp
from torch.utils.data import DataLoader


if __name__ == "__main__":
    net = CN.CoughNet_Micro()
    net.load_state_dict(torch.load("./ChNet_Micro.mod"))
    fs, audio_arr = sp.get_audio_arr("./audio/mix.wav")
    print ("audio sampling freq: ", fs)
    print ("audio length in sample: ", audio_arr.shape[0])
    f, t, gram, fps = sp.spectrogram(audio_arr, fs, show=False)
    cd = CD.CoughDataset(f, t, gram, fps)
    dataloader = DataLoader(dataset=cd, batch_size=4, shuffle=True)
    
    len = cd.__len__()
    
    correct = 0
    wrong = 0
    false_positive = 0
    false_negative = 0

    for i, data in enumerate(dataloader, 0):
        inputs, labels = data[0], data[1]
        predict = net(inputs)
        if (i > 10000):
            break
        for j in range (4):
            if round(predict[j].item()) != round(labels[j].item()):
                wrong += 1
                if round(labels[j].item()) == 1:
                    false_negative += 1
                else:
                    false_positive += 1
            else:
                correct += 1
    print("correct: ", correct)
    print("wrong: ", wrong)
    print("false positive: ", false_positive)
    print("false negative: ", false_negative)
    