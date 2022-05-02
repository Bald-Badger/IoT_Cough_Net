from torch.utils.data import Dataset, DataLoader


class CoughDataset(Dataset):
    section_len = 0.4 # 0.4 sec rolling window
    positive_time_range = [[6.52, 6.75], [7.02, 7.2], [14.6, 14.85], [23.1, 23.35], [23.48, 23.68], [23.95, 24.05], [39.58, 39.78], [40.09, 40.29], [46.29, 46.39], [57.99, 58.19], [58.34, 58.56], [58.72, 58.79], [70.51, 70.74], [71.09, 71.25], [75.75, 75.93], [76.17, 76.27], [81.99, 82.23], [82.3, 82.46], [82.64, 82.75], [87.31, 87.43], [94.62, 94.83], [94.88, 95.1], [96.76, 96.91]]
    

    def __init__(self, f=[], t=[], gram=[], fps=0):
        self.window_len = round(fps * self.section_len)
        self.nframe = len(t)
        self.ds = []
        print(t[0:100])
        for i in range (self.nframe - self.window_len):
            start_time = t[i]
            end_time = t[i+self.window_len]
            positive = self.enclose(start_time, end_time)
            self.ds.append({'data':gram[i:i+self.window_len], 'label':positive})
            
    
    def __len__(self):
        return len(self.ds)
    
    
    def __getitem__(self, index):
        return self.ds[index]
    

    def enclose(self, head=0.0, tail=0.0):
        result = False
        for r in self.positive_time_range:
            if head < r[0] and tail > r[1]:
                result = True
                break
        return result

