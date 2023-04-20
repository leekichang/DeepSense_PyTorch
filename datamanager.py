import torch
import numpy as np
from torch.utils.data import Dataset

class DM(Dataset):
    def __init__(self, datapath) -> None:
        super().__init__()
        self.X, self.Y = np.load(f'{datapath}/X.npy'), np.load(f'{datapath}/Y.npy')
    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx,:]).float()
        Y = torch.tensor(self.Y[idx]).long()
        return X, Y
    
    def __len__(self):
        self.len = len(self.Y)
        return self.len