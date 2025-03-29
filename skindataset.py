import torch
from torch.utils.data import Dataset
import numpy as np

class SkinDataset(Dataset):

    def __init__(self, data_path='data/skins.npy', number_of_samples=None):

        self.skins = np.load(data_path)

        if number_of_samples:
            self.skins = self.skins[:number_of_samples]

    def __len__(self):
        return len(self.skins)
    
    def __getitem__(self, idx):
        skin = self.skins[idx]

        # Turn to tensor and scale between -1 and 1
        skin = torch.from_numpy(skin).float()
        skin = skin.permute(2, 0, 1) # Change from HWC to CHW
        skin = (skin - 127.5) / 127.5 # Scale between -1 and 1

        return skin
    