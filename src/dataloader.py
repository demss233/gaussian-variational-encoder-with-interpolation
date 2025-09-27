import numpy as np
import pandas as pd
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from .data import process

class Configure(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.transform = transforms.Compose([transforms.ToTensor()])
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        age = self.dataframe['age'].iloc[idx]
        ethnicity = self.dataframe['ethnicity'].iloc[idx]
        gender = self.dataframe['gender'].iloc[idx]
        
        pixels = self.dataframe['pixels'].iloc[idx]
        pixels = np.array(pixels.tolist())
        pixels = pixels.reshape(48, 48) / 255
        pixels = np.array(pixels,'float32')
        
        pixels = self.transform(pixels)
        sample = {'image': pixels, 'age': age, 'ethnicity': ethnicity, 'gender': gender}
        
        return sample


def get_loaders(train_df, eval_df, batch_size, num_workers):
    train_dataset = Configure(train_df)
    eval_dataset = Configure(eval_df)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 0)
    eval_dataloader = DataLoader(eval_dataset, batch_size = batch_size, shuffle = False, num_workers = 0)
    return train_dataset, eval_dataset, train_dataloader, eval_dataloader 