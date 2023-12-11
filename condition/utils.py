import os, shutil, random
from pathlib import Path
#from kaggle import api
import torch

from torch.utils.data import DataLoader , Dataset
import pandas as pd




def mk_folders(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

class CSVDataset(Dataset):
    def __init__(self, args):
        self.path = args.dataset_path
        self.data_frame = pd.read_csv(args.dataset_path)
        self.m = args.spd_size

    def __getitem__(self,idx):
        sample = self.data_frame.iloc[idx,14:].values    
        sample = sample.reshape(self.m, self.m)
        Y = self.data_frame.iloc[idx,1:14].values   
        return torch.from_numpy(sample), torch.from_numpy(Y)
    

    def __len__(self):
        return len(self.data_frame)



