import os
import glob
import numpy as np
import xarray as xr

import torch
from torch.utils import data


class CalipsoGOES(data.Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.files = glob.glob(f'{self.directory}/*.nc')
       
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        f = self.files[idx]
        ds = xr.open_dataset(f)

        x = ds['Rad'].values # (band, lat, lon) (values in K)
        x = (x - 150) / (350 - 150)
        x = torch.Tensor(x).float()

        y = ds['Layer_Top_Altitude'].values # (layers,)
        y = y[:1]
        y = torch.Tensor(y).float()
        y[y==-9999] = -10.
        #Stratospheric features reported during daylight -- especially those reported above 20 km
        #between 60N and 60S -- are often noise artifacts and should be treated with suspicion.
        y[y > 20] = -10


        #y = y / 10 - 0.5  # scale top layer (value in kms)

        return x, y[:1]

if __name__ == '__main__':
    directory = '/nobackupp10/tvandal/cloud-height/data/calipso_goes_pairs/'
    dataset = CalipsoGOES(directory)

    for i in range(100):
        x, y = dataset[i]
        print(x.shape, y.shape, y)
