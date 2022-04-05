import os
import glob
import numpy as np
import xarray as xr

import torch
from torch.utils import data

from nexai.data.geo import stats

class CalipsoGOES(data.Dataset):
    def __init__(self, directory, patch_size=9, mode='train'):
        self.directory = directory
        self.files = sorted(glob.glob(f'{self.directory}/*.nc'))
        self.patch_size = patch_size
        
        N = len(self.files)
        if mode == 'train':
            self.files = self.files[:int(0.8*N)]
        elif mode == 'valid':
            self.files = self.files[int(0.8*N):int(0.9*N)]
        elif mode == 'test':
            self.files = self.files[int(0.9*N):]
            
        bands = list(range(7,17))
        self.mu, self.sd = stats.get_sensor_stats('G16')
        self.mu = np.array([self.mu[b-1] for b in bands])[:,np.newaxis,np.newaxis]
        self.sd = np.array([self.sd[b-1] for b in bands])[:,np.newaxis,np.newaxis]
        
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        f = self.files[idx]
        try:
            ds = xr.open_dataset(f)
        except FileNotFoundError:
            pass
            return self.__getitem__(idx-1)
        except:
            print('remove', f)
            os.remove(f)
            #raise OSError(f)
        #finally:
            return self.__getitem__(idx-1)
        
        r = int((self.patch_size - 1) / 2)
        try:
            x = ds['Rad'].values # (band, lat, lon) (values in K)
        except  KeyError:
            os.remove(f)
            del self.files[idx]
            return self.__getitem__(idx-1)

        c, h, w  = x.shape
        md = int(h / 2)
        x = x[:,md-r:md+r+1,md-r:md+r+1]

        #x = (x - 150) / (350 - 150)
        
        x = (x - self.mu) / self.sd
        
        if np.mean(np.isfinite(x)) != 1.:
            print('found nan: remove', idx, f)
            os.remove(f)
            return self.__getitem__(idx-1)
        
        x = torch.Tensor(x).float()

        y = ds['Layer_Top_Altitude'].values # (layers,) # range (0, ~10)
        y = y[:1]
        y = torch.Tensor(y).float()
        
        y[y != -9999] = (y[y != -9999] - 5) / 10
        
        #y[y==-9999] = -10.
        
        #clouds = (y!=-9999)
        
        #Stratospheric features reported during daylight -- especially those reported above 20 km
        #between 60N and 60S -- are often noise artifacts and should be treated with suspicion.
        #y[y > 20] = -10
        #y = y / 10 - 0.5  # scale top layer (value in kms)

        return x, y[:1,np.newaxis,np.newaxis]
    
def iterate_training_dataset(directory):
    dataset = CalipsoGOES(directory)
    print("length", len(dataset))
    data_params = {'batch_size': 8, 'shuffle': True,
                   'num_workers': 8, 'pin_memory': False}
    training_generator = data.DataLoader(dataset, **data_params)
    
    labels = []
    for i, (x, y) in enumerate(training_generator):
        #print(i*16)
        labels.append(y)
        
       #if i == 100:
       #     break
    #print(labels)

if __name__ == '__main__':
    directory = '/nobackupp10/tvandal/cloud-height/data/calipso_goes_pairs_w_noclouds/'
    iterate_training_dataset(directory)
