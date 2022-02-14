import os
import xarray as xr
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms

from . import goesr
from . import stats

from .. import flow_transforms
from ..preprocess import image_histogram_equalization

class DatasetL1b(data.Dataset):
    def __init__(self, data_directory, 
                 product='ABI-L1b-RadF',
                 channel=9,
                 year=None,
                 dayofyear=None,
                 transform=None,
                 resolution_km=2.):
        self.data_directory = data_directory
        self.goes = goesr.GOESL1b(product=product,
                                  channels=[channel],
                                  data_directory=self.data_directory)
        self.files = self.goes.local_files(year=year, dayofyear=dayofyear)
        self.N = self.files.shape[0] - 3

        if product == 'ABI-L1b-RadF':
            size_2km = (10848, 10848)
        elif product == 'ABI-L1b-RadC':
            size_2km = (3000, 5000)

        #new_size = (int(size_1km[0] / resolution_km), 
        #            int(size_1km[1] / resolution_km))
        new_size = (1024, 1024)
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize(new_size),
                                            transforms.ToTensor()])
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        sample_files = self.files.values[[idx,idx+1],0] # consecutive snapshots
        samples = []
        for f in sample_files:
            #data = goesr.L1bBand(f).open_dataset()
            #x = data['Rad'].values.astype(np.float32)
            x = np.load(f).astype(np.float32)
            x = (x - 150) / (350 - 150)
            x[~np.isfinite(x)] = 0.
            x = self.transform(x)
            samples.append(x)
        return samples

class L1bPatches(data.Dataset):
    def __init__(self, data_directory, time_step=1, size=512, bands=[9,], mode='train'):
        self.patch_folders = [os.path.join(data_directory, f) for f in os.listdir(data_directory)]
        
        self.patches = [L1bPatch(f, time_step=time_step, size=size, bands=bands, mode=mode) for f in self.patch_folders]
        self.patches = data.ConcatDataset(self.patches)
        self.N = len(self.patches)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.patches[idx]

class L1bPatch(data.Dataset):
    def __init__(self, data_directory, time_step=1, size=512, bands=[9,], mode='train'):
        self.files = sorted([os.path.join(data_directory, f) for f in os.listdir(data_directory)])
        
        N = len(self.files)
        if mode == 'train':
            self.files = self.files[:int(N*0.8)]
        elif mode == 'valid':
            self.files = self.files[int(N*0.8):int(N*0.9)]
        elif mode == 'test':
            self.files = self.files[int(N*0.9):]
        
        self.N = len(self.files)-time_step
        self.time_step = time_step
        self.bands = bands
        self.rand_transform = transforms.Compose([
            transforms.RandomCrop((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        sample_files = [self.files[idx], self.files[idx+self.time_step]]
        N_samples = len(sample_files)
        N_bands = len(self.bands)
        
        ds = [xr.open_dataset(f) for f in sample_files]
        
        x = np.concatenate([d['Rad'].sel(band=self.bands).values for d in ds], 0)
        x[~np.isfinite(x)] = 0.
        
        mask = torch.Tensor((x[0] == x[0]).copy()).float()
        
        x = image_histogram_equalization(x)
        x = self.rand_transform(torch.Tensor(x))
        x = x.unsqueeze(1)
        #x = [x[i:i+N_bands] for i in range(N_samples)]
        return x, mask
    

class L1bResized(data.Dataset):
    def __init__(self, data_directory, time_step=1, new_size = (1024, 1024), jitter=6, band=9):
        self.files = sorted([os.path.join(data_directory, f) for f in os.listdir(data_directory)])
        self.N = len(self.files)-time_step
        self.time_step = time_step
        self.jitter = jitter
        self.new_size = new_size
        jitter_size = (new_size[0] + jitter, new_size[1] + jitter)
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize(jitter_size),
                                            transforms.ToTensor()])
        self.mu, self.sd = stats.get_sensor_stats("ABI")        
        self.mu = self.mu[band-1]
        self.sd = self.sd[band-1]
        
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        sample_files = [self.files[idx], self.files[idx+self.time_step]]
        print(sample_files)
        samples = []
        for f in sample_files:
            if (f[-3:] == '.nc') or (f[-4:] == '.nc4'):
                data = xr.open_dataset(f)
                x = data['Rad'].values.astype(np.float32)
                x = np.copy(x)
            elif f[-3:] == 'npy':
                x = np.load(f).astype(np.float32)
            x = (x - self.mu) / self.sd
            x[~np.isfinite(x)] = 0.
            x = self.transform(x)
            samples.append(x)

        #if np.random.uniform() > 0.5:
        #    samples = [torch.flipud(s) for s in samples]
        #if np.random.uniform() > 0.5:
        #    samples = [torch.fliplr(s) for s in samples]
        #if np.random.uniform() > 0.5:
        #    samples = [torch.rot90(s, 1, [1, 2]) for s in samples]        
    
        if self.jitter > 0:
            ix = np.random.choice(range(self.jitter))
            iy = np.random.choice(range(self.jitter))
            samples = [s[:,ix:ix+self.new_size[0],iy:iy+self.new_size[1]] for s in samples]
    
        #factor1 = np.random.uniform(0.9,1.1)
        #factor2 = np.random.uniform(0.9,1.1)
        #samples = [s*factor1 for s in samples]
        #samples[1] = samples[1]*factor2
        mask = (samples[0] != 0).float()
        return samples, mask

class L1bResizedThreeFrame(data.Dataset):
    def __init__(self, data_directory, time_step=6, new_size = (1024, 1024), mu=0, sd=1, jitter=10):
        self.files = sorted([os.path.join(data_directory, f) for f in os.listdir(data_directory)])
        self.N = len(self.files)-time_step
        self.time_step = time_step
        self.jitter = jitter
        self.new_size = new_size
        jitter_size = (new_size[0] + jitter, new_size[1] + jitter)
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize(jitter_size),
                                            transforms.ToTensor()])
        self.mu = mu
        self.sd = sd
        
    def get_statistics(self, n=20):
        print("Generating statistics")
        mu, sd = [], []
        rand_indices = np.random.choice(range(len(self)), n)
        for j in rand_indices:
            mu.append(self[j][0][0].mean().numpy())
            sd.append(self[j][0][0].std().numpy())
            
        self.mu = np.nanmean(mu)
        self.sd = np.nanmean(sd)
        print(f"Mean={self.mu}, Std={self.sd}")
        return self.mu, self.sd
        
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
       # random select middle frame [idx+1, idx+2, ..., idx+self.time_step-1]
        mids = np.arange(idx+1, idx+self.time_step)
        mid_idx = np.random.choice(mids, 1)[0]
        t = (mid_idx-idx) / self.time_step
        
        sample_files = [self.files[idx], self.files[mid_idx], self.files[idx+self.time_step]]
        samples = []
        for f in sample_files:
            if f[-3:] == '.nc':
                data = xr.open_dataset(f)
                x = data['Rad'].values.astype(np.float32)
                x = np.copy(x)
            elif f[-3:] == 'npy':
                x = np.load(f).astype(np.float32)
                x = np.copy(x)
                
            x = (x - self.mu) / self.sd            
            x[~np.isfinite(x)] = 0.
            x = self.transform(x)
            samples.append(x)

        if np.random.uniform() > 0.5:
            samples = [torch.flipud(s) for s in samples]
        if np.random.uniform() > 0.5:
            samples = [torch.fliplr(s) for s in samples]
        if np.random.uniform() > 0.5:
            samples = [torch.rot90(s, 1, [1, 2]) for s in samples]        
    
        if self.jitter > 0:
            ix = np.random.choice(range(self.jitter), 1)[0]
            iy = np.random.choice(range(self.jitter), 1)[0]
            samples = [s[:,ix:ix+self.new_size[0],iy:iy+self.new_size[1]] for s in samples]
            
        factor1 = np.random.uniform(0.9,1.1)
        #factor2 = np.random.uniform(0.9,1.1)
        samples = [s*factor1 for s in samples]
        #samples[1] = samples[1]*factor2
        #mask = (samples[0] != 0).float()
        
        return samples, t

## SloMo Training Dataset on NOAA GOES MesoScale
class MesoscaleThreeFrame(data.Dataset):
    def __init__(self, example_directory='/nobackupp10/tvandal/data/training_data/optical_flow/mesoscale/9Min-16Channels-Train-pt/',
                 channels=list(range(0,15)), n_upsample=9, n_overlap=3, train=True):
        self.example_directory = example_directory
        if not os.path.exists(self.example_directory):
            os.makedirs(self.example_directory)
        self._example_files()
        self.n_upsample = n_upsample
        self.n_overlap = n_overlap
        self.train = train
        self.channels = [c-1 for c in channels]

    def _example_files(self):
        self.example_files = [os.path.join(self.example_directory, f) for f in
                              os.listdir(self.example_directory) if 'npy' == f[-3:]]
        self.N_files = len(self.example_files)

    def _check_directory(self, year, day):
        yeardayfiles = [f for f in self.example_files if '%4i_%03i' % (year, day) in f]
        if len(list(yeardayfiles)) > 0:
            return True
        return False

    def transform(self, block):
        n_select = self.n_upsample + 1

        # randomly shift temporally
        i = np.random.choice(range(0,self.n_overlap))
        block = block[i:n_select+i]

        # randomly shift vertically 
        i = np.random.choice(range(0,6))
        block = block[:,:,i:i+128]

        # randomly shift horizontally
        i = np.random.choice(range(0,6))
        block = block[:,:,:,i:i+128]

        # randomly rotate image
        k = int((np.random.uniform()*4) % 4)
        if k > 0:
            block = np.rot90(block, axes=(2,3))

        # randomly flip 
        if np.random.uniform() > 0.5:
            block = np.flip(block, axis=2)#.copy()

        return block.copy()

    def __len__(self):
        return self.N_files

    def __getitem__(self, idx):
        f = self.example_files[idx]

        try:
            block = np.load(f)
        except ValueError:
            os.remove(f)
            del self.example_files[idx]
            self.N_files -= 1
            return self.__getitem__(idx-1)

        if self.train:
            block = self.transform(block)
        return_index = np.random.choice(range(1, self.n_upsample))
        I0 = torch.from_numpy(block[0, self.channels]).float()
        I1 = torch.from_numpy(block[-1, self.channels]).float()
        IT = torch.from_numpy(block[return_index, self.channels]).float()
        #sample = torch.stack([I0, IT, I1], dim=0)

        #except ValueError as err:
        #    print("Error", err)
        #    os.remove(f)
        #    del self.example_files[idx]
        #    self.N_files -= 1
        #    raise TypeError("Cannot load file: {}".format(f))

        return (I0, I1, IT), (return_index / (1.*self.n_upsample))


