'''
Apply calipso cloud height model to l1b
arguments: time, sensor, product
'''

import os, sys
import datetime as dt
import numpy as np
import xarray as xr
import glob

import torch
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.metrics
    
from nexai.data import calipso
from nexai.data.geo import goesr
from train import get_model
from dataloader import CalipsoGOES
    
from nexai.data.geo import stats

class CloudHeight(object):
    def __init__(self, 
                 model, 
                 checkpoint_file,  # surface wind model
                 product='ABI-L1b-RadC', 
                 bands=list(range(7,17))):
        
        # load surface wind model
        self.model = model#.cuda()
        checkpoint = torch.load(checkpoint_file)
        global_step = checkpoint['global_step']
        print(f'Loading checkout from step {global_step}')
        try:
            self.model.module.load_state_dict(checkpoint['model'])
        except:
            self.model.load_state_dict(checkpoint['model'])
        self.geo = goesr.GOESL1b(product=product,
                                 channels=bands)
        self.model = self.model.cuda()
        self.model.train(False)
        self.mu, self.sd = stats.get_sensor_stats('G16')
        self.mu = np.array([self.mu[b-1] for b in bands])[:,np.newaxis,np.newaxis]
        self.sd = np.array([self.sd[b-1] for b in bands])[:,np.newaxis,np.newaxis]
        
        
    def forward(self, t):
        files = self.geo.snapshot_file(t.year, t.timetuple().tm_yday, 
                                       t.hour, t.minute)

        #out_file = os.path.join(write_dir, f'{t}_{lat}_{lon}.nc')
        #if os.path.exists(out_file):
        #    continue

        geo_ds = []
        for idx, f in files.iteritems():
            ds = goesr.L1bBand(f).reproject_to_latlon()
            ds['Rad'] = ds['Rad'].expand_dims(dim={'band': 1})
            geo_ds.append(ds)

        geo_ds = xr.concat(geo_ds, dim='band')
        geo_ds = geo_ds.assign_coords(coords=dict(band=files.index.values))
        
        #geo_ds = geo_ds.sel(lat=slice(20,50), lon=slice(-105,-65))
        
        
        x = geo_ds['Rad'].values
        #x = (x - 150) / (350 - 150)
        x = (x - self.mu) / self.sd
        
        c, h, w = x.shape
        
        tile_size = 512
        stride = 500        
        
        # (1, C, H, W)
        #x_tensor = torch.Tensor(x).float().unsqueeze(0)
        # (1, C * tile_size, L)
        #x_unfold = torch.nn.functional.unfold(x_tensor, (tile_size, tile_size))
        #print('unfold', x_tensor.shape, x_unfold.shape)
        #return
    
        r = int((tile_size - 1) // 2)
        y = np.zeros((h, w))
        counter = np.zeros((h, w))
        pad = 0
        for i in range(0, h, stride):
            for j in range(0, w, stride):
                i = min(h-tile_size, i)
                j = min(w-tile_size, j)
                x_patch = x[np.newaxis,:,i:i+tile_size,j:j+tile_size]
                x_tensor = torch.Tensor(x_patch).float().cuda()
                output = self.model(x_tensor).cpu().detach().numpy()
                probs = output[:,:1]
                values = output[:,1:]
                values = (values * 10) + 5
                
                cloudy = probs > 0.5
                result = np.where(cloudy, values, np.zeros_like(values))
                
                y[i+pad:i+tile_size-pad, j+pad:j+tile_size-pad] += result[0,0] #result[0,0]
                counter[i+pad:i+tile_size-pad, j+pad:j+tile_size-pad] += 1
                
        print(counter)
        heights = y / counter #[r:-r,r:-r]
        
        
        geo_ds['CTH'] = xr.DataArray(heights, coords=[("lat", geo_ds.lat.values), ("lon", geo_ds.lon.values)])
        return geo_ds
    

def make_cloud_height_map():
    patch_size = 1
    model = get_model(input_size=patch_size)
    checkpoint_file = 'models/patchsize-1-classify-round5/checkpoint.pth.tar'
    runner = CloudHeight(model, 
                         checkpoint_file, 
                         product='ABI-L1b-RadF')
    
    t = dt.datetime.now() - dt.timedelta(hours=1, days=1)
    ds = runner.forward(t)
    
    print(ds)
    
    fig, axs = plt.subplots(1,2,figsize=(10,5))
    axs = axs.flatten()
    
    axs[0].imshow(ds['Rad'].sel(band=12).values[::-1], vmin=220, vmax=320, cmap='jet')
    
    im = axs[1].imshow(ds['CTH'].values[::-1], cmap='rainbow', vmin=0, vmax=12)
    #axs[1].set_colorbar()
    fig.colorbar(im, ax=axs[1], shrink=0.5)
    plt.savefig('heights_map.png', dpi=200)
    plt.close()
    #print(np.histogram(heights.flatten()))
    

        
def test_set_results():
    patch_size = 1
    
    model = get_model(input_size=patch_size).cuda()
    checkpoint_file = 'models/patchsize-1-classify-round5/checkpoint.pth.tar'
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model'])
    
    #runner = CloudHeight(model, checkpoint_file)
    data_path = '/nobackupp10/tvandal/cloud-height/data/calipso_goes_pairs_2/' 
    dataset = CalipsoGOES(data_path, patch_size=patch_size, mode='test')
    
    data = []
    
    N = 1000
    arr = np.random.choice(np.arange(len(dataset)), N)
    labels = []
    for i in arr:
        x, label = dataset[i]
        x = x.unsqueeze(0).cuda()
        prediction = model(x)
        data.append(prediction.cpu().detach().numpy())
        labels.append(label.unsqueeze(0))

        #print(prediction, label)
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    
    print(data.shape, labels.shape)
    
    cloudy = (labels != -9999.)
    predclass = (data[:,:1] > 0.5)
    
    accuracy = (cloudy == predclass).mean()
    f1 = sklearn.metrics.f1_score(cloudy.flatten().astype(bool), predclass.flatten().astype(bool))
    
    print(f"Cloud classification accuracy: {accuracy}")
    print(f"F1 score: {f1}")

    data[:,1:] = (data[:,1:] * 10) + 5
    labels[cloudy] = (labels[cloudy] * 10) + 5
    
    print("Pearson", scipy.stats.pearsonr(data[:,1:][cloudy], labels[cloudy]))
    err = data[:,1:][cloudy] - labels[cloudy]
    bias = np.mean(err)
    mae = np.abs(err).mean()
    print(f"MAE: {mae}")
    print(f"Bias: {bias}")

    plt.scatter(labels[cloudy].flatten(), data[:,1:][cloudy].flatten())
    plt.xlim([0,10])
    plt.ylim([0,10])
    plt.xlabel("Calipso Label (km)")
    plt.ylabel("Cloud Height Prediction")
    plt.savefig("heights.png")
    plt.close()
        
if __name__ == '__main__':
    make_cloud_height_map()
    test_set_results()