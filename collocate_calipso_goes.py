import os, sys
import numpy as np
import xarray as xr
import glob

from nexai.data import calipso
from nexai.data.geo import goesr

from mpi4py import MPI

comm = MPI.COMM_WORLD
MPI_RANK = comm.Get_rank()
MPI_SIZE = comm.Get_size()

# Calipso directory
calipso_dir = '/nobackupp10/tvandal/data/calipso/'
calipso_files = glob.glob(calipso_dir + '*/*/*.hdf')
    

# GOES-16 directory
goes_dir = '/nex/datapool/geonex/public/GOES16/NOAA-L1B/'
bands = np.arange(7,17)
GEO = goesr.GOESL1b(data_directory=goes_dir,
                    product='ABI-L1b-RadF',
                    channels=bands)

# where to save data
write_dir='./data/calipso_goes_pairs_w_qa/'
if not os.path.exists(write_dir):
    os.makedirs(write_dir)

def match_calipso_file_to_goes(f, write_dir):
    radius = 4
    cal_ds = calipso.CalipsoFile(f).file_to_xarray()    
    prev_files = None
    for n, t in enumerate(cal_ds.time.values):
        lat = cal_ds.sel(time=t).lats.values
        lon = cal_ds.sel(time=t).lons.values
        
        
        if (lon > -30) or (lon < -130) or (np.abs(lat) > 45):
            continue

        ts = t.astype('M8[ms]').astype('O')
        files = GEO.snapshot_file(ts.year, ts.timetuple().tm_yday, 
                      ts.hour, ts.minute)
        
        out_file = os.path.join(write_dir, f'{t}_{lat}_{lon}.nc')
        if os.path.exists(out_file):
            continue
        
        if (prev_files == files).all():
            pass
        else:
            geo_ds = []
            for idx, f in files.iteritems():
                ds = goesr.L1bBand(f).reproject_to_latlon()
                ds['Rad'] = ds['Rad'].expand_dims(dim={'band': 1})
                geo_ds.append(ds)

            geo_ds = xr.concat(geo_ds, dim='band')
            geo_ds = geo_ds.assign_coords(coords=dict(band=files.index.values))

        lat_idx = np.argmin(np.abs(geo_ds.lat.values - lat))
        lon_idx = np.argmin(np.abs(geo_ds.lon.values - lon))

        patch = geo_ds.isel(lat=slice(lat_idx-radius, lat_idx+radius+1),
                            lon=slice(lon_idx-radius, lon_idx+radius+1))
        
        sample = cal_ds.sel(time=t).merge(patch)
        sample.to_netcdf(out_file)
        print(f"wrote to {out_file}")
        prev_files = files
        
        
        
        
    return


# Split files in train,valid,test
N = len(calipso_files)
calipso_train = calipso_files[:int(0.8*N)]
calipso_valid = calipso_files[int(0.8*N):int(0.9*N)]
calipso_test = calipso_files[int(0.9*N):]

# iterate calipso files
for i in range(MPI_RANK, len(calipso_files), MPI_SIZE):
    match_calipso_file_to_goes(calipso_files[i], write_dir)
