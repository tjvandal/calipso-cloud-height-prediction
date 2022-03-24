from pyhdf.SD import SD, SDC
import xarray as xr
import numpy as np
import os
import datetime as dt

class CalipsoFile:
    def __init__(self, file):
        self.file = file
                
    def file_to_xarray(self):
        cal = SD(self.file, SDC.READ)
        indicies = np.arange(len(cal.select('Latitude')[:]))
        ds = {}

        cad_score = cal.select('CAD_Score')[:,0]
        num_layers = cal.select('Number_Layers_Found')[:,0]
        rows = (np.abs(cad_score) == 100) + (cad_score == -127)
        
        times = cal.select('Profile_UTC_Time')[:,0][rows]
                
        #  'yymmdd.ffffffff'
        profile_time_to_dt = lambda t: dt.datetime(year=2000+int(t[:2]),
                                                   month=int(t[2:4]),
                                                   day=int(t[4:6]))\
                                       + dt.timedelta(days=float(t[6:]))
        
        times = [profile_time_to_dt(t) for t in times.astype(str)]        
        ds['lats'] = xr.DataArray(cal.select('Latitude')[:,0][rows], 
                            dims=['time'], 
                            coords=dict(time=times))
        ds['lons'] = xr.DataArray(cal.select('Longitude')[:,0][rows], 
                            dims=['time'], 
                            coords=dict(time=times))
        
        variables = ['Layer_Top_Altitude', 'Layer_Base_Altitude', 
                     'Layer_Top_Pressure', 'Layer_Base_Pressure',
                     'Surface_Top_Altitude_532']
        layers = np.arange(10)
        for v in variables:
            x = cal.select(v)[:][rows]
            if x.shape[1] == 1:
                ds[v] = xr.DataArray(x[:,0], coords=[times], dims=['time'])
            elif x.shape[1] == len(layers):
                ds[v] = xr.DataArray(x, coords=[times, layers], dims=['time', 'layer'])
                
            
        ds = xr.Dataset(ds)
        
        # no cloud when no layer is present but QA is good
        
        #print('after cad', np.histogram(ds['Layer_Top_Altitude'].values[:,0].flatten()))
        
        return ds