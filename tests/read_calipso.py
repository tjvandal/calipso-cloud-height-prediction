import os, sys
import glob

sys.path.append('..')

from nexai.data import calipso

calipso_dir = '/nobackupp10/tvandal/data/calipso/'

def read_file(file):
    obj = calipso.CalipsoFile(file)
    ds = obj.file_to_xarray()
    return ds    
    
def match_with_l1b(file):
    ds = read_file(file)
    
    print(ds['time'])
    

if __name__ == '__main__':
    calipso_dir = '/nobackupp10/tvandal/data/calipso/'
    file_test = os.path.join(calipso_dir, '2019/01/CAL_LID_L2_01kmCLay-Standard-V4-20.2019-01-17T04-46-46ZD.hdf')
    match_with_l1b(file_test)