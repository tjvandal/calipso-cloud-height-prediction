# Cloud height prediction from GOES-16/17

## Install 

## Training Process

### Generate training dataset

Collocated calipso observations with GOES-16/17 thermal bands. 

GOES-16/17 data is located on NEX: `/nex/datapool/geonex/public/`

Calispo data path on stored on /nobackupp10: `/nobackupp10/tvandal/data/calipso/` 

Run command: `python collocate_calipso_goes.py` or `qsub collocate_calipso_goes.pbs`

### Model 

A fully convolutional model with spatial context only applied in layer 1 is applicable on arbitrarily sized images. Outputs two channels, cloud/no cloud classification and a regression value. 

```       
model = nn.Sequential(
    nn.Conv2d(10, 64, kernel_size=1, stride=1, padding=0),
    nn.ReLU(inplace=True), 
    nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
    nn.ReLU(inplace=True),
    #nn.Dropout(0.10),
    nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0),
) 
```

### Training 

`python train.py --model_path [path]  --data_path [data path] --batch_size [64] --lr [1e-3] ...`

Tensorboard logged at `model_path/`.  `tensorboard --logdir model_path`

### Testing

Cloud vs no cloud classification accuracy. Regression bias/variance statistics.  
`test.py`: Generates maps of cloud height and results on test set. 

## Application

Generate a cloud top height dataset from GOES-16 in 2020. Make a short video of cloud heights with corresponding geocolor imagery. 

Code: `inference.py` (in progress)

## Comparison with NOAA CTP

Pull NOAA CTP product onto /nobackupp10 


## References 

NOAA Derived motion winds: https://www.goes-r.gov/products/baseline-derived-motion-winds.html

NOAA Cloud top height: https://www.star.nesdis.noaa.gov/goesr/documents/ATBDs/Baseline/ATBD_GOES-R_Cloud%20Height_v3.0_July%202012.pdf 

Calipso: https://www-calipso.larc.nasa.gov/resources/calipso_users_guide/data_summaries/layer/index_v420.php#layer_top_altitude <br>
Opacity_Flag: If the surface was detected (i.e., the lidar surface altitude field does not contain fill values) then there are no opaque layers in the column. 

Cloudsat geoprof-lidar: https://www.cloudsat.cira.colostate.edu/data-products/2b-geoprof-lidar
wget -nd -N -r --user=[user_name] --ask-password ftp://ftp.cloudsat.cira.colostate.edu/2B-GEOPROF-LIDAR.P2_R05/2019
