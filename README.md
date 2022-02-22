# cloud-height


## References 

NOAA Derived motion winds: https://www.goes-r.gov/products/baseline-derived-motion-winds.html

NOAA Cloud top height: https://www.star.nesdis.noaa.gov/goesr/documents/ATBDs/Baseline/ATBD_GOES-R_Cloud%20Height_v3.0_July%202012.pdf 

Calipso: https://www-calipso.larc.nasa.gov/resources/calipso_users_guide/data_summaries/layer/index_v420.php#layer_top_altitude <br>
Opacity_Flag: If the surface was detected (i.e., the lidar surface altitude field does not contain fill values) then there are no opaque layers in the column. 

Cloudsat geoprof-lidar: https://www.cloudsat.cira.colostate.edu/data-products/2b-geoprof-lidar
wget -nd -N -r --user=[user_name] --ask-password ftp://ftp.cloudsat.cira.colostate.edu/2B-GEOPROF-LIDAR.P2_R05/2019
