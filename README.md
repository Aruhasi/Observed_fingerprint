# Observed fingerprint of global warming
This repository contains all codes which are used to producing the whole figures contained in this paper.
## 1. Data preprocessing
The scripts used preprocessed large ensemble climate data which can be downloaded from https://aims2.llnl.gov/projects/cmip6/. The observational dataset can be downloaded at HadCRUT5: https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.0.2.0/download.html, Berkeley Earth Surface temperature data: https://berkeleyearth.org/data/, NOAAGlobalTemp data:https://www.ncei.noaa.gov/products/land-based-station/noaa-global-temp.

## 2. Method:
------
- All datasets were interpolated onto 2 x 2 global grid using the bilinear interpolation method of Climate Data Operator (CDO)
- Run *src* to store calculations of trends for the various time segments (from 10 to 73 years in observations and models) and their trend significance test.
```
├── Data_preparing             # annual mean SAT anomalies, GSAT timeseries, observed trend pattern calculations
├── src
│   ├── Data_Preoricess.py         # second level contains the pre-processing codes \n
│   ├── SAT_function.py       # store the defined subfunction for calculation inbetween main script 
```
- Run individual scripts for figures
```
├── Figure1             # first level is the name for the model / reanalysis
├── Figure2
└── Figure3       # some pre-processing codes that apply to all models
```

## 3. Enviornment requirements
The code has been tested on Linux system. 

`enviroment.yml` provides a list of python dependences. 
an environment can be created by following code:
```bash
conda env create -f environment.yml
conda activate mykernel
```