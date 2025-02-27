# Observed fingerprint of global warming
This repository contains all codes which are used to producing the whole figures contained in this paper.
## 1. Data preprocessing
All the adopted data in this study are publicly available. HadCRUT5 near surface temperature data version 5.0.1.0: \url{https://www.metoffice.gov.uk/hadobs/hadcrut5/data/current/download.html};
The CMIP6 data are freely available at ESGF \url{https://aims2.llnl.gov/projects/cmip6/}.

## 2. Code structure

```
├── CanESM2             # first level is the name for the model / reanalysis
├── CESM1_CAM5
├── CR20
├── CR20_allens
├── ERA5
├── ERA5_allens
├── GFDL_CM3
├── MK36
├── MPI_GE
├── MPI_GE_onepct
│   ├── codes           # second level contains the pre-processing codes \n
│   ├── composite       # some results such as composite analysis
│   ├── EOF_result      # EOF analysis results
│   ├── zg              # variables such as 'zg', 'ts'
│   ├── zg_Aug          # also variables that are separated by months
│   ├── zg_Jul
│   ├── zg_Jun
│   ├── ts  
└── zodes_for_all       # some pre-processing codes that apply to all models
```

## 3. Enviornment requirements
The code has been tested on Linux system. 

`enviroment.yml` provides a list of python dependences. 
an environment can be created by following code:
```bash
conda env create -f environment.yml
conda activate mykernel
```