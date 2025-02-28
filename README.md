# Observed fingerprint of global warming
This repository contains all codes which are used to producing the whole figures contained in this paper.
## 1. Data preprocessing
The scripts preprocess large ensemble climate data, which can be downloaded from:
- [CMIP6 Large Ensemble Data](https://aims2.llnl.gov/projects/cmip6/)

The observational datasets can be accessed at:
- **HadCRUT5:** [HadCRUT5 Data](https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.0.2.0/download.html)
- **Berkeley Earth Surface Temperature Data:** [Berkeley Earth Data](https://berkeleyearth.org/data/)
- **NOAAGlobalTemp Data:** [NOAA Global Temp](https://www.ncei.noaa.gov/products/land-based-station/noaa-global-temp)

---

## 2. Methodology:
- All datasets were interpolated onto a **2° x 2° global grid** using the **bilinear interpolation** method in Climate Data Operators (CDO).
- The `src` directory contains scripts for calculating trends across different time segments (ranging from **10 to 73 years** in both observations and models) and conducting trend significance tests.

### **Project Directory Structure**
```
├── Data_preparing            # Annual mean SAT anomalies, GSAT timeseries, and observed trend pattern calculations
├── src
│   ├── Data_Preoricess.py    # Pre-processing scripts
│   ├── SAT_function.py       # store the defined subfunction for calculation inbetween main script 
```
- Run individual scripts to generate specific figures:
```
├── Figure1
|   ├──Figure1_HadCRUT5_GSAT_trend_95%_Plotting.ipynb        
├── Figure2
|   ├──Figure2.py
|── Figure3
|   ├──Figure3_plotting.py
|── Figure4
|   ├──Figure4_plotting.py
```
