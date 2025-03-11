# Observed Fingerprint of Global Warming

This repository contains all the codes used to produce the figures presented in the accompanying paper.

## 1. Environment Setup

The code is designed to run on a **Linux system** using a **Conda environment**. All necessary dependencies, including the required Python version and additional packages, are specified in the `environment.yml` file.

### Steps to Set Up the Environment

1. **Create the Conda Environment**  
   Run the following command to create the environment:

   ```bash
   conda env create -f environment.yml
   ```
2. **Activate the Environment**
    After creation, activate the environment using

    ```bash
    conda activate mykernel
    ```
    Note:
    - Replace mykernel with the actual name specified in the environment.yml file if it's different.
    - Ensure that Conda is properly installed on your system.
    - If any issues arise, verify the Python version and dependencies listed in environment.yml.
---
## 2. Data Preparation and Preprocessing

### Data Availability

The scripts preprocess large ensemble climate data, which can be downloaded from the following sources:

- **CMIP6 Large Ensemble Data:** [CMIP6 Data](https://aims2.llnl.gov/projects/cmip6/)

The observational datasets are accessible from:

- **HadCRUT5:** [Download Link](https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.0.2.0/download.html)  
- **Berkeley Earth Surface Temperature Data:** [Download Link](https://berkeleyearth.org/data/)  
- **NOAAGlobalTemp Data:** [Download Link](https://www.ncei.noaa.gov/products/land-based-station/noaa-global-temp)  

### Preprocessing and Subfunctions

- All datasets were interpolated onto a **2° x 2° global grid** using the **bilinear interpolation** method via **Climate Data Operators (CDO)**.

- The `src` directory contains scripts for key computational processes used by the main code. For example:

  - **Trend Calculation:** The subordinate function `src/SAT_function/calculate_trend_ols` calculates trends across different time segments, ranging from **10 to 73 years** in both observational datasets and model simulations.  
  - **Significance Testing:** The script `src/SAT_function/apply_mannkendall` applies the Mann-Kendall test to assess trend significance.

---

## 3. Code Structure 

- **project-root/**
```
│ ├── src/ # Source code for data processing 
    ├── Data_Preoricess.py    # Pre-processing function used in the main code
│   ├── SAT_function.py       # subordinate functions used in the main code
│ ├── data/ # Directory for input datasets ├── figures/ # Directory for generated figures 
└── README.md # Project overview and setup instruction
├── Data_preparing            # Annual mean SAT anomalies, GSAT timeseries, and observed trend pattern calculations
├── src

```
- **Run individual scripts to generate specific figures:**
```
├── Figure1/
|   ├──Figure1_HadCRUT5_GSAT_trend_95%_Plotting.ipynb        
├── Figure2/
|   ├──Figure2.py
|── Figure3/
|   ├──Figure3_plotting.py
|── Figure4/
|   ├──Figure4_plotting.py
├── Extended_Figs/
    ├──Extended_Fig**.py or .ipynb for plotting

```
---
## 4. How to run
For those wanting to try it out: The best is to use the same enviornment as used here. 
 1.

---
## 5. Contact
For any questions, please contact Hasi Aru (mailto:josie.aruhasi@mpimet.mpg.de).
