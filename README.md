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
- Key functions in the src directory:
    - calculate_trend_ols: Computes trends over segments of 10-73 years.
    - apply_mannkendall: Conducts Mann-Kendall significance testing.
---

## 3. Code Structure 

```
├── src/                     # Source code
│   ├── Data_Preprocess.py   # Pre-processing functions
│   ├── SAT_function.py      # Subordinate functions
├── data/                    # Demo input datasets
├── Data_preparing/          # Annual mean SAT anomalies, GSAT timeseries, trend calculations
├── Figure1/                 # Scripts for Figure 1
├── Figure2/                 # Scripts for Figure 2
├── Figure3/                 # Scripts for Figure 3
├── Figure4/                 # Scripts for Figure 4
├── Figure5/                 # Scripts for Figure 5
└── Extended_Figs/           # Scripts for extended figures
```
---
## 4. How to Run

To reproduce the analysis, it's strongly recommended to use the **same environment** as described in the setup section. The analysis process involves several key steps:

---

### **Overview of Analysis Workflow**

1. **Calculate the Large Ensemble GSAT Timeseries**  
   - This serves as the regressor for subsequent analysis.

2. **Perform Ordinary Least Squares (OLS) Regression**  
   - Calculate the **regression coefficient (Beta)** and **intercept (Alpha)** between **SAT anomalies** and **GSAT timeseries**.

3. **Segmented Trend Calculation**  
   - Based on the reconstructed forced and unforced SAT anomalies, calculate the segmented trends.

4. **Generate Global SAT Trend Maps**  
   - Create global maps showing SAT trends for both **human-forced** and **internal variability** contributions.

---
### **Step-by-Step Example: Generating Figure 1**

1. Separate Human-Forced and Internal Variability Trends  
- Navigate to the following Jupyter notebook:  
  `Observed_fingerprint/Data_preparing/HadCRUT5_GSAT_forced_unforced_trend_separation_Beta_Alpha.ipynb`

- **Inputs Required:**  
   - `Observed_fingerprint/data/GMSAT_SMILEs_ENS_annual_timeseries_obtained_basedOn_ModelENS.nc`  
   - `Observed_fingerprint/data/tas_HadCRUT5_annual_anomalies.nc`

- **Process:**  
   - Execute the notebook to separate human-forced and internal variability components of SAT anomalies.

- **Expected Output:**  
   - Processed datasets containing human-forced and internal variability SAT anomalies.


2. Calculate Segmented Trend Patterns  
- Navigate to:  
  `/Figure1/HadCRUT5_GSAT_forced_unforced_trend_pattern.ipynb`

- **Input Required:**  
   - Use the output dataset from **Step 1**.

- **Process:**  
   - Calculate the segmented trend patterns.  
   - The trends are calculated for various lengths (from **10 to 73 years**) within the period **1950–2022** using:  
     ```python
     np.arange(10, 74, 1)
     ```

- **Reference for Trend Length Calculation:**  
   - See the detailed process in:  
     `/Figure3/ACCESS/ACCESS_Forced_trend_calculation.ipynb`

- **Expected Output:**  
   - Defined trend length patterns for both forced and unforced SAT anomalies.

3. Plot the Final Figure 
- Navigate to:  
  `/Figure1/Figure1_HadCRUT5_GSAT_trend_95%_Plotting.ipynb`

- **Input Required:**  
   - Use the trend data generated from **Step 2**.

- **Process:**  
   - Execute the plotting script to generate **Figure 1**, as presented in the main text.

- **Expected Output:**  
   - Final visualization of the global SAT trend with **95% confidence intervals**.

---

### **Processing Time Considerations**

- The total processing time depends on the following factors:  
   - **Chunk Size:** Larger chunks may increase processing speed but require more memory.  
   - **Running Window:** The time complexity increases with longer trend lengths.  
   - For instance, a **73-year trend** calculation runs significantly faster than 10-year segments for constructing internal variability distribution.  
   - Optimal settings can be adjusted in the relevant scripts to balance speed and memory usage.

- *Tips*: 
    - Ensure that all datasets are correctly preprocessed and located in the specified directories.  
    - It's recommended to run the notebooks **step by step** and verify outputs before proceeding to the next stage.  
    - If you encounter memory issues, consider reducing the chunk size or optimizing data processing steps.

---
## 5. Contact
For inquiries, contact Hasi Aru at josie.aruhasi@mpimet.mpg.de.
