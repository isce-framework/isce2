## ISCE-2 Stack Processors

Read the document for each stack processor for details.

+ [topsStack](./topsStack/README.md)
+ [stripmapStack](./stripmapStack/README.md)
+ [alosStack](./alosStack/alosStack_tutorial.txt)

### Installation

To use a stack processor you need to:

#### 1. Install ISCE as usual

#### 2. Setup paths for stack processors

The stack processors do not show up in the install directory of your isce software. They can be found in the isce source directory. Thus, extra path setup is needed.

2.1 Add the following path to your `${PYTHONPATH}` environment vavriable:

```bash
export ISCE_STACK={full_path_to_your_contrib/stack}
export PYTHONPATH=${PYTHONPATH}:${ISCE_STACK}
```

2.2 Depending on which stack processor you want to use, add the following path to your `${PATH}` environment variable:

+ For Sentinel-1 TOPS data

```bash
export PATH=${PATH}:${ISCE_STACK}/topsStack
```

+ For StripMap data

```bash
export PATH=${PATH}:${ISCE_STACK}/stripmapStack
```

+ For ALOS-2 data

```bash
export PATH=${PATH}:${ISCE_STACK}/alosStack
```

#### Important Note: ####

There are naming conflicts between topsStack and stripmapStack scripts. Therefore users **MUST** have the path of **ONLY ONE stack processor in their $PATH at a time**, to avoid the naming conflicts.

### References

Users who use the stack processors may refer to the following literatures:

For TOPS stack processing:

+ Fattahi, H., Agram, P., & Simons, M. (2017). A Network-Based Enhanced Spectral Diversity Approach for TOPS Time-Series Analysis. _IEEE Trans. Geosci. Remote Sens., 55_(2), 777-786, doi: [10.1109/TGRS.2016.2614925](https://doi.org/10.1109/TGRS.2016.2614925)

1. Ionospheric correction

+ Liang, C., Agram, P., Simons, M., & Fielding, E. J. (2019). Ionospheric Correction of InSAR Time Series Analysis of C-band Sentinel-1 TOPS Data. _IEEE Trans. Geosci. Remote Sens., 59_(9), 6755-6773, doi: [10.1109/TGRS.2019.2908494](https://doi.org/10.1109/TGRS.2019.2908494).

For StripMap stack processor and ionospheric phase estimation:

+ Fattahi, H., Simons, M., & Agram, P. (2017). InSAR Time-Series Estimation of the Ionospheric Phase Delay: An Extension of the Split Range-Spectrum Technique. _IEEE Trans. Geosci. Remote Sens., 55_(10), 5984-5996, doi: [10.1109/TGRS.2017.2718566](https://doi.org/10.1109/TGRS.2017.2718566)

For ALOS and ALOS-2 stack processing:

1. ScanSAR or multi-mode InSAR processing

+ Liang, C., & Fielding, E. J. (2017). Interferometry With ALOS-2 Full-Aperture ScanSAR Data. _IEEE Trans. Geosci. Remote Sens., 55_(5), 2739-2750, doi: [10.1109/TGRS.2017.2653190](https://doi.org/10.1109/TGRS.2017.2653190)

2. Ionospheric correction, burst-by-burst ScanSAR processing, and burst-mode spectral diversity (SD) or 
multi-aperture InSAR (MAI) processing

+ Liang, C., & Fielding, E. J. (2017). Measuring Azimuth Deformation With L-Band ALOS-2 ScanSAR Interferometry. _IEEE Trans. Geosci. Remote Sens., 55_(5), 2725-2738, doi: [10.1109/TGRS.2017.2653186](https://doi.org/10.1109/TGRS.2017.2653186)

3. Ionospheric correction

+ Liang, C., Liu, Z., Fielding, E. J., & Bürgmann, R. (2018). InSAR Time Series Analysis of L-Band Wide-Swath SAR Data Acquired by ALOS-2. _IEEE Trans. Geosci. Remote Sens., 56_(8), 4492-4506, doi: [10.1109/TGRS.2018.2821150](https://doi.org/10.1109/TGRS.2018.2821150)
