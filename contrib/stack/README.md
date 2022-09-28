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

2.1 Add the following path to your `${PYTHON_PATH}` environment vavriable:

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

+ H. Fattahi, P. Agram, and M. Simons, “A network-based enhanced spectral diversity approach for TOPS time-series analysis,” IEEE Trans. Geosci. Remote Sens., vol. 55, no. 2, pp. 777–786, Feb. 2017. (https://ieeexplore.ieee.org/abstract/document/7637021/)

For StripMap stack processor and ionospheric phase estimation:

+ H. Fattahi, M. Simons, and P. Agram, "InSAR Time-Series Estimation of the Ionospheric Phase Delay: An Extension of the Split Range-Spectrum Technique", IEEE Trans. Geosci. Remote Sens., vol. 55, no. 10, 5984-5996, 2017. (https://ieeexplore.ieee.org/abstract/document/7987747/)

For ALOS and ALOS-2 stack processing:

1. ScanSAR or multi-mode InSAR processing

+ C. Liang and E. J. Fielding, "Interferometry with ALOS-2 full-aperture ScanSAR data," IEEE Transactions on Geoscience and Remote Sensing, vol. 55, no. 5, pp. 2739-2750, May 2017.

2. Ionospheric correction, burst-by-burst ScanSAR processing, and burst-mode spectral diversity (SD) or 
multi-aperture InSAR (MAI) processing

+ C. Liang and E. J. Fielding, "Measuring azimuth deformation with L-band ALOS-2 ScanSAR interferometry," IEEE Transactions on Geoscience and Remote Sensing, vol. 55, no. 5, pp. 2725-2738, May 2017.

3. Ionospheric correction

+ C. Liang, Z. Liu, E. J. Fielding, and R. Bürgmann, "InSAR time series analysis of L-band wide-swath SAR data acquired by ALOS-2," IEEE Transactions on Geoscience and Remote Sensing, vol. 56, no. 8, pp. 4492-4506, Aug. 2018.

