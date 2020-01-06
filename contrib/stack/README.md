## Stack Processors

Read the document for each stack processor for details.

+ [stripmapStack](./stripmapStack/README.md)
+ [topsStack](./topsStack/README.md)

### Installation

To use the TOPS or Stripmap stack processors you need to:

1. Install ISCE as usual

2. Depending on which stack processor you need to try, add the path of the folder containing the python scripts to your `$PATH` environment variable as follows:
   - add the full path of your **contrib/stack/topsStack** to `$PATH` to use the topsStack for processing a stack of Sentinel-1 TOPS data
   - add the full path of your **contrib/stack/stripmapStack** to `$PATH` to use the stripmapStack for processing a stack of StripMap data

Note: The stack processors do not show up in the install directory of your isce software. They can be found in the isce source directory. 

#### Important Note: ####

There might be conflicts between topsStack and stripmapStack scripts (due to comman names of different scripts). Therefore users **MUST only** have the path of **one stack processor in their $PATH environment at a time**, to avoid conflicts between the two stack processors.

### References

Users who use the stack processors may refer to the following literatures:

For StripMap stack processor and ionospheric phase estimation:

+ H. Fattahi, M. Simons, and P. Agram, "InSAR Time-Series Estimation of the Ionospheric Phase Delay: An Extension of the Split Range-Spectrum Technique", IEEE Trans. Geosci. Remote Sens., vol. 55, no. 10, 5984-5996, 2017. (https://ieeexplore.ieee.org/abstract/document/7987747/)

For TOPS stack processing:

+ H. Fattahi, P. Agram, and M. Simons, “A network-based enhanced spectral diversity approach for TOPS time-series analysis,” IEEE Trans. Geosci. Remote Sens., vol. 55, no. 2, pp. 777–786, Feb. 2017. (https://ieeexplore.ieee.org/abstract/document/7637021/)
