# PyCuAmpcor - Amplitude Cross-Correlation with GPU

## Contents

  * [1. Introduction](#1-introduction)
  * [2. Installation](#2-installation)
  * [3. User Guide](#3-user-guide)
  * [4. List of Parameters](#4-list-of-parameters)
  * [5. List of Procedures](#5-list-of-procedures)

## 1. Introduction

Ampcor (Amplitude cross correlation) in InSAR processing offers an estimate of spatial displacements (offsets) with the feature tracking method (also called as speckle tracking or pixel tracking). The offsets are in dimensions of a pixel or sub-pixel (with additional oversampling).

In practice, we

  * choose a rectangle window, $R(x,y)$, from the reference image, serving as the template,

 * choose a series of windows of the same size, $S(x+u, y+v)$, from the search image; the search windows are shifted in location by $(u,v)$;

  * perform cross-correlation between the search windows with the reference window, to obtain the normalized correlation surface $c(u,v)$;

  * find the maximum of $c(u,v)$ while its location, $(u_m,v_m)$, provides an estimate of the offset.

A detailed formulation can be found, e.g., by J. P. Lewis with [the frequency domain approach](http://scribblethink.org/Work/nvisionInterface/nip.html).

PyCuAmpcor follows the same procedure as the FORTRAN code, ampcor.F, in ROIPAC. In order to optimize the performance on GPU, some implementations are slightly different. In the [list the procedures](#5-list-of-procedures), we show the detailed steps of PyCuAmpcor, as well as their differences.

## 2. Installation

### 2.1 Installation with ISCE2

PyCuAmpcor is included in [ISCE2](https://github.com/isce-framework/isce2), and can be compiled/installed by CMake or SCons, together with ISCE2. An installation guide can be found at [isce-framework](https://github.com/isce-framework/isce2#building-isce).

Some special notices for PyCuAmpcor:

* PyCuAmpcor now uses the GDAL VRT driver to read image files. The memory-map accelerated I/O is only supported by GDAL version >=3.1.0. Earlier versions of GDAL are supported, but run slower.

* PyCuAmpcor offers a debug mode which outputs intermediate results. For end users, you may disable the debug mode by

    * CMake, use the Release build type *-DCMAKE_BUILD_TYPE=Release*
    * SCons, it is disabled by default with the -DNDEBUG flag in SConscript

* PyCuAmpcor requires CUDA-Enabled GPUs with compute capabilities >=2.0. You may  specify the targeted architecture by

    * CMake, add the flag *-DCMAKE_CUDA_FLAGS="-arch=sm_60"*, sm_35 for K40/80, sm_60 for P100, sm_70 for V100.

    * SCons, modify the *scons_tools/cuda.py* file by adding *-arch=sm_60* to *env['ENABLESHAREDNVCCFLAG']*.

  Note that if the *-arch* option is not specified, CUDA 10 uses sm_30 as default while CUDA 11 uses sm_52 as default. GPU architectures with lower compute capabilities will not run the compiled code properly.

### 2.2 Standalone Installation

You may also install PyCuAmpcor as a standalone package.

```bash
    # go to PyCuAmpcor source directory
    cd contrib/PyCuAmpcor/src
    # edit Makefile to provide the correct gdal include path and gpu architecture to NVCCFLAGS
    # call make to compile
    make
    # install 
    python3 setup.py install  
 ```

## 3. User Guide

The main procedures of PyCuAmpcor are implemented with CUDA/C++. A Python interface to configure and run PyCuAmpcor is offered. Sample python scripts are provided in *contrib/PyCuAmpcor/examples* directory.

### 3.1 cuDenseOffsets.py

*cuDenseOffsets.py*, as also included in InSAR processing stacks, serves as a general purpose script to run PyCuAmpcor. It uses *argparse* to pass parameters, either from a command line

```bash
cuDenseOffsets.py -r 20151120.slc.full -s 20151214.slc.full --outprefix ./20151120_20151214/offset --ww 64 --wh 64 --oo 32 --kw 300 --kh 100 --nwac 32 --nwdc 1 --sw  20 --sh 20 --gpuid 2
 ```

 or by a shell script

 ```
#!/bin/bash
reference=./merged/SLC/20151120/20151120.slc.full # reference image name 
secondary=./merged/SLC/20151214/20151214.slc.full # secondary image name
ww=64  # template window width
wh=64  # template window height
sw=20   # (half) search range along width
sh=20   # (half) search range along height
kw=300   # skip between windows along width
kh=100   # skip between windows along height
mm=0   # margin to be neglected 
gross=0  # whether to use a varying gross offset
azshift=0 # constant gross offset along height/azimuth 
rgshift=0 # constant gross offset along width/range
deramp=0 # 0 for mag (TOPS), 1 for complex linear ramp, 2 for complex no deramping  
oo=32  # correlation surface oversampling factor 
outprefix=./merged/20151120_20151214/offset  # output prefix
outsuffix=_ww64_wh64   # output suffix
gpuid=0   # GPU device ID
nstreams=2 # number of CUDA streams
usemmap=1 # whether to use memory-map i/o
mmapsize=8 # buffer size in GB for memory map
nwac=32 # number of windows in a batch along width
nwdc=1  # number of windows in a batch along height

rm $outprefix$outsuffix*
cuDenseOffsets.py --reference $reference --secondary $secondary --ww $ww --wh $wh --sw $sw --sh $sh --mm $mm --kw $kw --kh $kh --gross $gross --rr $rgshift --aa $azshift --oo $oo --deramp $deramp --outprefix $outprefix --outsuffix $outsuffix --gpuid $gpuid  --usemmap $usemmap --mmapsize $mmapsize --nwac $nwac --nwdc $nwdc 
 ```

Note that in PyCuAmpcor, the following names for directions are equivalent:
* row, height, down, azimuth, along the track.
* column, width, across, range, along the sight.

In the above script, the computation starts from the (mm+sh, mm+sw) pixel in the reference image, take a series of template windows of size (wh, ww) with a skip (sh, sw), cross-correlate with the corresponding windows in the secondary image, and iterate till the end of the images. The output offset fields are stored in *outprefix+outputsuffix+'.bip'*, which is in BIP format, i.e., each pixel has two bands of float32 data, (offsetDown, offsetAcross). The total number of pixels is given by the total number of windows (numberWindowDown, numberWindowAcross), which is computed by the script and also saved to the xml file.

If you are interested in a particular region instead of the whole image, you may specify the location of the starting pixel (in reference image) and the number of windows desired by adding

```
--startpixelac $startPixelAcross --startpixeldw $startPixelDown --nwa $numberOfWindowsAcross --nwd $numberOfWindowsDown
```

PyCuAmpcor supports two types of gross offset fields,
* static (--gross=0), i.e., a constant shift between reference and secondary images. The static gross offsets can be passed by *--rr $rgshift --aa $azshift*. Note that the margin as well as the starting pixel may be adjusted.
* dynamic (--gross=1), i.e., shifts between reference windows and secondary windows are varying in different locations. This is helpful to reduce the search range if you have a prior knowledge of the estimated offset fields, e.g., the velocity model of glaciers. You may prepare a BIP input file of the varying gross offsets (same format as the output offset fields), and use the option *--gross-file $grossOffsetFilename*. If you need the coordinates of reference windows, you may run *cuDenseOffsets.py* at first to find out the location of the starting pixel and the total number of windows. The coordinate for the starting pixel of the (iDown, iAcross) window will be (startPixelDown+iDown\*skipDown, startPixelAcross+iAcross\*skipAcross).

### 3.2 Customized Python Scripts

If you need more control of the computation, you may follow the examples to create your own Python script. The general steps are
* create a PyCuAmpcor instance
```python
# if installed with ISCE2
from isce.contrib.PyCuAmpcor.PyCuAmpcor import PyCuAmpcor
# if standalone
from PyCuAmpcor import PyCuAmpcr
# create an instance
objOffset = PyCuAmpcor()
```

* set various parameters, e.g., (see a [list of configurable parameters](#4-list-of-parameters) below)
```python
objOffset.referenceImageName="20151120.slc.full.vrt"
...
objOffset.windowSizeWidth = 64
...
```

* ask CUDA/C++ to check/initialize parameters
```python
objOffset.setupParams()
```

* set up the starting pixel(s) and gross offsets
```python
objOffset.referenceStartPixelDownStatic = objOffset.halfSearchRangeDown 
objOffset.referenceStartPixelAcrossStatic = objOffset.halfSearchRangeDown
# if static gross offset
objOffset.setConstantGrossOffset(0, 0)
# if dynamic gross offset, computed and stored in vD, vA 
objOffset.setVaryingGrossOffset(vD, vA)
# check whether all windows are within the image range
objOffset.checkPixelInImageRange() 
```

* and finally, run PyCuAmpcor
```python
objOffset.runAmpcor()
```

## 4. List of Parameters

**Image Parameters**

| PyCuAmpcor           | Notes                     |
| :---                 | :----                     |
| referenceImageName   | The file name of the reference/template image |
| referenceImageHeight | The height of the reference image |
| referenceImageWidth  | The width of the reference image |
| secondaryImageName   | The file name of the secondary/search image   |
| secondaryImageHeight | The height of the secondary image |
| secondaryImageWidth  | The width of the secondary image |
| grossOffsetImageName | The output file name for gross offsets  |
| offsetImageName      | The output file name for dense offsets  |
| snrImageName         | The output file name for signal-noise-ratio of the correlation |
| covImageName         | The output file name for variance of the correlation surface |

PyCuAmpcor now uses exclusively the GDAL driver to read images, only single-precision binary data are supported. (Image heights/widths are still required as inputs; they are mainly for dimension checking.  We will update later to read them with the GDAL driver). Multi-band is not currently supported, but can be added if desired.

The offset output is arranged in BIP format, with each pixel (azimuth offset, range offset). In addition to a static gross offset (i.e., a constant for all search windows), PyCuAmpcor supports varying gross offsets as inputs (e.g., for glaciers, users can compute the gross offsets with the velocity model for different locations and use them as inputs for PyCuAmpcor.

The offsetImage only outputs the (dense) offset values computed from the cross-correlations. Users need to add offsetImage and grossOffsetImage to obtain the total offsets.

The dimension/direction names used in PyCuAmpcor are:
* the inner-most dimension x(i): row, height, down, azimuth, along the track.
* the outer-most dimension y(j): column, width, across, range, along the sight.

Note that ampcor.F and GDAL in general use y for rows and x for columns.

Note also PyCuAmpcor parameters refer to the names used by the PyCuAmpcor Python class. They may be different from those used in C/C++/CUDA, or the cuDenseOffsets.py args.

**Process Parameters**

| PyCuAmpcor           | Notes                     |
| :---                 | :----                     |
| devID                | The CUDA GPU to be used for computation, usually=0, or users can use the CUDA_VISIBLE_DEVICES=n enviromental variable to choose GPU |
| nStreams | The number of CUDA streams to be used, recommended=2, to overlap the CUDA kernels with data copying, more streams require more memory which isn't alway better |
| useMmap              | Whether to use memory map cached file I/O, recommended=1, supported by GDAL vrt driver (needs >=3.1.0) and GeoTIFF |
| mmapSize             | The cache size used for memory map, in units of GB. The larger the better, but not exceed 1/4 the total physical memory. |
| numberWindowDownInChunk |  The number of windows processed in a batch/chunk, along lines |
| numberWindowAcrossInChunk | The number of windows processed in a batch/chunk, along columns |

Many windows are processed together to maximize the usage of GPU cores; which is called as a Chunk. The total number of windows in a chunk is limited by the GPU memory. We recommend
numberWindowDownInChunk=1, numberWindowAcrossInChunk=10, for a window size=64.


**Search Parameters**

| PyCuAmpcor           | Notes    |
| :---                 | :----                     |
| skipSampleDown       | The skip in pixels for neighboring windows along height |
| skipSampleAcross     | The skip in pixels for neighboring windows along width |
| numberWindowDown     | the number of windows along height |
| numberWindowAcross   | the number of windows along width  |
| referenceStartPixelDownStatic | the starting pixel location of the first reference window - along height component |
|referenceStartPixelAcrossStatic | the starting pixel location of the first reference window - along width component |

The C/C++/CUDA program accepts inputs with the total number of windows (numberWindowDown, numberWindowAcross) and the starting pixels of each reference window. The purpose is to establish multiple-threads/streams processing. Therefore, users are required to provide/compute these inputs, with tools available from PyCuAmpcor python class. The cuDenseOffsets.py script also does the job.

We provide some examples below, assuming a PyCuAmpcor class object is created as

```python
    objOffset = PyCuAmpcor()
```

**To compute the total number of windows**

We use the line direction as an example, assuming parameters as

```
   margin # the number of pixels to neglect at edges
   halfSearchRangeDown # the half of the search range
   windowSizeHeight # the size of the reference window for feature tracking
   skipSampleDown # the skip in pixels between two reference windows
   referenceImageHeight # the reference image height, usually the same as the secondary image height
```

and the number of windows may be computed along lines as

```python
   objOffset.numberWindowDown = (referenceImageHeight-2*margin-2*halfSearchRangeDown-windowSizeHeight) // skipSampleDown
```

If there is a gross offset, you may also need to subtract it when computing the number of windows.

The output offset fields will be of size (numberWindowDown, numberWindowAcross). The total number of windows numberWindows = numberWindowDown\*numberWindowAcross.

**To compute the starting pixels of reference/secondary windows**

The starting pixel for the first reference window is usually set as

```python
   objOffset.referenceStartPixelDownStatic = margin + halfSearchRangeDown
   objOffset.referenceStartPixelAcrossStatic = margin + halfSearchRangeAcross
```

you may also choose other values, e.g., for a particular region of the image, or a certain location for debug purposes.


With a constant gross offset, call

```python
   objOffset.setConstantGrossOffset(grossOffsetDown, grossOffsetAcross)
```

to set the starting pixels of all reference and secondary windows.

The starting pixel for the secondary window will be (referenceStartPixelDownStatic-halfSearchRangeDown+grossOffsetDown, referenceStartPixelAcrossStatic-halfSearchRangeAcross+grossOffsetAcross).

For cases you choose a varying grossOffset, you may use two numpy arrays to pass the information to PyCuAmpcor, e.g.,

```python
    objOffset.referenceStartPixelDownStatic = objOffset.halfSearchRangeDown + margin
    objOffset.referenceStartPixelAcrossStatic = objOffset.halfSearchRangeAcross + margin
    vD = np.random.randint(0, 10, size =objOffset.numberWindows, dtype=np.int32)
    vA = np.random.randint(0, 1, size = objOffset.numberWindows, dtype=np.int32)
    objOffset.setVaryingGrossOffset(vD, vA)
```

to set all the starting pixels for reference/secondary windows.

Sometimes, adding a large gross offset may cause the windows near the edge to be out of range of the orignal image. To avoid memory access errors, call

```python
   objOffset.checkPixelInImageRange()
```

to verify. If an out-of-range error is reported, you may consider to increase the margin or reduce the number of windows.

## 5. List of Procedures

The following procedures apply to one pair of reference/secondary windows, which are iterated through the whole image.

### 5.1 Read a window from Reference/Secondary images

* Load a window of size (windowSizeHeight, windowSizeWidth) from a starting pixel from the reference image

* Load a larger chip of size (windowSizeHeight+2\*halfSearchRangeDown, windowSizeWidth+2\*halfSearchRangeAcross) from the secondary image, the starting position is shifted by (-halfSearchRangeDown, -halfSearchRangeAcross) from the starting position of the reference image (may also be shifted additionally by the gross offset). The secondary chip can be viewed as a set of windows of the same size as the reference window, but shifted in locations varied within the search range.

**Parameters**

| PyCuAmpcor          | CUDA variable       | ampcor.F equivalent   | Notes                     |
| :---                | :---                | :----                 | :---                      |
| windowSizeHeight    | windowSizeHeightRaw | i_wsyi                |Reference window height     |
| windowSizeWidth     | windowSizeWidthRaw  | i_wsxi                |Reference window width      |
| halfSearchRangeDown | halfSearchRangeDownRaw | i_srchy            | half of the search range along lines |
| halfSearchRangeAcross | halfSearchRangeAcrossRaw | i_srchx            | half of the search range along  |


**Difference to ROIPAC**
No major difference


### 5.2 Perform cross-correlation and obtain an offset in units of the pixel size

* Take amplitudes (real) of the signals (complex or real) in reference/secondary windows
* Compute the normalized correlation surface between reference and secondary windows: the resulting correlation surface is of size (2\*halfSearchRangeDown+1, 2\*halfSearchRangeAcross+1); two cross-correlation methods are offered, time domain or frequency domain algorithms.
* Find the location of the maximum/peak in correlation surface.
* Around the peak position, extract a smaller window from the correlation surface for statistics, such as signal-noise-ratio (SNR), variance.

This step provides an initial estimate of the offset, usually with a large search range. In the following, we will zoom in around the peak, and oversample the windows with a smaller search range.


**Parameters**

| PyCuAmpcor          | CUDA variable       | ampcor.F equivalent   | Notes                     |
| :---                | :---                | :----                 | :---                      |
| algorithm           | algorithm           | N/A                   |  the cross-correlation computation method 0=Freq 1=time   |
| corrStatWindowSize  | corrStatWindowSize  | 21               | the size of correlation surface around the peak position used for statistics, may be adjusted   |


**Difference to ROIPAC**

* ROIPAC only offers the time-domain algorithm. The frequency-domain algorithm is faster and is set as default in PyCuAmpcor.
* ROIPAC proceeds from here only for windows with *good* match, or with high coherence. To maintain parallelism, PyCuAmpcor proceeds anyway while leaving the *filtering* to users in post processing.


### 5.3 Extract a smaller window from the secondary window for oversampling

* From the secondary window, we extract a smaller window of size (windowSizeHeightRaw+2\*halfZoomWindowSizeRaw, windowSizeWidthRaw+2\*halfZoomWindowSizeRaw) with the center determined by the peak position. If the peak position, e.g., along height, is OffsetInit (taking values in \[0, 2\*halfSearchRangeDownRaw\]), the starting position to extract will be OffsetInit+halfSearchRangeDownRaw-halfZoomWindowSizeRaw.

**Parameters**

| PyCuAmpcor          | CUDA variable       | ampcor.F equivalent   | Notes                     |
| :---                | :---                | :----                 | :---                      |
| N/A                 | halfZoomWindowSizeRaw  | i_srchp(p)=4       |  The smaller search range to zoom-in. In PyCuAmpcor, is determined by zoomWindowSize/(2\*rawDataOversamplingFactor)

**Difference to ROIPAC**

ROIPAC extracts the secondary window centering at the correlation surface peak. If the peak locates near the edge, zeros are padded if the extraction zone exceeds the window range. In PyCuAmpcor, the extraction center may be shifted away from peak to warrant all pixels being in the range of the original window.


### 5.4 Oversampling reference and (extracted) secondary windows

* Oversample both the reference and the (extracted) secondary windows by a factor of 2, which is to avoid aliasing in the complex multiplication of the SAR images. The oversampling is performed with FFT (zero padding), same as in ROIPAC.
* A deramping procedure is in general required for complex signals before oversampling, to shift the band center to 0. The procedure is only designed to remove a linear phase ramp. It doesn't work for TOPSAR, whose ramp goes quadratic. Instead, the amplitudes are taken before oversampling.
* the amplitudes (real) are then taken for each pixel of the complex signals in reference and secondary windows.

**Parameters**

| PyCuAmpcor          | CUDA variable       | ampcor.F equivalent   | Notes                     |
| :---                | :---                | :----                 | :---                      |
| rawDataOversamplingFactor | rawDataOversamplingFactor | i_ovs=2   | the oversampling factor for reference and secondary windows, use 2 for InSAR SLCs. |
| derampMethod        | derampMethod        | 1 or no effect on TOPS | Only for complex: 0=take mag (TOPS), 1=linear deramp (default), else=skip deramp.

**Difference to ROIPAC**

ROIPAC enlarges both windows to a size which is a power of 2; ideal for FFT. PyCuAmpcor uses their original sizes for FFT.

ROIPAC always performs deramping with Method 1, to obtain the ramp by averaging the phase difference between neighboring pixels. For TOPS mode, users need to specify 'mag' as the image *datatype* such that the amplitudes are taken before oversampling. Therefore, deramping has no effect. In PyCuAmpcor, derampMethod=0 is equivalent to *datatype='mag'*, taking amplitudes but skipping deramping. derampMethod=1 always performs deramping, no matter the 'complex' or 'real' image datatypes.

### 5.5 Cross-Correlate the oversampled reference and secondary windows

* cross-correlate the oversampled reference and secondary windows.
* other procedures are needed to obtain the normalized cross-correlation surface, such as calculating and subtracting the mean values.
* the resulting correlation surface is of size (2\*halfZoomWindowSizeRaw\*rawDataOversamplingFactor+1, 2\*halfZoomWindowSizeRaw\*rawDataOversamplingFactor+1). We cut the last row and column to make it an even sequence, or the size 2\*halfZoomWindowSizeRaw\*rawDataOversamplingFactor=zoomWindowSize.

**Parameters**

| PyCuAmpcor          | CUDA variable       | ampcor.F equivalent   | Notes                     |
| :---                | :---                | :----                 | :---                      |
| corrSurfaceZoomInWindow | zoomWindowSize  | i_cw   | The size of correlation surface of the (anti-aliasing) oversampled reference/secondary windows, also used to set halfZoomWindowSizeRaw. Set it to 16 to be consistent with ROIPAC. |

**Difference to ROIPAC**

In ROIPAC, an extra resizing step is performed on the correlation surface, from (2\*halfZoomWindowSizeRaw\*rawDataOversamplingFactor+1, 2\*halfZoomWindowSizeRaw\*rawDataOversamplingFactor+1) to (i_cw, i_cw), centered at the peak (in ROIPAC, the peak seeking is incorporated in the correlation module while is seperate in PyCuAmpcor). i_cw is a user configurable variable; it could be smaller or bigger than 2\*i_srchp\*i_ovs+1=17 (fixed), leading to extraction or enlargement by padding 0s. This procedure is not performed in PyCuAmpcor, as it makes little difference in the next oversampling procedure.

### 5.6 Oversample the correlation surface and find the peak position

* oversample the (real) correlation surface by a factor oversamplingFactor, or the resulting surface is of size (zoomWindowSize\*oversamplingFactor, zoomWindowSize\*oversamplingFactor) Two oversampling methods are offered, oversamplingMethod=0 (FFT, default), =1(sinc).
* find the peak position in the oversampled correlation surface, OffsetZoomIn, in range zoomWindowSize\*oversamplingFactor.
* calculate the final offset, from OffsetInit (which is the starting position of secondary window extraction in 2.4),

   offset = (OffsetInit-halfSearchRange)+OffsetZoomIn/(oversamplingFactor\*rawDataOversamplingFactor)

Note that this offset does not include the pre-defined gross offset. Users need to add them together if necessary.


**Parameters**

| PyCuAmpcor          | CUDA variable       | ampcor.F equivalent   | Notes                     |
| :---                | :---                | :----                 | :---                      |
| corrSurfaceOverSamplingFactor | oversamplingFactor  | i_covs   | The oversampling factor for the correlation surface |
| corrSurfaceOverSamplingMethod | oversamplingMethod | i_sinc_fourier=i_sinc | The oversampling method 0=FFT, 1=sinc. |

**Difference to ROIPAC**

ROIPAC by default uses the sinc interpolator (the FFT method is included but one needs to change the FORTRAN code to switch). For since interpolator, there is no difference in implementations. For FFT, ROIPAC always enlarges the window to a size in power of 2.
