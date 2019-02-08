#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Paul Rosen, Piyush Agram, Heresh Fattahi, David Bekaert
# Organization: Jet Propulsion Laboratory, California Institute of Technology
# Copyright 2018 by the California Institute of Technology.
# ALL RIGHTS RESERVED.
# United States Government Sponsorship acknowledged. Any commercial use must be
# negotiated with the Office of Technology Transfer at the
# California Institute of Technology.
#
# This software may be subject to U.S. export control laws.
# By accepting this software, the user agrees to comply with all applicable U.S.
# export laws and regulations. User has the responsibility to obtain export
# licenses,  or other export authority as may be required before exporting
# such information to foreign countries or providing access to foreign persons.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


This repository is meant for Jupyter notebooks meant for instruction at UNAVCO Short Course for InSAR Theory and Processing in Aug 2018.


Instructors
-----------

1. Paul Rosen (JPL)
2. Gareth Funning (UC Riverside)
3. Scott Baker (UNAVCO)
4. Piyush Agram (JPL)
5. David Bekaert (JPL)
6. Heresh Fattahi (JPL)

Organization of this folder
---------------------------

This folder is organized as follows:

.
├── StartHere\_JupyterHubAccess.ipynb
│ 
├── Atmosphere
│   ├── Ionosphere
│   └── Troposphere
│ 
├── DataAccess
│   ├── GRFN.ipynb
│   └── SSARA.ipynb
│ 
├── GDAL
│   ├── 01\_IntroToRasterData.ipynb
│   ├── 02\_RasterDataManipulation.ipynb
│   └── 03\_RasterProjection.ipynb
│ 
├── Stripmap
│   ├── notebook\_docs
│   └── stripmapApp.ipynb
│ 
├── TOPS
│   ├── Tops.ipynb
│   └── support\_docs
│ 
├── Theory
│   ├── Principles\ and\ Theory.ipynb
│   └── InSARPrinciplesTheory\_UNAVCO\_18
│ 
├── TimeSeries
│   ├── 01\_PrepIgramStack.ipynb
│   ├── 02\_SBASInvert.ipynb
│   ├── 03\_TimefnInvert.ipynb
│   ├── GRFN
│   ├── Intro\_to\_GIAnT.ipynb
│   ├── Step0\_prepGIAnT.ipynb
│   ├── TimefnInvert.py
│   └── plotts\_notebook.ipynb 


Recommended order of Tutorials
-------------------------------

The suggested sequence of notebooks is as follows:

1.  StartHere_JupyterHubAccess.ipynb
    - Simple intro to Jupyter environments. We recommend using the Jupyter Notebook environment.

2.  Theory/Principles\ and\ Theory.ipynb
    - Theory of radar interferometry and radar signal properties

3.  Dataaccess/SSARA.ipynb
    - Access to SAR data from archives via SSARA 

4. Stripmap/stripmapApp.ipynb
    - Stripmap interferometry

5. TOPS/Tops.ipynb
    - TOPS interferometry

6. GDAL
    a. 01\_IntroToRasterData.ipynb
    b. 02\_RasterDataManipulation.ipynb
    c. 03\_RasterProjection.ipynb

7. Atmosphere
    a. Troposphere/Tropo.ipynb
    b. Ionosphere/stripmapApp_ionosphere.ipynb

8.  Timeseries
    a. Step0\_prepGIAnT.ipynb
    b. 01\_PrepIgramStack.ipynb
    c. 02\_SBASInvert.ipynb
    d. plotts\_notebook.ipynb
    e. 03\_TimefnInvert.ipynb
