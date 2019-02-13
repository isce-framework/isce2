#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 Author: Paul Rosen, Piyush Agram, Heresh Fattahi, David Bekaert
 Organization: Jet Propulsion Laboratory, California Institute of Technology
 Copyright 2018 by the California Institute of Technology.
 ALL RIGHTS RESERVED.
 United States Government Sponsorship acknowledged. Any commercial use must be
 negotiated with the Office of Technology Transfer at the
 California Institute of Technology.

 This software may be subject to U.S. export control laws.
 By accepting this software, the user agrees to comply with all applicable U.S.
 export laws and regulations. User has the responsibility to obtain export
 licenses,  or other export authority as may be required before exporting
 such information to foreign countries or providing access to foreign persons.

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


Jupyter setup
-------------

**OSX with Macports (assuming python36)**
  - sudo port install py36-jupyter
  - sudo pip-3.6 install jupyterlab
  - sudo pip-3.6 install jupyter_contrib_nbextensions
  - sudo jupyter-3.6 contrib nbextension install --user
  - sudo pip-3.6 install jupyter_nbextensions_configurator
  - sudo jupyter-3.6 nbextensions_configurator enable --user
  - sudo pip-3.6 install RISE
  - sudo jupyter-nbextension-3.6 install rise --py --sys-prefix


**With Anaconda**
  - python3 -m pip install jupyterlab
  - python3 -m pip install jupyter_contrib_nbextensions
  - python3 -m pip install jupyter_nbextensions_configurator
  - conda install -c conda-forge RISE


Recommended order of Tutorials
-------------------------------

The suggested sequence of notebooks is as follows:

1. [Theory/Principles and Theory.ipynb](Theory/Principles%20and%20Theory.ipynb)
   - Theory of radar signal properties and interferometry

2. [Dataaccess/SSARA.ipynb](DataAccess/SSARA.ipynb)
   - Access to SAR data from archives via SSARA 

3. [Stripmap/stripmapApp.ipynb](Stripmap/stripmapApp.ipynb)
   - Stripmap interferometry

4. [TOPS/Tops.ipynb](TOPS/Tops.ipynb)
   - TOPS interferometry

5. GDAL
   - [01 IntroToRasterData.ipynb](GDAL/01_IntroToRasterData.ipynb)
   - [02 RasterDataManipulation.ipynb](GDAL/02_RasterDataManipulation.ipynb)
   - [03 RasterProjection.ipynb](GDAL/03_RasterProjection.ipynb)

6. Atmosphere
   - [Troposphere/Tropo.ipynb](Atmosphere/Troposphere/Tropo.ipynb)
   - [Ionosphere/stripmapApp\_ionosphere.ipynb](Atmosphere/Ionosphere/stripmapApp_ionosphere.ipynb)

7. TimeSeries
   - [Step0 prepGIAnT.ipynb](TimeSeries/Step0_prepGIAnT.ipynb)
   - [01\_PrepIgramStack.ipynb](TimeSeries/01_PrepIgramStack.ipynb)
   - [02\_SBASInvert.ipynb](TimeSeries/02_SBASInvert.ipynb)
   - [plotts\_notebook.ipynb](TimeSeries/plotts_notebook.ipynb)
   - [03\_TimefnInvert.ipynb](TimeSeries/03_TimefnInvert.ipynb)


   
   
