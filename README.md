# ISCE2

[![CircleCI](https://circleci.com/gh/isce-framework/isce2.svg?style=svg)](https://circleci.com/gh/isce-framework/isce2)

This is the Interferometric synthetic aperture radar Scientific Computing
Environment (ISCE).  Its initial development was funded by NASA's Earth Science
Technology Office (ESTO) under the Advanced Information Systems Technology
(AIST) 2008 and is currently being funded under the NASA-ISRO SAR (NISAR)
project.

THIS IS RESEARCH CODE PROVIDED TO YOU "AS IS" WITH NO WARRANTIES OF CORRECTNESS.
USE AT YOUR OWN RISK.

This software is open source under the terms of the the Apache License. Its export
classification is 'EAR99 NLR', which entails some restrictions and responsibilities.
Please read the accompanying LICENSE.txt and LICENSE-2.0 files.

ISCE is a framework designed for the purpose of processing Interferometric
Synthetic Aperture Radar (InSAR) data.  The framework aspects of it have been
designed as a general software development framework.  It may have additional
utility in a general sense for building other types of software packages.  In
its InSAR aspect ISCE supports data from many space-borne satellites and one
air-borne platform.  We continue to increase the number of sensors supported.
At this time the sensors that are supported are the following: ALOS, ALOS2,
COSMO_SKYMED, ENVISAT, ERS, KOMPSAT5, RADARSAT1, RADARSAT2, RISAT1, Sentinel1,
TERRASARX, UAVSAR and SAOCOM1A.

## Contents

1. [Software Dependencies](#software-dependencies)
   - [Installing dependencies with Anaconda](#with-anaconda)
   - [Installing dependencies with Macports](#with-macports)
   - [Note On 'python3' Exectuable Convention](#python3-convention)
   - [License required for dependencies to enable some workflows in ISCE](#license-required-for-dependencies-to-enable-some-workflows-in-isce)
2. [Building ISCE](#building-isce)
   - [SCons](#scons-recommended)
     - [Configuration control: SCONS\_CONFIG\_DIR and SConfigISCE](#configuration-control)
     - [Install ISCE](#install-isce)
   - [CMake](#cmake-experimental)
   - [Setup Your Environment](#setup-your-environment)
3. [Running ISCE](#running-isce)
   - [Running ISCE from the command line](#running-isce-from-the-command-line)
   - [Running ISCE in the Python interpreter](#running-isce-in-the-python-interpreter)
   - [Running ISCE with steps](#running-isce-with-steps)
   - [Running ISCE stack processors](./contrib/stack/README.md)
   - [Notes on Digital Elevation Models (DEMs)](#notes-on-digital-elevation-models)
4. [Input Files](#input-files)
5. [Component Configurability](#component-configurability)
   - [Component Names: Family and Instance](#component-names-family-and-instance)
   - [Component Configuration Files: Locations, Names, Priorities](#component-configuration-files-locations-names-priorities)
   - [Component Configuration Help](#component-configuration-help)
6.  [User community Forums](#user-community-forums)

------

## 1. Software Dependencies

### Basic:

* gcc >= 4.8+  (with C++11 support)
* fftw >= 3.2.2 (with single precision support)
* Python >= 3.5  (3.6 preferred)
* scons >= 2.0.1
* curl - for automatic DEM downloads
* GDAL and its Python bindings >= 2.2

### Optional:
#### For a few sensor types:

* hdf5 >= 1.8.5 and h5py >= 1.3.1  - for COSMO-SkyMed, Kompsat5, and 'Generic' sensor

#### For mdx (image visualization tool) options:

* Motif libraries and include files
* ImageMagick - for mdx production of kml file (advanced feature)
* grace - for mdx production of color table and line plots (advanced feature)

#### For the "unwrap 2 stage" option:

RelaxIV and Pulp are required.  Information on getting these packages if
you want to try the unwrap 2 stage option:

* RelaxIV (a minimum cost flow relaxation algorithm coded in C++ by
Antonio Frangioni and Claudio Gentile at the University of Pisa,
based on the Fortran code developed by Dimitri Bertsekas while
at MIT) is available at https://github.com/frangio68/Min-Cost-Flow-Class.
The RelaxIV files should be placed in the directory: 'contrib/UnwrapComp/src/RelaxIV' so that ISCE will compile it properly.

* PULP: Use easy\_install or pip to install it or else clone it from,
https://github.com/coin-or/pulp.  Make sure the path to the installed
pulp.py is on your PYTHONPATH environment variable (it should be the case
if you use easy\_install or pip).

#### For splitSpectrum and GPU modules:

* cython3 - must have an executable named cython3 (use a symbolic link)
* cuda - for GPUtopozero and GPUgeo2rdr
* opencv - for split spectrum

### With Anaconda

The conda requirements file is shown below:
```bash
cython
gdal
git
h5py
libgdal
pytest
numpy
fftw
scipy
basemap
scons
opencv
```

With the above contents in a textfile named "requirements.txt"

```bash
> conda install --yes --file requirements.txt
```

Ensure that you create a link in the anaconda bin directory for cython3.


### With Macports

The following ports (assuming gcc7 and python36) are needed on OSX 

```bash
gcc7
openmotif
python36
fftw-3 +gcc7 
fftw-3-single +gcc7
xorg-libXt +flat_namespace
git 
hdf5 +gcc7 
h5utils
netcdf +gcc7
netcdf-cxx
netcdf-fortran
postgresql95
postgresql95-server
proj
cairo
scons
opencv +python36
ImageMagick
gdal +expat +geos +hdf5 +netcdf +postgresql95 +sqlite3
py36-numpy +gcc7 +openblas
py36-scipy +gcc7 +openblas
py36-matplotlib +cairo +tkinter
py36-matplotlib-basemap
py36-h5py
py36-gdal
```

### Python3 Convention

We follow the convention of most package managers in using the executable
'python3' for Python3.x and 'python' for Python2.x.  This makes it easy to turn
Python code into executable commands that know which version of Python they
should invoke by naming the appropriate version at the top of the executable
file (as in #!/usr/bin/env python3 or #!/usr/bin/env python).  Unfortunately,
not all package managers (such as macports) follow this convention.  Therefore,
if you use one of a package manager that does not create the 'python3'
executable automatically, then you should place a soft link on your path to
have the command 'python3' on your path.  Then you will be able to execute an
ISCE application such as 'stripmapApp.py as "> stripmapApp.py" rather than as
"> /path-to-Python3/python stripmapApp.py".

### License required for dependencies to enable some workflows in ISCE

Some of the applications, or workflows (such as insarApp.py and isceApp.py),
in ISCE that may be familiar to users will not work with this open source version
of ISCE without obtaining licensed components.  WinSAR users who have downloaded
ISCE from the UNAVCO website (https://winsar.unavco.org/software/isce) have signed
the licence agreement and will be given access to those licensed components.  Others
wanting to use those specific workflows and components may be able to sign the
agreement through UNAVCO if they become members there.  Further instructions will
be available for a possible other procedure for obtaining a license directly from
the supplier of those components.

ISCE provides workflows that do not require the licensed components that
may be used effectively and that will be supported going forward by the ISCE team.
Users that need to work with newly processed data along with older processed data
may require those licensed components as a convenience unless they also reprocess
the older data with the same workflows available in this open source release.


-------

## Building ISCE

### SCons (recommended)

#### Configuration control

Scons requires that configuration information be present in a directory
specified by the environment variable SCONS\_CONFIG\_DIR.  First, create a
build configuration file, called SConfigISCE and place it in your chosen
SCONS\_CONFIG\_DIR.  The SConfigISCE file should contain the following
information, note that the #-symbol denotes a comment and does not need
to be present in the SConfigISCE file:

NOTE: Locations vary from system to system, so make sure to use the appropriate location.
      The one listed here are just for illustrative purpose.

```bash
# The directory in which ISCE will be built
PRJ_SCONS_BUILD = $ISCE_BUILD_ROOT/isce

# The directory into which ISCE will be installed
PRJ_SCONS_INSTALL = $ISCE_INSTALL_ROOT/isce

# The location of libraries, such as libstdc++, libfftw3 (for most system
# it's /usr/lib and/or /usr/local/lib/ and/or /opt/local/lib)
LIBPATH = $YOUR_LIB_LOCATION_HOME/lib64 $YOUR_LIB_LOCATION_HOME/lib

# The location of Python.h. If you have multiple installations of python
# make sure that it points to the right one
CPPPATH = $YOUR_PYTHON_INSTALLATION_LOCATION/include/python3.xm $YOUR_PYTHON_INSTALLATION_LOCATION/lib/python3.x/site-packages/numpy/core/include

# The location of the fftw3.h (most likely something like /usr/include or
# /usr/local/include /opt/local/include
FORTRANPATH =  $YOUR_FFTW3_INSTALLATION_LOCATION/include

# The location of your Fortran compiler. If not specified it will use the system one
FORTRAN = $YOUR_COMPILER_LOCATION/bin/gfortran

# The location of your C compiler. If not specified it will use the system one
CC = $YOUR_COMPILER_LOCATION/bin/gcc

# The location of your C++ compiler. If not specified it will use the system one
CXX = $YOUR_COMPILER_LOCATION/bin/g++

#libraries needed for mdx display utility
MOTIFLIBPATH = /opt/local/lib       # path to libXm.dylib
X11LIBPATH = /opt/local/lib         # path to libXt.dylib
MOTIFINCPATH = /opt/local/include   # path to location of the Xm
                                    # directory with various include files (.h)
X11INCPATH = /opt/local/include     # path to location of the X11 directory
                                    # with various include files

#Explicitly enable cuda if needed
ENABLE_CUDA = True
CUDA_TOOLKIT_PATH = $YOUR_CUDA_INSTALLATION  #/usr/local/cuda
```

In the above listing of the SConfigISCE file,  ISCE\_BUILD\_ROOT and
ISCE\_INSTALL\_ROOT may be actual environment variables that you create or else
you can replace them with the actual paths you choose to use for the build files
and the install files.  Also, in the following the capitalization of 'isce' as
lower case does matter.  This is the case-sensitive package name that Python
code uses for importing isce.

#### Install ISCE

```bash
cd isce
scons install
```

For a verbose install run:

```bash
scons -Q install
```

The scons command also allows you to explicitly specify the name of the
SConfigISCE file, which could be used to specify an alternative file for
(say SConfigISCE\_NEW) which must still be  located in the same
SCONS\_CONFIG\_DIR, run

```bash
scons install --setupfile=SConfigISCE_NEW
```

This will build the necessary components and install them into the location
specified in the configuration file as PRJ\_SCONS\_INSTALL.


##### Note about compiling ISCE after an unsuccessful build.

When building ISCE, scons will check the list of header files and libraries that
ISCE requires.  Scons will cache the results of this dependency checking.  So,
if you try to build ISCE and scons tells you that you are missing headers or
libraries, then you should remove the cached files before trying to build ISCE
again after installing the missing headers and libraries.  The cached files are
config.log, .sconfig.dblite, and the files in directory .sconf_temp.  You should
run the following command while in the top directory of the ISCE source (the
directory containing the SConstruct file):

```bash
> rm -rf config.log .sconfig.dblite .sconf_temp .sconsign.dblite
```

and then try "scons install" again.

The same also applies for rebuilding with SCons after updating the code, e.g.
via a `git pull`. If you encounter issues after such a change, it's recommended
to remove the cache files and build directory and do a fresh rebuild.

### CMake (experimental)
Make sure you have the following prerequisites:
* CMake ≥ 3.13
* GCC ≥ 4.8  (with C++11 support)
* Python ≥ 3.5
* Cython
* FFTW 3
* GDAL

```sh
git clone https://github.com/isce-framework/isce2
cd isce2
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/my/isce/install/location
make install
```

#### Additional cmake configuration options

CMake uses `CMAKE_PREFIX_PATH` as a global prefix for finding packages,
which can come in handy when using e.g. Anaconda:

```sh
cmake [...] -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
```

On macOS, cmake will also look for systemwide "frameworks",
which is usually not what you want when using Conda or Macports.

```sh
cmake [...] -DCMAKE_FIND_FRAMEWORK=NEVER
```

For packagers, the `PYTHON_MODULE_DIR` can be used to specify ISCE2's
package installation location relative to the installation prefix

```sh
cmake [...] -DPYTHON_MODULE_DIR=lib/python3.8m/site-packages
```

### Setup Your Environment

Once everything is installed, you will need to set the following environment
variables to run the programs included in ISCE ($ISCE_INSTALL_ROOT may be an
environment variable you created [above](#configuration-control) or else replace it with the actual
path to where you installed ISCE):

```bash
export PYTHONPATH=$ISCE\_INSTALL\_ROOT:$PYTHONPATH
```

and to put the executable commands in the ISCE applications directory on your
PATH for convenience,

```bash
export ISCE_HOME=$ISCE_INSTALL_ROOT/isce
export PATH=$ISCE_HOME/applications:$PATH
```

An optional environment variable is $ISCEDB.  This variable points to a
directory in which you may place xml files containing global preferences. More
information on this directory and the files that you might place there is
given below in Section on [Input Files](#input-files).  For now you can ignore this environment variable.

To test your installation and your environment, do the following:

```bash
> python3
>>> import isce
>>> isce.version.release_version
```
-----

## Running ISCE

### Running ISCE from the command line

Copy the example xml files located in the example directory in the ISCE source
tree to a working directory and modify them to point to your own data.  Run
them using the command:

```bash
> $ISCE_HOME/applications/stripmapApp.py isceInputFile.xml
```

or (with $ISCE\_HOME/applications on your PATH) simply,

```bash
> stripmapApp.py isceInputFile.xml
```

The name of the input file on the command line is arbitrary.  ISCE also looks
for appropriately named input files in the local directory

You can also ask ISCE for help from the command line:

```bash
> stripmapApp.py --help
```

This will tell you the basic command and the options for the input file.
Example input files are also given in the 'examples/input\_files' directory.

As explained in the [Component Configurability](#component-configurability) section below, it is also possible
to run stripmapApp.py without giving an input file on the command line.  ISCE will
automatically find configuration files for applications and components if they
are named appropriately.

### Running ISCE in the Python interpreter

It is also possible to run ISCE from within the Python interpreter.  If you have
an input file named insarInputs.xml you can do the following:

```bash
%> python3
>>> import isce
>>> from stripmapApp import Insar
>>> a = Insar(name="stripmapApp", cmdline="insarInputs.xml")
>>> a.configure()
>>> a.run()
```

(As explained in the [Component Configurability](#component-configurability) section below, if the file
insarInputs.xml were named stripmapApp.xml or insar.xml, then the 'cmdline' input
on the line creating 'a' would not be necessary.  The file 'stripmapApp.xml' would
be loaded automatically because when 'a' is created above it is given the name
'stripmapApp'.  A file named 'insar.xml' would also be loaded automatically if it
exists because the code defining stripmapApp.py gives all instances of it the
'family' name 'insar'.  See the Component Configurability section below for
details.)

### Running ISCE with steps

An other way to run ISCE is the following:

```bash
stripmapApp.py insar.xml --steps
```

This will run stripmapApp.py from beginning to end as is done without the
\-\-steps option, but with the added feature that the workflow state is
stored in files after each step in the processing using Python's pickle
module. This method of running stripmapApp.py is only a little slower
and it uses extra disc space to store the pickle files, but it
provides some advantage for debugging and for stopping and starting a
workflow at any predetermined point in the flow.

The full options for running stripmapApp.py with steps is the following:

```bash
stripmapApp.py insar.xml [--steps] [--start=<s>] [--end=<s>] [--dostep=<s>]
```

where "\<s\>" is the name of a step.  To see the full ordered list of steps
the user can issue the following command:

```bash
stripmapApp.py insar.xml --steps --help
```

The \-\-steps option was explained above.
The \-\-start and \-\-end option can be used together to process a range of steps.
The \-\-dostep option is used to process a single step.

For the \-\-start and \-\-dostep options to work, of course, requires that the
steps preceding the starting step have been run previously because the
state of the work flow at the beginning of the first step to be run must
be stored from a previous run.

An example for using steps might be to execute the end-to-end workflow
with \-\-steps to store the state of the workflow after every step as in,

```bash
stripmapApp.py insar.xml --steps
```

Then use \-\-steps to rerun some of the steps (perhaps you made a code
modification for one of the steps and want to test it without starting
from the beginning) as in

```bash
stripmapApp.py insar.xml --start=<step-name1> --end=<step-name2>
```

or to rerun a single step as in

```bash
stripmapApp.py insar.xml --dostep=<step-name>
```

Running stripmapApp.py with \-\-steps also enables one to enter the Python
interpreter after a run and load the state of the workflow at any stage
and introspect the objects in the flow and play with them as follows,
for example:

```bash
%> python3
>>> import isce
>>> f = open("PICKLE/formslc")
>>> import pickle
>>> a = pickle.load(f)
>>> o = f.getReferenceOrbit()
>>> t, x, p, off = o._unpackOrbit()
>>> print(t)
>>> print(x)
```

Someone with familiarity of the inner workings of ISCE can exploit
this mode of interacting with the pickle object to discover much about
the workflow states and also to edit the state to see its effect
on a subsequent run with \-\-dostep or \-\-start.

### Running [ISCE stack processors](./contrib/stack/README.md)

### Notes on Digital Elevation Models

- ISCE will automatically download SRTM Digital Elevation Models when you run an
application that requires a DEM.  In order for this to work follow the next 2
instructions:

1. You will need to have a user name and password from urs.earthdata.nasa.gov and
you need to include LPDAAC applications to your account.

    a. If you don't already have an earthdata username and password,
       you can set them at https://urs.earthdata.nasa.gov/

    b. If you already have an earthdata account, please ensure that
       you add LPDAAC applications to your account:
         - Login to earthdata here: https://urs.earthdata.nasa.gov/home
         - Click on my applications on the profile
         - Click on “Add More Applications”
         - Search for “LP DAAC”
         - Select “LP DAAC Data Pool” and “LP DAAC OpenDAP” and approve.

2. create a file named .netrc with the following 3 lines:

```bash
machine urs.earthdata.nasa.gov
    login your_earthdata_login_name
    password your_earthdata_password
```

3. set permissions to prevent others from viewing your credentials:

```bash
> chmod go-rwx .netrc
```

- When you run applications that require a dem, such as stripmapApp.py, if a dem
component is provided but the dem is referenced to the EGM96 geo reference (which
is the case for SRTM DEMs) it will be converted to have the  WGS84 ellipsoid as its
reference.  A new dem file with suffix wgs84 will be created.

- If no dem component is specified as an input a EGM96 will be automatically
downloaded (provided you followed the preceding instructions to register at
earthdata) and then it will be converted into WGS84.

- If you define an environment variable named DEMDB to contain the path to a
directory, then ISCE applications will download the DEM (and water body mask files
into the directory indicated by DEMDB.  Also ISCE applications will look for the
DEMs in the DEMDB directory and the local processing directory before downloading
a new DEM.  This will prevent ISCE from downloading multiple copies of a DEM if
you work with data in different subdirectories that cover similar geographic
locations.


## Input Files

Input files are structured 'xml' documents.  This section will briefly
introduce their structure using a special case appropriate for processing ALOS
data.  Examples for the other sensor types can be found in the directory
'examples/input\_files'.

The basic (ALOS) input file looks like this (indentation is optional):

### stripmapApp.xml (Option 1)

```xml
<stripmapApp>
<component name="stripmapApp">
    <property name="sensor name">ALOS</property>
    <component name="Reference">
        <property name="IMAGEFILE">
            /a/b/c/20070215/IMG-HH-ALPSRP056480670-H1.0__A
        </property>
        <property name="LEADERFILE">
            /a/b/c/20070215/LED-ALPSRP056480670-H1.0__A
        </property>
        <property name="OUTPUT">20070215</property>
    </component>
    <component name="Secondary">
        <property name="IMAGEFILE">
            /a/b/c/20061231/IMG-HH-ALPSRP049770670-H1.0__A
        </property>
        <property name="LEADERFILE">
            /a/b/c/20061231/LED-ALPSRP049770670-H1.0__A
        </property>
        <property name="OUTPUT">20061231</property>
    </component>
</component>
</stripmapApp>
```

The data are enclosed between an opening tag and a closing tag.  The \<stripmapApp\>
tag is closed by the \<\/stripmapApp\> tag for example.  This outer tag is necessary
but its name has no significance.  You can give it any name you like.  The
other tags, however, need to have the names shown above.  There are 'property',
and 'component' tags shown in this example.

The component tags have names that match a Component name in the ISCE code.
The component tag named 'stripmapApp' refers to the configuration information for
the Application (which is a Component) named "stripmapApp".  Components contain
properties and other components that are configurable.  The property tags
give the values of a single variable in the ISCE code.  One of the properties
defined in stripmapApp.py is the "sensor name" property.  In the above example
it is given the value ALOS.  In order to run stripmapApp.py two images need to
be specified.  These are defined as components named 'Reference' and 'Secondary'.
These components have properties named 'IMAGEFILE', 'LEADERFILE', and 'OUTPUT'
with the values given in the above example.

NOTE: the capitalization of the property and component names are not of any
importance.  You could enter 'imagefile' instead of 'IMAGEFILE', for example,
and it would work correctly.  Also extra spaces in names that include spaces,
such as "sensor name" do not matter.

There is a lot of flexibility provided by ISCE when constructing these input
files through the use of "catalog" tags and "constant" tags.

A "catalog" tag can be used to indicate that the contents that would normally
be found between an opening ad closing "component" tag are defined in another
xml file.  For example, the stripmapApp.xml file shown above could have been split
between three files as follows:

### stripmapApp.xml (Option 2)

```xml
<stripmapApp>
    <component name="insar">
        <property  name="Sensor name">ALOS</property>
        <component name="reference">
            <catalog>20070215.xml</catalog>
        </component>
        <component name="secondary">
            <catalog>20061231.xml</catalog>
        </component>
    </component>
</stripmapApp>
```

#### 20070215.xml

```xml
<component name="Reference">
    <property name="IMAGEFILE">
        /a/b/c/20070215/IMG-HH-ALPSRP056480670-H1.0__A
    </property>
    <property name="LEADERFILE">
        /a/b/c/20070215/LED-ALPSRP056480670-H1.0__A
    </property>
    <property name="OUTPUT">20070215 </property>
</component>
```

#### 20061231.xml

```xml
<component name="Secondary">
    <property name="IMAGEFILE">
        /a/b/c/20061231/IMG-HH-ALPSRP049770670-H1.0__A
    </property>
    <property name="LEADERFILE">
        /a/b/c/20061231/LED-ALPSRP049770670-H1.0__A
    </property>
    <property name="OUTPUT">20061231</property>
</component>
```
### rtcApp.xml 
The inputs are Sentinel GRD zipfiles
```xml
<rtcApp>
    <constant name="dir">/Users/data/sentinel1 </constant>
    <component name="rtcApp">
        <property name="sensor name">sentinel1</property>
        <property name="posting">100</property>
        <property name="polarizations">[VV, VH]</property>
        <property name="epsg id">32618</property>
        <property name="geocode spacing">100</property>
        <property name="geocode interpolation method">bilinear</property>
        <property name="apply thermal noise correction">True</property>
        <component name="reference">
        <property name="safe">$dir$/rtcApp/data/S1A_IW_GRDH_1SDV_20181221T225104_20181221T225129_025130_02C664_B46C.zip</property>
        <property name="orbit directory">$dir$/orbits</property>
        <property name="output directory">$dir$/rtcApp/output</property>
        </component>
    </component>
</rtcApp>
```
-----

## Component Configurability

In the examples for running stripmapApp.py ([Here](#running-isce-from-the-command-line) and [Here](#running-isce-in-the-python-interpreter) above) the input
data were entered by giving the name of an 'xml' file on the command line.  The
ISCE framework parses that 'xml' file to assign values to the configurable
variables in the isce Application stripmapApp.py.  The Application executes
several steps in its workflow.  Each of those steps are handled by a Component
that is also configurable from input data.  Each component may be configured
independently from user input using appropriately named and placed xml files.
This section will explain how to name these xml files and where to place them.

### Component Names: Family and Instance

Each configurable component has two "names" associated with it.  These names
are used in locating possible configuration xml files for those components. The
first name associated with a configurable component is its "family" name.  For
stripmapApp.py, the family name is "insar".  Inside the stripmapApp.py file an
Application is created from a base class named Insar.  That base class defines
the family name "insar" that is given to every instance created from it.  The
particular instance that is created in the file stripmapApp.py is given the
'instance name' 'stripmapApp'.  If you look in the file near the bottom you will
see the line,

```python
insar = Insar(name="stripmapApp")
```

This line creates an instance of the class Insar (that is given the family name
'insar' elsewhere in the file) and gives it the instance name "stripmapApp".

Other applications could be created that could make several different instances
of the Insar.  Each instance would have the family name "insar" and would be
given a unique instance name.  This is possible for every component.  In the
above example xml files instances name "Reference" and "Secondary" of a family named
"alos" are created.

### Component Configuration Files: Locations, Names, Priorities

The ISCE framework looks for xml configuration files when configuring every
Component in its flow in 3 different places with different priorities.  The
configuration sequence loads configuration parameters found in these xml files
in the sequence lowest to highest priority overwriting any parameters defined
as it moves up the priority sequence.  This layered approach allows a couple
of advantages.  It allows the user to define common parameters for all instances
in one file while defining specific instance parameters in files named for those
specific instances.  It also allows global preferences to be set in a special
directory that will apply unless the user overrides them with a higher priority
xml file.

The priority sequence has two layers.  The first layer is location of the xml
file and the second is the name of the file.  Within each of the 3 location
priorities indicated below, the filename priority goes from 'family name' to
'instance name'.  That is, within a given location priority level, a file
named after the 'family name' is loaded first and then a file with the
'instance name' is loaded next and overwrites any property values read from the
'family name' file.

The priority sequence for location is as follows:

(1)  The highest priority location is on the command line.  On the command line
the filename can be anything you choose.  Configuration parameters can also be
entered directly on the command line as in the following example:

```bash
> stripmapApp.py insar.reference.output=reference_c.raw
```

This example indicates that the variable named 'output' of the Component
named 'reference' belonging to the Component (or Application) named 'insar'
will be given the name "reference\_c.raw".

The priority sequence on the command line goes from lowest priority on the left
to highest priority on the right.  So, if we use the command line,

```bash
> stripmapApp.py myInputFile.xml insar.reference.output=reference_c.raw
```

where the myInputFile.xml file also gives a value for the insar reference output
file as reference\_d.raw, then the one defined on the right will win, i.e.,
reference\_c.raw.

(2) The next priority location is the local directory in which stripmapApp.py is
executed.  Any xml file placed in this directory named according to either the
family name or the instance name for any configurable component in ISCE will be
read while configuring the component.

(3) If you define an environment variable named ISCEDB, you can place xml files
with family names or instance names that will be read when configuring
Configurable Components.  These files placed in the ISCEDB directory have the
lowest priority when configuring properties of the Components.  The files placed
in the ISCEDB directory can be used to define global settings that will apply
unless the xml files in the local directory or the command line override those
preferences.

### Component Configuration Structure

However, the component tag has to have the family name of the Component/
Application.  In the above examples you see
that the outermost component tag has the name "insar", which is the family name
of the class Insar of which stripmapApp is an instance.


### Component Configuration Help

At this time there is limited information about component configurability
through the command

```bash
> stripmapApp.py --help
```

Future deliveries will improve this situation.  In the meantime we describe
here how to discover from the code which Components and parameters are
configurable.  One note of caution is that it is possible for a parameter
to appear to be configurable from user input when the particular flow will
not allow this degree of freedom.  Experience and evolving documentation will
be of use in determining these cases.

How to find out whether a component is configurable, what its configurable
parameters are, what "name" to use in the xml file, and what name to give to
the xml file.

Let's take as an example, Ampcor.py, which is in components/mroipac/ampcor.

Open it in an editor and search for the string "class Ampcor".  It is on
line 263.  You will see that it inherits from Component.  This is the minimum
requirement for it to be a configurable component.

Now look above that line and you will see several variable names being set
equal to a call to Component.Parameter.  These declarations define these
variables as configurable parameters.  They are entered in the "parameter\_list"
starting on line 268.  That is the method by which these Parameters are made
configurable parameters of the Component Nstage.

Each of the parameters defines the "public\_name", which is the "name" that you
would enter in the xml file.  For instance if you want to set the gross offset
in range, which is defined starting on line 88 in the variable
ACROSS\_GROSS\_OFFSET, then you would use an xml tag like the following (assuming
you have determined that the gross offset in range is about 150 pixels):

```xml
<property name="ACROSS_GROSS_OFFSET">150</property>
```

Now, to determine what to call the xml file and what "name" to use in the
component tag.  A configurable component has a "family" name and an instance
"name".  It is registered as having these names by calling the
Component.\_\_init\_\_ constructor, which is done on line 806.  On that line you
will see that the call to \_\_init\_\_ passes  'family=self.class.family' and
'name=name' to the Component constructor (super class of Ampcor).  The family
name is given as "nstage" on line 265.  The instance name is passed as the
value of the 'name=name' and was passed to it from whatever program created it.
Nstage is created in  components/isceobj/StripmapProc/runRefineSecondaryTiming.py where
it is given the name 'reference_offset1'  on line 35. If you are setting a parameter that
should be the same for all uses of Ampcor, then you can use the
family name 'ampcor' for the name of the xml file as 'ampcor.xml'.  It is more
likely that you will want to use the instance name 'reference\_offset1.xml'
Use the family name 'ampcor' for the component tag 'name'.

Example for SLC matching use of Ampcor:

Filename: reference\_offset1.xml:

```xml
<dummy>
<component name="ampcor">
    <property name="ACROSS_GROSS_OFFSET">150</property>
</component>
</dummy>
```

## User community forums

Read helpful information and participate in discussion with
the user/developer community on GitHub Discussions:

https://github.com/isce-framework/isce2/discussions
