============
Installation
============

Obtain the ISCE source code from the download-site. Unpack the tarball in a temporary folder, referred to below as *ISCE_SRC*. You can delete that folder after the complete installation of ISCE.

Before installing ISCE, several software packages - called dependencies - need to be installed. The dependencines can be installed using standard repository management tools or with the custom installation script included in the ISCE distribution or manually.

.. note::
   To install ISCE and its dependencies, you will need approximately 3.1 GB of free space on the hard drive, or 400 MB after removing the source and build directories.


*********************************************
Dependencies with repository management tools
*********************************************

The dependencies for ISCE can be installed using standard repository management tools like *yum* or *apt* on Linux and *Macports* on OS X platforms. We list simple commands for installing various dependencies for OS X and Ubuntu platforms. Repository management tools typically install the software in standard locations, e.g, /usr, /usr/local or /opt/local. If this is not desired, we suggest that the users install dependencies using the install script provided with the ISCE distribution.

.. tabularcolumns:: |p{1.5cm}|p{6.5cm}|p{6cm}|

+-----------+-----------------------------------------+------------------------------+
| Package   |     OS X                                |      Ubuntu 12.04            |
+===========+=========================================+==============================+
| Compilers | - ``sudo port install mp-gcc46``        | ``sudo yum install``         |
|           | - ``sudo port select gcc mp-gcc46``     | ``build-essential gfortran`` | 
+-----------+-----------------------------------------+------------------------------+
| Python    | - ``sudo port install python27``        | ``sudo yum install``         |
|           | - ``sudo port select python python27``  | ``python2.7-dev``            |
+-----------+-----------------------------------------+------------------------------+
| FFTW3     | - ``sudo port install fftw-3 +gcc46``   | ``sudo yum install``         |
|           | - ``sudo port install fftw-3-single``   | ``libfftw3-3 libfftw3-dev``  |
|           |   ``+ gcc46``                           |                              |
+-----------+-----------------------------------------+------------------------------+
| X11 files | ``sudo port install openmotif``         | ``sudo yum install``         |
|           |                                         | ``lesstif2 lesstif2-dev``    |
|           |                                         | ``libxt-dev``                |
+-----------+-----------------------------------------+------------------------------+
| HDF5      | ``sudo port install py27-h5py``         | ``sudo yum install``         |
|           |                                         | ``python-h5py``              |
|           |                                         | ``libhdf5-serial-dev``       |
+-----------+-----------------------------------------+------------------------------+
| scons     | ``sudo port install scons``             | ``sudo yum install  scons``  |
+-----------+-----------------------------------------+------------------------------+
| numpy     | ``sudo port install py27-numpy +gcc46`` | ``sudo yum install``         |
|           |                                         | ``python-numpy``             |
+-----------+-----------------------------------------+------------------------------+

Once you have installed these dependencies, ISCE can be installed using the provided installation script ( :ref:`isce_only_install` ) or manually ( :ref:`build_isce` ).

.. note::
   On Linux distributions other than Ubuntu 12.04, users must identify equivalent packages to the ones listed above and use the correct package names in the *yum* or *apt* commands. 

************************
With Installation Script
************************

This distribution includes a script that is designed to download, build and install all relevant packages needed for ISCE. The installation script is a bash script called *install.sh*, located in the setup directory of *ISCE_SRC*. Before running it, you should first ``cd`` to the setup directory. To get a quick help, issue the following command:

``./install.sh -h``

.. note::
   To build all the dependencies, you need the following packages to be preinstalled on your computer: gcc, g++, make, m4. Use your favorite package manager to install them (see `Tested Platforms`_).
   

Quick Installation
==================

It is recommended to install ISCE and all its dependencies at the same time by means of the installation script. For quick installation, use *install.sh* with the -p option (-p as in prefix):

``./install.sh -p INSTALL_FOLDER``

where *INSTALL_FOLDER* is the ISCE root folder where everything will be installed. *INSTALL_FOLDER* should be a local directory away from the system areas to avoid conflicts and so that administration privileges are not needed.

.. warning:: Do not use ISCE_SRC or any directory within the source tree as the installation folder.


Understanding the Script
========================

The *install.sh* bash script checks some system parameters before installing the dependencies and then ISCE.

1. The script checks for *gcc*, *g++*, *make* and *m4*, needed to build other packages. Your system should already come with both compilers *gcc* and *g++*. Any version will do. The required version of *gcc* (and *g++*) will be installed later by the script. *m4* is needed to create makefiles and *make* to run them. If you do not have any of those packages, you need to install it manually before using the installation script (see `Tested Platforms`_).

2. The script checks that you have Python installed and that its version is later than the required one. The script will also look for the *Python.h* file to make sure that you have the development package of Python. If not, Python will be installed by the script.

3. The script downloads, unpacks, builds and installs all the relevant packages needed for ISCE (see `ISCE Prerequisites`_). The file *setup_config.py* contains a list of places where the packages currently exist (i.e. where they should be downloaded from). By commenting out a particular package with a # at the beginning of the line, you can prevent that package from being installed, for example because an appropriate version is already installed on your system elsewhere. If the specified server for a particular package in this file is not available, then you can simply browse the web for a different server for this package and replace it in the *setup_config.py* file.

4. After checking some system parameters, the script generates a config file for *scons* to install ISCE, called *SConfigISCE*, located in the directory *$HOME/.isce*.

5. The script calls *scons* to install ISCE, using parameters from the *SConfigISCE* file.

6. Once ISCE is installed, a *.isceenv* file is placed in the directory *$HOME/.isce*. You have to source that file to export the environment variables each time you want to run ISCE: ``source ~/.isce/.isceenv``

.. note:: If an error occurs during the installation, the script exits and displays an error message. Try to fix it or send a copy of the message to the ISCE team. Once the error is fixed, you can run the script again (see `Adding Options`_).


Adding Options
==============

You can pass some options to the script so that the installation does not start from the beginning. You might want to download or install some packages only, especially after an abnormal script termination. Or you might want to install ISCE only, if all the dependencies are already installed. Again, it is recommended to use the **quick installation** step ; add options to the script only if you want to save time or reinstall a few packages.


Choosing Your Dependencies
--------------------------

By default, the script will download, unpack and install all the dependencies given in the *setup_config.py* file. If at some point, any of the dependencies has already been downloaded, unpacked or installed in the *INSTALL_FOLDER*, you can control the behaviour of the script with three extra options: -d -u -i, along with the -p option.

* ``-d DEP_LIST``: download the list of dependencies
* ``-u DEP_LIST``: unpack the list of dependencies
* ``-i DEP_LIST``: install the list of dependencies

where *DEP_LIST* can be **ALL** | **NONE** | **dep1,dep2...** (a comma-separated string, with no space). The dependencies can be: **GMP,MPFR,MPC,GCC,SCONS,FFTW,SZIP,HDF5,NUMPY,H5PY**

You can thus customize the installation with the following command:
``./install.sh -p INSTALL_FOLDER -d DEP_LIST -u DEP_LIST -i DEP_LIST``

Note that if an option is omitted, it defaults to NONE. But at least one of the three options (-d -u -i) has to be given, otherwise it equals to a quick installation.

-d) If a package has already been dowloaded to the *INSTALL_FOLDER*, you do not need to download it again. Specify only the packages you want to download with the **-d option** (those packages will then be untarred and installed).

-u) It might take time to untar some packages. You might want to skip that step if it has already been done inside the *INSTALL_FOLDER*. Specify only the dependencies that you want to unpack with the **-u option** (those dependencies will then be installed too). You do not need to pass those already given with the -d option.

-i) To install specific packages, pass them to the **-i option**. You do not need to pass those already given with the -d and -u options.

.. note:: At each step (download, unpack, install), the script processes all the specified packages before moving to the next step. If the script fails somewhere, you can just start from that step after fixing the bug.

.. note:: After installing the dependencies, the script will go on with the installation of ISCE, based on the generated *SConfigISCE* file.


Possible Combinations
----------------------

The following table shows how you can combine the three options -d, -u and -i to customize the installation of the dependencies. In any case, ISCE will be built after the specified dependencies are installed.

.. tabularcolumns:: |p{1cm}|p{1cm}|p{1cm}|p{3cm}|p{3cm}|p{3cm}|

+--------+--------+--------+------------------+------------------+------------------+
| \-d    | \-u    | \-i    | download         | unpack           | install          |
+========+========+========+==================+==================+==================+
| NONE   | NONE   | NONE   | nothing          | nothing          | nothing          |
+--------+--------+--------+------------------+------------------+------------------+
| NONE   | NONE   | list I | nothing          | nothing          | list I           |
+--------+--------+--------+------------------+------------------+------------------+
| NONE   | NONE   | ALL    | nothing          | nothing          | everything       |
+--------+--------+--------+------------------+------------------+------------------+
| NONE   | list U | NONE   | nothing          | list U           | list U           |
+--------+--------+--------+------------------+------------------+------------------+
| NONE   | list U | list I | nothing          | list U           | lists U & I      |
+--------+--------+--------+------------------+------------------+------------------+
| NONE   | list U | ALL    | nothing          | list U           | everything       |
+--------+--------+--------+------------------+------------------+------------------+
| NONE   | ALL    | \*     | nothing          | everything       | everything       |
+--------+--------+--------+------------------+------------------+------------------+
| list D | NONE   | NONE   | list D           | list D           | list D           |
+--------+--------+--------+------------------+------------------+------------------+
| list D | NONE   | list I | list D           | list D           | lists D & I      |
+--------+--------+--------+------------------+------------------+------------------+
| list D | NONE   | ALL    | list D           | list D           | everything       |
+--------+--------+--------+------------------+------------------+------------------+
| list D | list U | NONE   | list D           | lists D & U      | lists D & U      |
+--------+--------+--------+------------------+------------------+------------------+
| list D | list U | list I | list D           | lists D & U      | lists D & U & I  |
+--------+--------+--------+------------------+------------------+------------------+
| list D | list U | ALL    | list D           | lists D & U      | everything       |
+--------+--------+--------+------------------+------------------+------------------+
| list D | ALL    | \*     | list D           | everything       | everything       |
+--------+--------+--------+------------------+------------------+------------------+
| ALL    | \*     | \*     | everything       | everything       | everything       |
+--------+--------+--------+------------------+------------------+------------------+

.. note:: Where NONE is present, you can just omit that option... except when all three are NONE: give at least one option with NONE to restrict the installation to the ISCE package. For example, the following combinations are equivalent: ``-d NONE -u NONE -i NONE`` and ``-d NONE -i NONE`` and ``-i NONE``

.. note:: The symbol * means that the argument for that particular option does not matter.

.. _isce_only_install:

Installing ISCE Only
--------------------

If you have all the dependencies already installed, you might want to install the ISCE package only. Two possibilities are offered:

1. Pass NONE to the three options -d, -u and -i (see note in previous section):
``./install.sh -p INSTALL_FOLDER -i NONE``

Here the script generates a SConfigISCE based on your system configuration and sets up the environment for the installation.


2. Pass the *SConfigISCE* file as an argument to the -c option:
``./install.sh [-p INSTALL_FOLDER] -c SConfigISCE_FILE``

Here the environment variables are supposed to have been set up for the installation so that the script can find all it needs. You might need to pass the *INSTALL_FOLDER* with the -p option so the script knows where the dependencies have been installed.

Use the **-c option** if you have edited the *SConfigISCE* file generated by the script, e.g. to add path to X11 or Open Motif libraries. Or if you have created the *SConfigISCE* file manually, e.g. after a manual installation.

   
Tested Platforms
================

.. warning:: The following packages need to be preinstalled on your computer: gcc, g++, make, m4. If not, use a package manager to do so (check examples in the third column of the table below).

.. warning:: On a 64-bit platform, you need to have the C standard library so that gcc can generate code for 32-bit platform. To get it: ``sudo apt-get install libc6-dev-i386`` or ``sudo yum install glibc-devel.i686`` or ``sudo zypper install glibc-devel-32bit``


+---------------------------+----------+------------------------------------------------+-------------------+
| Operating system          | Platform | Installing prerequisites                       | Results           |
+===========================+==========+================================================+===================+
| Ubuntu 10.04 lucid        |  32-bit  | ``sudo apt-get install gcc g++ make m4``       | OK                |
+---------------------------+----------+------------------------------------------------+-------------------+
| Ubuntu 12.04 precise      |  64-bit  | ``sudo apt-get install gcc g++ make m4``       | OK                |
+---------------------------+----------+------------------------------------------------+-------------------+
| Linux Mint 13 Maya        |  64-bit  | ``sudo apt-get install gcc g++ make m4``       | OK                |
+---------------------------+----------+------------------------------------------------+-------------------+
| openSUSE 12.1             |  32-bit  | ``sudo zypper install gcc gcc-c++ make m4``    | OK                |
+---------------------------+----------+------------------------------------------------+-------------------+
| Fedora 17 Desktop Edition |  64-bit  | ``sudo yum install gcc gcc-c++ make m4``       | OK                |
+---------------------------+----------+------------------------------------------------+-------------------+
| Mac OS X Lion 10.7.2      |  64-bit  | *install Xcode*                                | OK                |
+---------------------------+----------+------------------------------------------------+-------------------+
| CentOS 6.3                |  64-bit  | ``sudo yum install gcc gcc-c++ make m4``       | OK                |
+---------------------------+----------+------------------------------------------------+-------------------+


*******************
Manual Installation
*******************

If you would prefer to install all the required packages by hand, read carefully the following sections and the installation guides accompanying the packages.


ISCE Prerequisites
==================

To compile ISCE, you will first need the following prerequisites:

* gcc >= 4.3.5 (C, C++, and Fortran compiler collection)
* fftw 3.2.2 (Fourier transform routines)
* Python >= 2.6 (Interpreted programming language)
* scons >= 2.0.1 (Software build system)
* For COSMO-SkyMed support

  - hdf5 >= 1.8.5 (Library for the HDF5 scientific data file format)
  - h5py >= 1.3.1 (Python interface to the HDF5 library)

Many of these prerequisites are available through package managers such as *MacPorts*, *Homebrew* and *Fink* on the Mac OS X operating system, *yum* on Fedora Linux, and *apt-get/aptitude* on Ubuntu. The only prerequisites that require special build procedures is fftw 3.2.2, the remaining prerequisites can be installed using the package managers listed above. At the very minimum, you should attempt to build all of the prerequisites, as well as ISCE itself with a set of compilers from the same build/version. This will reduce the possibility of build-time and run-time issues.


Building gcc
------------

Building gcc from source code can be a difficult undertaking. Refer to the detailed directions at http://gcc.gnu.org/install/ for further help.

On a Mac OS operating system, you can install Xcode to get gcc and some other tools. See https://developer.apple.com/xcode/

Building fftw-3.2.2
-------------------

* Get fftw-3.2.2 from http://www.fftw.org/fftw-3.2.2.tar.gz
* Untar the file *fftw-3.2.2.tar.gz* using ``tar -zxvf fftw-3.2.2.tar.gz``
* Go into the directory that was just created with ``cd fftw-3.2.2``
* Configure the build process by running ``./configure --enable-single --enable-shared --prefix=<directory>``
  where <directory> is the full path to an installation location where you have write access.
* Build the code using ``make``
* Finally, install fftw using ``make install``

Building python
---------------

* Get the Python source code from http://www.python.org/ftp/python/2.7.2/Python-2.7.2.tgz
* Untar the file *Python-2.7.2.tgz* using ``tar -zxvf Python-2.7.2.tgz``
* Go into the directory that was just created with ``cd Python-2.7.2``
* Configure the build process by running ``./configure --prefix=<directory>``
  where <directory> is the full path to an installation location where you have write access.
* Build Python by typing ``make``
* Install Python by typing ``make install``

Building scons
--------------

.. warning:: Ensure that you build scons using the python executable built in the previous step!

* Get scons from http://prdownloads.sourceforge.net/scons/scons-2.0.1.tar.gz
* Untar the file *scons-2.0.1.tar.gz* using ``tar -zxvf scons-2.0.1.tar.gz``
* Go into the directory that was just created with ``cd scons-2.0.1.tar.gz``
* Build scons by typing ``python setup.py build``
* Install scons by typing ``python setup.py install``

Building hdf5
-------------

.. note:: Only necessary for COSMO-SkyMed support

* Get the source code from http://www.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8.8/src/hdf5-1.8.8.tar.gz
* Untar the file *hdf5-1.8.8.tar.gz* using ``tar -zxvf hdf5.1.8.8.tar.gz``
* Go into the directory that was just created with ``cd hdf5-1.8.8``
* Configure the build process by running ``./configure --prefix=<directory>``
  where <directory> is the full path to an installation location where you have write access.
* Build hdf5 by typing ``make``
* Install hdf5 by typing ``make install``

Building h5py
-------------

.. note:: Only necessary for COSMO-SkyMed support

.. warning:: Ensure that you have Numpy and HDF5 already installed

.. warning:: Ensure that you build h5py using the python executable built in a few steps back!

* Get the h5py source code from http://h5py.googlecode.com/files/h5py-1.3.1.tar.gz
* Untar the file *h5py-1.3.1.tar.gz* using ``tar -zxvf h5py-1.3.1.tar.gz``
* Go into the directory that was just created with ``cd h5py-1.3.1``
* Configure the build process by running ``python setup.py configure -hdf5=<HDF5_DIR>``
* Build h5py by typing ``python setup.py build``
* Install h5py by typing ``python setup.py install``

.. note:: Once all these packages are built, you must setup your PATH and LD_LIBRARY_PATH variables in the unix shell to ensure that these packages are used for compiling and linking rather than the default system packages.

.. note:: If you use a pre-installed version of python to build numpy or h5py, you might need to have write access to the folder *dist-packages* or *site-packages* of python. If you are not root, you can install a python package in another folder and setup PYTHONPATH variable to point to the *site-packages* of that folder.


.. _build_isce:

Building ISCE
=============


Creating *SConfigISCE* File
---------------------------

Scons requires that configuration information be present in a directory specified by the environment variable SCONS_CONFIG_DIR. First, create a build configuration file, called *SConfigISCE* and place it in your chosen SCONS_CONFIG_DIR. The *SConfigISCE* file should contain the following information, note that the #-symbol denotes a comment and does not need to be present in the *SConfigISCE* file.::


        # The directory in which ISCE will be built
        PRJ_SCONS_BUILD = $HOME/build/isce-build
        # The directory into which ISCE will be installed
        PRJ_SCONS_INSTALL = $HOME/isce
        # The location of libraries, such as libstdc++, libfftw3
        LIBPATH = $HOME/lib64 $HOME/lib
        # The location of Python.h
        CPPPATH = $HOME/include/python2.7
        # The location of your Fortran compiler 
        FORTRAN = $HOME/bin/gfortran
        # The location of your C compiler
        CC = $HOME/bin/gcc
        # The location of your C++ compiler
        CXX = $HOME/bin/g++

        #libraries needed for mdx display utility
        MOTIFLIBPATH = /opt/local/lib     # path to libXm.dylib
        X11LIBPATH = /opt/local/lib       # path to libXt.dylib
        MOTIFINCPATH = /opt/local/include # path to location of the Xm directory with .h files
        X11INCPATH = /opt/local/include   # path to location of the X11 directory with .h files


.. warning:: The C, C++, and Fortran compilers should all be the same version to avoid build and run-time issues.


Installing ISCE
---------------

Untar the file *isce.tar.gz* to the folder *ISCE_SRC*

Now, ensure that your PYTHONPATH environment variable includes the ISCE configuration directory located in the ISCE source tree e.g.

``export PYTHONPATH=<ISCE_SRC>/configuration``

Create the environment variable SCONS_CONFIG_DIR that contains the path where *SConfigISCE* is stored:

``export SCONS_CONFIG_DIR=<PATH_TO_SConfigISCE_FOLDER>``

.. warning:: The path for SCONS_CONFIG_DIR should not end with '/'

.. note:: The configuration folder and SCONS_CONFIG_DIR are only required during the ISCE build phase, and is not needed once ISCE is installed.


Once everything is setup appropriately, issue the command

``scons install``

from the root of the isce source tree. This will build the necessary components into the directory specified in the configuration file as PRJ_SCONS_BUILD and install them into the location specified by PRJ_SCONS_INSTALL.


.. _Setting_Up_Environment_Variables:

Setting Up Environment Variables
--------------------------------

After the installation, each time you want to run ISCE, you need to setup PYTHONPATH and add a new environment variable ISCE_HOME:

``export ISCE_HOME=<isce_directory>``
where <isce_directory> is the directory specified in the configuration file as PRJ_SCONS_INSTALL

``export PYTHONPATH=$ISCE_HOME/components; <parent_of_isce_directory>``
where <parent_of_isce_directory> is the parent directory of ISCE_HOME.

***************************************
Special Notes on Creating Documentation
***************************************

Generating Documentation
========================

ISCE documentation is generated from rst files that are based on the markup syntax called reStructuredText_.

.. _reStructuredText: http://docutils.sourceforge.net/rst.html

To generate the documentation, navigate to the *docs/manual* folder inside the ISCE source tree. There, use the Makefile: ``make html`` or ``make latexpdf`` according to the type of output you want. Issue the command ``make`` to have a list of available output types.


Prerequisites
=============

To convert rst files, you need to have Sphinx_ installed (get it with your package manager or from Sphinx website).

If you want to build Sphinx from source, you might need to have Python compiled with zlib_ and the Python module setuptools_. To generate LaTex files, install first the LaTex_ software.

.. _Sphinx: http://pypi.python.org/pypi/Sphinx
.. _setuptools: http://pypi.python.org/pypi/setuptools
.. _zlib: http://zlib.net
.. _LaTex: http://www.latex-project.org


