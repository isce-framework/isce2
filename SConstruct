#!/usr/bin/env python

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2010 California Institute of Technology. ALL RIGHTS RESERVED.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# United States Government Sponsorship acknowledged. This software is subject to
# U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
# (No [Export] License Required except when exporting to an embargoed country,
# end user, or in support of a prohibited end use). By downloading this software,
# the user agrees to comply with all applicable U.S. export laws and regulations.
# The user has the responsibility to obtain export licenses, or other export
# authority as may be required before exporting this software to any 'EAR99'
# embargoed foreign country or citizen of those countries.
#
# Author: Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



import os
import sys

if sys.version_info[0] == 2:
    print('Building with scons from python2')
else:
    raw_input = input
    print('Building with scons from python3')

if 'SCONS_CONFIG_DIR' in os.environ:
    sconsConfigDir = os.environ['SCONS_CONFIG_DIR']
else:
    print("Error. Need to set the variable SCONS_CONFIG_DIR in the shell environment")
    raise Exception

from configuration import sconsConfigFile
#allow scons to take the input argument --setupfile=someOtherFile to allow change of the default SConfigISCE
AddOption('--setupfile',dest='setupfile',type='string',default='SConfigISCE')
AddOption('--isrerun',dest='isrerun',type='string',default='no')
AddOption('--skipcheck',dest='skipcheck', action='store_true', default=False)

env = Environment(ENV = os.environ)
sconsSetupFile = GetOption('setupfile')
isrerun = GetOption('isrerun')
skipcheck = GetOption('skipcheck')

sconsConfigFile.setupScons(env,sconsSetupFile)
#add some information that are necessary to build the framework such as specific includes, libpath and so on
buildDir = env['PRJ_SCONS_BUILD']
libPath = os.path.join(buildDir,'libs')
#this is the directory where all the built library are put so they can easily be found during linking
env['PRJ_LIB_DIR'] = libPath

# add the libPath to the LIBPATH environment that is where all the libs are serched
env.AppendUnique(LIBPATH = [libPath])
# add the modPath to the FORTRANMODDIR environment that is where all the fortran mods are searched

#not working yet
modPath = os.path.join(buildDir,'mods')
env['FORTRANMODDIR'] =  modPath
env.AppendUnique(FORTRANPATH = [modPath])
env.AppendUnique(F90PATH = [modPath])
env.AppendUnique(F77PATH = [modPath])
#add the includes needed by the framework
imageApiInc = os.path.join(buildDir,'components/iscesys/ImageApi/include')
dataCasterInc = os.path.join(buildDir,'components/iscesys/ImageApi/DataCaster/include')
lineAccessorInc = os.path.join(buildDir,'components/isceobj/LineAccessor/include')
stdOEInc =  os.path.join(buildDir,'components/iscesys/StdOE/include')
utilInc =  os.path.join(buildDir,'components/isceobj/Util/include')
utilLibInc =  os.path.join(buildDir,'components/isceobj/Util/Library/include')

env.AppendUnique(CPPPATH = [imageApiInc,dataCasterInc,lineAccessorInc,stdOEInc,utilInc,utilLibInc])
env['HELPER_DIR'] = os.path.join(env['PRJ_SCONS_INSTALL'],'helper')
env['HELPER_BUILD_DIR'] = os.path.join(env['PRJ_SCONS_BUILD'],'helper')

#put the pointer function createHelp in the environment so it can be access anywhere
from configuration.buildHelper import createHelp
env['HELP_BUILDER'] = createHelp
#Create an env variable to hold all the modules added to the sys.path by default.
#They are the same as the one in in __init__.py in the same directory of this file
moduleList = []
installDir = env['PRJ_SCONS_INSTALL']
moduleList.append(os.path.join(installDir,'applications'))
moduleList.append(os.path.join(installDir,'components'))
env['ISCEPATH'] = moduleList
env.PrependUnique(LIBS=['gdal'])
Export('env')


inst = env['PRJ_SCONS_INSTALL']

####new part
#####PSA. Check for header files and libraries up front
confinst = Configure(env)
hdrparams = [('python3 header', 'Python.h', 'Install python3-dev or add path to Python.h to CPPPATH'),
          ('fftw3', 'fftw3.h', 'Install fftw3 or libfftw3-dev or add path to fftw3.h to CPPPATH and FORTRANPATH'),
          ('hdf5', 'hdf5.h', 'Install HDF5 of libhdf5-dev or add path to hdf5.h to CPPPATH'),
          ('X11', 'X11/Xlib.h', 'Install X11 or libx11-dev or add path to X11 directory to X11INCPATH'),
          ('Xm', 'Xm/Xm.h', 'Install libXm or libXm-dev or add path to Xm directory to MOTIFINCPATH'),
          ('openmp', 'omp.h', 'Compiler not built with OpenMP. Use a different compiler or add path to omp.h to CPPPATH'),]

allflag  = False
for (name,hname,msg) in hdrparams:
    if not (confinst.CheckCHeader(hname) or confinst.CheckCXXHeader(hname)):
        print('Could not find: {0} header for {1}'.format(hname, name))
        print('Error: {0}'.format(msg))
        allflag = True

libparams=  [('libhdf5', 'hdf5', 'Install hdf5 or libhdf5-dev'),
          ('libfftw3f', 'fftw3f', 'Install fftw3 or libfftw3-dev'),
          ('libXm', 'Xm', 'Install Xm or libXm-dev'),
          ('libXt', 'Xt', 'Install Xt or libXt-dev')]

for (name,hname,msg) in libparams:
    if not confinst.CheckLib(hname):
        print('Could not find: {0} lib for {1}'.format(hname, name))
        print('Error: {0}'.format(msg))
        allflag = True

if env.FindFile('fftw3.f', env['FORTRANPATH']) is None:
    print('Checking for F include fftw3 ... no')
    print('Could not find: fftw3.f header for fftw3')
    print('Error: Install fftw3 or libfftw3-dev or add path to FORTRANPATH')
    allflag = True
else:
    print('Checking for F include fftw3 ... yes'.format(name))


###This part added to handle GDAL and C++11
gdal_version = os.popen('gdal-config --version').read()
print('GDAL version: {0}'.format(gdal_version))
try:
    gdal_majorversion = int(gdal_version.split('.')[0])
    gdal_subversion = int(gdal_version.split('.')[1])
except:
    raise Exception('gdal-config not found. GDAL does not appear to be installed ... cannot proceed. If you have installed gdal, ensure that you have path to gdal-config in your environment')

env['GDALISCXX11'] = None
if (gdal_majorversion > 2) or (gdal_subversion >= 3):
    env['GDALISCXX11'] = 'True' 


##Add C++11 for GDAL checks
#Save default environment if C++11
if env['GDALISCXX11']:
    preCXX11 = confinst.env['CXXFLAGS']
    confinst.env.Replace(CXXFLAGS=preCXX11 + ['-std=c++11'])

if not confinst.CheckCXXHeader('gdal_priv.h'):
    print('Could not find: gdal_priv.h for gdal')
    print('Install gdal or add path to gdal includes to CPPPATH')
    allflag = True

if not confinst.CheckLib('gdal'):
    print('Could not find: libgdal for gdal')
    print('Install gdal or include path to libs to LIBPATH')
    allflag = True

###If C++11, revert to original environment
if env['GDALISCXX11']:
    confinst.env.Replace(CXXFLAGS=preCXX11)


###Decide whether to complain or continue
if (allflag and not skipcheck):
    print('Not all components of ISCE will be installed and can result in errors.')
    raw_input('Press Enter to continue.... Ctrl-C to exit')
elif (allflag and skipcheck):
    print('Not all components of ISCE will be installed and can result in errors.')
    print('User has requested to skip checks. Expect failures ... continuing')
else:
    print('Scons appears to find everything needed for installation')

try:
    # Older versions of scons do not have CheckProg, so 'try' to use it
    if confinst.CheckProg('cython3'):
        env['CYTHON3'] = True
    else:
        print('cython3 is not installed. Packages that depend on cython3 will not be installed.')
        env['CYTHON3'] = False
except:
    # If CheckProg is not available set env['CYTHON3'] = True and hope for the best
    # If the cython3 link does not exist, then a later error should prompt the user to
    # create the cython3 link to their cython installed as cython.
    env['CYTHON3'] = True
    pass

env = confinst.Finish()
###End of new part

### GPU branch-specific modifications
if 'ENABLE_CUDA' in env and env['ENABLE_CUDA'].upper() == 'TRUE':
    print('User requested compilation with CUDA, if available')
    try:
        env.Tool('cuda', toolpath=['scons_tools'])
        env['GPU_ACC_ENABLED'] = True
        print("CUDA-relevant libraries and toolkit found. GPU acceleration may be enabled.")
    except:
        env['GPU_ACC_ENABLED'] = False
        print("CUDA-relevant libraries or toolkit not found. GPU acceleration will be disabled.")
else:
    print('User did not request CUDA support. Add ENABLE_CUDA = True to SConfigISCE to enable CUDA support')
    env['GPU_ACC_ENABLED'] = False

### End of GPU branch-specific modifications


env.Install(inst, '__init__.py')
env.Install(inst, 'release_history.py')

if not os.path.exists(inst):
    os.makedirs(inst)

v = 0
if isrerun == 'no':
    cmd = 'scons -Q install --isrerun=yes'
    if skipcheck:
        cmd += ' --skipcheck'
    v = os.system(cmd)
if v == 0:
    env.Alias('install',inst)
    applications = os.path.join('applications','SConscript')
    SConscript(applications)
    components = os.path.join('components','SConscript')
    SConscript(components)
    defaults = os.path.join('defaults','SConscript')
    SConscript(defaults)
    library = os.path.join('library','SConscript')
    SConscript(library)
    contrib = os.path.join('contrib','SConscript')
    SConscript(contrib)

    if 'test' in sys.argv:
        #Run the unit tests
        env['Test'] = True
    else:
        #Don't run tests.
        #This option only installs test support package for future test runs.
        env['Test'] = False

    tests = os.path.join('test', 'SConscript')
    SConscript(tests)
