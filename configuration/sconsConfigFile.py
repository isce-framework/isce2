#!/usr/bin/env python3

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

COMPILER_OPTIONS ={'COMPILER_OPTIMIZATION':'-O2','COMPILER_WARNING':'-Wall'}
LINKER_OPTIONS ={'LINKFLAGS':'-fopenmp'} # some systems don't need it, but ubuntu does
GFORTRANFLAGS = ['-ffixed-line-length-none' ,'-fno-second-underscore',    '-fPIC','-fno-range-check']
GCCFLAGS = ['-fPIC']
if 'DEVELOPER' in os.environ:
    GFORTRANFLAGS.append('-fbounds-check')
    GCCFLAGS.append('-fbounds-check')
class SconsConfig(object):
    def __init__(self):
        self.dir = None
        self.file = None
        return
sconsConfig = SconsConfig()

def readConfigFile(fileName):

    fin = open(fileName)
    allLines = fin.readlines()
    retDict = {}
    for line in allLines:
        if line.startswith('#'):#remove comments at the beginning of a line
            continue

        if line.find('#'):# remove comments at the end of a line
            indx = line.find('#')
            line = line[0:indx]

        line = substitute_env(line)   #replace '$VAR' string swith their value from the environment

        lineS =line.split('=')
        if len(lineS) == 1:#variable not defined
            continue

        key = lineS[0].strip()
        valueS = lineS[1].split()
        retDict[key] = valueS[0] if (len(valueS)==1) else valueS

    return retDict

def substitute_env(s):
    import re
    import os

    envs = re.findall(r"\$(\w*)",s)
    for x in envs:
        if x in os.environ.keys():
            s = s.replace("$"+x,os.environ[x])
        else:
            print(" ")
            print("Variable, $%s, used in the configuration file\n\n%s\n\nis not defined in the shell environment." % (x,os.path.join(sconsConfig.dir,sconsConfig.file)))
            print("Please correct this situation and try again.")
            print("(Either add that variable to the environment or edit the\nconfiguration file to use a variable in the environment\nor no variable at all).")
            print(" ")
            sys.exit(1)

    return s

def newList(lizt):
    #scons may add its CLVar to the list, which has base class UserList
    from collections import UserList
    rLizt = []
    if isinstance(lizt,list):
        rLizt.extend(lizt)
    elif isinstance(lizt,UserList):
        rLizt.extend(lizt)
    elif isinstance(lizt,str):
        rLizt.extend(lizt.split())
    else:
        print("ERROR: unexpected list type in newList")

    return rLizt

def mergeLists(list1,list2):
    retList = newList(list1)
    otherList = newList(list2)

    for el2 in otherList:
        if not retList.count(el2):
           retList.append(el2)

    return retList

def initCompilerFlags(flag,initDict,dict):

    if 'C' in flag:
        dkey = 'CCFLAGS'
    elif 'FORT' in flag:
        dkey = 'FORTRANFLAGS'
    elif 'LINK' in flag:
        dkey = 'LINKFLAGS'

    coList = []
    for cokey in initDict.keys():
        coList.append(initDict[cokey])

    if dkey in dict:
        if(dict[dkey]):#make sure that there is something otherwise newList in mergeLists fails
            return mergeLists(coList,dict[dkey])
        else:
            return coList
    else:
        return coList

def setupSunOs(dict):

        dict['LINKFLAGS'] = initCompilerFlags('LINKFLAGS',LINKER_OPTIONS,dict)
        dict['FORTRANFLAGS'] = initCompilerFlags('FORTRANFLAGS',COMPILER_OPTIONS,dict)
        dict['CCFLAGS'] = initCompilerFlags('CCFLAGS',COMPILER_OPTIONS,dict)
        dict['CCFLAGS'] = mergeLists(dict['CCFLAGS'],GCCFLAGS)

        if os.path.basename(dict['FORTRAN']).count('gfortran'):

            dict['LINKFLAGS'] = mergeLists(dict['LINKFLAGS'],'--allow-shlib-undefined')
            dict['FORTRANFLAGS'] = mergeLists(dict['FORTRANFLAGS'],GFORTRAN_COMPILE_FLAGS)
            dict['LIBS'] = ['gfortran']
            dict['FORTRANMODDIRPREFIX'] = '-J'


        dict['LIBS'] = mergeLists(dict['LIBS'], ['m'])
        if not 'STDCPPLIB' in dict:
            if not 'LIBPATH' in dict:
                print("Missing information. Either the variable STDC++LIB has to be set in the SConfig file or the LIBPATH needs to be set to be  \
                      able to deduce the right stdc++ library. Try to look for libstdc++*.so in the /usr/lib directory.")
                raise Exception
            else:# try to guess stdc++ from LIBPATH
                libstd = ''
                found = False
                libpath = dict['LIBPATH']
                if(isinstance(libpath,str)):
                    libpath = [libpath]
                for dir in libpath:
                    if not os.path.exists(dir):
                        continue
                    listDir = os.listdir(dir)
                    for file in listDir:
                        if file.startswith('libstdc++'):
                            libstd = 'stdc++'
                            found = True
                            break
                    if found:
                        break

                if not found:
                    print("Error. Cannot locate the stdc++ library in the directories specified by LIBPATH in the SConfig file.")
                    raise Exception
                dict['LIBS'] = mergeLists(dict['LIBS'],[libstd])
        else:
            dict['LIBS'] = mergeLists(dict['LIBS'],[dict['STDCPPLIB']])

        return dict


def setupLinux(dict):

        dict['LINKFLAGS'] = initCompilerFlags('LINKFLAGS',LINKER_OPTIONS,dict)
        dict['FORTRANFLAGS'] = initCompilerFlags('FORTRANFLAGS',COMPILER_OPTIONS,dict)
        dict['CCFLAGS'] = initCompilerFlags('CCFLAGS',COMPILER_OPTIONS,dict)
        dict['CCFLAGS'] = mergeLists(dict['CCFLAGS'],GCCFLAGS)

        if os.path.basename(dict['FORTRAN']).count('gfortran'):

            dict['LINKFLAGS'] = mergeLists(dict['LINKFLAGS'],'-Wl,-undefined,suppress')
            dict['FORTRANFLAGS'] = mergeLists(dict['FORTRANFLAGS'],GFORTRANFLAGS)
            dict['LIBS'] = ['gfortran']
            dict['FORTRANMODDIRPREFIX'] = '-J'


        dict['LIBS'] = mergeLists(dict['LIBS'], ['m'])
        if not 'STDCPPLIB' in dict:
            if not 'LIBPATH' in dict:
                print("Missing information. Either the variable STDC++LIB has to be set in the SConfig file or the LIBPATH needs to be set to be  \
                      able to deduce the right stdc++ library. Try to look for libstdc++*.so in the /usr/lib directory.")
                raise Exception
            else:# try to guess stdc++ from LIBPATH
                libstd = ''
                found = False
                libpath = dict['LIBPATH']
                if(isinstance(libpath,str)):
                    libpath = [libpath]
                for dir in libpath:
                    if not os.path.exists(dir):
                        continue
                    listDir = os.listdir(dir)
                    for file in listDir:
                        if file.startswith('libstdc++'):
                            libstd = 'stdc++'
                            found = True
                            break
                    if found:
                        break

                if not found:
                    print("Error. Cannot locate the stdc++ library in the directories specified by LIBPATH in the SConfig file.")
                    raise Exception
                dict['LIBS'] = mergeLists(dict['LIBS'],[libstd])
        else:
            dict['LIBS'] = mergeLists(dict['LIBS'],[dict['STDCPPLIB']])

        return dict


def setupDarwin(dict):

        dict['LINKFLAGS'] = initCompilerFlags('LINKFLAGS',LINKER_OPTIONS,dict)
        dict['FORTRANFLAGS'] = initCompilerFlags('FORTRANFLAGS',COMPILER_OPTIONS,dict)
        dict['CCFLAGS'] = initCompilerFlags('CCFLAGS',COMPILER_OPTIONS,dict)
        dict['CCFLAGS'] = mergeLists(dict['CCFLAGS'],GCCFLAGS)

        if os.path.basename(dict['FORTRAN']).count('gfortran'):

            dict['LINKFLAGS'] = mergeLists(dict['LINKFLAGS'],'-Wl,-undefined,dynamic_lookup')
            dict['FORTRANFLAGS'] = mergeLists(dict['FORTRANFLAGS'],GFORTRANFLAGS)
            dict['LIBS'] = ['gfortran']
            dict['FORTRANMODDIRPREFIX'] = '-J'


        dict['LIBS'] = mergeLists(dict['LIBS'], ['m'])
        if not 'STDCPPLIB' in dict:
            if not 'LIBPATH' in dict:
                print("Missing information. Either the variable STDC++LIB has to be set in the SConfig file or the LIBPATH needs to be set to be  \
                      able to deduce the right stdc++ library. Try to look for libstdc++*.dylib in the /usr/lib directory.")
                raise Exception
            else:# try to guess stdc++ from LIBPATH
                libstd = ''
                found = False
                libpath = dict['LIBPATH']
                if(isinstance(libpath,str)):
                    libpath = [libpath]
                for dir in libpath:
                    if not os.path.exists(dir):
                        continue
                    listDir = os.listdir(dir)
                    for file in listDir:
                        if file.startswith('libstdc++') and file.endswith('.dylib'):
                            libstd = file[3:(len(file) - 6)]
                            found = True
                            break
                    if found:
                        break

                if not found:
                    print("Error. Cannot locate the stdc++ library in the directories specified by LIBPATH in the SConfig file.")
                    raise Exception
                dict['LIBS'] = mergeLists(dict['LIBS'],[libstd])
        else:
            dict['LIBS'] = mergeLists(dict['LIBS'],[dict['STDCPPLIB']])

        return dict

def setupCompilers(dict):
    dict['LDMODULEPREFIX'] = ''
    if dict['SYSTEM_TYPE'].lower() == 'darwin':
        dict = setupDarwin(dict)
    elif dict['SYSTEM_TYPE'].lower() == 'linux':
        dict = setupLinux(dict)
    elif dict['SYSTEM_TYPE'].lower() == 'sunos':
        dict = setupSunOs(dict)
    else:
        print('System not supported. Supported ones are Darwin, Linux and SunOs. Use uname to find out the system type.')
        raise Exception

    if 'CPPDEFINES'  in dict:
        dict['CPPDEFINES'] = mergeLists(dict['CPPDEFINES'], ['NEEDS_F77_TRANSLATION', 'F77EXTERNS_LOWERCASE_TRAILINGBAR'])
    else:
        dict['CPPDEFINES'] = ['NEEDS_F77_TRANSLATION', 'F77EXTERNS_LOWERCASE_TRAILINGBAR']

    dict['F90FLAGS'] = []
    for val in dict['FORTRANFLAGS']:
        if val == '-ffixed-line-length-none':
            val = '-ffree-line-length-none'
        dict['F90FLAGS'].append(val)

    return dict

def setupArchitecture(dict):
    import platform as PL
    platform = PL.architecture()
    flag = ''
    if (platform[0] == '64bit'):
        flag = '-m64'
    elif (platform[0] == '32bit'):
        flag = '-m32'
    listKeys = ['CCFLAGS','FORTRANFLAGS','LINKFLAGS','F90FLAGS']
    for key in listKeys:
        if dict[key].count('-m32') or dict[key].count('-m64'):
            if dict[key].count('-m32'):#if choice if different from user's warn but leave the way it is
                if not (flag == '-m32'):
                    print('################################################################################')
                    print('Warning. The software will be compiled as 32 bit on a 64 bit machine. Most likely will not work. Change the flag to -m64 or comment out this flag and let the system figure it out.')
                    print('################################################################################')
            else:
                if not (flag == '-m64'):
                    print('################################################################################')
                    print('Warning. The software will be compiled as 64 bit on a 32 bit machine. Most likely will not work. Change the flag to -m32 or comment out this flag and let the system figure it out.')
                    print('################################################################################')
        else:#flag not present, add it
            dict[key].append(flag)

def setupScons(env,fileName = None):

    envDictionary = env.Dictionary()
    if 'SCONS_CONFIG_DIR' in os.environ:
        sconsConfigDir = os.environ['SCONS_CONFIG_DIR']
    else:
        print("Error. Need to set the variable SCONS_CONFIG_DIR in the shall environment")
        raise Exception
    if fileName == None:
        fileName = 'SConfig'

    sconsConfig.dir = sconsConfigDir
    sconsConfig.file = fileName

    retDict = readConfigFile(sconsConfigDir + '/' + fileName)
    if not  'SYSTEM_TYPE' in retDict:
        retDict['SYSTEM_TYPE'] = os.uname()[0]
    if  'FORTRAN' not  in retDict:#if not present then use default
        retDict['FORTRAN'] = env['FORTRAN']

    if  'F77' not  in retDict:#if not present then use default
        retDict['F77'] = retDict['FORTRAN']
    if  'F90' not  in retDict:#if not present then use default
        retDict['F90'] = retDict['FORTRAN']
    if  'F95' not  in retDict:#if not present then use default
        retDict['F95'] = retDict['FORTRAN']
    #if CXX is not explicitly defined, but CC is, then assume that CXX is in the same dir
    #unfortunatelly one cannot just use gcc to compile cpp code, since it generates that right obj code, but does not link the g++ libraries

    if (('CC'  in retDict) and ('CXX' not  in retDict)):# use g++ in the same directory where CC was defined.
        (head,tail) = os.path.split(retDict['CC'])
        slash = ''
        if not (head == ''):
            slash = '/'
        gpp = head + slash + 'g++'
        retDict['CXX']= gpp

    if ('CXX' not  in retDict):#if not present then use default
        retDict['CXX']= env['CXX']

    if ('CC' not  in retDict):#if not present then use default
        retDict['CC']= env['CC']


    for key, val in COMPILER_OPTIONS.items():
        if key not in retDict: #key not in SConfig file
            if key in env: #get value from environment if it is defined there
               retDict[key] = env[key]
            else: #or else use default defined at top of this file
               retDict[key] = val

    for key, val in LINKER_OPTIONS.items():
        if key not in retDict: #key not in SConfig file
            if key in env: #get value from environment if it is defined there
               retDict[key] = env[key]
            else: #or else use default defined at top of this file
               retDict[key] = val

    retDict = setupCompilers(retDict)
    setupArchitecture(retDict)
    for key in retDict.keys():
            if isinstance(retDict[key],list):
                for value in retDict[key]:
                    exec('env.AppendUnique(' + key + ' = [\'' + value + '\'])')

            else:# assume is a string
                exec('env.Replace(' + key + ' = \'' + retDict[key] + '\')')
    return env









