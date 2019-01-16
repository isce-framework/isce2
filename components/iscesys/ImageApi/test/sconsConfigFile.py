#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                                  Giangi Sacco
#                        NASA Jet Propulsion Laboratory
#                      California Institute of Technology
#                        (C) 2009  All Rights Reserved
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#!/usr/bin/env python3
import os
import sys
def readConfigFile(fileName):


    fin = open(fileName)
    allLines = fin.readlines()
    retDict = {}
    homeStr = os.environ['HOME']
    for line in allLines:
        if line.startswith('#'):#remove comments at the beginning of a line
            continue

        if line.find('#'):# remove comments at the end of a line
            indx = line.find('#')
            line = line[0:indx]
        lineS =line.split('=')
        if len(lineS) == 1:#variable not defined
            continue
        value =  []
        #value = ''
        key = lineS[0].strip()
        valueS = lineS[1].split()
        if len(valueS) == 1:
            valueS[0] = valueS[0].replace('$HOME',homeStr)#replace (if exists) the word $HOME with the env value
            retDict[key] = valueS[0]
        else:
            for i in range(len(valueS)):
                valueS[i] = valueS[i].replace('$HOME',homeStr)#replace (if exists) the word $HOME with the env value
                value.append(valueS[i])
                #value += " " + valueS[i]
            retDict[key] = value
    return retDict

def mergeLists(list1,list2):
    retList = list1
    for el2 in list2:
        if not list1.count(el2):
           list1.append(el2)
    return retList

def setupSunOs(dict):
        if os.path.basename(dict['FORTRAN']).count('gfortran'):

            if 'LINKFLAGS' in dict:
                if isinstance(dict['LINKFLAGS'],list):
                    dict['LINKFLAGS'] = mergeLists(dict['LINKFLAGS'], ['-Wall','--allow-shlib-undefined'])
                else:
                    dict['LINKFLAGS'] = [dict['LINKFLAGS'],'-Wall','--allow-shlib-undefined']

            else:
                dict['LINKFLAGS'] = ['-Wall','--allow-shlib-undefined']

            if 'FORTRANFLAGS' in dict:
                if isinstance(dict['FORTRANFLAGS'],list):
                    dict['FORTRANFLAGS'] = mergeLists(dict['FORTRANFLAGS'], ['-ffixed-line-length-none' ,'-fno-second-underscore',  '-O3' , '-Wall','-fPIC','-fno-range-check'])
                else:
                    dict['FORTRANFLAGS'] =[dict['FORTRANFLAGS'], '-ffixed-line-length-none' ,'-fno-second-underscore',  '-O3' , '-Wall','-fPIC','-fno-range-check']
            else:
                dict['FORTRANFLAGS'] = ['-ffixed-line-length-none' ,'-fno-second-underscore' ,  '-O3','-Wall','-fPIC','-fno-range-check']

            dict['LIBS'] = ['gfortran']
            dict['FORTRANMODDIRPREFIX'] = '-J'


        if 'CCFLAGS' in dict:
            if isinstance(dict['CCFLAGS'],list):
                dict['CCFLAGS'] = mergeLists(dict['CCFLAGS'], ['-O3', '-Wall','-fPIC'])
            else:
                dict['CCFLAGS'] = [dict['CCFLAGS'], '-O3', '-Wall','-fPIC']
        else:
            dict['CCFLAGS'] = ['-O3', '-Wall','-fPIC']

        dict['LIBS'] = mergeLists(dict['LIBS'], ['m'])
        if not 'STDCPPLIB' in dict:
            if not 'LIBPATH' in dict:
                print("Missing information. Either the variable STDC++LIB has to be set in the SConfig file or the LIBPATH needs to be set to be  \
                      able to deduce the right stdc++ library. Try to look for libstdc++*.so in the /usr/lib directory.")
                raise Exception
            else:# try to guess stdc++ from LIBPATH
                libstd = ''
                found = False
                for dir in dict['LIBPATH']:
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

        if os.path.basename(dict['FORTRAN']).count('gfortran'):

            if 'LINKFLAGS' in dict:
                if isinstance(dict['LINKFLAGS'],list):
                    dict['LINKFLAGS'] = mergeLists(dict['LINKFLAGS'], ['-Wall','-Wl,-undefined,suppress'])
                else:
                    dict['LINKFLAGS'] = [dict['LINKFLAGS'],'-Wall','-Wl,-undefined,suppress']

            else:
                dict['LINKFLAGS'] = ['-Wall','-Wl,-undefined,suppress']

            if 'FORTRANFLAGS' in dict:
                if isinstance(dict['FORTRANFLAGS'],list):
                    dict['FORTRANFLAGS'] = mergeLists(dict['FORTRANFLAGS'], ['-ffixed-line-length-none' ,'-fno-second-underscore',  '-O3' , '-Wall','-fPIC','-fno-range-check'])
                else:
                    dict['FORTRANFLAGS'] =[dict['FORTRANFLAGS'], '-ffixed-line-length-none' ,'-fno-second-underscore',  '-O3' , '-Wall','-fPIC','-fno-range-check']
            else:
                dict['FORTRANFLAGS'] = ['-ffixed-line-length-none' ,'-fno-second-underscore' ,  '-O3','-Wall','-fPIC','-fno-range-check']

            dict['LIBS'] = ['gfortran']
            dict['FORTRANMODDIRPREFIX'] = '-J'


        if 'CCFLAGS' in dict:
            if isinstance(dict['CCFLAGS'],list):
                dict['CCFLAGS'] = mergeLists(dict['CCFLAGS'], ['-O3', '-Wall','-fPIC'])
            else:
                dict['CCFLAGS'] = [dict['CCFLAGS'], '-O3', '-Wall','-fPIC']
        else:
            dict['CCFLAGS'] = ['-O3', '-Wall','-fPIC']

        dict['LIBS'] = mergeLists(dict['LIBS'], ['m'])
        if not 'STDCPPLIB' in dict:
            if not 'LIBPATH' in dict:
                print("Missing information. Either the variable STDC++LIB has to be set in the SConfig file or the LIBPATH needs to be set to be  \
                      able to deduce the right stdc++ library. Try to look for libstdc++*.so in the /usr/lib directory.")
                raise Exception
            else:# try to guess stdc++ from LIBPATH
                libstd = ''
                found = False
                for dir in dict['LIBPATH']:
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

        if os.path.basename(dict['FORTRAN']).count('gfortran'):

            if 'LINKFLAGS' in dict:
                if isinstance(dict['LINKFLAGS'],list):
                    dict['LINKFLAGS'] = mergeLists(dict['LINKFLAGS'], ['-Wall','-Wl,-undefined,dynamic_lookup'])
                else:
                    dict['LINKFLAGS'] = [dict['LINKFLAGS'], '-Wall','-Wl,-undefined,dynamic_lookup']
            else:
                dict['LINKFLAGS'] = ['-Wall','-Wl,-undefined,dynamic_lookup']


            if 'FORTRANFLAGS' in dict:
                if isinstance(dict['FORTRANFLAGS'],list):
                    dict['FORTRANFLAGS'] = mergeLists(dict['FORTRANFLAGS'], ['-ffixed-line-length-none' ,'-fno-second-underscore',  '-O3' , '-Wall','-fPIC','-fno-range-check'])
                else:
                    dict['FORTRANFLAGS'] =[dict['FORTRANFLAGS'], '-ffixed-line-length-none' ,'-fno-second-underscore',  '-O3' , '-Wall','-fPIC','-fno-range-check']
            else:
                dict['FORTRANFLAGS'] = ['-ffixed-line-length-none' ,'-fno-second-underscore' ,  '-O3','-Wall','-fPIC','-fno-range-check']


            dict['FORTRANMODDIRPREFIX'] = '-J'
            dict['LIBS'] = ['gfortran']


        if 'CCFLAGS' in dict:
            if isinstance(dict['CCFLAGS'],list):
                dict['CCFLAGS'] = mergeLists(dict['CCFLAGS'], ['-O3','-Wall','-fPIC'])
            else:
                dict['CCFLAGS'] = [dict['CCFLAGS'], '-O3','-Wall','-fPIC']
        else:
            dict['CCFLAGS'] = ['-O3','-Wall','-fPIC']


        dict['LIBS'] = mergeLists(dict['LIBS'], ['m'])
        if not 'STDCPPLIB' in dict:
            if not 'LIBPATH' in dict:
                print("Missing information. Either the variable STDC++LIB has to be set in the SConfig file or the LIBPATH needs to be set to be  \
                      able to deduce the right stdc++ library. Try to look for libstdc++*.dylib in the /usr/lib directory.")
                raise Exception
            else:# try to guess stdc++ from LIBPATH
                libstd = ''
                found = False
                for dir in dict['LIBPATH']:
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


    retDict = setupCompilers(retDict)
    setupArchitecture(retDict)
    for key in retDict.keys():
            if isinstance(retDict[key],list):
                for value in retDict[key]:
                    exec('env.AppendUnique(' + key + ' = [\'' + value + '\'])')

            else:# assume is a string
                exec('env.Replace(' + key + ' = \'' + retDict[key] + '\')')
    return env









