#!/usr/bin/env python3

#
# This scripts downloads, unpacks and installs the packages required by the
# InSAR Scientific Computing Environment (ISCE). It is called by the bash script
# install.sh and requires the script setup_config.py.
#
# Authors : Eric Gurrola, Kosal Khun, Marco Lavalle
# Date    : April 2013
# Version : 2.0

from __future__ import print_function
import sys
import os
import urllib
import getopt
import re
import shutil
import subprocess
import datetime
import time
import platform
import traceback
import barthread


VARENV           = ['PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'SCONS_CONFIG_DIR', 'ISCE_HOME'] #environment variables
THIS_FOLDER      = os.path.dirname(os.path.abspath(__file__)) #folder containing this file
CONFIG_FOLDER    = os.path.join(os.path.dirname(THIS_FOLDER), 'configuration') #name of configuration folder in the ISCE source tree
SCONS_CONFIG_DIR = os.path.join(os.getenv('HOME'), '.isce') #folder where config file will be written
CONFIG_FILE      = 'SConfigISCE' #name of config file to be created
BASH_FILE        = os.path.join(SCONS_CONFIG_DIR, '.isceenv') #source this file in order to define environment variables
SETUP_CONFIG     = 'setup_config' #name of file (without .py) inside THIS_FOLDER, with dependencies to be downloaded
CONTACT          = "isceteam@gmail.com" #email address shown when an error happens
SETUP_LOG        = os.path.join(SCONS_CONFIG_DIR, 'setup.log') #log file for the installation (inside the SCONS_CONFIG_DIR)
LOGFILE          = None #the log file object referring to SETUP_LOG
VERBOSE          = False #default verbose value
WORKING          = None #barthread display

def usage():
    """
    Print a message about how to use install.sh
    """
    print("%s must be called by install.sh\n" % os.path.basename(__file__))
    subprocess.check_call(os.path.join(THIS_FOLDER, "install.sh -h"), shell=True)


def print2log(msg, withtime=True, cmd=False):
    """
    Output the message displayed by the setup script
    to LOGFILE and to the standard output
    """
    global LOGFILE
    print(msg)
    if cmd:
        msg = "Issuing command:\n\t%s" % msg
    if withtime:
        now = datetime.datetime.today()
        msg = "%s >> %s" % (now.isoformat(), msg)
    LOGFILE.write((msg + '\n').encode('utf-8'))
    LOGFILE.flush()
    os.fsync(LOGFILE)


def executeCommand(command, logfile, critical=True, executable='bash'):
    """
    Take a command and add extra code so that messages are
    logged to a file (logfile) and displayed on standard output.
    The exit status of the command (and not the exit status of tee) is returned to subprocess.
    If critical, the program exits.
    executable is the shell to use.
    """
    global WORKING
    if logfile is not None:
        print2log("Output messages of this command can be found in file %s" % logfile)
    if VERBOSE:
        if logfile is None:
            loggedcommand = command
        else:
            loggedcommand = "%s 2>&1 | tee -a %s; exit ${PIPESTATUS[0]}" % (command, logfile)
    else:
        if logfile is None:
            loggedcommand = "%s > /dev/null" % command
        else:
            loggedcommand = "%s >> %s 2>&1" % (command, logfile)
        WORKING = barthread.BarThread()
    try:
        subprocess.check_call(loggedcommand, shell=True, executable=executable)
        if WORKING:
            WORKING.stop()
            WORKING = None
        else:
            print2log("Done")
    except subprocess.CalledProcessError as e:
        if WORKING:
            WORKING.stop(False)
            WORKING = None
        print2log("...An error occurred with exit status %s. You can find more details in the file %s" % (e.returncode, logfile))
        if critical:
            sys.exit(1)
        else:
            print2log("...Non critical error, command skipped.")


def printenv(msg):
    msg = "************\n" + msg
    for var in VARENV:
        try:
            env = os.environ[var]
        except KeyError:
            env = ""
        msg += "%s=%s\n" % (var, env)
    msg += "************"
    print2log(msg)


def changedir(folder):
    print2log("cd %s" % folder, cmd=True)
    os.chdir(folder)


def createfolder(folder):
    print2log("mkdir -p %s" % folder, cmd=True)
    os.makedirs(folder)


def removefolder(folder):
    """
    Remove a folder using shutil.rmtree
    If fails, use removeall()
    """
    if os.path.exists(folder):
        print2log("rm -rf %s" % folder, cmd=True)
        try:
            shutil.rmtree(folder)
        except OSError:
            removeall(folder)


def removeall(folder):
    """
    Remove a folder recursively using os.remove
    """
    if not os.path.isdir(folder):
        return
    files = os.listdir(folder)
    for f in files:
        fullpath = os.join(folder, f)
        if os.path.isfile(fullpath):
            os.remove(fullpath)
        elif os.path.isdir(fullpath):
            removeall(fullpath)
            os.rmdir(fullpath)


def downloadfile(url, fname, repeat=1):
    counter = 0
    while counter < repeat:
        try:
            response = urllib.request.urlopen(url)
            break
        except urllib.request.URLError as e:
            counter += 1
            if hasattr(e, 'reason'):
                print2log("Failed to reach server. Reason: %s" % e.reason)
                if counter == repeat:
                    return False
                time.sleep(1) #wait 1 second
            elif hasattr(e, 'code'):
                print2log("The server couldn't fulfill the request. Error code: %s" % e.code)
                return False
    data = response.read()
    with open(fname, 'wb') as code:
        code.write(data)
    return True



class InstallItem(object):
    """
    This class allows unpacking and installation of a package.
    """

    def __init__(self, item, paths):
        self.item = item
        self.paths = paths
        self.flags = None;
        self.getFlags()
        self.this_src = None
        self.this_bld = None


    def getFlags(self):
        """
        Get the flags used to install the item.
        """
        user_flags = self.item.properties['user_flags']
        flags_list = self.item.properties['flags_list']
        SPC = " "
        if user_flags:
            if type(user_flags) in (list, tuple):
                FL = user_flags
            elif type(flags) is str:
                FL = user_flags.split()
            else:
                print2log("ProgError: user_flags for %s must be a list or a string" % self.item.name)
                sys.exit(1)
        elif flags_list:
            if type(flags_list) in (list, tuple):
                FL = [SPC, "--prefix=" + self.paths.prefix]
                FL.extend(flags_list)
            else:
                print2log("ProgError: flags_list for %s must be a list" % self.item.name)
                sys.exit(1)
        else:
            FL = [SPC, "--prefix=" + self.paths.prefix]
        self.flags = SPC.join(FL)


    def unpack(self, toUnpack=True):
        """
        Get the folder where the package will be untarred
        Unpack the item if toUnpack=True
        """
        global WORKING
        destfile = self.item.destfile
        destfolder, fname = os.path.split(destfile)
        ns = fname.split('.')
        if ('tar' in ns) or ('tgz' in ns) and (ns.index('tgz') == len(ns)-1):
            if ns[-1] == 'tar':
                flag = "xvf"
                self.this_src = destfile[:-4]
            elif ns[-1] in ['tgz']:
                flag = "xzvf"
                self.this_src = destfile[:-4]
            elif ns[-1] in ['gz']:
                flag = "xzvf"
                self.this_src = destfile[:-7]
            elif ns[-1] in ['bz2']:
                flag = "xjvf"
                self.this_src = destfile[:-8]
            else:
                print2log("Unknown tar file type: %s" % fname)
                sys.exit(1)

            if toUnpack:
                print2log("Unpacking %s ..." % fname)
                if not os.path.isfile(destfile):
                    print2log("Could not find file %s. Please download it first using -d %s" % (destfile, self.item.name))
                    sys.exit(1)
                changedir(self.paths.src)
                command = "tar -%s %s" % (flag, destfile)
                print2log(command, cmd=True)
                if not VERBOSE:
                    command += " > /dev/null"
                WORKING = barthread.BarThread()
                subprocess.check_call(command, shell=True)
                WORKING.stop()
                WORKING = None
        else:
            print2log("...unsupported archive scheme for %s" % destfile)
            sys.exit(1)


    def install(self):
        """
        Install the item.
        Method can be config (make) or setup (setup.py)
        """
        env = self.item.properties['environment']
        method = self.item.properties['installation_method']
        if env:
            self.setEnv(env)
        cwd = self.cd_this_bld()
        build_folder = os.path.basename(self.this_bld)
        print2log("Installing %s ..." % build_folder)
        if not os.path.isdir(self.this_src):
            print2log("Could not find folder %s. Please download and unpack %s first." % (self.this_src, self.item.name))
            sys.exit(1)

        if method == 'config':
            builddir = self.this_bld
            if self.item.properties['prestring']:
                prestr = self.item.properties['prestring'] + " "
            else:
                prestr = ""
            if platform.system().lower() == "freebsd":
                make = "gmake" #for FreeBSD, use gmake instead of make
            else:
                make = "make"
            commands = [("configure", prestr + os.path.join(self.this_src, "configure") + self.flags, True),
                        ("build", make, True),
                        ("install", make + " install", True)]
        elif method == 'setup':
            #we build in src folder rather than in build folder (some setup.py won't work otherwise)
            builddir = self.this_src
            if "--prefix=" in self.flags:
                #replace --prefix=path by --home=path (the python module will be installed in path/lib(64)/python/)
                self.flags = self.flags.replace("--prefix=", "--home=")
            #execute: setup.py configure with flags
            commands = [("setup", "python " + os.path.join(self.this_src, "setup.py configure " + self.flags), False)]
            #previous command gives an error if configure is not needed, the script will then skip "configure"
            #execute setup.py install
            commands.append(("setup", "python " + os.path.join(self.this_src, "setup.py install " + self.flags), True))
        else:
            print2log("ProgError: Unknown installation method for %s." % self.item.name)
            sys.exit(1)

        changedir(builddir)
        printenv("Current values of environment variables:\n")
        for (step, command, critical) in commands:
            print2log(command, cmd=True)
            logfile = "%s_%s.log" %  (os.path.join(self.this_bld, self.item.name), step)
            executeCommand(command, logfile, critical)

        changedir(cwd)
        if env:
            self.restoreEnv()
        print2log("Installation of %s done" % self.item.name)


    def cd_this_bld(self):
        """
        Return the current directory
        and create build directory
        """
        cwd = os.getenv('PWD')
        self.this_bld = os.path.join(self.paths.bld, os.path.basename(self.this_src))
        removefolder(self.this_bld)
        createfolder(self.this_bld)
        return cwd


    def setEnv(self, vars_dict):
        """
        Save current environment and update environment with variables in vars_dict
        """
        self.env = {}
        for var, val in vars_dict.items():
            self.env[var] = os.getenv(var)
            os.environ[var] = val


    def restoreEnv(self):
        """
        Restore environment saved by setEnv
        """
        for var, val in self.env.items():
            if val:
                os.environ[var] = val
            else:
                os.environ.pop(var)



class Paths(object):
    """
    This class allows the creation of subdirectories below prefix
    """

    def __init__(self, prefix, python_version):
        self.prefix = prefix
        paths = []
        for folder in ["src", "bin", "lib", "include", "build"]:
            path = os.path.join(prefix, folder)
            if not os.path.isdir(path):
                createfolder(path)
            paths.append(path)
        (self.src, self.bin, self.lib, self.inc, self.bld) = tuple(paths)
        pkg_dir = ':'.join( [ os.path.join(self.lib + bits, "python") for bits in ['64', '', '32'] ] )
        self.pkg = pkg_dir



class URLItem(object):
    """
    This class defines an item (i.e., a dependency) with its url and properties
    """

    def __init__(self, name, urls, properties):
        self.name = name
        self.urls = urls
        keys = ('installation_method', 'flags_list', 'user_flags', 'prestring', 'environment')
        if len(keys) != len(properties):
            print2log("ProgError: Please check that the properties given are correct for %s in class ISCEDeps." % name)
            sys.exit(1)
        self.properties = dict(zip(keys, properties))



class ISCEDeps(object):
    """
    This class prepares the environment and installs dependencies,
    before installing ISCE.
    """

    dependency_list = ["GMP", "MPFR", "MPC", "GCC", "SCONS", "FFTW", "SZIP", "HDF5", "NUMPY", "H5PY"] #list of packages that can be installed, order matters! use uppercase!
    deplog_key = ["skipped", "downloaded", "unpacked", "installed"] #dependency log


    def __init__(self, **kwargs):
        global VERBOSE
        try:
            VERBOSE = kwargs["verbose"]
        except:
            pass
        version = sys.version_info
        self.python_version = "{}.{}".format(version.major, version.minor)
#        p = subprocess.Popen(['python3', '-V'], stdout=subprocess.PIPE)
#        x =  p.communicate()[0]
#        stv = ''.join([e if isinstance(e,str) else e.decode("utf-8") for e in x])
#        self.python_version = "%s.%s" % tuple(stv.split(' ')[1].split('.')[:2])
# python_version is now the python3 version (not the sys.version_info)
        self.uname = kwargs["uname"]
        self.bash_vars = [] #environment variables to be written in bash file
        self.dependency_log = {}
        for key in self.deplog_key:
            self.dependency_log[key] = []
        self.prefix = kwargs["prefix"]
        if self.prefix: #if prefix given
            self.paths = Paths(self.prefix, self.python_version)
        else:
            self.paths = None

        try:
            #config file is given: skip installation of dependencies
            self.config = kwargs["config"]
            return
        except KeyError:
            #config file not given
            self.config = None

        #read setup_config.py
        setup_config = readSetupConfig(SETUP_CONFIG + '.py')
        properties = {} # dictionary of properties for each item to be installed

        GCC = kwargs["gcc"]
        GXX = kwargs["gpp"]
        prestring = "CC=" + GCC + " CXX=" + GXX
        env = { #use the latest compilers installed with gcc
            'CC': os.path.join(self.paths.bin, "gcc"),
            'F77': os.path.join(self.paths.bin, "gfortran")
            }
        #to add a new item:
        #properties[name_of_item] = (installation_method, flags_list, user_flags, prestring, environment)
        properties["GMP"] = ("config", ["--enable-cxx"], None, prestring, None)
        properties["MPFR"] = ("config", ["--with-gmp=" + self.prefix], None, prestring, None)
        properties["MPC"] = ("config", ["--with-gmp=" + self.prefix, "--with-mpfr=" + self.prefix], None, None, None)
        properties["GCC"] = ("config", ["--with-gmp=" + self.prefix, "--with-mpfr=" + self.prefix, "--enable-languages=c,c++,fortran", "--enable-threads"], None, prestring, None)
        properties["SCONS"] = ("setup", [], None, None, None)
        properties["FFTW"] = ("config", ["--enable-single", "--enable-shared"], None, None, env)
        properties["SZIP"] = ("config", [], None, None, None)
        properties["HDF5"] = ("config", ["--enable-fortran", "--enable-cxx"], None, None, None)
        properties["NUMPY"] = ("setup", [], None, None, None)
        properties["H5PY"] = ("setup", ["--hdf5=" + self.prefix], None, None, None)
        """ TODO: we can try to support the installation of the following packages if needed
        properties["MOTIF"] = ("config", [], None, None, None)
        properties["SPHINX"] = ("setup", [], None, None, None)
        properties["XT"] = ("config", [], None, None, None)
        properties["XP"] = ("config", [], None, None, None)
        """

        self.urlitems = {}
        #install dependencies
        for dep in self.dependency_list:
            self.make_urls(setup_config, dep, properties[dep])

        toDownload = kwargs["download"]
        toUnpack = kwargs["unpack"]
        toInstall = kwargs["install"]
        if not (toDownload + toUnpack + toInstall): # none given: do everything
            toDownload = self.dependency_list
            toUnpack = self.dependency_list
            toInstall = self.dependency_list
        else: # at least one is given
            toDownload = self.getDepList(toDownload) # get list of dependencies to download
            toUnpack = self.getDepList(toUnpack) # get list of dependencies to unpack
            toUnpack.extend(toDownload) # add depedencies from download list
            toUnpack = ",".join(toUnpack) # make list back into a comma-separated string
            toUnpack = self.getDepList(toUnpack) # remove duplicata and reorder dependencies
            toInstall = self.getDepList(toInstall) # get list of dependencies to install
            toInstall.extend(toUnpack) # add dependencies from unpack list (and download list)
            toInstall = ",".join(toInstall) # make list into a comma-separated string
            toInstall = self.getDepList(toInstall) # remove duplicata and reorder dependencies
        self.toDownload = toDownload
        self.toUnpack = toUnpack
        self.toInstall = toInstall


    def getDepList(self, depList):
        """
        Take a string and return a list of dependencies
        The list is ordered according to self.dependency_list
        """
        if depList.upper() == "NONE" or depList == "":
            return []
        elif depList.upper() == "ALL":
            return list(self.dependency_list)
        else:
            fill = False
            if depList.endswith('+'): #given string ends with >
                fill = True
                depList = depList[:-1]
            givenList = depList.upper().split(",")
            if fill:
                #get last element of given list
                last = givenList[-1]
                #find where last element is located in dependency_list
                index = self.dependency_list.index(givenList[-1])
                #append all dependencies following last element
                givenList.extend(self.dependency_list[index+1:])
            depList = []
            for dep in self.dependency_list:
                if dep in givenList and dep in self.urlitems:
                    depList.append(dep)
            return depList


    def make_urls(self, config, dependency, properties):
        """
        Check if a dependency is in config file
        And add corresponding URLItem object to self.urlitems
        """
        try:
            urls = config[dependency]
            item = URLItem(dependency, urls, properties)
            urlpath, fname = os.path.split(urls[0])
            item.destfile = os.path.join(self.paths.src, fname)
            self.urlitems[dependency] = item
        except AttributeError:
            self.dependency_log["skipped"].append(dependency)
            print2log("Item %s not given in %s.py. Proceeding as if it has already been installed and hoping for the best..." % (dependency, SETUP_CONFIG))


    def run(self):
        """
        Run main script for installing dependencies and ISCE
        """
        self.prepare() #prepare environment for installation
        self.installIsce() #install isce with Scons
        self.createBashFile() #create a bash file with environment variables


    def unpackinstall(self):
        """
        Unpack the dependencies in self.toUnpack (if needed)
        then install those in self.toInstall
        """
        insList = []
        for dep in self.toInstall:
            item = self.urlitems[dep]
            ins = InstallItem(item, self.paths)
            if dep in self.toUnpack:
                ins.unpack(True)
                self.dependency_log["unpacked"].append(ins.item.name)
            else:
                ins.unpack(False)
            insList.append(ins)
        for ins in insList:
            ins.install()
            self.dependency_log["installed"].append(ins.item.name)


    def download(self):
        """
        Download the dependencies specified in self.toDownload
        """
        global WORKING
        for dep in self.toDownload:
            item = self.urlitems[dep]
            for url in item.urls:
                urlpath, fname = os.path.split(url)
                print2log("Downloading %s from %s to %s" % (fname, urlpath, self.paths.src))
                WORKING = barthread.BarThread()
                response = downloadfile(url, item.destfile, repeat=2)
                if response:
                    if os.path.exists(item.destfile):
                        self.dependency_log["downloaded"].append(item.name)
                        WORKING.stop()
                        WORKING = None
                        break
                    else:
                        continue
            if not os.path.exists(item.destfile):
                msg = "Cannot download %s. Please check your internet connection and make sure that the download url for %s in %s.py is correct.\n"
                msg += "You might also consider installing the package manually: see tutorial." % (fname, item.name, SETUP_CONFIG)
                print2log(msg)
                sys.exit(1)


    def createConfigFile(self):
        """
        Create SConfigISCE file
        """
        MANDATORY_VARS = ['PRJ_SCONS_BUILD', 'PRJ_SCONS_INSTALL', 'LIBPATH', 'CPPPATH', 'FORTRAN', 'CC', 'CXX', 'FORTRANPATH'] # ML added FORTRANPATH 2014-04-02
        OPTIONAL_VARS = ['MOTIFLIBPATH', 'X11LIBPATH', 'MOTIFINCPATH', 'X11INCPATH']
        mandatory_ok = True
        optional_ok = True
        msg = "Creating configuration file...\n"
        self.config_values['PRJ_SCONS_BUILD'] = os.path.join(self.paths.bld, 'isce_build')
        msg += "ISCE will be built in %s\n" % self.config_values['PRJ_SCONS_BUILD']
        self.config_values['PRJ_SCONS_INSTALL'] = os.getenv('ISCE_HOME')
        msg += "ISCE will be installed in %s\n" % self.config_values['PRJ_SCONS_INSTALL']
        libpath = []
        for bits in ["64", "", "32"]:
            if os.path.isdir(self.paths.lib + bits):
                libpath.append(self.paths.lib + bits)
        self.updatePath('LD_LIBRARY_PATH', libpath)
        libpath = os.getenv('LD_LIBRARY_PATH').split(':')
        self.config_values['LIBPATH'] = " ".join(libpath)
        msg += "Libraries will be checked inside %s\n" % self.config_values['LIBPATH']

        print(os.path.join('python' + self.python_version + 'm', 'Python.h'))
        CPPPATH = self.getFilePath(os.path.join('python' + self.python_version + 'm', 'Python.h')) # ML added +'m' on 2014-04-02 to reflect new location
        self.config_values['CPPPATH'] = os.path.join(CPPPATH, 'python' + self.python_version + 'm') # ML added +'m' on 2014-04-02 to reflect new location
        print(os.path.join(CPPPATH, 'python' + self.python_version + 'm'))
        if CPPPATH:
            msg += "Python.h was found in %s\n" % self.config_values['CPPPATH']
        else:
            mandatory_ok = False
            msg += "Python.h could NOT be found. Please edit the file %s and add the location of Python.h for the variable CPPPATH\n" % CONFIG_FILE

        fortranpath = self.getFilePath('fftw3.f')
        self.config_values['FORTRANPATH'] = fortranpath
        if fortranpath:
            msg += "fftw3.f was found in %s\n" % self.config_values['FORTRANPATH']
        else:
            mandatory_ok = False
            msg += "fftw3.f could NOT be found. Please edit the file %s and add the location of fftw3.f for the variable FORTRANPATH\n" % CONFIG_FILE

        COMPILERS = [
            ('Fortran', 'FORTRAN', 'gfortran'), #(compiler name, variable name, executable name)
            ('C', 'CC', 'gcc'),
            ('C++', 'CXX', 'g++')
            ]
        for compiler in COMPILERS:
            path = self.getFilePath(compiler[2])
            self.config_values[compiler[1]] = os.path.join(path, compiler[2])
            if path:
                msg += "The path of your %s compiler is %s\n" % (compiler[0], self.config_values[compiler[1]])
            else:
                mandatory_ok = False
                msg += "No %s compiler has been found. Please edit the file %s and add the location of your %s compiler for the variable %s\n" % (compiler[0], CONFIG_FILE, compiler[0], compiler[1])

        if self.uname == 'Darwin': #Mac OS
            ext = 'dylib'
        else: #should be Linux (doesn't work with Windows)
            ext = 'so'
        MDX_DEP = [
            ('MOTIFLIBPATH', 'libXm.' + ext), #(variable name, library name)
            ('X11LIBPATH', 'libXt.' + ext),
            ('MOTIFINCPATH', os.path.join('Xm', 'Xm.h')),
            ('X11INCPATH', os.path.join('X11', 'X.h'))
            ]
        for dep in MDX_DEP:
            path = self.getFilePath(dep[1])
            self.config_values[dep[0]] = path
            if path:
                msg += "The path of %s is %s\n" % (dep[1], path)
            else:
                optional_ok = False
                msg += "%s has NOT been found. Please edit the file %s and add the location of %s for the variable %s\n" % (dep[1], CONFIG_FILE, dep[1], dep[0])

        config_vars = MANDATORY_VARS
        if optional_ok:
            config_vars.extend(OPTIONAL_VARS)
        else:
            print2log("Could not find libraries for building mdx.")
        f = open(os.path.join(SCONS_CONFIG_DIR, CONFIG_FILE), 'wb')
        for var in config_vars:
            f.write("%s=%s\n" % (var, self.config_values[var]))
        f.close()
        print2log(msg)

        if not mandatory_ok: #config file is not complete...
            msg = "You need to edit the file %s located in %s, before going further.\n" % (CONFIG_FILE, SCONS_CONFIG_DIR)
            msg += "Then run the following command to install ISCE:\n"
            msg += "./install.sh -p %s -c %s" % (self.prefix, os.path.join(SCONS_CONFIG_DIR, CONFIG_FILE))
            print2log(msg, False)
            sys.exit(1)


    def getFilePath(self, name):
        """
        Return a path containing the file 'name'. The path is searched inside env var PATH.
        """
        path_found = ""
        for path in os.getenv('PATH').split(':'):
            if path_found:
                break
            if os.path.isfile(os.path.join(path, name)): #name found inside path
                path_found = path
            else:
                dirname, basename = os.path.split(path)
                if basename == 'bin': #if path ends with 'bin'
                    for folder in ['lib64', 'lib', 'lib32', 'include']: #look inside lib and include folders
                        if os.path.isfile(os.path.join(dirname, folder, name)):
                            path_found = os.path.join(dirname, folder)
                            break
        return path_found


    def installIsce(self):
        """
        Install ISCE
        """
        print2log("Installing ISCE...")
        os.environ['PYTHONPATH'] += ":" + CONFIG_FOLDER #add config folder to pythonpath
        if self.paths:
            self.updatePath('PATH', [self.paths.bin])

        changedir(os.path.dirname(THIS_FOLDER))
        command = "scons install"
        printenv("Current values of environnement variables:\n")
        logfile = "%s.log" % self.config_values['PRJ_SCONS_BUILD']
        print2log(command, cmd=True)
        executeCommand(command, logfile)


    def createBashFile(self):
        """
        Create file with environment variables
        """
        f = open(BASH_FILE, 'wb')
        for var in self.bash_vars:
            goodpaths = []
            exp, val = var.split('=')
            paths = val.split(':')
            for path in paths:
                if os.path.isdir(path):
                    goodpaths.append(path)
            f.write("%s=%s\n" % (exp, ':'.join(goodpaths)))
        f.close()
        msg = "ISCE INSTALLATION DONE\n"
        msg += "ISCE has been successfully installed!\n"
        msg += "ISCE applications are located in %s\n" % self.config_values['PRJ_SCONS_INSTALL']
        msg += "Environment variables needed by ISCE are defined in the file %s\n" % BASH_FILE
        msg += "Before running ISCE, source this file in order to add the variables to your environment:\n"
        msg += "    source %s\n" % BASH_FILE
        msg += "You can source the file in your .bashrc file so that the variables are automatically defined in your shell."
        print2log(msg)


    def prepare(self):
        """
        Prepare environment for installation
        """
        self.config_values = {} #variable values to be written to config file (or extracted from config file if given)
        if self.config: #config file is given by user (packages are supposed to be pre-installed)
            self.readConfigFile() #read file and update self.config_values
            self.setEnvironment()
        else: #config file not given
            self.setEnvironment()
            self.download() #download packages...
            self.unpackinstall() #...and install them
            self.createConfigFile() #create the config file for Scons
        for var in ['PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'ISCE_HOME']:
            self.bash_vars.append("export %s=%s" % (var, os.getenv(var)))


    def updatePath(self, varname, pathlist):
        """
        Append all paths in pathlist at the beginning of env variable varname.
        """
        if type(pathlist) is list:
            oldpath = os.getenv(varname)
            if oldpath: #env not empty
                oldpath = oldpath.split(':')
                for path in oldpath:
                    if path not in pathlist: #path not in pathlist
                        pathlist.append(path) #add it at the end of pathlist
            pathlist = ':'.join(pathlist)
        os.environ[varname] = pathlist


    def setEnvironment(self):
        """
        Set environment variables
        """
        #Initial values of environment variables
        printenv("Preparing environment\nInitial values of environment variables:\n")

        pythonpath = []
        if self.config:
            try:
                key = 'PRJ_SCONS_INSTALL'
                isce_home = self.config_values[key]
                key = 'LIBPATH'
                lib_path = self.config_values[key].split()
            except KeyError:
                print2log("Make sure that %s is present in %s" % (key, self.config))
                sys.exit(1)

            config_dir, config_file = os.path.split(self.config)
            if config_file != CONFIG_FILE: #make a copy of config file if it's not located in SCONS_CONFIG_DIR
                config_copy = os.path.join(SCONS_CONFIG_DIR, CONFIG_FILE)
                shutil.copy(self.config, config_copy)
                if os.path.isfile(config_copy): #check that file has been copied
                    self.config = config_copy
                    print2log("The config file has been moved to %s" % self.config)
                    config_dir = SCONS_CONFIG_DIR
                else:
                    msg = "Could not copy %s to %s\n" % (self.config, config_copy)
                    msg += "Please do it manually, then run this command from the setup directory:\n"
                    msg += "./install.sh -c %s" % self.config
                    print2log(msg)
                    sys.exit(1)
        else: #config file not given
            isce_home = os.path.join(self.prefix, 'isce')
            config_dir = SCONS_CONFIG_DIR
            lib_path = []
            for bits in ["64", "", "32"]:
                lib_path.append(self.paths.lib + bits)

        if self.paths:
            pythonpath.append(self.paths.pkg)

        pythonpath.extend([isce_home, os.path.join(isce_home, 'applications'), os.path.join(isce_home, 'components'), self.prefix]) # added prefix folder to PYTHONPATH 2/12/13
        VAR_TO_UPDATE = {
            'PYTHONPATH': pythonpath,
            'LD_LIBRARY_PATH': lib_path,
            'SCONS_CONFIG_DIR': config_dir,
            'ISCE_HOME': isce_home,
            }
        if self.paths:
            VAR_TO_UPDATE['PATH'] = [self.paths.bin]
        if not self.config:
            # when installing and using gcc, there's a multiarch problem debuting with Ubuntu Natty and Debian Wheezy
            # we need to give explicitly the search path
            # http://wiki.debian.org/Multiarch/
            if platform.system().lower() == "linux":
                #distname, version, distid = platform.linux_distribution()
                #if (distname.lower() == "ubuntu" and version >= "11") or (distname.lower() == "debian" and version >= "7" ):
                machine = platform.machine()
                if os.path.isdir("/usr/lib/%s-linux-gnu/" % machine):
                    VAR_TO_UPDATE['LIBRARY_PATH'] = ["/usr/lib/%s-linux-gnu/" % machine] #precompilation search path for libraries
                    VAR_TO_UPDATE['LD_LIBRARY_PATH'].extend(VAR_TO_UPDATE['LIBRARY_PATH'])
                if os.path.isdir("/usr/include/%s-linux-gnu/" % machine):
                    VAR_TO_UPDATE['CPATH'] = ["/usr/include/%s-linux-gnu" % machine] #precompilation search path for include files

        for var, pathlist in VAR_TO_UPDATE.items():
            self.updatePath(var, pathlist)
        os.environ['PATH'] += ":%s" % os.path.join(os.getenv('ISCE_HOME'), 'applications') #add applications folder to the path
        printenv("New values of environment variables:\n")


    def readConfigFile(self):
        """
        Read config file passed with option -c
        """
        f = open(self.config, 'rb')
        lines = f.readlines()
        for line in lines:
            m = re.match("([^#].*?)=([^#]+?)$", line.strip().decode('utf-8'))
            if m:
                var = m.group(1).strip()
                val = m.group(2).strip()
                self.config_values[var] = val
        f.close()


def readSetupConfig(setup_config):
    """
    Read setup_config file where urls are given
    """
    params = {}
    f = open(setup_config, 'rb')
    lines = f.readlines()
    for line in lines:
        m = re.match("([^#].*?)=([^#]+?)$", line.strip().decode('utf-8'))
        if m:
            var = m.group(1).strip()
            val = m.group(2).strip().replace('"', '')
            if var in params.keys():
                params[var].append(val)
            else:
                params[var] = [val]
    f.close()
    return params


def checkArgs(args):
    """
    Check arguments passed to this python file
    """
    try:
        opts, args = getopt.getopt(args, "h", ["help", "prefix=", "ping=", "config=", "uname=", "download=", "unpack=", "install=", "gcc=", "gpp=", "verbose"])
    except getopt.GetoptError as err:
        print2log("ProgError: %s" % str(err))
        usage()
        sys.exit(2)

    ok = True
    ping = ""
    verbose = False
    kwargs = {}
    for o, a in opts:
        if o in ("-h", "--help"):
            ok = False
            break
        elif o == "--ping":
            ping = a
        elif o == "--verbose":
            kwargs[o[2:]] = True
        elif o in ["--prefix", "--config", "--uname",
                   "--download", "--unpack", "--install",
                   "--gcc", "--gpp"]:
            kwargs[o[2:]] = a
        else:
            print2log("ProgError: unhandled option: %s" % o)
            ok = False
            break
    if not (ok and ping == "pong"):
        usage()
        sys.exit(2)
    try:
        kwargs["--prefix"] = os.path.abspath(kwargs["--prefix"])
    except KeyError:
        pass
    try:
        kwargs["--config"] = os.path.abspath(kwargs["--config"])
    except KeyError:
        pass

    return kwargs



if __name__ == "__main__":
    step = 0
    witherror = True
    try:
        if not os.path.isdir(SCONS_CONFIG_DIR):
            createfolder(SCONS_CONFIG_DIR)
        LOGFILE = open(SETUP_LOG, 'ab') #open SETUP_LOG for appending
        print2log("=" * 60, False)
        msg = "Starting setup script:\n"
        msg += " ".join(sys.argv) + "\n"
        msg += "-" * 60
        print2log(msg)
        step = 1
        #get arguments from command line
        kwargs = checkArgs(sys.argv[1:])
        print2log("Checking command line... done")
        step = 2
        a = ISCEDeps(**kwargs)
        print2log("Initializing script... done")
        step = 3
        print2log("Starting installation...")
        a.run()
        witherror = False
    except KeyboardInterrupt:
        print2log("Program interrupted by user.")
    except Exception:
        if step == 0:
            msg = "Error when reading script"
        elif step == 1:
            msg = "Error when checking command line:"
        elif step == 2:
            msg = "Error when initializing script:"
        elif step == 3:
            msg = "The script has ended unexpectedly.\n"
            msg += "##### DEPENDENCIES #####\n"
            for key in a.deplog_key:
                try:
                    msg += "%s: %s\n" % (key, ", ".join(a.dependency_log[key]))
                except KeyError:
                    msg += "%s: none\n" % key
            msg += "If you run this installation again, you might want to use advanced options for the script. See tutorial.\n\n"
        print2log("%s\n%s" % (msg, traceback.format_exc()))
    finally:
        if WORKING:
            WORKING.stop(False)
            WORKING = None
        print("-" * 60)
        print("All the displayed messages have been logged to the file %s." % SETUP_LOG)
        print("-" * 60)
        msg = "For any questions, contact %s\n" % CONTACT
        if witherror:
            msg += "The setup script ended with errors."
        else:
            msg += "ISCE seems to have been installed correctly."
        print2log(msg)
        LOGFILE.close()
