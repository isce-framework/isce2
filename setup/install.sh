#!/bin/bash
#
# This scripts installs the prerequisites for the InSAR
# Scientific Computing Environment (ISCE)
#
# Author  : Kosal Khun, Marco Lavalle
# Date    : 2012-06-28
# Version : 1.1
# Update  : 2014-04-02 updated to support python3 (ML)

BASENAME=$0 #command line with relative path to this file
ARGS=$@
BASEDIR=$(pwd) #get base folder where command has been issued
cd $(dirname $BASENAME) #go to setup folder
SETUPDIR=$(pwd) #get absolute path of setup folder

i_folder="INSTALL_FOLDER"
c_file="CONFIG_FILE"
dep_list="GMP,MPFR,MPC,GCC,SCONS,FFTW,SZIP,HDF5,NUMPY,H5PY"
SETUPLOG="$HOME/.isce/setup.log"

printhelp () {
    echo $1
    echo "Use the option -h to get more information."
    exit 1
}

print2log () {
    cmd=0
    if [[ -n $2 ]]; then
        if [[ $2 == "d" ]]; then
            now=$(date +"%c")
            msg="$now >> $1"
        elif [[ $2 == "c" ]]; then
            cmd=1
            msg="running command: $1"
        else
            msg=$1
        fi
    else
        msg=$1
    fi
    echo -e $msg >> $SETUPLOG
    echo -e $1
    if [[ $cmd == 1 ]]; then
        $1
    fi
}


usage () {
    echo -e "Usage: $BASENAME -p $i_folder [OPTION]... | -c $c_file"
    echo "Install the software ISCE and its dependencies."
    echo "List of dependencies: $dep_list"
    echo
    echo -e " -p $i_folder"
    echo -e "\tInstall everything to the directory $i_folder."
    echo -e "\t- Packages will be downloaded and unpacked into $i_folder/src"
    echo -e "\t- Binaries, libraries and includes will be put into"
    echo -e "\t$i_folder/bin, $i_folder/lib, $i_folder/include"
    echo -e "\t- Environment variables and ISCEConfig file will be setup"
    echo -e "\tto install ISCE at $i_folder/isce"
    echo -e "\tUse only this option for fast installation."
    echo
    echo -e " -c $c_file"
    echo -e "\tThe script will skip the installation of the dependencies"
    echo -e "\tand will use the given $c_file to install ISCE."
    echo -e "\tUse only if all required dependencies are already installed"
    echo -e "\tor if the configuration file has been changed manually."
    echo -e "\tYou need to pass the -p argument"
    echo -e "\tif you have used this script to install python"
    echo -e "\tand some packages like numpy, h5py."
    echo
    echo "Additional options:"
    echo -e " -v\tVerbose"
    echo -e " -b PYTHON_PATH\n\tTell the script where python is located."
    echo -e "\tPYTHON_PATH must be the full path to the python file."
    echo
    echo "Use the following options to customize your installation:"
    echo -e " -d NONE | ALL | dep1,dep2..."
    echo -e "\tDownload the given dependencies (see list)."
    echo -e "\tDefault: NONE"
    echo -e "\tThe names must be separated by a comma, with no space."
    echo -e "\tThe packages not in the list must be present"
    echo -e "\tin $i_folder/src"
    echo -e " -u NONE | ALL | dep1,dep2..."
    echo -e "\tUntar the given dependencies (see list)"
    echo -e "\tin addition to those given with -d"
    echo -e "\tDefault: NONE (the script will use list given with -d)"
    echo -e "\tThe packages not in the list must be already"
    echo -e "\tunpacked in $i_folder/src"
    echo -e " -i NONE | ALL | dep1,dep2..."
    echo -e "\tInstall the given dependencies (see list)"
    echo -e "\tin addition to those given with -d and -u"
    echo -e "\tDefault: NONE (the script will use list given with -d and -u)"
    echo -e "\tThe packages not in the list should be already"
    echo -e "\tinstalled in $i_folder"
    echo
    echo -e " ***warning***"
    echo -e "\tPython 3.x does not support Scons, therefore the install script"
    echo -e "\trequires python 2.x to run Scons and python >= 3.2.0 to run ISCE."
    echo -e "\tThe script assumes that python 2.x is available in the path as python,"
    echo -e "\tand downloads and installs python3 if not found."
    echo

}


startlog () {
    LOGDIR=$(dirname $SETUPLOG)
    mkdir $LOGDIR
    print2log ""
    print2log "=========================================="
    print2log "Starting install.sh script:" d
    print2log "\t$BASENAME $ARGS"
    print2log "------------------------------------------"
    print2log "current directory: $BASEDIR"
}


changedir () {
    if [[ -n $1 && -d $1 ]]; then
        print2log "cd $1" c
        print2log "current directory: $(pwd)"
    else
        print2log "error in changedir: no directory given"
        exit 1
    fi
}

VERBOSE=
while getopts 'p:b:c:vd:u:i:h' opt; do
    case $opt in
	h)
	    usage
	    exit 0
	    ;;
        b)
            PYTHONBIN=$OPTARG
            ;;
        p)
	    PREFIX=$OPTARG
	    ;;
        c)
	    CONFIG=$OPTARG
	    ;;
        v)
            VERBOSE=--verbose
            ;;
        d)
	    DO_DOWNLOAD=$OPTARG
	    ;;
        u)
	    DO_UNPACK=$OPTARG
	    ;;
        i)
	    DO_INSTALL=$OPTARG
	    ;;
	\?)
	    printhelp "error in arguments"
	    ;;
        *)
	    usage;;
    esac
done


if [[ -z "$PREFIX" && -z "$CONFIG" ]]; then #neither prefix nor config are given
    printhelp "missing arguments -p or -c" #exit
fi

if [[ -n "$PYTHONBIN" ]]; then #python path given
    if [[ ! -e "$PYTHONBIN" ]]; then #could not find python
        printhelp "could not find $PYTHONBIN: please, give the full path to python" #exit
    fi
else
    PYTHONBIN=$(which python)
fi


startlog
UNAME=$(uname) #OS or kernel name
print2log "checking uname... $UNAME"

##### IF CONFIG IS GIVEN ####
if [[ -n "$CONFIG" ]]; then #skip installation of dependencies
    cd $BASEDIR #come back to initial folder
    if [[ ! -f $CONFIG ]]; then
	print2log "config file $CONFIG could not be found: please check the folder or create the file first"
        printhelp #exit
    else
	cd $(dirname $CONFIG)
	CONFIG=$(pwd)/$(basename $CONFIG) #get absolute path of config file
    fi
    if [[ -n "$PREFIX" ]]; then #if prefix given
	cd $BASEDIR #come back to initial folder
	if [[ ! -d $PREFIX ]]; then #if folder doesn't exist
	    print2log "directory $PREFIX doesn't exist"
            printhelp #exit
	else
	    cd $PREFIX
	    PREFIX=$(pwd) #get absolute path of prefix
	fi
    fi

    print2log "cd $SETUPDIR" c #go to setup folder
    $PYTHONBIN setup.py --ping=pong --uname=${UNAME} --config=${CONFIG} --prefix=${PREFIX} $VERBOSE
    exit 1
fi

#### WITH PREFIX - NO CONFIG ####
#checking prerequisites: gcc g++ make m4
check="gcc g++ make m4"
arrCheck=(${check// / })
missing=
for ((c=0; c < ${#arrCheck[@]}; c++))
do
    x=${arrCheck[c]}
    print2log "checking for $x"
    which[c]=$(command -v $x)
    if [[ -n "${which[c]}" ]]; then #path of app returned
	print2log "result: $($x --version | head -1)"
	if [[ "$x" == "gcc" ]]; then
	    GCC=${which[c]}
	elif [[ "$x" == "g++" ]]; then
	    GPP=${which[c]}
	fi
    else #app not found
	print2log "result: no"
	missing="$missing $x"
    fi
done
if [[ -z "$missing" ]]; then #prerequisites already installed
    print2log "$check already installed"
else #missing packages
    print2log "missing package(s):$missing"
    print2log "check your PATH or install them manually"
    exit 1
fi

#checking destination folder
cd $BASEDIR
if [[ ! -d $PREFIX ]]; then
    print2log "directory $PREFIX does not exists"
    print2log "mkdir -p $PREFIX" c
    if [ $? ]; then
	print2log "...done"
	cd $PREFIX
	PREFIX=$(pwd) #get absolute path of prefix
    else
	print2log "could not create ${PREFIX}"
	print2log "$i_folder must be a local directory where you have write permissions"
	exit 1
    fi
fi

changedir $PREFIX
INSTALL_DIR=$(pwd) #absolute path of installation folder
DOWNLOAD_DIR=$INSTALL_DIR/src #download folder
BUILD_DIR=$INSTALL_DIR/build #build folder
SOURCE_DIR=$INSTALL_DIR/src #source folder (where downloaded package is untarred)

export PATH=${INSTALL_DIR}/bin:${PATH} #make sure that python is searched in INSTALL_DIR first
if [[ -n $LD_LIBRARY_PATH ]]; then
    export LD_LIBRARY_PATH=${INSTALL_DIR}/lib:${LD_LIBRARY_PATH}
else
    export LD_LIBRARY_PATH=${INSTALL_DIR}/lib
fi
print2log "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

print2log "checking for Python"
REQUIRED=3.2.0 #the minimum version of python needed - ML updated on 2014-04-03 from 2.6.6 
APPVER=3.4.0 #the version that will be installed - ML updated on 2014-04-03 from 2.7.3
#Check if Python is installed and the version is correct
PYTHONBIN3=$(which python3) # ML added PYTHONBIN3 on 2014-04-02
PYTHON=$(command -v $PYTHONBIN3) #ML PYTHONBIN3 in place of PYTHONBIN
if [[ -n "${PYTHON}" ]]; then #python is installed
    #VERSION=$(${PYTHON} -V 2>&1 | awk -F" " '{print $2}')
    VERSION=$(${PYTHON} -c "import sys; print('.'.join(map(str, sys.version_info[:3])))")
    print2log "result: Python $VERSION"
    if [[ "$VERSION" < "2.9.9" ]]; then #not supported yet - ML updated from  > "2.9.9" on 2014-04-02
        answer_ok=0
        print2log "Python $VERSION is no longer supported."
        until [[ $answer_ok == 1 ]]; do
            echo -n "Python $VERSION is no longer supported. Would you like the script to install python $APPVER? (Y/n) "
            read -n 1 answer
            if [[ -z "$answer" ]]; then
                answer=y
            else
                echo
            fi
            if [[ "$answer" == "n" || "$answer" == "N" ]]; then
                print2log "ISCE needs at least python $REQUIRED" # ML updated to Python 3.x
                print2log "Python installation skipped by user: exiting script"
                exit 1
            fi
            if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
                answer_ok=1
            fi
        done

    elif [[ "$VERSION" < "$REQUIRED" ]]; then #not correct version!
	print2log "you must have python >= $REQUIRED"
    else #correct version, now check if it's python-devel
	#first, check if python has distutils
	print2log "checking for module distutils"
	distutils_result=$(${PYTHON} -c "import distutils" 2>&1)
	if [[ -z "$distutils_result" ]]; then #distutils ok
	    print2log "result: yes"
	    #check for Python include path
	    print2log "checking for Python include path"
	    python_path=$(${PYTHON} -c "import distutils.sysconfig; print (distutils.sysconfig.get_python_inc ());")
	    if [[ -n "${python_path}" ]]; then #include path found
		print2log "result: ${python_path}"
		#check for Python.h
		print2log "checking for Python.h"
		pythonH="${python_path}/Python.h"
		if [[ -f ${pythonH} ]]; then #Python.h found
		    print2log "result: ${pythonH}"
		    python_ok="Yoohoo!"
		else
		    print2log "result: Python.h not found in ${python_path}"
		fi
	    else
		print2log "result: Python include path not found"
	    fi
	else
	    print2log "result: Python module distutils not found"
	fi
    fi
else #python is NOT installed
    print2log "result: Python not found"
fi


if [[ -n "$python_ok" ]]; then #correct version of python is already installed
    print2log "your python3 version is correct."
    bindir=$INSTALL_DIR/bin
    if [[ $(dirname $PYTHONBIN3) != "$bindir" ]]; then  # ML PYTHONBIN3 in place of PYTHONBIN
        if [[ ! -d $bindir ]]; then
            print2log "mkdir -p $bindir" c
        elif [[ -e $bindir/python ]]; then
            print2log "rm $bindir/python" c
        fi
        print2log "ln -s $PYTHONBIN3 $bindir/python3" c
    fi

else #python has to be installed
    print2log "you don't have the correct version of Python"
    print2log "the script will download and install Python $APPVER"

    changedir $INSTALL_DIR
    if [ ! -d $DOWNLOAD_DIR ]; then
	print2log "mkdir $DOWNLOAD_DIR" c
    fi
    if [ ! -d $SOURCE_DIR ]; then
	print2log "mkdir $SOURCE_DIR" c
    fi
    if [ ! -d $BUILD_DIR ]; then
	print2log "mkdir $BUILD_DIR" c
    fi

    APPFILE="Python-${APPVER}.tgz" #ML updated from tar.bz2 to tgz on 2014-04-02 
    APP_DIR="Python-${APPVER}"
    URL="https://www.python.org/ftp/python/${APPVER}/${APPFILE}"
    print2log "downloading Python archive file..."
    changedir $DOWNLOAD_DIR
    if which curl >/dev/null; then
	print2log "curl -O --insecure $URL" c
    else
	print2log "wget $URL --no-check-certificate" c #ML added --no-check-certificate on 2014-04-02
    fi
    if [[ ! -e $APPFILE ]]; then #file not downloaded!
	print2log "error while trying to download file from $URL"
	print2log "please check your internet connection and the URL"
	exit 1
    fi
    print2log "...done"

    print2log "unpacking source files..."
    changedir $SOURCE_DIR
    print2log "rm -Rf $APP_DIR" c
    print2log "tar -xf ${DOWNLOAD_DIR}/${APPFILE}" c
    print2log "...done"

    print2log "building files..."
    changedir $BUILD_DIR
    print2log "rm -Rf $APP_DIR" c
    print2log "mkdir $APP_DIR" c
    changedir $APP_DIR

    if [[ -n $VERBOSE ]]; then
        redirect="2>&1 | tee -a"
        redirect2=
    else
        redirect=">>"
        redirect2="2>&1"
    fi
    if [[ $UNAME == 'Darwin' ]]; then #Mac OS
	print2log "${SOURCE_DIR}/${APP_DIR}/configure \
	    --with-dyld \
	    --prefix=${PREFIX}" c
#	    --enable-unicode=ucs4 \
#	    --program-suffix=.exe $redirect ${BUILD_DIR}/${APP_DIR}/PYTHON_configure.log $redirect2" c
    else
	print2log "${SOURCE_DIR}/${APP_DIR}/configure \
	    --prefix=${PREFIX} \
	    --enable-shared" c
# $redirect ${BUILD_DIR}/${APP_DIR}/PYTHON_configure.log $redirect2" c
#            --enable-unicode=ucs4 \
#	    --program-suffix=.exe \
    fi

    print2log "make" c # $redirect ${BUILD_DIR}/${APP_DIR}/PYTHON_build.log $redirect2" c
    print2log "make install" c # $redirect ${BUILD_DIR}/${APP_DIR}/PYTHON_install.log $redirect2" c

    PYTHONBIN3=${INSTALL_DIR}/bin/python3  # ML changed to python3 on 2014-04-02
    if command -v $PYTHONBIN3; then #python has been installed
	print2log "...done"
    else
	print2log "python could not be installed"
	print2log "please install python >= ${REQUIRED} manually"
	print2log "if you already have it installed, make sure that\nyour PATH has been changed accordingly"
	exit 1
    fi
fi

print2log "\ninstalling the other dependencies..."
changedir $SETUPDIR
$PYTHONBIN setup.py --ping=pong --uname=${UNAME} --gcc=${GCC} --gpp=${GPP} --prefix=${INSTALL_DIR} --download=${DO_DOWNLOAD} --unpack=${DO_UNPACK} --install=${DO_INSTALL} $VERBOSE
