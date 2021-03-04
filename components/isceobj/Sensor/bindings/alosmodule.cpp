//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Copyright 2009 California Institute of Technology. ALL RIGHTS RESERVED.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 
// United States Government Sponsorship acknowledged. This software is subject to
// U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
// (No [Export] License Required except when exporting to an embargoed country,
// end user, or in support of a prohibited end use). By downloading this software,
// the user agrees to comply with all applicable U.S. export laws and regulations.
// The user has the responsibility to obtain export licenses, or other export
// authority as may be required before exporting this software to any 'EAR99'
// embargoed foreign country or citizen of those countries.
//
// Author: Giangi Sacco
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#include <Python.h>
#include <iostream>
#include "alosmodule.h"

using namespace std;

static const char * const __doc__ = "module for ALOS_pre_process";

PyModuleDef moduledef = {
    // header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "alos",
    // module documentation string
    __doc__,
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1,
    alos_methods,
};

// initialization function for the module
// *must* be called PyInit_alos
PyMODINIT_FUNC
PyInit_alos()
{
    // create the module using moduledef struct defined above
    PyObject * module = PyModule_Create(&moduledef);
    // check whether module creation succeeded and raise an exception if not
    if (!module) {
        return module;
    }
    // otherwise, we have an initialized module
    // and return the newly created module
    return module;
}

PyObject *alos_C(PyObject* self,PyObject *args)
{
    char *imageFile,*leaderFile,*outFile;
    int image_i;
    struct PRM inputPRM;
    struct PRM outputPRM;
    struct GLOBALS globals;

    if(!PyArg_ParseTuple(args,"sssi",&leaderFile,&imageFile,&outFile, &image_i))
    {
        return NULL;
    }
    strcpy(inputPRM.led_file,leaderFile);
    // They call it the input_file, since it is used as input to the remaining
    // functions
    strcpy(inputPRM.input_file,outFile);
    inputPRM.near_range = -1; // Near Range
    inputPRM.RE = -1; // Local Earth Radius
    inputPRM.fd1 = 0.0; // Doppler centroid
    inputPRM.chirp_ext = 1000; // Chirp Extension
    inputPRM.nrows = 16384; // Number of rows to use
    inputPRM.num_patches = 1000; // Number of patches to use
    // There are other options that are globals and are not listed in the PRM
    // structure
    globals.imagefilename = imageFile;
    globals.quad_pol = 0; // Is this quad polarization data?
    globals.ALOS_format = 0; // Is this an ERSDAC product?
    globals.force_slope = 0; // Should we force a chirp slope?
    globals.forced_slope = 1.0; // If so, what is its value?
    globals.dopp = 0; // Are we calculating a doppler?
    globals.tbias = 0.0; // Is there a time bias to fix poor orbits?

    ALOS_pre_process(inputPRM,&outputPRM,globals,image_i);

    PyObject * dict = PyDict_New();
    createDictionaryOutput(&outputPRM,dict);
    return Py_BuildValue("N", dict);
}

PyObject *alose_C(PyObject* self,PyObject *args)
{
    char *imageFile,*leaderFile,*outFile;
    int image_i;
    struct PRM inputPRM;
    struct PRM outputPRM;
    struct GLOBALS globals;

    if(!PyArg_ParseTuple(args,"sssi",&leaderFile,&imageFile,&outFile, &image_i))
    {
        return NULL;
    }
    strcpy(inputPRM.led_file,leaderFile);
    // They call it the input_file, since it is used as input to the remaining
    // functions
    strcpy(inputPRM.input_file,outFile);
    inputPRM.near_range = -1; // Near Range
    inputPRM.RE = -1; // Local Earth Radius
    inputPRM.fd1 = 0.0; // Doppler centroid
    inputPRM.chirp_ext = 1000; // Chirp Extension
    inputPRM.nrows = 16384; // Number of rows to use
    inputPRM.num_patches = 1000; // Number of patches to use
    // There are other options that are globals and are not listed in the PRM
    // structure
    globals.imagefilename = imageFile;
    globals.quad_pol = 0; // Is this quad polarization data?
    globals.ALOS_format = 1; // Is this an ERSDAC product?
    globals.force_slope = 0; // Should we force a chirp slope?
    globals.forced_slope = 1.0; // If so, what is its value?
    globals.dopp = 0; // Are we calculating a doppler?
    globals.tbias = 0.0; // Is there a time bias to fix poor orbits?

    ALOS_pre_process(inputPRM,&outputPRM,globals,image_i);

    PyObject * dict = PyDict_New();
    createDictionaryOutput(&outputPRM,dict);
    return Py_BuildValue("N", dict);
}


PyObject * createDictionaryOutput(struct PRM * prm, PyObject * dict)
{
    double vel;
    double fd,fdd,fddd;
    double dr,daz;
    int lookssquare;
    double cosinc, sininc, range;
    double sol = 299792458.;
    /*  velocity in orbit  */
    vel=prm->vel/sqrt(prm->RE/(prm->RE+prm->ht));
    /*  Doppler in prfs */
    fd=prm->fd1/prm->prf;
    fdd=prm->fdd1/prm->prf;
    fddd=prm->fddd1/prm->prf;
    /* looks for ~square pixels */
    range=prm->near_range+sol/2./prm->fs*prm->num_rng_bins/2.;
    cosinc=(prm->RE*prm->RE +
        range*range-((prm->RE+prm->ht)*(prm->RE+prm->ht)))/2./prm->RE/range;
    sininc=sqrt(1-cosinc*cosinc);
    dr=sol/2./prm->fs/sininc;
    daz=vel/prm->prf*(prm->RE/(prm->RE+prm->ht));
    lookssquare=dr/daz+0.5;
    if(lookssquare == 2)lookssquare=4;
    if(lookssquare == 3)lookssquare=4;

    Py_ssize_t len = 3;
    PyObject  * dopCoef = PyList_New(len);
    PyObject * floatVal = PyFloat_FromDouble((double)fd);
    PyList_SetItem(dopCoef,0,floatVal);//steals the reference
    floatVal = PyFloat_FromDouble((double)fdd);
    PyList_SetItem(dopCoef,1,floatVal);
    floatVal = PyFloat_FromDouble((double)fdd);
    PyList_SetItem(dopCoef,2,floatVal);
    //does not steal -> iuse Py_xdecref
    PyDict_SetItemString(dict, "DOPPLER_CENTROID_COEFFICIENTS", dopCoef);
    Py_XDECREF(dopCoef);
    PyObject * intVal = PyLong_FromLong((long) prm->bytes_per_line);
    PyDict_SetItemString(dict,"NUMBER_BYTES_PER_LINE",intVal);
    Py_XDECREF(intVal);
    intVal = PyLong_FromLong((long) prm->good_bytes);
    PyDict_SetItemString(dict,"NUMBER_GOOD_BYTES",intVal);
    Py_XDECREF(intVal);



    intVal = PyLong_FromLong((long) prm->num_lines);
    PyDict_SetItemString(dict,"NUMBER_LINES",intVal);
    Py_XDECREF(intVal);


    intVal = PyLong_FromLong((long) prm->num_rng_bins);
    PyDict_SetItemString(dict,"NUMBER_RANGE_BIN",intVal);
    Py_XDECREF(intVal);
    intVal = PyLong_FromLong((long) lookssquare);
    PyDict_SetItemString(dict,"NUMBER_AZIMUTH_LOOKS",intVal);
    Py_XDECREF(intVal);
    intVal = PyLong_FromLong((long)prm->chirp_ext);
    PyDict_SetItemString(dict,"RANGE_CHIRP_EXTENSION_POINTS",intVal);
    Py_XDECREF(intVal);
    floatVal = PyFloat_FromDouble((double)prm->RE);
    PyDict_SetItemString(dict,"PLANET_LOCAL_RADIUS",floatVal);
    Py_XDECREF(floatVal);
    floatVal = PyFloat_FromDouble((double)vel);
    PyDict_SetItemString(dict,"BODY_FIXED_VELOCITY",floatVal);
    Py_XDECREF(floatVal);
    floatVal = PyFloat_FromDouble((double)prm->ht);
    PyDict_SetItemString(dict,"SPACECRAFT_HEIGHT",floatVal);
    Py_XDECREF(floatVal);
    floatVal = PyFloat_FromDouble((double)prm->near_range);
    PyDict_SetItemString(dict,"RANGE_FIRST_SAMPLE",floatVal);
    Py_XDECREF(floatVal);
    floatVal = PyFloat_FromDouble((double)prm->prf);
    PyDict_SetItemString(dict,"PRF",floatVal);
    Py_XDECREF(floatVal);
    floatVal = PyFloat_FromDouble((double)prm->xmi);
    PyDict_SetItemString(dict,"INPHASE_VALUE",floatVal);
    Py_XDECREF(floatVal);
    floatVal = PyFloat_FromDouble((double)prm->xmq);
    PyDict_SetItemString(dict,"QUADRATURE_VALUE",floatVal);
    Py_XDECREF(floatVal);
    floatVal = PyFloat_FromDouble((double)prm->az_res);
    PyDict_SetItemString(dict,"AZIMUTH_RESOLUTION",floatVal);
    Py_XDECREF(floatVal);
    floatVal = PyFloat_FromDouble((double)prm->fs);
    PyDict_SetItemString(dict,"RANGE_SAMPLING_RATE",floatVal);
    Py_XDECREF(floatVal);
    floatVal = PyFloat_FromDouble((double)prm->chirp_slope);
    PyDict_SetItemString(dict,"CHIRP_SLOPE",floatVal);
    Py_XDECREF(floatVal);
    floatVal = PyFloat_FromDouble((double)prm->pulsedur);
    PyDict_SetItemString(dict,"RANGE_PULSE_DURATION",floatVal);
    Py_XDECREF(floatVal);
    floatVal = PyFloat_FromDouble((double)prm->lambda);
    PyDict_SetItemString(dict,"RADAR_WAVELENGTH",floatVal);
    Py_XDECREF(floatVal);
    PyObject * strVal = PyUnicode_FromString(prm->iqflip);
    PyDict_SetItemString(dict,"IQ_FLIP",strVal);
    Py_XDECREF(strVal);
    floatVal = PyFloat_FromDouble((double)prm->SC_clock_start);
    PyDict_SetItemString(dict,"SC_CLOCK_START",floatVal);
    Py_XDECREF(floatVal);
    floatVal = PyFloat_FromDouble((double)prm->SC_clock_stop);
    PyDict_SetItemString(dict,"SC_CLOCK_STOP",floatVal);
    Py_XDECREF(floatVal);
    floatVal = PyFloat_FromDouble((double)prm->near_range);
    PyDict_SetItemString(dict,"NEAR_RANGE",floatVal);
    Py_XDECREF(floatVal);
    intVal = PyLong_FromLong((long)prm->first_sample);
    PyDict_SetItemString(dict,"FIRST_SAMPLE",intVal);
    Py_XDECREF(intVal);

    return Py_BuildValue("i", 1);
}
