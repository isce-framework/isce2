/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright 2019 California Institute of Technology. ALL RIGHTS RESERVED.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * United States Government Sponsorship acknowledged. This software is subject to
 * U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
 * (No [Export] License Required except when exporting to an embargoed country,
 * end user, or in support of a prohibited end use). By downloading this software,
 * the user agrees to comply with all applicable U.S. export laws and regulations.
 * The user has the responsibility to obtain export licenses, or other export
 * authority as may be required before exporting this software to any 'EAR99'
 * embargoed foreign country or citizen of those countries.
 *
 * Authors: Piyush Agram, Yang Lei
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */



#include <Python.h>
#include <string>
#include "geogridOptical.h"
#include "geogridOpticalmodule.h"

static const char * const __doc__ = "Python extension for geogrid";

PyModuleDef moduledef = {
    //header
    PyModuleDef_HEAD_INIT,
    //name of the module
    "geogridOptical",
    //module documentation string
    __doc__,
    //size of the per-interpreter state of the module;
    -1,
    geogrid_methods,
};

//Initialization function for the module
PyMODINIT_FUNC
PyInit_geogridOptical()
{
    PyObject* module = PyModule_Create(&moduledef);
    if (!module)
    {
        return module;
    }
    return module;
}

PyObject* createGeoGridOptical(PyObject* self, PyObject *args)
{
    geoGridOptical* ptr = new geoGridOptical;
    return Py_BuildValue("K", (uint64_t) ptr);
}

PyObject* destroyGeoGridOptical(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    if (!PyArg_ParseTuple(args, "K", &ptr))
    {
        return NULL;
    }

    if (((geoGridOptical*)(ptr))!=NULL)
    {
        delete ((geoGridOptical*)(ptr));
    }
    return Py_BuildValue("i", 0);
}

PyObject* setEPSG(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    int code1, code2;
    if (!PyArg_ParseTuple(args, "Kii", &ptr, &code1, &code2))
    {
        return NULL;
    }
    ((geoGridOptical*)(ptr))->epsgDem = code1;
    ((geoGridOptical*)(ptr))->epsgDat = code2;
    return Py_BuildValue("i", 0);
}


PyObject* setChipSizeX0(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    double chipSizeX0;
    if (!PyArg_ParseTuple(args, "Kd", &ptr, &chipSizeX0))
    {
        return NULL;
    }
    ((geoGridOptical*)(ptr))->chipSizeX0 = chipSizeX0;
    return Py_BuildValue("i", 0);
}

PyObject* setGridSpacingX(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    double gridSpacingX;
    if (!PyArg_ParseTuple(args, "Kd", &ptr, &gridSpacingX))
    {
        return NULL;
    }
    ((geoGridOptical*)(ptr))->gridSpacingX = gridSpacingX;
    return Py_BuildValue("i", 0);
}

PyObject* setRepeatTime(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    double repeatTime;
    if (!PyArg_ParseTuple(args, "Kd", &ptr, &repeatTime))
    {
        return NULL;
    }
    ((geoGridOptical*)(ptr))->dt = repeatTime;
    return Py_BuildValue("i", 0);
}

PyObject* setOpticalImageDimensions(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    int wid, len;
    if (!PyArg_ParseTuple(args, "Kii", &ptr, &wid, &len))
    {
        return NULL;
    }

    ((geoGridOptical*)(ptr))->nPixels = wid;
    ((geoGridOptical*)(ptr))->nLines = len;
    return Py_BuildValue("i", 0);
}

PyObject* setXParameters(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    double r0, rspace;
    if (!PyArg_ParseTuple(args, "Kdd", &ptr, &r0, &rspace))
    {
        return NULL;
    }

    ((geoGridOptical*)(ptr))->startingX = r0;
    ((geoGridOptical*)(ptr))->XSize = rspace;
    return Py_BuildValue("i", 0);
}

PyObject* setYParameters(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    double t0, prf;
    if (!PyArg_ParseTuple(args, "Kdd", &ptr, &t0, &prf))
    {
        return NULL;
    }

    ((geoGridOptical*)(ptr))->startingY = t0;
    ((geoGridOptical*)(ptr))->YSize = prf;
    return Py_BuildValue("i", 0);
}

PyObject* setXLimits(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    double x0, x1;
    if (!PyArg_ParseTuple(args, "Kdd", &ptr, &x0, &x1))
    {
        return NULL;
    }

    ((geoGridOptical*)(ptr))->xmin = x0;
    ((geoGridOptical*)(ptr))->xmax = x1;
    return Py_BuildValue("i", 0);
}

PyObject* setYLimits(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    double x0, x1;
    if (!PyArg_ParseTuple(args, "Kdd", &ptr, &x0, &x1))
    {
        return NULL;
    }

    ((geoGridOptical*)(ptr))->ymin = x0;
    ((geoGridOptical*)(ptr))->ymax = x1;
    return Py_BuildValue("i", 0);
}

PyObject* setDEM(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    char* name;
    if (!PyArg_ParseTuple(args, "Ks", &ptr, &name))
    {
        return NULL;
    }

    ((geoGridOptical*)(ptr))->demname = std::string(name);
    return Py_BuildValue("i", 0);
}

PyObject* setVelocities(PyObject *self, PyObject* args)
{
    uint64_t ptr;
    char *vx;
    char *vy;
    if (!PyArg_ParseTuple(args, "Kss", &ptr, &vx, &vy))
    {
        return NULL;
    }

    ((geoGridOptical*)(ptr))->vxname = std::string(vx);
    ((geoGridOptical*)(ptr))->vyname = std::string(vy);
    return Py_BuildValue("i", 0);
}

PyObject* setSearchRange(PyObject *self, PyObject* args)
{
    uint64_t ptr;
    char *srx;
    char *sry;
    if (!PyArg_ParseTuple(args, "Kss", &ptr, &srx, &sry))
    {
        return NULL;
    }
    
    ((geoGridOptical*)(ptr))->srxname = std::string(srx);
    ((geoGridOptical*)(ptr))->sryname = std::string(sry);
    return Py_BuildValue("i", 0);
}

PyObject* setChipSizeMin(PyObject *self, PyObject* args)
{
    uint64_t ptr;
    char *csminx;
    char *csminy;
    if (!PyArg_ParseTuple(args, "Kss", &ptr, &csminx, &csminy))
    {
        return NULL;
    }
    
    ((geoGridOptical*)(ptr))->csminxname = std::string(csminx);
    ((geoGridOptical*)(ptr))->csminyname = std::string(csminy);
    return Py_BuildValue("i", 0);
}

PyObject* setChipSizeMax(PyObject *self, PyObject* args)
{
    uint64_t ptr;
    char *csmaxx;
    char *csmaxy;
    if (!PyArg_ParseTuple(args, "Kss", &ptr, &csmaxx, &csmaxy))
    {
        return NULL;
    }
    
    ((geoGridOptical*)(ptr))->csmaxxname = std::string(csmaxx);
    ((geoGridOptical*)(ptr))->csmaxyname = std::string(csmaxy);
    return Py_BuildValue("i", 0);
}

PyObject* setStableSurfaceMask(PyObject *self, PyObject* args)
{
    uint64_t ptr;
    char *ssm;
    if (!PyArg_ParseTuple(args, "Ks", &ptr, &ssm))
    {
        return NULL;
    }
    
    ((geoGridOptical*)(ptr))->ssmname = std::string(ssm);
    return Py_BuildValue("i", 0);
}

PyObject* setSlopes(PyObject *self, PyObject* args)
{
    uint64_t ptr;
    char *sx;
    char *sy;
    if (!PyArg_ParseTuple(args, "Kss", &ptr, &sx, &sy))
    {
        return NULL;
    }

    ((geoGridOptical*)(ptr))->dhdxname = std::string(sx);
    ((geoGridOptical*)(ptr))->dhdyname = std::string(sy);
    return Py_BuildValue("i", 0);
}

PyObject* setWindowLocationsFilename(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    char* name;
    if (!PyArg_ParseTuple(args, "Ks", &ptr, &name))
    {
        return NULL;
    }

    ((geoGridOptical*)(ptr))->pixlinename = std::string(name);
    return Py_BuildValue("i", 0);
}

PyObject* setWindowOffsetsFilename(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    char* name;
    if (!PyArg_ParseTuple(args, "Ks", &ptr, &name))
    {
        return NULL;
    }

    ((geoGridOptical*)(ptr))->offsetname = std::string(name);
    return Py_BuildValue("i", 0);
}

PyObject* setWindowSearchRangeFilename(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    char* name;
    if (!PyArg_ParseTuple(args, "Ks", &ptr, &name))
    {
        return NULL;
    }
    
    ((geoGridOptical*)(ptr))->searchrangename = std::string(name);
    return Py_BuildValue("i", 0);
}

PyObject* setWindowChipSizeMinFilename(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    char* name;
    if (!PyArg_ParseTuple(args, "Ks", &ptr, &name))
    {
        return NULL;
    }
    
    ((geoGridOptical*)(ptr))->chipsizeminname = std::string(name);
    return Py_BuildValue("i", 0);
}

PyObject* setWindowChipSizeMaxFilename(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    char* name;
    if (!PyArg_ParseTuple(args, "Ks", &ptr, &name))
    {
        return NULL;
    }
    
    ((geoGridOptical*)(ptr))->chipsizemaxname = std::string(name);
    return Py_BuildValue("i", 0);
}

PyObject* setWindowStableSurfaceMaskFilename(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    char* name;
    if (!PyArg_ParseTuple(args, "Ks", &ptr, &name))
    {
        return NULL;
    }
    
    ((geoGridOptical*)(ptr))->stablesurfacemaskname = std::string(name);
    return Py_BuildValue("i", 0);
}

PyObject* setRO2VXFilename(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    char* name;
    if (!PyArg_ParseTuple(args, "Ks", &ptr, &name))
    {
        return NULL;
    }

    ((geoGridOptical*)(ptr))->ro2vx_name = std::string(name);
    return Py_BuildValue("i", 0);
}

PyObject* setRO2VYFilename(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    char* name;
    if (!PyArg_ParseTuple(args, "Ks", &ptr, &name))
    {
        return NULL;
    }
    
    ((geoGridOptical*)(ptr))->ro2vy_name = std::string(name);
    return Py_BuildValue("i", 0);
}


PyObject* setNodataOut(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    int nodata;
    if (!PyArg_ParseTuple(args, "Ki", &ptr, &nodata))
    {
        return NULL;
    }
    
    ((geoGridOptical*)(ptr))->nodata_out = nodata;
    return Py_BuildValue("i", 0);
}



PyObject* getXOff(PyObject *self, PyObject *args)
{
    int var;
    uint64_t ptr;
    
    if (!PyArg_ParseTuple(args, "K", &ptr))
    {
        return NULL;
    }
    
    var = ((geoGridOptical*)(ptr))->pOff;
    return Py_BuildValue("i",var);
}

PyObject* getYOff(PyObject *self, PyObject *args)
{
    int var;
    uint64_t ptr;
    
    if (!PyArg_ParseTuple(args, "K", &ptr))
    {
        return NULL;
    }
    
    var = ((geoGridOptical*)(ptr))->lOff;
    return Py_BuildValue("i",var);
}

PyObject* getXCount(PyObject *self, PyObject *args)
{
    int var;
    uint64_t ptr;
    
    if (!PyArg_ParseTuple(args, "K", &ptr))
    {
        return NULL;
    }
    
    var = ((geoGridOptical*)(ptr))->pCount;
    return Py_BuildValue("i",var);
}

PyObject* getYCount(PyObject *self, PyObject *args)
{
    int var;
    uint64_t ptr;
    
    if (!PyArg_ParseTuple(args, "K", &ptr))
    {
        return NULL;
    }
    
    var = ((geoGridOptical*)(ptr))->lCount;
    return Py_BuildValue("i",var);
}

PyObject* getXPixelSize(PyObject *self, PyObject *args)
{
    double var;
    uint64_t ptr;
    
    if (!PyArg_ParseTuple(args, "K", &ptr))
    {
        return NULL;
    }
    
    var = ((geoGridOptical*)(ptr))->X_res;
    return Py_BuildValue("d",var);
}

PyObject* getYPixelSize(PyObject *self, PyObject *args)
{
    double var;
    uint64_t ptr;
    
    if (!PyArg_ParseTuple(args, "K", &ptr))
    {
        return NULL;
    }
    
    var = ((geoGridOptical*)(ptr))->Y_res;
    return Py_BuildValue("d",var);
}

PyObject* setDtUnity(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    double dt_unity;
    if (!PyArg_ParseTuple(args, "Kd", &ptr, &dt_unity))
    {
        return NULL;
    }
    ((geoGridOptical*)(ptr))->dt_unity = dt_unity;
    return Py_BuildValue("i", 0);
}

PyObject* setMaxFactor(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    double max_factor;
    if (!PyArg_ParseTuple(args, "Kd", &ptr, &max_factor))
    {
        return NULL;
    }
    ((geoGridOptical*)(ptr))->max_factor = max_factor;
    return Py_BuildValue("i", 0);
}

PyObject* setUpperThreshold(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    double upper_thld;
    if (!PyArg_ParseTuple(args, "Kd", &ptr, &upper_thld))
    {
        return NULL;
    }
    ((geoGridOptical*)(ptr))->upper_thld = upper_thld;
    return Py_BuildValue("i", 0);
}

PyObject* setLowerThreshold(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    double lower_thld;
    if (!PyArg_ParseTuple(args, "Kd", &ptr, &lower_thld))
    {
        return NULL;
    }
    ((geoGridOptical*)(ptr))->lower_thld = lower_thld;
    return Py_BuildValue("i", 0);
}


PyObject* geogridOptical(PyObject* self, PyObject* args)
{
    uint64_t ptr;
    if (!PyArg_ParseTuple(args, "K", &ptr))
    {
        return NULL;
    }

    ((geoGridOptical*)(ptr))->geogridOptical();
    return Py_BuildValue("i", 0);
}
