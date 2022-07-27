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


#ifndef geogridmodule_h
#define geogridmodule_h

#include <Python.h>
#include <stdint.h>

extern "C"
{
        PyObject * createGeoGrid(PyObject*, PyObject*);
        PyObject * destroyGeoGrid(PyObject*, PyObject*);
        PyObject * geogrid(PyObject *, PyObject *);
        PyObject * setRadarImageDimensions(PyObject *, PyObject *);
        PyObject * setRangeParameters(PyObject *, PyObject *);
        PyObject * setAzimuthParameters(PyObject*, PyObject *);
        PyObject * setRepeatTime(PyObject *, PyObject *);
    
        PyObject * setDEM(PyObject *, PyObject *);
        PyObject * setVelocities(PyObject*, PyObject*);
        PyObject * setSearchRange(PyObject*, PyObject*);
        PyObject * setChipSizeMin(PyObject*, PyObject*);
        PyObject * setChipSizeMax(PyObject*, PyObject*);
        PyObject * setStableSurfaceMask(PyObject*, PyObject*);
        PyObject * setSlopes(PyObject*, PyObject*);
        PyObject * setOrbit(PyObject *, PyObject *);
        PyObject * setLookSide(PyObject *, PyObject *);
        PyObject * setNodataOut(PyObject *, PyObject *);
    
        PyObject * setDtUnity(PyObject *, PyObject *);
        PyObject * setMaxFactor(PyObject *, PyObject *);
        PyObject * setUpperThreshold(PyObject*, PyObject *);
        PyObject * setLowerThreshold(PyObject *, PyObject *);

        PyObject * setWindowLocationsFilename(PyObject *, PyObject *);
        PyObject * setWindowOffsetsFilename(PyObject *, PyObject *);
        PyObject * setWindowSearchRangeFilename(PyObject *, PyObject *);
        PyObject * setWindowChipSizeMinFilename(PyObject *, PyObject *);
        PyObject * setWindowChipSizeMaxFilename(PyObject *, PyObject *);
        PyObject * setWindowStableSurfaceMaskFilename(PyObject *, PyObject *);
        PyObject * setRO2VXFilename(PyObject *, PyObject *);
        PyObject * setRO2VYFilename(PyObject *, PyObject *);
        PyObject * setEPSG(PyObject *, PyObject *);
        PyObject * setIncidenceAngle(PyObject *, PyObject *);
        PyObject * setChipSizeX0(PyObject *, PyObject *);
        PyObject * setGridSpacingX(PyObject *, PyObject *);
        PyObject * setXLimits(PyObject *, PyObject *);
        PyObject * setYLimits(PyObject *, PyObject *);
        PyObject * getXPixelSize(PyObject *, PyObject *);
        PyObject * getYPixelSize(PyObject *, PyObject *);
        PyObject * getXOff(PyObject *, PyObject *);
        PyObject * getYOff(PyObject *, PyObject *);
        PyObject * getXCount(PyObject *, PyObject *);
        PyObject * getYCount(PyObject *, PyObject *);
}

static PyMethodDef geogrid_methods[] =
{
        {"createGeoGrid_Py", createGeoGrid, METH_VARARGS, " "},
        {"destroyGeoGrid_Py", destroyGeoGrid, METH_VARARGS, " "},
        {"geogrid_Py", geogrid, METH_VARARGS, " "},
        {"setRadarImageDimensions_Py", setRadarImageDimensions, METH_VARARGS, " "},
        {"setRangeParameters_Py", setRangeParameters, METH_VARARGS, " "},
        {"setAzimuthParameters_Py", setAzimuthParameters, METH_VARARGS, " "},
        {"setRepeatTime_Py", setRepeatTime, METH_VARARGS, " "},
        {"setDEM_Py", setDEM, METH_VARARGS, " "},
        {"setEPSG_Py", setEPSG, METH_VARARGS, " "},
        {"setIncidenceAngle_Py", setIncidenceAngle, METH_VARARGS, " "},
        {"setChipSizeX0_Py", setChipSizeX0, METH_VARARGS, " "},
        {"setGridSpacingX_Py", setGridSpacingX, METH_VARARGS, " "},
        {"setVelocities_Py", setVelocities, METH_VARARGS, " "},
        {"setSearchRange_Py", setSearchRange, METH_VARARGS, " "},
        {"setChipSizeMin_Py", setChipSizeMin, METH_VARARGS, " "},
        {"setChipSizeMax_Py", setChipSizeMax, METH_VARARGS, " "},
        {"setStableSurfaceMask_Py", setStableSurfaceMask, METH_VARARGS, " "},
        {"setSlopes_Py", setSlopes, METH_VARARGS, " "},
        {"setOrbit_Py", setOrbit, METH_VARARGS, " "},
        {"setLookSide_Py", setLookSide, METH_VARARGS, " "},
        {"setNodataOut_Py", setNodataOut, METH_VARARGS, " "},
        {"setDtUnity_Py", setDtUnity, METH_VARARGS, " "},
        {"setMaxFactor_Py", setMaxFactor, METH_VARARGS, " "},
        {"setUpperThreshold_Py", setUpperThreshold, METH_VARARGS, " "},
        {"setLowerThreshold_Py", setLowerThreshold, METH_VARARGS, " "},
        {"setXLimits_Py", setXLimits, METH_VARARGS, " "},
        {"setYLimits_Py", setYLimits, METH_VARARGS, " "},
        {"getXPixelSize_Py", getXPixelSize, METH_VARARGS, " "},
        {"getYPixelSize_Py", getYPixelSize, METH_VARARGS, " "},
        {"getXOff_Py", getXOff, METH_VARARGS, " "},
        {"getYOff_Py", getYOff, METH_VARARGS, " "},
        {"getXCount_Py", getXCount, METH_VARARGS, " "},
        {"getYCount_Py", getYCount, METH_VARARGS, " "},
        {"setWindowLocationsFilename_Py", setWindowLocationsFilename, METH_VARARGS, " "},
        {"setWindowOffsetsFilename_Py", setWindowOffsetsFilename, METH_VARARGS, " "},
        {"setWindowSearchRangeFilename_Py", setWindowSearchRangeFilename, METH_VARARGS, " "},
        {"setWindowChipSizeMinFilename_Py", setWindowChipSizeMinFilename, METH_VARARGS, " "},
        {"setWindowChipSizeMaxFilename_Py", setWindowChipSizeMaxFilename, METH_VARARGS, " "},
        {"setWindowStableSurfaceMaskFilename_Py", setWindowStableSurfaceMaskFilename, METH_VARARGS, " "},
        {"setRO2VXFilename_Py", setRO2VXFilename, METH_VARARGS, " "},
        {"setRO2VYFilename_Py", setRO2VYFilename, METH_VARARGS, " "},
        {NULL, NULL, 0, NULL}
};
#endif //geoGridmodule_h

