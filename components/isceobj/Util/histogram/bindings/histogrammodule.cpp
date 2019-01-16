//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Copyright 2012 California Institute of Technology. ALL RIGHTS RESERVED.
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
#include "histogrammodule.h"

// A C++ extension is required for this code since
// ctypes does not currently allow interfacing with C++ code
// (name-mangling and all).

static const char * __doc__ = "module for p2.cpp";

PyModuleDef moduledef = {
    // header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "histogram",
    // module documentation string
    __doc__,
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1,
    histogram_methods,
};

// initialization function for the module
// *must* be called PyInit_filter
PyMODINIT_FUNC
PyInit_histogram()
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

PyObject *realHistogram_C(PyObject *self, PyObject *args)
{
    uint64_t imagePointer;
    int width, height, bands;
    int nBins,i,j,k,ind;
    double nullValue, key;
    DataAccessor* img;
    double *line;
    double *qt;
    double *val;

    //Histogram objects
    p2_t *hists;


    //Parse command line
    if(!PyArg_ParseTuple(args, "Kid", &imagePointer, &nBins,&nullValue))
    {
      return NULL;
    }

    //Get image dimensions
    img = (DataAccessor*) imagePointer;
    bands = img->getBands();
    width = img->getWidth();
    height = img->getNumberOfLines();

    std::cout << "Dimensions: " << width << " " << height << "\n";
    //Allocate memory for one line of data
    line = new double[width*bands];
    qt = new double[nBins + 1];
    val = new double[nBins + 1];

    //Create histogram objects
    hists = new p2_t[bands];
    for(k=0; k<bands; k++)
        hists[k].add_equal_spacing(nBins);


    //For each line
    for(i=0; i<height; i++)
    {
        img->getLineSequential((char*) line);

        //For each band
        for(k=0; k<bands; k++)
        {
            //For each pixel
            for(j=0; j<width;j++)
            {
                ind = j*bands + k;
                key = line[ind];
                if (key != nullValue)
                    hists[k].add(key);
            }
        }
    }


//    for(k=0;k<bands;k++)
//        hists[k].report();

    //Delete line 
    delete [] line;

    //Convert to Python Lists
    PyObject *list = PyList_New(bands); 

    for (k=0;k<bands;k++)
    {
        hists[k].getStats(qt, val);
        PyObject *qlist = PyList_New(nBins + 1);
        PyObject *vlist = PyList_New(nBins + 1);

        for(i=0; i< (nBins+1); i++)
        {
            PyObject* listEl = PyFloat_FromDouble(qt[i]);
            if(listEl == NULL)
            {
                std::cout << "Error in file " << __FILE__   << " at line " << __LINE__ << ". Cannot set list element" << endl;
                exit(1);
            }            
            PyList_SetItem(qlist,i, listEl);

            listEl = PyFloat_FromDouble(val[i]);
            if(listEl == NULL)
            {
                std::cout << "Error in file " << __FILE__   << " at line " << __LINE__ << ". Cannot set list element" << endl;
                exit(1);
            }
            PyList_SetItem(vlist,i, listEl);
        }

        PyObject *tuple = PyTuple_New(2);
        PyTuple_SetItem(tuple, 0, qlist);
        PyTuple_SetItem(tuple, 1, vlist);

        PyList_SetItem(list, k, tuple);
    }

    //Delete stats arrays
    delete [] qt;
    delete [] val;

    //Delete the histogram object
    delete [] hists;

    return Py_BuildValue("N",list);
}


