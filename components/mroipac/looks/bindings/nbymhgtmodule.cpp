//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                                  Giangi Sacco
//                        NASA Jet Propulsion Laboratory
//                      California Institute of Technology
//                  (C) 2009-2010  All Rights Reserved
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#include <Python.h>
#include "nbymhgtmodule.h"
#include <cmath>
#include <sstream>
#include <iostream>
#include <string>
#include <stdint.h>
#include <vector>
#include "ImageAccessor.h"
using namespace std;

static const char * const __doc__ = "Python extension for nbymhgt.F";

PyModuleDef moduledef = {
    // header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "nbymhgt",
    // module documentation string
    __doc__,
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1,
    nbymhgt_methods,
};

// initialization function for the module
PyMODINIT_FUNC
PyInit_nbymhgt()
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



PyObject * nbymhgt_C(PyObject* self, PyObject* args)
{
    char * inputImage;
    char * outputImage;
    int lenIn = 0;
    int lenOut = 0;
    char enIn = 'n';
    char enOut = 'n';
    string inmode = "read";
    string typeIn = "FLOAT";
    string typeOut = "FLOAT";
    string outmode = "write";
    int width = 0;
    int navg = 0;//average along width
    int mavg = 0;//average along length
    int flag = 0;
    int length = -1;

    PyObject * dictionary = NULL;
    //put explicity the mandatory args and in a dictionary the optionals
    if(!PyArg_ParseTuple(args, "s#s#iii|O", &inputImage,&lenIn,&outputImage,&lenOut,&width,&navg,&mavg,&dictionary))
    {
        return NULL;
    }
    if((dictionary != NULL))
    {
        PyObject * lengthPy = PyDict_GetItemString(dictionary,"LENGTH");
        if(lengthPy != NULL)
        {
            length = (int) PyLong_AsLong(lengthPy);
        }
        PyObject * enInPy = PyDict_GetItemString(dictionary,"INPUT_ENDIANNESS");
        if(enInPy != NULL)
        {
            char * inEndian = PyBytes_AsString(enInPy);
            enIn = inEndian[0];
        }
        PyObject * enOutPy = PyDict_GetItemString(dictionary,"OUTPUT_ENDIANNESS");
        if(enOutPy != NULL)
        {
            char * outEndian = PyBytes_AsString(enOutPy);
            enOut = outEndian[0];
        }

    }
    int wido = width/navg;
    vector<float> b1(2*width*mavg,0);
    vector<float> bout(2*wido,0);

    string infile = inputImage;
    string outfile = outputImage;


    ImageAccessor IAIn;
    ImageAccessor IAOut;
    if( enIn == 'n')//use as default the machine endianness
    {
        enIn = IAIn.getMachineEndianness();
    }
    if( enOut == 'n')//use as default the machine endianness
    {
        enOut = IAOut.getMachineEndianness();
    }
    IAIn.initImageAccessor(infile,inmode,enIn,typeIn,2*width);
    IAOut.initImageAccessor(outfile,outmode,enOut,typeOut,2*wido);
    if(length == -1)
    {
        length = IAIn.getFileLength();
    }
    int outLength = length/mavg;
    int indX = 0;
    int indY = 0;
    int numEl = 2*width*mavg;
    for(int i = 0; i < outLength; ++i)
    {
        indY = i*mavg + 1;
        indX = 1;
        IAIn.getSequentialElements((char *) &b1[0], indY,indX,numEl);
        for(int j = 0; j < wido; ++j)
        {
            int numGood = 0;
            for(int k = 0; k < navg; ++k)
            {
                for(int l = 0; l < mavg; ++l)
                {
                    if(b1[k + j*navg + 2*l*width] > flag)
                    {
                        bout[j] += b1[k + j*navg + 2*l*width];
                        bout[j + wido] += b1[k + j*navg + 2*l*width + width];
                        ++numGood;
                    }
                }
            }
            if(numGood)
            {
                bout[j] /= numGood;
                bout[j + wido] /= numGood;
            }
        }
        IAOut.setLineSequential((char *) &bout[0]);
        bout.assign(2*wido,0);
    }
    IAIn.finalizeImageAccessor();
    IAOut.finalizeImageAccessor();
    return Py_BuildValue("i", 0);

}
