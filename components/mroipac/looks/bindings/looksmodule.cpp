//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                                  Giangi Sacco
//                        NASA Jet Propulsion Laboratory
//                      California Institute of Technology
//                  (C) 2009-2010  All Rights Reserved
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#include <Python.h>
#include "looksmodule.h"
#include <cmath>
#include <complex>
#include <sstream>
#include <iostream>
#include <string>
#include <stdint.h>
#include <vector>
#include "DataAccessor.h"
using namespace std;

static const char * const __doc__ = "Python extension for looks.F";

PyModuleDef moduledef = {
    // header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "looks",
    // module documentation string
    __doc__,
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1,
    looks_methods,
};

// initialization function for the module
// *must* be called PyInit_formslc
PyMODINIT_FUNC
PyInit_looks()
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

template <typename T>
int takeLooks(DataAccessor *IAIn, DataAccessor* IAout, int ld, int la);

template <typename T>
int takeLookscpx(DataAccessor *IAIn, DataAccessor* IAout, int ld, int la);


PyObject * looks_C(PyObject* self, PyObject* args)
{
    uint64_t inptr, outptr;
    int nd,na;
    DataAccessor *in;
    DataAccessor *out;
    char *dtype;
    string type;
    int retVal;

    if(!PyArg_ParseTuple(args, "KKiis", &inptr,&outptr,&nd,&na,&dtype))
    {
        return NULL;
    }

    type = dtype;
    in = (DataAccessor*) inptr;
    out = (DataAccessor*) outptr;

    if (type == "byte" || type == "BYTE" || type == "char" || type == "CHAR")
    {
        retVal = takeLooks<char>(in,out,nd,na);
    }
    else if (type == "short" || type == "SHORT")
    {
        retVal = takeLooks<short>(in,out,nd,na);
    }
    else if (type == "int" || type == "INT")
    {
        retVal = takeLooks<int>(in,out,nd,na);
    }
    else if (type == "long" || type == "LONG")
    {
        retVal = takeLooks<long>(in,out,nd,na);
    }
    else if (type == "float" || type == "FLOAT")
    {
        retVal = takeLooks<float>(in,out,nd,na);
    }
    else if (type == "double" || type == "DOUBLE")
    {
      retVal = takeLooks<double>(in,out,nd,na);
    }
    else if (type == "cbyte" || type == "CBYTE" || type == "cchar"
      || type == "CCHAR")
    {
        retVal = takeLookscpx<char>(in,out,nd,na);
    }
    else if (type == "cshort" || type == "CSHORT")
    {
        retVal = takeLookscpx<short> (in,out,nd,na);
    }
    else if (type == "cint" || type == "CINT")
    {
        retVal = takeLookscpx<int> (in,out,nd,na);
    }
    else if (type == "clong" || type == "CLONG")
    {
        retVal = takeLookscpx<long> (in,out,nd,na);
    }
    else if (type == "cfloat" || type == "CFLOAT")
    {
        retVal = takeLookscpx<float>(in,out,nd,na);
    }
    else if (type == "cdouble" || type == "CDOUBLE")
    {
        retVal = takeLookscpx<double>(in,out,nd,na);
    }
    else
    {
        cout << "Error. Unrecognized data type " << type << endl;

        ERR_MESSAGE;
    }

    return Py_BuildValue("i", retVal);
}


template<typename T>
int takeLooks(DataAccessor *IAIn, DataAccessor* IAout, int ld, int la)
{

    int na = IAIn->getWidth();
    int nd = IAIn->getNumberOfLines();
    int bands = IAIn->getBands();
    int nfull = na * bands;

    vector<double> bdbl(nfull,0);
    vector<T > bout(nfull,0);
    vector<T > ain(nfull,0);
    bool eofReached = false;

    int lineCount = 0;
    double norm = ld*la;
    int naout = (na/la) * la;
    int ndout = (nd/ld) * ld;
    int nfullout = naout*bands;
    int retVal;

    lineCount = 0;

    for(int line = 0; line < ndout; line += ld)
    {
        eofReached = false;

        for(int i = 0; i < ld; ++i)
        {
            int lineToGet = (line + i);
            retVal = IAIn->getLine((char *) &ain[0], lineToGet);

            if (retVal == -1)
            {
                eofReached = true;
                break;
            }

            for(int j = 0; j < nfull; j++)
            {
                bdbl[j] += ain[j];
            }

        }

        int jpix=0;
        for(int j=0; j<naout; j += la)
        {
            for(int b=0; b < bands; ++b)
            {
                double sum=0.0;
                for(int k = 0; k < la; ++k)
                {
                    sum += bdbl[(j+k)*bands+b];
                }
                bout[jpix*bands+b] = static_cast<T>(sum/norm);

            }
            ++jpix;
        }

        int lineToSet = lineCount;
        IAout->setLine((char *) &bout[0], lineToSet);
        bdbl.assign(nfull,0.0);
        bout.assign(nfull,0.0);
        ++lineCount;
    }
    return 0;
}



template<typename T>
int takeLookscpx(DataAccessor *IAIn, DataAccessor* IAout, int ld, int la)
{

    int na = IAIn->getWidth();
    int nd = IAIn->getNumberOfLines();
    int bands = IAIn->getBands();
    int nfull = na * bands;

    vector<complex<double> > bdbl(nfull,0);
    vector<complex<T> > bout(nfull,0);
    vector<complex<T> > ain(nfull,0);
    bool eofReached = false;

    int lineCount = 0;
    double norm = ld*la;
    int naout = (na/la) * la;
    int ndout = (nd/ld) * ld;
    int nfullout = naout*bands;
    int retVal;


    lineCount = 0;

    for(int line = 0; line < ndout; line += ld)
    {
        eofReached = false;

        for(int i = 0; i < ld; ++i)
        {
            int lineToGet = (line + i);
            retVal = IAIn->getLine((char *) &ain[0], lineToGet);

            if (retVal == -1)
            {
                eofReached = true;
                break;
            }

            for(int j = 0; j < nfull; j++)
            {
                bdbl[j] += complex<double>(ain[j].real(), ain[j].imag());
            }

        }

        int jpix=0;
        for(int j=0; j<naout; j += la)
        {
            for(int b=0; b < bands; ++b)
            {
                complex<double> sum(0.0,0.0);
                for(int k = 0; k < la; ++k)
                {
                    sum += bdbl[(j+k)*bands+b];
                }
                bout[jpix*bands+b] = complex<T> (static_cast<T>(sum.real()/norm), static_cast<T>(sum.imag()/norm)) ;
            }
            ++jpix;
        }

        int lineToSet = lineCount;
        IAout->setLine((char *) &bout[0], lineToSet);
        bdbl.assign(nfull,0.0);
        bout.assign(nfull,0.0);
        ++lineCount;
    }
    return 0;
}
