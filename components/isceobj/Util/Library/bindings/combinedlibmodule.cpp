#include <Python.h>
#include "combinedlibmodule.h"
#include <stdint.h>
#include <iostream>
#include <string>
using namespace std;

static const char * const __doc__ = "module for combined lib";

PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "combinedlib",
    __doc__,
    -1,
    combinedlib_methods,
};

PyMODINIT_FUNC
PyInit_combinedlibmodule()
{
    PyObject * module = PyModule_Create(&moduledef);
    if (!module) {
        return module;
    }
    return module;
}


PyObject* exportOrbitToC(PyObject* self, PyObject* args)
{
    int nvec,i,j;
    int basis;
    double data[7];
    PyObject* list;
    cOrbit *orb = new cOrbit;
    if( orb == NULL)
    {
        cout << "Insufficient memory to allocate orbit at " << __FILE__ << endl;
        exit(1);
    }

    if(!PyArg_ParseTuple(args, "iO", &basis, &list))
    {
        return NULL;
    }
    if(!PyList_Check(list))
    {
        cout << "Expecting a list object for 2nd argument at " << __FILE__ << endl;
        exit(1);
    }

    nvec = (int) PyList_Size(list);
    initOrbit(orb, nvec, basis);

    for(i=0; i<nvec; i++)
    {
        PyObject * listEl = PyList_GetItem(list, i);
        if(listEl == NULL)
        {
            cout << "Error in retrieving state vector number " << i <<" at " << __FILE__ << endl;
            exit(1);
        }
        if(!PyList_Check(listEl))
        {
            cout << "Expecting a list of 7 double numbers at " << __FILE__ << endl;
            exit(1);
        }

        for(j=0;j<7;j++)
        {
            PyObject *val = PyList_GetItem(listEl, j);
            if(val == NULL)
            {
                cout << "Error retrieving state vector entry " << j << " at " << __FILE__ << endl;
                exit(1);
            }

            data[j] = (double) PyFloat_AsDouble(val);
            if(PyErr_Occurred() != NULL)
            {
                cout << "Error in translating Python List to double array at " << __FILE__ << endl;
                exit(1);
            }
        }

        setStateVector(orb, i, data[0], &(data[1]), &(data[4]));
    }

    return Py_BuildValue("K", (uint64_t) orb);
}

PyObject* exportPegToC(PyObject *self, PyObject* args)
{

    double lat, lon, hdg;
    cPeg* peg = new cPeg;

    if(!PyArg_ParseTuple(args, "ddd", &lat, &lon, &hdg))
    {
        return NULL;
    }

    peg->lat = lat;
    peg->lon = lon;
    peg->hdg = hdg;

    return Py_BuildValue("K", (uint64_t) peg);
}

PyObject* exportEllipsoidToC(PyObject *self, PyObject* args)
{

    double a, e2;
    cEllipsoid* elp = new cEllipsoid;

    if(!PyArg_ParseTuple(args, "dd", &a, &e2))
    {
        return NULL;
    }

    elp->a = a;
    elp->e2 = e2;

    return Py_BuildValue("K", (uint64_t) elp);
}

PyObject* exportPoly1DToC(PyObject *self, PyObject *args)
{
    cPoly1d* poly = new cPoly1d;
    PyObject* list;
    int order,i;
    double mean, norm;

    if(!PyArg_ParseTuple(args, "iddO",&order,&mean,&norm,&list))
    {
        return NULL;
    }

    initPoly1d(poly, order);
    poly->mean = mean;
    poly->norm = norm;

    if(!PyList_Check(list))
    {
        cout << "Expecting a list of 1D polynomial coefficients at " << __FILE__ << endl;
        exit(1);
    }

    for(i=0; i<= order; i++)
    {
        PyObject* listEl = PyList_GetItem(list, i);
        if(listEl == NULL)
        {
            cout << "Expecting a double precision float from the list at " << __FILE__ << endl;
            exit(1);
        }

        poly->coeffs[i] = (double) PyFloat_AsDouble(listEl);

        if(PyErr_Occurred() != NULL)
        {
            cout << "Conversion from list element to double precision float failed at " << __FILE__ << endl;
            exit(1);
        }
    }

    return Py_BuildValue("K", (uint64_t) poly);
}

PyObject* exportPoly2DToC(PyObject *self, PyObject *args)
{
    cPoly2d* poly = new cPoly2d;
    int orders[2];
    double means[2], norms[2];

    PyObject* ord;
    PyObject* avg;
    PyObject* norm;
    PyObject* list;
    int nx, ny;
    double val;

    if(!PyArg_ParseTuple(args, "OOOO", &ord, &avg, &norm, &list))
    {
        return NULL;
    }

    if(!PyList_Check(ord))
    {
        cout << "Expected 1st argument to be a list of 2 integers at " << __FILE__ << endl;
        exit(1);
    }

    for(int i=0; i<2; i++)
    {
        PyObject* listEl = PyList_GetItem(ord, i);
        if(listEl == NULL)
        {
            cout << "Expecting an int from the list at " << __FILE__ << endl;
            exit(1);
        }
        orders[i] = (int) PyLong_AsLong(listEl);
        if(PyErr_Occurred() != NULL)
        {
            cout << "Conversion from list element to integer failed at " << __FILE__ << endl;
            exit(1);
        }
    }

    initPoly2d(poly, orders[0], orders[1]);

    if(!PyList_Check(avg))
    {
        cout << "Expected 2nd argument to be a list of 2 floats at " << __FILE__ << endl;
        exit(1);
    }

    for(int i=0; i<2;i++)
    {
        PyObject* listEl = PyList_GetItem(avg, i);
        if(listEl == NULL)
        {
            cout << "Expecting a double precision float from the list at " << __FILE__ << endl;
            exit(1);
        }
        means[i] = (double) PyFloat_AsDouble(listEl);
        if(PyErr_Occurred() != NULL)
        {
            cout << "Conversion from list element to double precision float failed at " << __FILE__ << endl;
            exit(1);
        }
    }
    poly->meanAzimuth = means[0];
    poly->meanRange = means[1];

    if(!PyList_Check(norm))
    {
        cout << "Expected 3rd argument to be a list of 2 floats at " << __FILE__ << endl;
        exit(1);
    }

    for(int i=0; i<2;i++)
    {
        PyObject* listEl = PyList_GetItem(norm, i);
        if(listEl == NULL)
        {
            cout << "Expecting a double precision float from the list at " << __FILE__ << endl;
            exit(1);
        }
        norms[i] = (double) PyFloat_AsDouble(listEl);
        if(PyErr_Occurred() != NULL)
        {
            cout << "Conversion from list element to double precision float failed at " << __FILE__ << endl;
            exit(1);
        }
    }
    poly->normAzimuth = norms[0];
    poly->normRange = norms[1];


    if(!PyList_Check(list))
    {
        cout << "Expected 4th argument to be a list of coefficients at " << __FILE__ << endl;
        exit(1);
    }

    ny = (int) PyList_Size(list);
    if( ny != (orders[0]+1))
    {
        cout << "Expected a list of size " << orders[0]+1 << " azimuth coeffs at " << __FILE__ << endl;
        exit(1);
    }

    for(int i=0; i< ny; i++)
    {
        PyObject* listEl = PyList_GetItem(list, i);
        if(listEl == NULL)
        {
            cout << "Failed to extract a list of range coeffs at " << __FILE__ << endl;
            exit(1);
        }

        if(!PyList_Check(listEl))
        {
            cout << "Expected a list of range coeffs at " << __FILE__ << endl;
            exit(1);
        }

        nx = (int) PyList_Size(listEl);

        if(nx != (orders[1]+1))
        {
            cout << "Expected a list of size " << orders[1]+1 << " range coeffs at " << __FILE__ << endl;
            exit(1);
        }

        for(int j=0; j< nx; j++)
        {
            PyObject* elem = PyList_GetItem(listEl, j);
            val = (double) PyFloat_AsDouble(elem);

            if(PyErr_Occurred() != NULL)
            {
                cout << "Error in converting double precision float from list of coeffs at " << __FILE__ << endl;
                exit(1);
            }

            setCoeff2d(poly, i,j,val);
        }
    }

    printPoly2d(poly);
    return Py_BuildValue("K", (uint64_t) poly);
}

PyObject* freeCOrbit(PyObject *self, PyObject* args)
{
    cOrbit* orb;
    uint64_t cptr;

    if(!PyArg_ParseTuple(args, "K",&cptr))
    {
        return NULL;
    }

    orb = (cOrbit*) cptr;

    deleteOrbit(orb);

    return Py_BuildValue("i", 0);
}

PyObject* freeCPoly2D(PyObject *self, PyObject* args)
{
    cPoly2d* poly;
    uint64_t cptr;

    if(!PyArg_ParseTuple(args, "K", &cptr))
    {
        return NULL;
    }

    poly = (cPoly2d*) cptr;
    deletePoly2d(poly);

    return Py_BuildValue("i", 0);
}

PyObject* freeCPoly1D(PyObject *self, PyObject* args)
{
    cPoly1d* poly;
    uint64_t cptr;

    if(!PyArg_ParseTuple(args, "K", &cptr))
    {
        return NULL;
    }

    poly = (cPoly1d*) cptr;
    deletePoly1d(poly);

    return Py_BuildValue("i", 0);
}

PyObject* freeCPeg(PyObject* self, PyObject* args)
{
    cPeg* peg;
    uint64_t cptr;

    if(!PyArg_ParseTuple(args, "K", &cptr))
    {
        return NULL;
    }

    peg = (cPeg*) cptr;
    delete peg;

    return Py_BuildValue("i", 0);
}

PyObject* freeCEllipsoid(PyObject* self, PyObject* args)
{
    cEllipsoid* elp;
    uint64_t cptr;

    if(!PyArg_ParseTuple(args, "K", &cptr))
    {
        return NULL;
    }

    elp = (cEllipsoid*) cptr;
    delete elp;

    return Py_BuildValue("i", 0);
}

PyObject* importPegFromC(PyObject *self, PyObject* args)
{
    cPeg* peg;
    uint64_t cptr;

    if(!PyArg_ParseTuple(args, "K", &cptr))
    {
        return NULL;
    }

    peg = (cPeg*) cptr;

    return Py_BuildValue("ddd", peg->lat, peg->lon, peg->hdg);
}

PyObject* importPoly1DFromC(PyObject *self, PyObject *args)
{
    uint64_t cptr;
    cPoly1d* poly;

    if(!PyArg_ParseTuple(args, "K", &cptr))
    {
        return NULL;
    }
    poly = (cPoly1d*) cptr;


    PyObject *list = PyList_New(poly->order + 1);
    for(int i=0; i< (poly->order+1); i++)
    {
        PyObject* listEl = PyFloat_FromDouble(poly->coeffs[i]);
        if (listEl == NULL)
        {
            cout << "Error in converting polynomial coefficient to list element at " << __FILE__ << endl;
            exit(1);
        }
        PyList_SetItem(list, i, listEl);
    }


    return Py_BuildValue("iddN",poly->order, poly->mean, poly->norm, list);
}

PyObject* importPoly2DFromC(PyObject *self, PyObject *args)
{
    uint64_t cptr;
    cPoly2d* poly;

    if(!PyArg_ParseTuple(args, "K", &cptr))
    {
        return NULL;
    }
    poly = (cPoly2d*) cptr;

    PyObject* list2d = PyList_New((poly->azimuthOrder+1)*(poly->rangeOrder+1));
    for(int i=0; i<=poly->azimuthOrder; i++)
    {
        for(int j=0; j<= poly->rangeOrder; j++)
        {
            PyObject* listEl = PyFloat_FromDouble(getCoeff2d(poly,i,j));
            if(listEl == NULL)
            {
                cout << "Error in converting polynomial2d coefficient to list element at " << __FILE__ << endl;
                exit(1);
            }
            PyList_SetItem(list2d, i*(poly->rangeOrder+1)+j, listEl);
            if(PyErr_Occurred() != NULL)
            {
                cout << "Error in setting list value at " << __FILE__ << endl;
                exit(1);
            }
        }

    }

    printPoly2d(poly);

    return Py_BuildValue("(ii)(dd)(dd) N",poly->azimuthOrder,
            poly->rangeOrder, poly->meanAzimuth, poly->meanRange,
            poly->normAzimuth, poly->normRange, list2d);
}

PyObject* importOrbitFromC(PyObject *self, PyObject* args)
{
    uint64_t cptr;
    cOrbit* orb;
    double tim;
    double pos[3], vel[3];

    if(!PyArg_ParseTuple(args, "K", &cptr))
    {
        return NULL;
    }
    orb = (cOrbit*) cptr;

    PyObject* list2d = PyList_New(orb->nVectors);
    for(int i=0; i< (orb->nVectors); i++)
    {
        getStateVector(orb, i, &tim, pos, vel);

//        cout << "Index: " << i << " out of " << orb->nVectors << "\n";
//        cout << "Time: " << tim << "\n";
//        cout << "Position: "<< pos[0] << " " << pos[1] << " " << pos[2] << "\n";
//        cout << "Velocity: "<< vel[0] << " " << vel[1] << " " << vel[2] << "\n";

        PyObject* list= PyList_New(7);
        if(PyErr_Occurred() != NULL)
        {
            PyErr_Print();
            cout << "Could not create a list out of a single state vector"<<endl;
            exit(1);
        }

        PyObject* obj = PyFloat_FromDouble(tim);
        if(PyErr_Occurred() != NULL)
        {
            PyErr_Print();
            cout << "Error in converting time to list element at " << __FILE__ << endl;
            exit(1);
        }

        PyList_SetItem(list, 0, obj);
        if(PyErr_Occurred() != NULL)
        {
            PyErr_Print();
            cout << "Error in converting time to list element at " << __FILE__ << endl;
            exit(1);
        }


        for(int j=0; j<3; j++)
        {
            PyObject *num = PyFloat_FromDouble(pos[j]);
            if(PyErr_Occurred() != NULL)
            {
                PyErr_Print();
                cout << "Error in converting positions to list elements at " << __FILE__ << endl;
                exit(1);
            }
            PyList_SetItem(list, j+1, num);
        }

        for(int j=0; j<3; j++)
        {
            PyObject *num = PyFloat_FromDouble(vel[j]);
            if(PyErr_Occurred() != NULL)
            {
                PyErr_Print();
                cout << "Error in converting velocities to list elements at " << __FILE__ << endl;
                exit(1);
            }
            PyList_SetItem(list, j+4, num);
        }

        PyList_SetItem(list2d, i, list);
    }

    return Py_BuildValue("iN", orb->basis, list2d);
}
