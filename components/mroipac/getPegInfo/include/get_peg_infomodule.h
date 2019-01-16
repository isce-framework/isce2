//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                                  Giangi Sacco
//                        NASA Jet Propulsion Laboratory
//                      California Institute of Technology
//                        (C) 2009-2010  All Rights Reserved
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#ifndef get_peg_infomodule_h
#define get_peg_infomodule_h

#include <Python.h>
#include <stdint.h>
#include "get_peg_infomoduleFortTrans.h"

extern "C"
{
        void get_peg_info_f();
        PyObject * get_peg_info_C(PyObject *, PyObject *);
        void setNumObservations_f(int *);
        PyObject * setNumObservations_C(PyObject *, PyObject *);
        void setStartLineSlc_f(int *);
        PyObject * setStartLineSlc_C(PyObject *, PyObject *);
        void setNumLinesInt_f(int *);
        PyObject * setNumLinesInt_C(PyObject *, PyObject *);
        void setNumLinesSlc_f(int *);
        PyObject * setNumLinesSlc_C(PyObject *, PyObject *);
        void setNumAzimuthLooksInt_f(int *);
        PyObject * setNumAzimuthLooksInt_C(PyObject *, PyObject *);
        void setPrfSlc_f(double *);
        PyObject * setPrfSlc_C(PyObject *, PyObject *);
        void setTimeSlc_f(double *);
        PyObject * setTimeSlc_C(PyObject *, PyObject *);
        void setTime_f(double *, int *);
        void allocate_r_time_f(int *);
        void deallocate_r_time_f();
        PyObject * allocate_r_time_C(PyObject *, PyObject *);
        PyObject * deallocate_r_time_C(PyObject *, PyObject *);
        PyObject * setTime_C(PyObject *, PyObject *);
        void setPositionVector_f(double *, int *, int *);
        void allocate_r_xyz1_f(int *,int *);
        void deallocate_r_xyz1_f();
        PyObject * allocate_r_xyz1_C(PyObject *, PyObject *);
        PyObject * deallocate_r_xyz1_C(PyObject *, PyObject *);
        PyObject * setPositionVector_C(PyObject *, PyObject *);
        void setVelocityVector_f(double *, int *, int *);
        void allocate_r_vxyz1_f(int *,int *);
        void deallocate_r_vxyz1_f();
        PyObject * allocate_r_vxyz1_C(PyObject *, PyObject *);
        PyObject * deallocate_r_vxyz1_C(PyObject *, PyObject *);
        PyObject * setVelocityVector_C(PyObject *, PyObject *);
        void setAccelerationVector_f(double *, int *, int *);
        void allocate_r_axyz1_f(int *,int *);
        void deallocate_r_axyz1_f();
        PyObject * allocate_r_axyz1_C(PyObject *, PyObject *);
        PyObject * deallocate_r_axyz1_C(PyObject *, PyObject *);
        PyObject * setAccelerationVector_C(PyObject *, PyObject *);
        void setPlanetGM_f(double *);
        PyObject * setPlanetGM_C(PyObject *, PyObject *);
        void setPlanetSpinRate_f(double *);
        PyObject * setPlanetSpinRate_C(PyObject *, PyObject *);
        void getPegLat_f(double *);
        PyObject * getPegLat_C(PyObject *, PyObject *);
        void getPegLon_f(double *);
        PyObject * getPegLon_C(PyObject *, PyObject *);
        void getPegHeight_f(double *);
        PyObject * getPegHeight_C(PyObject *, PyObject *);
        void getPegHeading_f(double *);
        PyObject * getPegHeading_C(PyObject *, PyObject *);
        void getVerticalFit_f(double *, int *);
        void allocate_r_af_f(int *);
        void deallocate_r_af_f();
        PyObject * allocate_r_af_C(PyObject *, PyObject *);
        PyObject * deallocate_r_af_C(PyObject *, PyObject *);
        PyObject * getVerticalFit_C(PyObject *, PyObject *);
        void getHorizontalFit_f(double *, int *);
        void allocate_r_cf_f(int *);
        void deallocate_r_cf_f();
        PyObject * allocate_r_cf_C(PyObject *, PyObject *);
        PyObject * deallocate_r_cf_C(PyObject *, PyObject *);
        PyObject * getHorizontalFit_C(PyObject *, PyObject *);
        void getVerticalVelocityFit_f(double *, int *);
        void allocate_r_afdot_f(int *);
        void deallocate_r_afdot_f();
        PyObject * allocate_r_afdot_C(PyObject *, PyObject *);
        PyObject * deallocate_r_afdot_C(PyObject *, PyObject *);
        PyObject * getVerticalVelocityFit_C(PyObject *, PyObject *);
        void getCrossTrackVelocityFit_f(double *, int *);
        void allocate_r_cfdot_f(int *);
        void deallocate_r_cfdot_f();
        PyObject * allocate_r_cfdot_C(PyObject *, PyObject *);
        PyObject * deallocate_r_cfdot_C(PyObject *, PyObject *);
        PyObject * getCrossTrackVelocityFit_C(PyObject *, PyObject *);
        void getAlongTrackVelocityFit_f(double *, int *);
        void allocate_r_sfdot_f(int *);
        void deallocate_r_sfdot_f();
        PyObject * allocate_r_sfdot_C(PyObject *, PyObject *);
        PyObject * deallocate_r_sfdot_C(PyObject *, PyObject *);
        PyObject * getAlongTrackVelocityFit_C(PyObject *, PyObject *);
        void getPegRadius_f(double *);
        PyObject * getPegRadius_C(PyObject *, PyObject *);
        void getGroundSpacing_f(double *);
        PyObject * getGroundSpacing_C(PyObject *, PyObject *);
        void getTranslationVector_f(double *, int *);
        void allocate_r_transVect_f(int *);
        void deallocate_r_transVect_f();
        PyObject * allocate_r_transVect_C(PyObject *, PyObject *);
        PyObject * deallocate_r_transVect_C(PyObject *, PyObject *);
        PyObject * getTranslationVector_C(PyObject *, PyObject *);
        void getTransformationMatrix_f(double *, int *, int *);
        void allocate_r_transfMat_f(int *,int *);
        void deallocate_r_transfMat_f();
        PyObject * allocate_r_transfMat_C(PyObject *, PyObject *);
        PyObject * deallocate_r_transfMat_C(PyObject *, PyObject *);
        PyObject * getTransformationMatrix_C(PyObject *, PyObject *);
        void getIntPosition_f(double *, int *, int *);
        void allocate_r_intPos_f(int *,int *);
        void deallocate_r_intPos_f();
        PyObject * allocate_r_intPos_C(PyObject *, PyObject *);
        PyObject * deallocate_r_intPos_C(PyObject *, PyObject *);
        PyObject * getIntPosition_C(PyObject *, PyObject *);
        void getIntVelocity_f(double *, int *, int *);
        void allocate_r_intVel_f(int *,int *);
        void deallocate_r_intVel_f();
        PyObject * allocate_r_intVel_C(PyObject *, PyObject *);
        PyObject * deallocate_r_intVel_C(PyObject *, PyObject *);
        PyObject * getIntVelocity_C(PyObject *, PyObject *);
        void getPegVelocity_f(double *, int *);
        void allocate_r_vxyzpeg_f(int *);
        void deallocate_r_vxyzpeg_f();
        PyObject * allocate_r_vxyzpeg_C(PyObject *, PyObject *);
        PyObject * deallocate_r_vxyzpeg_C(PyObject *, PyObject *);
        PyObject * getPegVelocity_C(PyObject *, PyObject *);
        void getPlatformSCHVelocity_f(double *, int *);
        void allocate_r_platvel_f(int *);
        void deallocate_r_platvel_f();
        PyObject * allocate_r_platvel_C(PyObject *, PyObject *);
        PyObject * deallocate_r_platvel_C(PyObject *, PyObject *);
        PyObject * getPlatformSCHVelocity_C(PyObject *, PyObject *);
        void getPlatformSCHAcceleration_f(double *, int *);
        void allocate_r_platacc_f(int *);
        void deallocate_r_platacc_f();
        PyObject * allocate_r_platacc_C(PyObject *, PyObject *);
        PyObject * deallocate_r_platacc_C(PyObject *, PyObject *);
        PyObject * getPlatformSCHAcceleration_C(PyObject *, PyObject *);
        void getTimeFirstScene_f(double *);
        PyObject * getTimeFirstScene_C(PyObject *, PyObject *);

}

static PyMethodDef get_peg_info_methods[] =
{
        {"get_peg_info_Py", get_peg_info_C, METH_VARARGS, " "},
        {"setNumObservations_Py", setNumObservations_C, METH_VARARGS, " "},
        {"setStartLineSlc_Py", setStartLineSlc_C, METH_VARARGS, " "},
        {"setNumLinesInt_Py", setNumLinesInt_C, METH_VARARGS, " "},
        {"setNumLinesSlc_Py", setNumLinesSlc_C, METH_VARARGS, " "},
        {"setNumAzimuthLooksInt_Py", setNumAzimuthLooksInt_C, METH_VARARGS, " "},
        {"setPrfSlc_Py", setPrfSlc_C, METH_VARARGS, " "},
        {"setTimeSlc_Py", setTimeSlc_C, METH_VARARGS, " "},
        {"allocate_r_time_Py", allocate_r_time_C, METH_VARARGS, " "},
        {"deallocate_r_time_Py", deallocate_r_time_C, METH_VARARGS, " "},
        {"setTime_Py", setTime_C, METH_VARARGS, " "},
        {"allocate_r_xyz1_Py", allocate_r_xyz1_C, METH_VARARGS, " "},
        {"deallocate_r_xyz1_Py", deallocate_r_xyz1_C, METH_VARARGS, " "},
        {"setPositionVector_Py", setPositionVector_C, METH_VARARGS, " "},
        {"allocate_r_vxyz1_Py", allocate_r_vxyz1_C, METH_VARARGS, " "},
        {"deallocate_r_vxyz1_Py", deallocate_r_vxyz1_C, METH_VARARGS, " "},
        {"setVelocityVector_Py", setVelocityVector_C, METH_VARARGS, " "},
        {"allocate_r_axyz1_Py", allocate_r_axyz1_C, METH_VARARGS, " "},
        {"deallocate_r_axyz1_Py", deallocate_r_axyz1_C, METH_VARARGS, " "},
        {"setAccelerationVector_Py", setAccelerationVector_C, METH_VARARGS, " "},
        {"setPlanetGM_Py", setPlanetGM_C, METH_VARARGS, " "},
        {"setPlanetSpinRate_Py", setPlanetSpinRate_C, METH_VARARGS, " "},
        {"getPegLat_Py", getPegLat_C, METH_VARARGS, " "},
        {"getPegLon_Py", getPegLon_C, METH_VARARGS, " "},
        {"getPegHeight_Py", getPegHeight_C, METH_VARARGS, " "},
        {"getPegHeading_Py", getPegHeading_C, METH_VARARGS, " "},
        {"allocate_r_af_Py", allocate_r_af_C, METH_VARARGS, " "},
        {"deallocate_r_af_Py", deallocate_r_af_C, METH_VARARGS, " "},
        {"getVerticalFit_Py", getVerticalFit_C, METH_VARARGS, " "},
        {"allocate_r_cf_Py", allocate_r_cf_C, METH_VARARGS, " "},
        {"deallocate_r_cf_Py", deallocate_r_cf_C, METH_VARARGS, " "},
        {"getHorizontalFit_Py", getHorizontalFit_C, METH_VARARGS, " "},
        {"allocate_r_afdot_Py", allocate_r_afdot_C, METH_VARARGS, " "},
        {"deallocate_r_afdot_Py", deallocate_r_afdot_C, METH_VARARGS, " "},
        {"getVerticalVelocityFit_Py", getVerticalVelocityFit_C, METH_VARARGS, " "},
        {"allocate_r_cfdot_Py", allocate_r_cfdot_C, METH_VARARGS, " "},
        {"deallocate_r_cfdot_Py", deallocate_r_cfdot_C, METH_VARARGS, " "},
        {"getCrossTrackVelocityFit_Py", getCrossTrackVelocityFit_C, METH_VARARGS, " "},
        {"allocate_r_sfdot_Py", allocate_r_sfdot_C, METH_VARARGS, " "},
        {"deallocate_r_sfdot_Py", deallocate_r_sfdot_C, METH_VARARGS, " "},
        {"getAlongTrackVelocityFit_Py", getAlongTrackVelocityFit_C, METH_VARARGS, " "},
        {"getPegRadius_Py", getPegRadius_C, METH_VARARGS, " "},
        {"getGroundSpacing_Py", getGroundSpacing_C, METH_VARARGS, " "},
        {"allocate_r_transVect_Py", allocate_r_transVect_C, METH_VARARGS, " "},
        {"deallocate_r_transVect_Py", deallocate_r_transVect_C, METH_VARARGS, " "},
        {"getTranslationVector_Py", getTranslationVector_C, METH_VARARGS, " "},
        {"allocate_r_transfMat_Py", allocate_r_transfMat_C, METH_VARARGS, " "},
        {"deallocate_r_transfMat_Py", deallocate_r_transfMat_C, METH_VARARGS, " "},
        {"getTransformationMatrix_Py", getTransformationMatrix_C, METH_VARARGS, " "},
        {"allocate_r_intPos_Py", allocate_r_intPos_C, METH_VARARGS, " "},
        {"deallocate_r_intPos_Py", deallocate_r_intPos_C, METH_VARARGS, " "},
        {"getIntPosition_Py", getIntPosition_C, METH_VARARGS, " "},
        {"allocate_r_intVel_Py", allocate_r_intVel_C, METH_VARARGS, " "},
        {"deallocate_r_intVel_Py", deallocate_r_intVel_C, METH_VARARGS, " "},
        {"getIntVelocity_Py", getIntVelocity_C, METH_VARARGS, " "},
        {"allocate_r_vxyzpeg_Py", allocate_r_vxyzpeg_C, METH_VARARGS, " "},
        {"deallocate_r_vxyzpeg_Py", deallocate_r_vxyzpeg_C, METH_VARARGS, " "},
        {"getPegVelocity_Py", getPegVelocity_C, METH_VARARGS, " "},
        {"allocate_r_platvel_Py", allocate_r_platvel_C, METH_VARARGS, " "},
        {"deallocate_r_platvel_Py", deallocate_r_platvel_C, METH_VARARGS, " "},
        {"getPlatformSCHVelocity_Py", getPlatformSCHVelocity_C, METH_VARARGS, " "},
        {"allocate_r_platacc_Py", allocate_r_platacc_C, METH_VARARGS, " "},
        {"deallocate_r_platacc_Py", deallocate_r_platacc_C, METH_VARARGS, " "},
        {"getPlatformSCHAcceleration_Py", getPlatformSCHAcceleration_C, METH_VARARGS, " "},
        {"getTimeFirstScene_Py", getTimeFirstScene_C, METH_VARARGS, " "},
        {NULL, NULL, 0, NULL}
};
#endif //get_peg_infomodule_h
