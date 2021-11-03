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

#ifndef GEOGRIDOPTICAL_H
#define GEOGRIDOPTICAL_H

#include <iostream>



struct geoGridOptical
{
    //DEM related inputs
    std::string demname;    //DEM
    std::string dhdxname;   //Slope in X
    std::string dhdyname;   //Slope in Y
    std::string vxname;     //Velocity in X
    std::string vyname;     //Velocity in Y
    std::string srxname;     //Search range in X
    std::string sryname;     //Search range in Y
    std::string csminxname;     //Chip size minimum in x
    std::string csminyname;     //Chip size minimum in y
    std::string csmaxxname;     //Chip size maximum in x
    std::string csmaxyname;     //Chip size maximum in y
    std::string ssmname;     //Stable surface mask
    int epsgDem, epsgDat;
    double chipSizeX0;
    double gridSpacingX;

    //Bounding box related
    double xmin, xmax;
    double ymin, ymax;

    //Radar image related inputs
    double startingX, startingY;
    double XSize, YSize;
    int nLines, nPixels;
    double dt;
    int nodata_out;
    int pOff, lOff, pCount, lCount;
    double X_res, Y_res;
    
    //dt-varying search range rountine parameters
    double dt_unity;
    double max_factor;
    double upper_thld, lower_thld;

    //Output file names
    std::string pixlinename;
    std::string offsetname;
    std::string searchrangename;
    std::string chipsizeminname;
    std::string chipsizemaxname;
    std::string stablesurfacemaskname;
    std::string ro2vx_name;
    std::string ro2vy_name;

    //Functions
    void computeBbox(double *);
    void geogridOptical();
    void cross_C(double r_u[3], double r_v[3], double r_w[3]);
    double dot_C(double r_v[3], double r_w[3]);
    double norm_C(double a[3]);
    void unitvec_C(double v[3], double u[3]);
};


#endif
