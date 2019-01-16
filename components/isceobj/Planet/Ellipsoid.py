#!/usr/bin/env python3 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2010 California Institute of Technology. ALL RIGHTS RESERVED.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# United States Government Sponsorship acknowledged. This software is subject to
# U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
# (No [Export] License Required except when exporting to an embargoed country,
# end user, or in support of a prohibited end use). By downloading this software,
# the user agrees to comply with all applicable U.S. export laws and regulations.
# The user has the responsibility to obtain export licenses, or other export
# authority as may be required before exporting this software to any 'EAR99'
# embargoed foreign country or citizen of those countries.
#
# Author: Eric Gurrola
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




from __future__ import print_function
import math
import numbers
from iscesys.Component.Component import Component
from isceobj.Util.geo import ellipsoid
## 
##  NB: The Heritage class is the oiginal ellipsoid's interface.
##      The geo/ellipsoid.Ellipsoid is mixed with to make
##      a class that is both new and backwards compatible.
##   A class to represent an ellipsoid.
##   The parameters maintained internally are the following:
##
##   a  = semi major axis - the greatest distance from center to ellipse
##   e2 = eccentricity-squared - the square of the ratio of the focal distance
##   (from center) and a
##
##   Other parameters can be used in describing an ellipsoid.
##   The other parameters that this class will compute are the following:
##
##   b  = semi minor axis = a * sqrt(1 - e**2) - the smallest distance from
##                                                 center to ellipse
##   e  = eccentricity = sqrt(e2)
##   c  = focal distance from center = a * e
##   f  = flattening = 1 - sqrt(1 - e**2) = (b-a)/a
##
##   Any of these four auxiliary ellipse parameters can be used to set the
##   eccentricity-squared. If e or f are equated to a value then e2 is set
##   accordingly.  If b or c are equated to a value, then e2 will be set using
##    the current value of a.  So, the correct value of a must be set prior
##   to setting b or c.
##
##   When you create an object of class Ellipsoid you have the opportunity to
##   set the semi major axis and eccentricity-squared.  If you don't give these
##   arguments in the constructor, then your object is initialized as a unit
##   sphere.
class Heritage(object):

    ## I made this a class variable because it is constant
    '''
    dictionaryOfVariables = {'SEMIMAJOR_AXIS': ['a',float,'mandatory'],
                             'ECCENTRICITY_SQUARED': ['e2', float,'mandatory'],
                             'MODEL':['model', str,'optional']
                             }
    '''
    def __init__(self):
        self.descriptionOfVariables = {}
        self.pegLat = 0.0
        self.pegLon = 0.0
        self.pegHdg = 0.0
        return None

    def __str__(self):
        retstr = "Semimajor axis: %s\n"
        retlst = (self.a,)
        retstr += "Eccentricity squared: %s\n"
        retlst += (self.e2,)
        return retstr % retlst

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model

    def get_a(self):
        return self.a

    def set_a(self,a):
        if a >= 0.0:
            self.a = a
        else:
            message = (
                "attempt to set ellipsoid semi major axis to negative value %f"+
                " is invalid"
                )
            raise ValueError(message % a)
        return None

    def get_e2(self):
        return self.e2

    def set_e2(self,e2):
        self.e2 = e2
        
    def get_e(self):
        return self.e2**0.5

    def set_e(self,e):
        self.sete2(e**2)

    def get_f(self):
        return 1.0 - (1.0-self.e2)**0.5

    def set_f(self,f):
        self.e2 = 1.0 - (1.0-f)**2

    def get_b(self):
        return self.a * (1.0-self.e2)**0.5

    def set_b(self,b):
        self.e2 = 1.0 - (b/self.a)**2

    def get_c(self):
        return self.c

    def set_c(self,c):
        self.c = c

#    model = property(get_model,set_model,doc="ellipse model,for instance WGS-84")
#    a  = property(get_a, set_a, doc="ellipse semi major axis")
#    e2 = property(get_e2, set_e2, doc="ellipse eccentricity-squared")
#    e  = property(get_e, set_e, doc="ellipse eccentricity")
#    f  = property(get_f, set_f, doc="ellipse flattening = 1 - sqrt(1-e2)")
#    b  = property(get_b, set_b, doc="ellipse semi minor axis --- should only be set after setting a")
#    c  = property(get_c, set_c, doc="ellipse distance from center to focus --- should only be set after setting a")

    ##
    #   Convert position relative to the ellipsoid from (x,y,z) to (lat,lon,height).
    #
    #    (x,y,z) is the cartesian coordinate of the point with origin at the center of the ellipsoid
    #    +x axis is defined by latitude = 0 degrees, longitude = 0 degrees
    #    +y axis is defined by latitude = 0 degrees, longitude = 90 degrees
    #    +z axis is defined by latitude = 90 degrees --- polar axis
    #
    #   (lat,lon,height) are defined in terms of a meridional plane containing the polar axis and the point,
    #    in which there an ellipse cross-section of the ellipsoid, and a normal line in that plane passing 
    #   through the point and is normal to the ellipsoid.
    #   lat is the geodetic latitude defined as the angle in this meridional plane between the normal line and the equatorial plane.
    #   lon is the longitude defined as the angle in the equatorial plane from the x axis to the meridional plane
    #   height is the signed distance from the ellipsoid surface to the point along the normal line            
    def xyz_to_llh_old(self,xyz):
        """xyz_to_llh(xyz): returns llh=(lat (deg), lon (deg), h (meters)) for the instance ellipsoid \ngiven the coordinates of a point at xyz=(z,y,z) (meters)\n"""

        r_llh = [None]*3   
        d_llh = [None]*3
           
        r_q2 = 1.0/(1.0 - self.e2)
        r_q = math.sqrt(r_q2)
        r_q3 = r_q2 - 1.0
        r_b = self.a*math.sqrt(1.0 - self.e2)

        r_llh[1] = math.atan2(xyz[1],xyz[0])

        r_p = math.sqrt(xyz[0]**2 + xyz[1]**2)
        r_tant = (xyz[2]/r_p)*r_q
        r_theta = math.atan(r_tant)
#        r_tant = math.atan2(r_q*xyz[2],r_p)
        r_tant = (xyz[2] + r_q3*r_b*math.sin(r_theta)**3)/(r_p - self.e2*self.a*math.cos(r_theta)**3)
        r_llh[0] =  math.atan(r_tant)
#        r_llh[0] = math.atan2((xyz[2] + r_q3*r_b*math.sin(r_theta)**3),(r_p - self.e2*self.a*math.cos(r_theta)**3))
        r_re = self.a/math.sqrt(1.0 - self.e2*math.sin(r_llh[0])**2)
        r_llh[2] = r_p/math.cos(r_llh[0]) - r_re
        
        d_llh[0] = math.degrees(r_llh[0])
        d_llh[1] = math.degrees(r_llh[1])
        d_llh[2] = r_llh[2] 
        return d_llh


    def xyz_to_llh(self,xyz):
        """xyz_to_llh(xyz): returns llh=(lat (deg), lon (deg), h (meters)) for the instance ellipsoid \n
        given the coordinates of a point at xyz=(z,y,z) (meters). \n
        Based on closed form solution of H. Vermeille, Journal of Geodesy (2002) 76:451-454. \n
        Handles simple list or tuples (xyz represents a single point) or a list of lists or tuples (xyz represents several points)"""

        a2 = self.a**2
        e4 = self.e2**2
        # just to cast back to single list once done
        onePosition = False
        if isinstance(xyz[0],numbers.Real):
            xyz = [xyz]
            onePosition = True
        
        r_llh = [0]*3
        d_llh = [[0]*3 for i in range(len(xyz))]
        for i in range(len(xyz)):
            xy2 = xyz[i][0]**2+xyz[i][1]**2
            p = (xy2)/a2
            q = (1.-self.e2)*xyz[i][2]**2/a2
            r = (p+q-e4)/6.
            s = e4*p*q/(4.*r**3)
            t = (1.+s+math.sqrt(s*(2.+s)))**(1./3.)
            u = r*(1.+t+1./t)
            v = math.sqrt(u**2+e4*q)
            w = self.e2*(u+v-q)/(2.*v)
            k = math.sqrt(u+v+w**2)-w
            D = k*math.sqrt(xy2)/(k+self.e2)

            
            r_llh[0] = math.atan2(xyz[i][2],D)
            r_llh[1] = math.atan2(xyz[i][1],xyz[i][0])
            r_llh[2] = (k+self.e2-1.)*math.sqrt(D**2+xyz[i][2]**2)/k
            
            d_llh[i][0] = math.degrees(r_llh[0])
            d_llh[i][1] = math.degrees(r_llh[1])
            d_llh[i][2] = r_llh[2] 
        if onePosition:
            return d_llh[0]
        else:
            return d_llh

    ##
    #    Convert position relative to the ellipsoid from (lat,lon,height) to (x,y,z).
    #
    #    (x,y,z) is the cartesian coordinate of the point with origin at the center of the ellipsoid
    #    +x axis is defined by latitude = 0 degrees, longitude = 0 degrees
    #    +y axis is defined by latitude = 0 degrees, longitude = 90 degrees
    #    +z axis is defined by latitude = 90 degrees --- polar axis
    #
    #    (lat,lon,height) are defined in terms of a meridional plane containing the polar axis and the point,
    #     in which there an ellipse cross-section of the ellipsoid, and a normal line in that plane passing 
    #    through the point and is normal to the ellipsoid.
    #    lat is the geodetic latitude defined as the angle in this meridional plane between the normal line and the equatorial plane.
    #    lon is the longitude defined as the angle in the equatorial plane from the x axis to the meridional plane
    #    height is the signed distance from the ellipsoid surface to the point along the normal line
    def llh_to_xyz(self,llh):
        
        """llh_to_xyz(llh): returns (z,y,z) (meters) coordinates of a point given the point at \nllh=(lat (deg), lon (deg), h (meters)) for the instance ellipsoid\n
        Handles simple list or tuples (xyz represents a single point) or a list of lists or tuples (xyz represents several points)
        """

        # just to cast back to single list once done
        onePosition = False
        if isinstance(llh[0],numbers.Real):
            llh = [llh]
            onePosition = True
        
        r_v = [[0]*3 for i in range(len(llh))]
        

        for i in range(len(llh)):
            r_lat = math.radians(llh[i][0])
            r_lon = math.radians(llh[i][1])
            hgt = llh[i][2]
            
            r_re = self.a/math.sqrt(1.0 - self.e2*math.sin(r_lat)**2)

            r_v[i][0] = (r_re + hgt)*math.cos(r_lat)*math.cos(r_lon)
            r_v[i][1] = (r_re + hgt)*math.cos(r_lat)*math.sin(r_lon)
            r_v[i][2] = (r_re*(1.0-self.e2) + hgt)*math.sin(r_lat)
        if onePosition:
            return r_v[0]
        else:
            return r_v

    ##
    # Compute the distance along a geodesic on the ellipsoid between the projection of two points onto the surface of the ellipsoid
    #These results are based on the memo
    #
    #"Summary of Mocomp Reference Line Determination Study" , IOM 3346-93-163
    #
    #and the paper
    #
    #"A Rigourous Non-iterative Procedure for Rapid Inverse Solution of Very
    #Long Geodesics" by E. M. Sadano, Bulletine Geodesique 1958
    def geo_dis(self,llh1,llh2):
        """geo_dis(llh1,llh2): returns geodesic distance (meters) for the instance ellipsoid \ngiven a starting position on the ellipsoid (at height zero) below the point \nllh1=(lat (deg), lon (deg), h (meters)) and ending position on the ellipsoid \nbelow the point llh2=(lat (deg), lon (deg), h (meters)). \n"""

        dis = 0.0
        if(self.e2 == 0):
            dis,hdg = self._sphericalDistance(llh1,llh2) #2013-06-03 Kosal: added _
        else:
            dis,hdg = self._ellipsoidalDistance(llh1,llh2) #2013-06-03 Kosal: added _
        return dis
    
    ##
    # Compute the heading (the angle from north) one would travel in going from one point on an ellipsoid to another
    def geo_hdg(self,llh1,llh2):                        
        """geo_hdg(llh1,llh2): returns the heading angle (degrees) for a geodesic on the instance ellipsoid \ngiven a starting position on the ellipsoid (at height zero) below the point llh1=(lat (deg), lon (deg), h (meters)) \nand ending position on the ellipsoid below llh2=(lat (deg), lon (deg), h (meters)). \n"""

        hdg = 0.0
        if(self.e2 == 0):
            dis,hdg = self._sphericalDistance(llh1,llh2) #2013-06-03 Kosal: added _
        else:
            dis,hdg = self._ellipsoidalDistance(llh1,llh2) #2013-06-03 Kosal: added _
        return hdg
            
    def _sphericalDistance(self,llh1,llh2):
        r_sinlati = math.sin(math.radians(llh1[0]))
        r_coslati = math.cos(math.radians(llh1[0]))
        r_sinlatf = math.sin(math.radians(llh2[0]))
        r_coslatf = math.cos(math.radians(llh2[0]))
        r_tanlatf = math.tan(math.radians(llh2[0]))
    
        r_t1 =  math.radians(llh2[1]) - math.radians(llh1[1])
        if (math.fabs(r_t1) > math.pi):
            r_t1 = (2.0*math.pi - math.fabs(r_t1))*math.copysign(1.0,-r_t1)
               
        r_sinlon = math.sin(r_t1)
        r_coslon = math.cos(r_t1)
        r_t2 = r_coslati*r_coslatf*r_coslon + r_sinlati*r_sinlatf
        r_t3 = r_coslati*r_tanlatf - r_sinlati*r_coslon
                
        r_geodis = self.a*math.acos(r_t2)
        r_geohdg = math.atan2(r_sinlon,r_t3)
        
        return r_geodis,r_geohdg
    
    def _ellipsoidalDistance(self,llh1,llh2):
        r_geodis = 0.0
        r_geohdg = None
        
        r_e = self.get_e()
        r_f = self.get_f()
                                      
        r_ep = r_e*r_f/(self.e2-r_f)
        r_n = r_f/self.e2
        
        r_sqrtome2 = math.sqrt(1.0 - self.e2)
        r_b0 = self.a*r_sqrtome2
        r_k1 = (16.0*self.e2*r_n**2 + r_ep**2)/r_ep**2
        r_k2 = (16.0*self.e2*r_n**2)/(16.0*self.e2*r_n**2 + r_ep**2)
        r_k3 = (16.0*self.e2*r_n**2)/r_ep**2
        r_k4 = (16.0*r_n - r_ep**2)/(16.0*self.e2*r_n**2 + r_ep**2)
        r_k5 = 16.0/(self.e2*(16.0*self.e2*r_n**2 + r_ep**2))
                
        r_tanlati = math.tan(math.radians(llh1[0]))
        r_tanlatf = math.tan(math.radians(llh2[0]))
        r_l  =  math.fabs(math.radians(llh2[1])-math.radians(llh1[1]))
        r_lsign = math.radians(llh2[1]) - math.radians(llh1[1])
        r_sinlon = math.sin(r_l)
        r_coslon = math.cos(r_l)
    
        r_tanbetai = r_sqrtome2*r_tanlati
        r_tanbetaf = r_sqrtome2*r_tanlatf
    
        r_cosbetai = 1.0/math.sqrt(1.0 + r_tanbetai**2)
        r_cosbetaf = 1.0/math.sqrt(1.0 + r_tanbetaf**2)
        r_sinbetai = r_tanbetai*r_cosbetai
        r_sinbetaf = r_tanbetaf*r_cosbetaf
    
        r_ac = r_sinbetai*r_sinbetaf
        r_bc = r_cosbetai*r_cosbetaf
    
        r_cosphi = r_ac + r_bc*r_coslon
        r_sinphi = math.copysign(1.0,r_sinlon)*math.sqrt(1.0 - min(r_cosphi**2,1.0))
    
        r_phi = math.fabs(math.atan2(r_sinphi,r_cosphi))
    
        if(self.a*math.fabs(r_phi) > 1e-6):    
            r_ca = (r_bc*r_sinlon)/r_sinphi
            r_cb = r_ca**2
            r_cc = (r_cosphi*(1.0 - r_cb))/r_k1
            r_cd = (-2.0*r_ac)/r_k1
            r_ce = -r_ac*r_k2
            r_cf = r_k3*r_cc
            r_cg = r_phi**2/r_sinphi
            
            r_x = ((r_phi*(r_k4 + r_cb) + r_sinphi*(r_cc + r_cd) + r_cg*(r_cf + r_ce))*r_ca)/r_k5
              
            r_lambda = r_l + r_x
            
            r_sinlam = math.sin(r_lambda)
            r_coslam = math.cos(r_lambda)
    
            r_cosph0 = r_ac + r_bc*r_coslam
            r_sinph0 = math.copysign(1.0,r_sinlam)*math.sqrt(1.0 - r_cosph0**2)
            r_phi0 = math.fabs(math.atan2(r_sinph0,r_cosph0))
            
            r_sin2phi = 2.0*r_sinph0*r_cosph0
    
            r_cosbeta0 = (r_bc*r_sinlam)/r_sinph0
            r_q = 1.0 - r_cosbeta0**2
            r_cos2sig = (2.0*r_ac - r_q*r_cosph0)/r_q
            r_cos4sig = 2.0*(r_cos2sig**2 - 0.5)
    
            r_ch = r_b0*(1.0 + (r_q*r_ep**2)/4.0 - (3.0*(r_q**2)*r_ep**4)/64.0)
            r_ci = r_b0*((r_q*r_ep**2)/4.0 - ((r_q**2)*r_ep**4)/16.0)
            r_cj = (r_q**2*r_b0*r_ep**4)/128.0
            
            r_t2 = (r_tanbetaf*r_cosbetai - r_coslam*r_sinbetai)
                                    
            r_sinlon = r_sinlam*math.copysign(1.0,r_lsign)
        
            r_cotalpha12 = (r_tanbetaf*r_cosbetai - r_coslam*r_sinbetai)/r_sinlam
            r_cotalpha21 = (r_sinbetaf*r_coslam - r_cosbetaf*r_tanbetai)/r_sinlam
        
            r_geodis = r_ch*r_phi0 + r_ci*r_sinph0*r_cos2sig - r_cj*r_sin2phi*r_cos4sig
            r_geohdg = math.atan2(r_sinlon,r_t2)    
        else:
            r_geodis = 0.0
            r_geohdg = None
            
        return r_geodis, r_geohdg
       
    ##
    # Compute the radius of curvature in the east direction on an ellipsoid
    def eastRadiusOfCurvature(self,llh):
        """eastRadiusOfCurvature(llh): returns Radius of Curvature (meters) \nin the East direction for the instance ellipsoid \ngiven a position llh=(lat (deg), lon (deg), h (meters))"""
        
        r_lat = math.radians(llh[0])

        reast = self.a/math.sqrt(1.0 - self.e2*math.sin(r_lat)**2)
        return reast

    ##
    # Compute the radius of curvature in the north direction on an ellipsoid
    def northRadiusOfCurvature(self,llh):
        """northRadiusOfCurvature(llh): returns Radius of Curvature (meters) \nin the North direction for the instance ellipsoid \ngiven a position llh=(lat (deg), lon (deg), h (meters))"""

        r_lat = math.radians(llh[0])

        rnorth = (self.a*(1.0 - self.e2))/(1.0 - self.e2*math.sin(r_lat)**2)**(1.5)
        return rnorth

    ##
    # Compute the radius of curvature on an ellipsoid
    def radiusOfCurvature(self,llh,hdg=0):
        """
        radiusOfCurvature(llh,[hdg]): returns Radius of Curvature (meters)
        in the direction specified by hdg for the instance ellipsoid
        given a position llh=(lat (deg), lon (deg), h (meters)).
        If no heading is given the default is 0, or North.
        """

        r_lat = math.radians(llh[0])
        r_hdg = math.radians(hdg)

        reast = self.eastRadiusOfCurvature(llh)
        rnorth = self.northRadiusOfCurvature(llh)

        #radius of curvature for point on ellipsoid
        rdir = (reast*rnorth)/(
            reast*math.cos(r_hdg)**2 + rnorth*math.sin(r_hdg)**2)

        #add height of the llh point
        return rdir + llh[2]

    ##
    # Compute the local, equivalent spherical radius
    def localRadius(self,llh):
        """
        localRadius(llh): returns the equivalent spherical radius (meters)
        for the instance ellipsoid given a position llh=(lat (deg), lon (deg),
        h (meters))
        """

        latg = math.atan(math.tan(math.radians(llh[0]))*self.a**2/self.get_b()**2)
        arg = math.cos(latg)**2/self.a**2 + math.sin(latg)**2/self.get_b()**2
        re = 1.0/math.sqrt(arg)

        return re

    def setSCH(self, pegLat, pegLon, pegHdg, pegHgt=0.0):
        """
        Set up an SCH coordinate system at the given peg point.
        Set a peg point on the ellipse at pegLat, pegLon, pegHdg (in degrees).
        Set the radius of curvature and the transformation matrix and offset
        vector needed to transform between (s,c,h) and ecef (x,y,z).
        """
        self.pegLat = pegLat
        self.pegLon = pegLon
        self.pegHdg = pegHdg
        self.pegHgt = pegHgt
        self.pegLLH = [pegLat, pegLon, pegHgt]

        #determine the radius of curvature at the peg point, i.e, the
        #the radius of the SCH sphere
        self.pegRadCur = self.radiusOfCurvature(self.pegLLH, pegHdg)

        #determine the rotation matrix (from radar_to_xyz.F)
        import numpy
        r_lat = numpy.radians(pegLat)
        r_lon = numpy.radians(pegLon)
        r_hdg = numpy.radians(pegHdg)
        r_clt = numpy.cos(r_lat)
        r_slt = numpy.sin(r_lat)
        r_clo = numpy.cos(r_lon)
        r_slo = numpy.sin(r_lon)
        r_chg = numpy.cos(r_hdg)
        r_shg = numpy.sin(r_hdg)

        ptm11 =  r_clt*r_clo
        ptm12 = -r_shg*r_slo - r_slt*r_clo*r_chg
        ptm13 =  r_slo*r_chg - r_slt*r_clo*r_shg
        ptm21 =  r_clt*r_slo
        ptm22 =  r_clo*r_shg - r_slt*r_slo*r_chg
        ptm23 = -r_clo*r_chg - r_slt*r_slo*r_shg
        ptm31 =  r_slt
        ptm32 =  r_clt*r_chg
        ptm33 =  r_clt*r_shg

        self.pegRotMatNP = numpy.matrix(
            [[ptm11, ptm12, ptm13],
             [ptm21, ptm22, ptm23],
             [ptm31, ptm32, ptm33]]
        )
        self.pegRotMatInvNP = self.pegRotMatNP.transpose()

        self.pegRotMat = self.pegRotMatNP.tolist()
        self.pegRotMatInv = self.pegRotMatInvNP.tolist()

        #find the translation vector as a column matrix
        self.pegXYZ = self.llh_to_xyz(self.pegLLH)
        self.pegXYZNP = numpy.matrix(self.pegXYZ).transpose()

        #Outward normal to ellispoid at the peg point
        self.pegNormal = [r_clt*r_clo, r_clt*r_slo, r_slt]
        self.pegNormalNP = numpy.matrix(self.pegNormal).transpose()

        #Offset Vector - to center of SCH sphere
        self.pegOVNP = self.pegXYZNP - self.pegRadCur*self.pegNormalNP
        self.pegOV = self.pegOVNP.transpose().tolist()[0]

        return

    def schbasis(self, posSCH):
        """
        xyzschMat = elp.schbasis(posSCH)
        Given an instance elp of an Ellipsoid with a peg point defined by a
        previous call to setSCH and an SCH position (as a list) return the
        transformation matrices from the XYZ frame to the SCH frame and the
        inverse from the SCH frame to the XYZ frame.  
        The returned object is a namedtuple with numpy matrices in elements
        named 'sch_to_xyz' and 'xyz_to_sch'
        sch_to_xyzMat = (elp.schbasis(posSCH)).sch_to_xyz
        xyz_to_schMat = (elp.schbasis(posSCH)).xyz_to_sch
        """

        import numpy
        r_coss = numpy.cos(posSCH[0]/self.pegRadCur)
        r_sins = numpy.sin(posSCH[0]/self.pegRadCur)
        r_cosc = numpy.cos(posSCH[1]/self.pegRadCur)
        r_sinc = numpy.sin(posSCH[1]/self.pegRadCur)

        r_matschxyzp = numpy.matrix([
            [-r_sins, -r_sinc*r_coss, r_coss*r_cosc],
            [ r_coss, -r_sinc*r_sins, r_sins*r_cosc],
            [ 0.0,     r_cosc,        r_sinc]])

        #compute sch to xyz matrix
        r_sch_to_xyzMat = self.pegRotMatNP*r_matschxyzp

        #get the inverse
        r_xyz_to_schMat = r_sch_to_xyzMat.transpose()

        from collections import namedtuple
        schxyzMat = namedtuple("schxyzMat", "sch_to_xyz  xyz_to_sch")

        return schxyzMat(r_sch_to_xyzMat, r_xyz_to_schMat)

    def sch_to_xyz(self, posSCH):
        """
        Given an sch coordinate system (defined by setSCH) and an input SCH
        point (a list), return the corresponding earth-centered-earth-fixed
        xyz position.
        """

        #compute the linear portion of the transformation

        #create the SCH sphere object
        sph = Ellipsoid()
        sph.a = self.pegRadCur
        sph.e2 = 0.

        import numpy
        #on SCH sphere, longitude = S/pegRadCur, latitude = C/pegRadCur,
        #height = H
        r_llh = [numpy.degrees(posSCH[1]/sph.a),
                 numpy.degrees(posSCH[0]/sph.a),
                 posSCH[2]]

        #convert sphere llh to sphere xyz coordinates
        r_schvt = numpy.matrix(sph.llh_to_xyz(r_llh)).transpose()

        #Rotate the sch position into the ecef orientation defined by the peg
        r_xyzv = self.pegRotMatNP*r_schvt

        #add the origin of the SCH sphere and return as list
        return ((r_xyzv + self.pegOVNP).transpose()).tolist()[0]

    def xyz_to_sch(self, posXYZ):
        """
        Given an sch coordinate system (defined by setSCH) and an input XYZ
        point (an earth-centered-earth-fixed point as a list), return the
        corresponding SCH position.
        """

        #create a spherical object of the radius of the SCH sphere
        sph = Ellipsoid()
        sph.a = self.pegRadCur
        sph.e2 = 0.

        #use numpy matrices for matrix manipulations
        import numpy
        r_xyz = numpy.matrix(posXYZ).transpose()

        #compute the xyz position relative to the SCH sphere origin
        r_xyzt = r_xyz - self.pegOVNP

        #Rotate the XYZ position from the ecef basis to the SCH sphere basis
        #defined by the peg, and pass the SCH sphere XYZ position to
        #llh_to_xyz to get the llh on the sch sphere
        r_xyzp = ((self.pegRotMatInvNP*r_xyzt).transpose()).tolist()[0]
        r_llh = sph.xyz_to_llh(r_xyzp)

        #S = SCH-sphere-radius*longitude, C = SCH-sphere-radius*latitude,
        #H = height above SCH sphere
        return [self.pegRadCur*numpy.radians(r_llh[1]),
                self.pegRadCur*numpy.radians(r_llh[0]),
                r_llh[2]]

    def schdot_to_xyzdot(self, posSCH, velSCH):
        """
        velXYZ = elp.schdot_to_xyzdot(posSCH, velSCH)
        where elp is an instance of Ellipsoid and posSCH, velSCH are the
        position and velocity in the SCH coordinate system defined by a
        previous call to elp.setSCH and posXYZ, velXYZ are the position
        and velocity in the ecef cartesian coordinate system.
        posSCH, velSCH, posXYZ, and velXYZ are all lists.
        """

        import numpy
        sch_to_xyzMat = (self.schbasis(posSCH)).sch_to_xyz
        velSCHNP = numpy.matrix(velSCH).transpose()
        velXYZNP = sch_to_xyzMat*velSCHNP
        velXYZ = velXYZNP.transpose().tolist()[0]
        posXYZ = self.sch_to_xyz(posSCH)
        return posXYZ, velXYZ

    def xyzdot_to_schdot(self, posXYZ, velXYZ):
        """
        posSCH, velSCH = elp.xyzdot_to_schdot(posXYZ, velXYZ)
        where elp is an instance of Ellipsoid and posXYZ, velXYZ are the
        position and velocity in the ecef cartesian coordinate system and
        posSCH, velSCH are the position and velocity in the SCH coordinate
        system defined by a previous call to elp.setSCH.
        posSCH, velSCH, posXYZ, and velXYZ are all lists.
        """

        import numpy
        posSCH = self.xyz_to_sch(posXYZ)
        xyz_to_schMat = (self.schbasis(posSCH)).xyz_to_sch
        velSCHNP = xyz_to_schMat*numpy.matrix(velXYZ).transpose()
        velSCH = velSCHNP.transpose().tolist()[0]

        return posSCH, velSCH

    def enubasis(self, posLLH):
        """
        xyzenuMat = elp.enubasis(posLLH)
        Given an instance elp of an Ellipsoid LLH position (as a list) return the
        transformation matrices from the XYZ frame to the ENU frame and the
        inverse from the ENU frame to the XYZ frame.  
        The returned object is a namedtuple with numpy matrices in elements
        named 'enu_to_xyz' and 'xyz_to_enu'
        enu_to_xyzMat = (elp.enubasis(posLLH)).enu_to_xyz
        xyz_to_enuMat = (elp.enubasis(posLLH)).xyz_to_enu
        """

        import numpy
        r_lat = numpy.radians(posLLH[0])
        r_lon = numpy.radians(posLLH[1])

        r_clt = numpy.cos(r_lat)
        r_slt = numpy.sin(r_lat)
        r_clo = numpy.cos(r_lon)
        r_slo = numpy.sin(r_lon)

        r_enu_to_xyzMat = numpy.matrix([
            [-r_slo, -r_slt*r_clo, r_clt*r_clo],
            [ r_clo, -r_slt*r_slo, r_clt*r_slo],
            [ 0.0  ,  r_clt      , r_slt]])

        r_xyz_to_enuMat = r_enu_to_xyzMat.transpose()

        from collections import namedtuple
        enuxyzMat = namedtuple("enuxyzMat", "enu_to_xyz  xyz_to_enu")

        return enuxyzMat(r_enu_to_xyzMat, r_xyz_to_enuMat)

    pass

SEMIMAJOR_AXIS = Component.Parameter(
    'a',
    public_name='SEMIMAJOR_AXIS',
    default=1.0,
    type=float,
    mandatory=False,
    intent='input',
    doc='Ellipsoid semimajor axis'
    )
ECCENTRICITY_SQUARED = Component.Parameter(
    'e2',
    public_name='ECCENTRICITY_SQUARED',
    default=0.0,
    type=float,
    mandatory=False,
    intent='input',
    doc='Ellipsoid eccentricity squared'
    )
MODEL = Component.Parameter(
    'model',
    public_name='MODEL',
    default='Unit Sphere',
    type=str,
    mandatory=False,
    intent='input',
    doc='Ellipsoid model'
    )

## This Ellipsoid is an amalgalm of the Heritage ellipsoid and the new one, as
## of 9/8/12: ellipsoid.Ellipsoid-- decorated properties  superceed explicit
## properties, while the getters and setter are retained for backwards
## compatability-- they now call the decorated  properties-- which still act
## as mutator methods, and allow an ellipsoid to be defined with "a" and any
## one of "e2", "e", "f", "finv"  "cosOE", "b", Since each one of these setters
## modifies e2, the value checking is left to that method-- an error is raised
## if e2 is not on  (0,1]. Other ellipsoid parameter properties are inherited
## from ellipsoid._OblateEllipsoid, including all the second and third
## flattenigs, eccentricity (though these are rarely used, as the need to do
## algebraic expansions in them is no longer extant). The  base-clase also has
## functions for the various radii-of-curvature and conversions between common,
## reduced, conformal, authalic, rectifying,  geocentric, and isometric
## latitudes as well as spheric and iterative "exact" solutions to great
## circle distance and bearing problems.  Another base-class:
## EllipsoidTransformations has the methods for computing ECEF<-->LLH
## (approximate or iterative exact) and for computing  the affine
## transformations to various tangent plane (LTP) coordinate systems...



class Ellipsoid(Component,ellipsoid.Ellipsoid, Heritage):
    
    
    parameter_list = (
                      SEMIMAJOR_AXIS,
                      ECCENTRICITY_SQUARED,
                      MODEL
                     )
    
    family = 'ellipsoid'
    
    def __init__(self,family='', name='', a=1.0, e2=0.0, model="Unit Sphere"):
        Component.__init__(self, family if family else  self.__class__.family, name=name)
        ellipsoid.Ellipsoid.__init__(self, a, e2, model=model)
        Heritage.__init__(self)
        return None
    
    #Make sure if init as Configurable that the base class gets initialized 
    def _configure(self):
        ellipsoid.Ellipsoid.__init__(self, self.a, self.e2, self.model)

    # \f$ c = a\epsilon \f$
    @property
    def c(self):
        return self.a*(self.e2)**0.5
    @c.setter
    def c(self,val):
        self.e2 = (val/self.a)**2
        pass

    @property
    def model(self):
        return self._model
    @model.setter
    def model(self, model):
        self._model = model
        pass

    pass
    
