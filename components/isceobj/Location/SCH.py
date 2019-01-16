
import math
from isceobj.Planet.Ellipsoid import Ellipsoid
from isceobj.Planet.AstronomicalHandbook import PlanetsData
from isceobj.Util.mathModule import MathModule as MM

class SCH(object):
    """A Class to convert between SCH and XYZ coordinates"""
    
    def __init__(self,peg=None):
        self.peg = peg
        self.r_ov = [0 for i in range(3)]
        self.M = [[0 for i in range(3)] for j in range(3)]
        self.invM = [[0 for i in range(3)] for j in range(3)]
        self.__initialize()
   
    def __initialize(self):
        self.initializeTranslationVector()
        self.initializeRotationMatrix()

    def initializeRotationMatrix(self):
        lat = math.radians(self.peg.getLatitude())
        lon = math.radians(self.peg.getLongitude())
        heading = math.radians(self.peg.getHeading())
                
        self.M[0][0] = math.cos(lat)*math.cos(lon)
        self.M[0][1] = -math.sin(heading)*math.sin(lon) - math.sin(lat)*math.cos(lon)*math.cos(heading)
        self.M[0][2] = math.sin(lon)*math.cos(heading) - math.sin(lat)*math.cos(lon)*math.sin(heading)
        self.M[1][0] = math.cos(lat)*math.sin(lon)
        self.M[1][1] = math.cos(lon)*math.sin(heading) - math.sin(lat)*math.sin(lon)*math.cos(heading)
        self.M[1][2] = -math.cos(lon)*math.cos(heading) - math.sin(lat)*math.sin(lon)*math.sin(heading)
        self.M[2][0] = math.sin(lat)
        self.M[2][1] = math.cos(lat)*math.cos(heading)
        self.M[2][2] = math.cos(lat)*math.sin(heading)
        
        self.invM = MM.matrixTranspose(self.M)
        
    def initializeTranslationVector(self):
        lat = math.radians(self.peg.getLatitude())
        lon = math.radians(self.peg.getLongitude())        
        radcur = self.peg.getRadiusOfCurvature() # Get the radius of curvature at the peg point
                       
        r_up = [0 for i in range(3)]
        r_p = [0 for i in range(3)]
                           
        r_up[0
                ] = math.cos(lat)*math.cos(lon)
        r_up[1] = math.cos(lat)*math.sin(lon)
        r_up[2] = math.sin(lat)
        
        # The Cartesian vector at the peg latitude and longitude at zero height
        r_p = self._calculateXYZ()
                
        for i in range(3):
            self.r_ov[i] = r_p[i] - radcur*r_up[i]

    def _calculateXYZ(self):
        """
        Calculate the cartesian coordinate of the point assuming the WGS-84 ellipsoid (to be fixed)
        """
        ellipsoid = Ellipsoid(a=PlanetsData.ellipsoid['Earth']['WGS-84'][0],
                              e2=PlanetsData.ellipsoid['Earth']['WGS-84'][1])
        llh = [self.peg.getLatitude(),self.peg.getLongitude(),0.0]
        xyz = ellipsoid.llh_to_xyz(llh)
        return xyz
                        
    def xyz_to_sch(self,xyz):        
        radcur = self.peg.getRadiusOfCurvature() # Get the radius of curvature at the peg point
        ellipsoid = Ellipsoid(a=radcur,e2=0.0)
        
        
                
        schvt = [0 for i in range(3)]
        rschv = [0 for i in range(3)]
        
        for i in range(3):
            schvt[i] = xyz[i] - self.r_ov[i]
        
        schv = MM.matrixVectorProduct(self.invM,schvt)        
        llh = ellipsoid.xyz_to_llh(schv)
        
        rschv[0] = radcur*math.radians(llh[1])
        rschv[1] = radcur*math.radians(llh[0])
        rschv[2] = llh[2]
        
        return rschv
    
    def sch_to_xyz(self,sch):        
        radcur = self.peg.getRadiusOfCurvature() # Get the radius of curvature at the peg point
        ellipsoid = Ellipsoid(a=radcur,e2=0.0)
        
        xyz = [0 for i in range(3)]
        llh = [0 for i in range(3)]
                
        llh[0] = math.degrees(sch[1]/radcur)
        llh[1] = math.degrees(sch[0]/radcur)
        llh[2] = sch[2]
        
        schv = ellipsoid.llh_to_xyz(llh)
        schvt = MM.matrixVectorProduct(self.M,schv)
        
        for i in range(3):
            xyz[i] = schvt[i] + self.r_ov[i]
            
        return xyz

    def vxyz_to_vsch(self,sch,vxyz):
        """
        Convert from cartesian velocity to sch velocity
        """
        schbasis = LocalSCH(peg=self.peg,sch=sch)
        vsch = schbasis.xyz_to_localsch(vxyz)

        return vsch

    def vsch_to_vxyz(self,sch,vsch):
        """
        Convert from sch velocity to cartesian velocity
        """
        schbasis = LocalSCH(peg=self.peg,sch=sch)
        vxyz = schbasis.localsch_to_xyz(vsch)

        return vxyz

class LocalSCH(SCH):
#    It almost might be better to define an SCH 'Location' object 
#    that can convert things between its local tangent plane and back

    def __init__(self,peg=None,sch=None):
        SCH.__init__(self,peg=peg)
        self.sch = sch
        self.sch2xyz = [[0 for i in range(3)] for j in range(3)]
        self.xyz2sch = [[0 for i in range(3)] for j in range(3)]

        self.__initialize()

    def __initialize(self):
        s = self.sch[0]/self.peg.getRadiusOfCurvature()
        c = self.sch[1]/self.peg.getRadiusOfCurvature()

        schxyzp = [[0 for i in range(3)] for j in range(3)]
        schxyzp[0][0] = -math.sin(s)
        schxyzp[0][1] = -math.sin(c)*math.cos(s)
        schxyzp[0][1] = math.cos(s)*math.cos(c)
        schxyzp[1][0] = math.cos(s)
        schxyzp[1][1] = -math.sin(c)*math.sin(s)
        schxyzp[1][2] = math.sin(s)*math.cos(c)
        schxyzp[2][0] = 0.0
        schxyzp[2][1] = math.cos(c)
        schxyzp[2][2] = math.sin(c)

        self.sch2xyz = MM.multiplyMatrices(self.M,schxyzp)
        self.xyz2sch = MM.matrixTranspose(self.sch2xyz)

    def xyz_to_localsch(self,xyz):
        sch = MM.matrixVectorProduct(self.xyz2sch,xyz)

        return sch

    def localsch_to_xyz(self,sch):
        xyz = MM.matrixVectorProduct(self.sch2xyz,sch)

        return xyz
