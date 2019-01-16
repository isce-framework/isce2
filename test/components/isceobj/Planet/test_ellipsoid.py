#!/usr/bin/env python3
#
# Author: Eric Gurrola
# Copyright 2015
#

import unittest
import numpy

import isce
from isceobj.Planet.Planet import Planet
from isceobj.Planet.Ellipsoid import Ellipsoid
from isceobj.Planet.AstronomicalHandbook import PlanetsData

class EllipsoidTest(unittest.TestCase):

    def setUp(self):
        self.ellipsoid = Ellipsoid(a=PlanetsData.ellipsoid['Earth']['WGS-84'][0],
                                   e2=PlanetsData.ellipsoid['Earth']['WGS-84'][1])
        self.llh1 = [40.0,-105.0,2000.0]
        self.llh2 = [40.15,-104.97,2119.0]

    def tearDown(self):
        pass

    def testDistance(self):
        ans = 16850.852914665338 # Result from Scott Hensley's Fortran code
        dis = self.ellipsoid.geo_dis(self.llh1,self.llh2)
        self.assertAlmostEqual(dis,ans,5)

    def testHeading(self):
        ans = 0.15228373938054995 # Result from Scott Hensley's Fortran code
        hdg = self.ellipsoid.geo_hdg(self.llh1,self.llh2)
        self.assertAlmostEqual(hdg,ans,9)

    def testCartesian(self):
        ans = [-1261499.8108277766,-4717861.0677524200,4092096.6400047773]
        xyz = self.ellipsoid.llh_to_xyz(self.llh2)
        for i in range(3):
            self.assertAlmostEqual(xyz[i],ans[i],2)

    def testCartesianRoundTrip(self):
        xyz = self.ellipsoid.llh_to_xyz(self.llh2)
        llh = self.ellipsoid.xyz_to_llh(xyz)
        for i in range(3):
            self.assertAlmostEqual(llh[i],self.llh2[i],2)

    def testRadiusOfCurvature(self):
        # Test east radius of curvature
        ans = 6386976.165976
        rcurv = self.ellipsoid.eastRadiusOfCurvature(self.llh1)
        self.assertAlmostEqual(ans,rcurv,3)

        # Test north radius of curvature
        ans = 6361815.825934
        rcurv = self.ellipsoid.northRadiusOfCurvature(self.llh1)
        self.assertAlmostEqual(ans,rcurv,3)

        # Test general radius of curvature
        ans = 6388976.165706277   #6386976.165976
        rcurv = self.ellipsoid.radiusOfCurvature(self.llh1,hdg=90.0)
        self.assertAlmostEqual(ans,rcurv,3)

        ans = 6363815.826433734   #6361815.825934
        rcurv = self.ellipsoid.radiusOfCurvature(self.llh1,hdg=0.0)
        self.assertAlmostEqual(ans,rcurv,3)

        ans = 6382667.441829258  #6380667.441906
        rcurv = self.ellipsoid.radiusOfCurvature(self.llh1,hdg=60.0)
        self.assertAlmostEqual(ans,rcurv,3)

        ans = 6356522.495223
        rcurv = self.ellipsoid.radiusOfCurvature([33.5340581084, 50.0, 0.0],
            hdg=-166.483356977)
        self.assertAlmostEqual(ans,rcurv,3)

    def notestLocalRadius(self):
        ans = 0.0
        rad = self.ellipsoid.localRadius(self.llh1)
        self.assertAlmostEqual(rad,ans,3)

    def testWGS84Ellipsoid(self):
        elp = Planet(pname="Earth").get_elp()
        a = 6378137.0
        e2 = 0.0066943799901
        b = 6356752.314
        self.assertAlmostEqual(a, elp.a, places=3)
        self.assertAlmostEqual(e2, elp.e2, places=10)
        self.assertAlmostEqual(b, elp.b, places=3)

    def testLATLON(self):
        elp = Planet(pname="Earth").get_elp()

        #From for_ellipsoid_test.F
        r_xyz = [7000000.0, -7500000.0, 8000000.0]
        r_llh = [38.038207425428674, -46.974934010881981, 6639569.3697941694]
        posLLH = elp.xyz_to_llh(r_xyz)
        for (a, b) in zip(r_llh[:2], posLLH[:2]):
            self.assertAlmostEqual(a, b, places=3)
        self.assertAlmostEqual(r_llh[2], posLLH[2], delta=.1)

        r_llh = [-33.0, 118.0, 2000.0]
        r_xyz = [-2514561.1100611691, 4729201.6284226896, -3455047.9192480515]
        posXYZ = elp.llh_to_xyz(r_llh)
        for (a, b) in zip(r_xyz, posXYZ):
            self.assertAlmostEqual(a, b, places=3)

    def testSETSCH(self):
        elp = Planet(pname="Earth").get_elp()
        elp.setSCH(66.0, -105.0, 36.0)

        #From for_ellipsoid_test.F
        r_radcur = 6391364.9560780991
        r_ov = [  -490.98983883031178,
                 -1832.3990245149471,
                -34854.866159332916]
        r_mat = [
            [-0.10527118956908345, 0.75904333077238850, -0.64247272211096140],
            [-0.39287742804503412, 0.56176045358432036,  0.72806079369889010],
            [ 0.91354545764260087, 0.32905685648333960,  0.23907380036690279]]
        r_matinv = [
            [-0.10527118956908345, -0.39287742804503412, 0.91354545764260087],
            [ 0.75904333077238850,  0.56176045358432036, 0.32905685648333960],
            [-0.64247272211096140,  0.72806079369889010, 0.23907380036690279]]

        self.assertAlmostEqual(r_radcur, elp.pegRadCur, places=3)
        for (a,b) in zip(r_ov, elp.pegOV):
            self.assertAlmostEqual(a, b, places=3)

        for i in range(3):
            for (a,b) in zip(r_mat[i], elp.pegRotMat[i]):
                self.assertAlmostEqual(a, b, places=3)

        for i in range(3):
            for (a,b) in zip(r_matinv[i], elp.pegRotMatInv[i]):
                self.assertAlmostEqual(a, b, places=3)


    def testConvertSCH(self):
        elp = Planet(pname="Earth").get_elp()
        elp.setSCH(66.0, -105.0, 36.0)

        #From for_ellipsoid_test.F
        #convert_sch_to_xyz, sch_to_xyz
        r_sch = [1468.0, -234.0, 7000.0]
        r_xyz = [-672788.46258740244, -2514950.4839521507, 5810769.7976823179]
        r_llh = [66.009415512068244, -104.97681810507400, 6999.9999703792855]

        posXYZ = elp.sch_to_xyz(r_sch)
        for (a,b) in zip(r_xyz,posXYZ):
            self.assertAlmostEqual(a, b, places=3)

        #convert_sch_to_xyz, xyz_to_sch
        r_xyz = [-672100.0, -2514000.0, 5811000.0]
        r_sch = [2599.1237664792707, 70.396218844576666, 6764.7576835183427]
        r_llh = [66.019224990424505, -104.96758302093188, 6764.7576984856278]
        posXYZ = elp.sch_to_xyz(r_sch)
        for (a,b) in zip(r_xyz,posXYZ):
            self.assertAlmostEqual(a, b, places=3)

    def testConvertSCHdot(self):
        elp = Planet(pname="Earth").get_elp()
        elp.setSCH(66.0, -105.0, 36.0)

        #From for_ellipsoid_test.F
        #convert_schdot_to_xyzdot, sch_to_xyz
        r_sch = [1468.0, -234.0, 7000.0]
        r_schdot = [800.0, -400.0, 100.0]
        r_xyz = [-672788.46258740244, -2514950.4839521507, 5810769.7976823179]
        r_xyzdot = [853.73728655948685, 118.98447071885982, 258.79594191185748]
        posXYZ, velXYZ = elp.schdot_to_xyzdot(r_sch,r_schdot)
        for (a,b) in zip(r_xyz,posXYZ):
            self.assertAlmostEqual(a, b, places=3)
        for (a,b) in zip(r_xyzdot,velXYZ):
            self.assertAlmostEqual(a, b, places=3)

        #convert_schdot_to_xyzdot, xyz_to_sch
        r_xyz = [-672100.0, -2514000.0, 5811000.0]
        r_xyzdot = [800.0, -400.0, 100.0]
        r_sch = [2599.1237664792707, 70.396218844576666, 6764.7576835183427]
        r_schdot = [415.39842327248573, -781.28909619852459, 164.41258499283407]
        posSCH, velSCH = elp.xyzdot_to_schdot(r_xyz,r_xyzdot)
        for (a,b) in zip(r_sch,posSCH):
            self.assertAlmostEqual(a, b, places=3)
        for (a,b) in zip(r_schdot,velSCH):
            self.assertAlmostEqual(a, b, delta=0.1)

    def testSCH1(self):
        elp = Planet(pname="Earth").get_elp()

        #S path on Earth equator West to East, origin at y=z=0
        elp.setSCH(0., 0., 90.)

        #SCH = [0.,0.,0.] => XYZ = [elp.a, 0., 0.]
        sch = [0.,0.,0.]
        xyz = elp.sch_to_xyz(sch)
        self.assertAlmostEqual(xyz[0], elp.a, places=3)
        self.assertAlmostEqual(xyz[1], 0., places=3)
        self.assertAlmostEqual(xyz[2], 0., places=3)
        sch1 = elp.xyz_to_sch(xyz)
        for (s,s1) in zip(sch, sch1):
            self.assertAlmostEqual(s, s1, places=3)

        #SCH = [(pi/2)*elp.a, 0, 0] => XYZ=[0., elp.a, 0.]
        sch = [numpy.pi*elp.a/2., 0., 0.]
        xyz = elp.sch_to_xyz(sch)
        self.assertAlmostEqual(xyz[0], 0., places=3)
        self.assertAlmostEqual(xyz[1], elp.a, places=3)
        self.assertAlmostEqual(xyz[2], 0., places=3)
        sch1 = elp.xyz_to_sch(xyz)
        for (s,s1) in zip(sch, sch1):
            self.assertAlmostEqual(s, s1, places=3)

        #SCH = [pi*elp.a, 0, 0] => XYZ=[-elp.a, 0., 0.]
        sch = [numpy.pi*elp.a, 0., 0.]
        xyz = elp.sch_to_xyz(sch)
        self.assertAlmostEqual(xyz[0], -elp.a, places=3)
        self.assertAlmostEqual(xyz[1], 0., places=3)
        self.assertAlmostEqual(xyz[2], 0., places=3)
# Round off causes degenerate case where lon = -180 and lon=180 are the same
# point and xyz(-sch) = xyz(+sch), but -sch != sch
#
#        sch1 = elp.xyz_to_sch(xyz)
#        print(sch1)
#        for (s,s1) in zip(sch, sch1):
#            self.assertAlmostEqual(s, s1, places=3)

        #SCH = [(3pi/2)*elp.a, 0, 0] => XYZ=[0., -elp.a, 0.]
        sch = [3*numpy.pi*elp.a/2., 0., 0.]
        xyz = elp.sch_to_xyz(sch)
        self.assertAlmostEqual(xyz[0], 0., places=3)
        self.assertAlmostEqual(xyz[1], -elp.a, places=3)
        self.assertAlmostEqual(xyz[2], 0., places=3)
#        sch1 = elp.xyz_to_sch(xyz)
#        for (s,s1) in zip(sch, sch1):
#            self.assertAlmostEqual(s, s1, places=3)

        #SCH = [2pi*elp.a, 0, 0] => XYZ=[elp.a, 0., 0.]
        sch = [2.*numpy.pi*elp.a, 0., 0.]
        xyz = elp.sch_to_xyz(sch)
        self.assertAlmostEqual(xyz[0], elp.a, places=3)
        self.assertAlmostEqual(xyz[1], 0., places=3)
        self.assertAlmostEqual(xyz[2], 0., places=3)
#Another sch degeneracy due to angle branch cut
#        sch1 = elp.xyz_to_sch(xyz)
#        for (s,s1) in zip(sch, sch1):
#            self.assertAlmostEqual(s, s1, places=3)

        #SCH = [0., (pi/2)*elp.a, elp.b-elp.a] => XYZ = [0., 0., elp.b]
        sch = [0., numpy.pi*elp.a/2., elp.b-elp.a]
        xyz = elp.sch_to_xyz(sch)
        self.assertAlmostEqual(xyz[0], 0., places=3)
        self.assertAlmostEqual(xyz[1], 0., places=3)
        self.assertAlmostEqual(xyz[2], elp.b, places=3)
#        sch1 = elp.xyz_to_sch(xyz)
#        for (s,s1) in zip(sch, sch1):
#            self.assertAlmostEqual(s, s1, places=3)

        #SCH = [0., pi*elp.a, 0.] => XYZ = [-elp.a, 0., 0.]
        sch = [0., numpy.pi*elp.a, 0.]
        xyz = elp.sch_to_xyz(sch)
        self.assertAlmostEqual(xyz[0], -elp.a, places=3)
        self.assertAlmostEqual(xyz[1], 0., places=3)
        self.assertAlmostEqual(xyz[2], 0., places=3)

        #SCH = [0., (3pi/2)*elp.a, elp.b-elp.a] => XYZ = [0., 0., -elp.b]
        sch = [0., 3.*numpy.pi*elp.a/2., elp.b-elp.a]
        xyz = elp.sch_to_xyz(sch)
        self.assertAlmostEqual(xyz[0], 0., places=3)
        self.assertAlmostEqual(xyz[1], 0., places=3)
        self.assertAlmostEqual(xyz[2], -elp.b, places=3)


    def testSCH2(self):
        elp = Planet(pname="Earth").get_elp()

        #Peg at North Pole, S path on prime meridian heading North to South
        elp.setSCH(90., 0., -90.)

        #SCH = [0.,0.,0.] => XYZ = [elp.b, 0., 0.]
        sch = [0.,0.,0.]
        xyz = elp.sch_to_xyz(sch)
        self.assertAlmostEqual(xyz[0], 0., places=3)
        self.assertAlmostEqual(xyz[1], 0., places=3)
        self.assertAlmostEqual(xyz[2], elp.b, places=3)
        sch1 = elp.xyz_to_sch(xyz)
        for (s,s1) in zip(sch, sch1):
            self.assertAlmostEqual(s, s1, places=3)

        #SCH = [pi*elp.pegRadCur, 0, elp.b+(elp.pegOV[2]-elp.pegRadCur)] =>
        #XYZ=[0., 0., -elp.b]
        sch = [numpy.pi*elp.pegRadCur, 0., elp.b+elp.pegOV[2]-elp.pegRadCur]
        xyz = elp.sch_to_xyz(sch)
        self.assertAlmostEqual(xyz[0], 0., places=3)
        self.assertAlmostEqual(xyz[1], 0., places=3)
        self.assertAlmostEqual(xyz[2], -elp.b, places=3)
        sch1 = elp.xyz_to_sch(xyz)
        for (s,s1) in zip(sch, sch1):
            self.assertAlmostEqual(s, s1, places=3)


    def testXYZSCH(self):
        elp = Planet(pname="Earth").get_elp()

        elp.setSCH(30., 60., 45.)
        sch = [-50000., 200000., 1000.]
        xyz = elp.sch_to_xyz(sch)
        sch1 = elp.xyz_to_sch(xyz)
        for (s,s1) in zip(sch, sch1):
            self.assertAlmostEqual(s, s1, places=3)

        xyz = [-4.e6, 10.e6, 1.e6]
        sch = elp.xyz_to_sch(xyz)
        xyz1 = elp.sch_to_xyz(sch)
        for (x,x1) in zip(xyz, xyz1):
            self.assertAlmostEqual(x, x1, places=3)

        elp.setSCH(65., -22., -30.)
        sch = [100000., -100000., 100000.]
        xyz = elp.sch_to_xyz(sch)
        sch1 = elp.xyz_to_sch(xyz)
        for (s,s1) in zip(sch, sch1):
            self.assertAlmostEqual(s, s1, places=3)

        xyz = [-1.e6, -2.e6, 100.e6]
        sch = elp.xyz_to_sch(xyz)
        xyz1 = elp.sch_to_xyz(sch)
        for (x,x1) in zip(xyz, xyz1):
            self.assertAlmostEqual(x, x1, places=3)

    def testSCHDOT(self):
        elp = Planet(pname="Earth").get_elp()

        elp.setSCH(0.,0.,90.)
        sch = [0.,0.,0.]
        schdot = [0.,0.,10.]
        xyz, xyzdot = elp.schdot_to_xyzdot(sch, schdot)
        ans = [10.0, 0.0, 0.0]
        for (x, x1) in zip(xyzdot, ans):
            self.assertAlmostEqual(x, x1, places=3)

        xyz = [elp.a, 0., 0.]
        sch1, schdot1 = elp.xyzdot_to_schdot(xyz, xyzdot)
        for (s, s1) in zip(schdot, schdot1):
            self.assertAlmostEqual(s, s1, places=3)

        elp.setSCH(30.,60.,30.)
        sch = [0.,0.,0.]
        schdot = [10.,0.,0.]
        xyz, xyzdot = elp.schdot_to_xyzdot(sch, schdot)
        ans = [-6.495190528383289, -1.2499999999999996, 7.500000000000001]
        for (x, x1) in zip(xyzdot, ans):
            self.assertAlmostEqual(x, x1, places=3)
        xyz = elp.sch_to_xyz(sch)
        sch1, schdot1 = elp.xyzdot_to_schdot(xyz, xyzdot)
        for (s, s1) in zip(schdot, schdot1):
            self.assertAlmostEqual(s, s1, places=3)

    def testDEBUG(self):
        elp = Planet(pname="Earth").get_elp()
        elp.setSCH(19.2796271, -155.282224, 58.9432911)
        posSCH =  [-58033.8, 0.0, 12494.4008]
        velSCH =  [234.84106135055595, 0.0, 12494.4008]
        posXYZ =  [-5511147.555045444, -2482080.457636343, 2068314.4442497757]
        velXYZ =  [-10652.45905403, -5017.70635173, 4184.84656172]
        p, v = elp.schdot_to_xyzdot(posSCH, velSCH)
        for (a,b) in zip(p, posXYZ):
            self.assertAlmostEqual(a, b, places=3)

        for (a,b) in zip(v, velXYZ):
            self.assertAlmostEqual(a, b, places=3)

if __name__ == "__main__":
#    unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(EllipsoidTest)
    #unittest.TextTestRunner(verbosity=2).run(suite)
    unittest.TextTestRunner(verbosity=0).run(suite)
