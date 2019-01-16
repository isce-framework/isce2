import unittest
from isceobj.Location.Peg import Peg
from isceobj.Location.SCH import SCH
from isceobj.Planet.AstronomicalHandbook import PlanetsData
from isceobj.Planet.Ellipsoid import Ellipsoid

class SCHTest(unittest.TestCase):

    def setUp(self):
        ellipsoid = Ellipsoid(a=PlanetsData.ellipsoid['Earth']['WGS-84'][0],
                              e2=PlanetsData.ellipsoid['Earth']['WGS-84'][1])
        peg = Peg(latitude=30.0,longitude=60.0,heading=45.0,ellipsoid=ellipsoid)
        self.xyz = ellipsoid.llh_to_xyz([30.1, 59.5, 650000.0])
        self.sch = SCH(peg=peg)

    def tearDown(self):
        pass

    def testInitializeTranslationVector(self):
        ans = [6968.2018617638387, 12069.279662064277, -13320.537019955460]
        self.sch.initializeTranslationVector()
        tvec = self.sch.r_ov
        for i in range(3):
            self.assertAlmostEquals(tvec[i],ans[i],5)

    def testInitializeRotationMatrix(self):
        ans = [[0.43301270188924235, -0.78914913099422490,0.43559574039886773], 
               [0.75000000000073663, 4.73671727434728518E-002, -0.65973960844047030],
               [0.50000000000147327, 0.61237243569363053, 0.61237243569675559]] 
        self.sch.initializeRotationMatrix()
        rotmat = self.sch.M
        for i in range(3):
            for j in range(3):
                self.assertAlmostEquals(rotmat[i][j],ans[i][j],5)

    def testXYZToSCH(self):
        ans = [-26156.370014733548, 41985.355842714926, 650000.43586986139]
        sch = self.sch.xyz_to_sch(self.xyz)
        for i in range(3):
            self.assertAlmostEquals(sch[i],ans[i],5)

    def testSCHToXYZ(self):
        ans = self.xyz
        xyz = self.sch.sch_to_xyz([-26156.370014733548, 41985.355842714926, 650000.43586986139])
        for i in range(3):
            self.assertAlmostEquals(xyz[i],ans[i],5)


if __name__ == "__main__":
    unittest.main()
