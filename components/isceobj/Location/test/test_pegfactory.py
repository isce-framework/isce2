import unittest
from isceobj.Planet.Ellipsoid import Ellipsoid
from isceobj.Planet.AstronomicalHandbook import PlanetsData
from isceobj.Location.Coordinate import Coordinate
from isceobj.Location.Peg import PegFactory

class PegFactoryTest(unittest.TestCase):

    def setUp(self):
        self.ellipsoid = Ellipsoid(a=PlanetsData.ellipsoid['Earth']['WGS-84'][0],
                                   e2=PlanetsData.ellipsoid['Earth']['WGS-84'][1])
        print (str(self.ellipsoid))

    def tearDown(self):
        pass

    def testFromEllipsoid(self):
        ans = 6356522.8174611665
        coord = Coordinate(latitude=33.5340581084, longitude=-110.699177108, height=0.0)
        peg = PegFactory.fromEllipsoid(coordinate=coord,heading=-166.483356977,ellipsoid=self.ellipsoid)
        self.assertAlmostEquals(ans,peg.radiusOfCurvature,5)

if __name__ == "__main__":
    unittest.main()
