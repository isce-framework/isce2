
import unittest
from mroipac.geolocate.Geolocate import Geolocate
from isceobj.Planet.Planet import Planet

class test_geolocate(unittest.TestCase):

    def setUp(self):
        # These are the state vectors for ERS-1 track 113 frame 2745 from 1993 01 09 near the scene start time
        self.pos = [-2503782.263,-4652987.799,4829281.081]
        self.vel = [-4002.34200000018,-3450.91900000069,-5392.36600000039]
        self.range = 831929.866545593
        self.squint = 0.298143953340833
        planet = Planet(pname='Earth')

        self.geolocate = Geolocate()
        self.geolocate.wireInputPort(name='planet',object=planet)

    def tearDown(self):
        pass

    def testGeolocate(self):
        ans = [42.457487,-121.276432]

        loc,lla,lia = self.geolocate.geolocate(self.pos,self.vel,self.range,self.squint)

        lat = loc.getLatitude()
        lon = loc.getLongitude()
        self.assertAlmostEquals(lat,ans[0],5)
        self.assertAlmostEquals(lon,ans[1],5)

    def testLookAngle(self):
        ans = 17.2150393
        loc,lla,lia = self.geolocate.geolocate(self.pos,self.vel,self.range,self.squint)
    
        self.assertAlmostEquals(lla,ans,5)

if __name__ == "__main__":
    unittest.main()
