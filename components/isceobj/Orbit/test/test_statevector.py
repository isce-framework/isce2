import datetime
import unittest
import isce
from isceobj.Orbit.Orbit import StateVector

class StateVectorTest(unittest.TestCase):

    def setUp(self):
            pass

    def tearDown(self):
            pass

    def testEqualCompare(self):
        """
        Test that __cmp__ returns true when the times are the same, but the
        positions and velocities are different.
        """
        sv1 = StateVector()
        time1 = datetime.datetime(year=2001,month=2,day=7,hour=12,minute=13,
                                  second=4)
        pos1 = [1.0,2.0,3.0]
        vel1 = [0.6,0.6,0.6]
        sv1.setTime(time1)
        sv1.setPosition(pos1)
        sv1.setVelocity(vel1)

        sv2 = StateVector()
        time2 = datetime.datetime(year=2001,month=2,day=7,hour=12,minute=13,
                                  second=4)
        pos2 = [2.0,3.0,4.0]
        vel2 = [0.7,0.7,0.7]
        sv2.setTime(time2)
        sv2.setPosition(pos2)
        sv2.setVelocity(vel2)

        self.assertTrue(sv1 == sv2)

    def testNotEqualCompare(self):
        """
        Test that __cmp__ returns false when the times are different, but the
        positions and velocities are the same.
        """
        sv1 = StateVector()
        time1 = datetime.datetime(year=2001,month=2,day=7,hour=12,minute=13,second=5)
        pos1 = [1.0,2.0,3.0]
        vel1 = [0.6,0.6,0.6]
        sv1.setTime(time1)
        sv1.setPosition(pos1)
        sv1.setVelocity(vel1)

        sv2 = StateVector()
        time2 = datetime.datetime(year=2001,month=2,day=7,hour=12,minute=13,second=4)
        pos2 = [1.0,2.0,3.0]
        vel2 = [0.6,0.6,0.6]
        sv2.setTime(time2)
        sv2.setPosition(pos2)
        sv2.setVelocity(vel2)

        self.assertFalse(sv1 == sv2)

    def testScalarVelocity(self):
        """
        Test that the scalar velocity returns the expected value
        """
        ans = 0.0288675134594813
        sv1 = StateVector()
        time1 = datetime.datetime(year=2001,month=2,day=7,hour=12,minute=13,
                                  second=5)
        pos1 = [1.0,2.0,3.0]
        vel1 = [0.0166666,0.0166666,0.0166666]
        sv1.setTime(time1)
        sv1.setPosition(pos1)
        sv1.setVelocity(vel1)

        vel = sv1.getScalarVelocity()
        self.assertAlmostEqual(ans,vel,5)

if __name__ == "__main__":
    unittest.main()
