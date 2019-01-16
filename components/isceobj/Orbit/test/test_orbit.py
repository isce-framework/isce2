import datetime
import logging
import unittest
from isceobj.Orbit.Orbit import Orbit, StateVector

class OrbitTest(unittest.TestCase):

    def setUp(self):
        logging.basicConfig()
        self.linearOrbit = Orbit()
        self.quadOrbit = Orbit()

        linpos, linvel = self.generateLinearSV(10,[[1.0,2.0,3.0]],[[1.0/60.0 for j in range(3)]])
        quadpos, quadvel = self.generateQuadraticSV(10,[[1.0,2.0,3.0]],0.1)

        dt = datetime.datetime(year=2010,month=1,day=1)

        for i in range(10):
            linsv = StateVector()
            quadsv = StateVector()
            linsv.setTime(dt)
            quadsv.setTime(dt)
            linsv.setPosition(linpos[i])
            linsv.setVelocity(linvel[i])
            quadsv.setPosition(quadpos[i])
            quadsv.setVelocity(quadvel[i])
            self.linearOrbit.addStateVector(linsv)
            self.quadOrbit.addStateVector(quadsv)

            dt = dt + datetime.timedelta(minutes=1)

    def tearDown(self):
        del self.linearOrbit
        del self.quadOrbit

    def generateLinearSV(self,num,pos,vel):
        for i in range(1,num):
            sv = [0.0 for j in range(3)]
            for j in range(3):
                sv[j] = pos[i-1][j]+vel[i-1][j]*60.0 
            pos.append(sv)
            vel.append(vel[0])
        return pos,vel

    def generateQuadraticSV(self,num,pos,rate):
        vel = [[0.0 for j in range(3)]]
        for t in range(1,num):
            newPos = [0.0 for j in range(3)]
            newVel = [0.0 for j in range(3)]
            for j in range(3):
                newPos[j] = pos[0][j] + rate*(t**2)
                newVel[j] = 2.0*rate*t/60.0
            pos.append(newPos)
            vel.append(newVel)
        return pos,vel

    def testAddStateVector(self):
        a = None
        self.assertRaises(TypeError,self.linearOrbit.addStateVector,a)

    def testLinearInterpolateOrbit(self):
        ans = [2.5,3.5,4.5]
        sv = self.linearOrbit.interpolateOrbit(datetime.datetime(year=2010,month=1,day=1,hour=0,minute=1,second=30),method='linear')
        pos = sv.getPosition()
        for i in range(3):
            self.assertAlmostEquals(pos[i],ans[i],5)

        ans = [1.225,2.225,3.225]
        sv = self.quadOrbit.interpolateOrbit(datetime.datetime(year=2010,month=1,day=1,hour=0,minute=1,second=30),method='linear')
        pos = sv.getPosition()
        for i in range(3):
            self.assertAlmostEquals(pos[i],ans[i],5)

    def testHermiteInterpolateOrbit(self):
        ans = [2.5,3.5,4.5]
        sv = self.linearOrbit.interpolateOrbit(datetime.datetime(year=2010,month=1,day=1,hour=0,minute=1,second=30),method='hermite')
        pos = sv.getPosition()
        for i in range(3):
            self.assertAlmostEquals(pos[i],ans[i],5)

        ans = [1.225,2.225,3.225]
        sv = self.quadOrbit.interpolateOrbit(datetime.datetime(year=2010,month=1,day=1,hour=0,minute=1,second=30),method='hermite')
        pos = sv.getPosition()
        for i in range(3):
            self.assertAlmostEquals(pos[i],ans[i],5)

    def testLegendreInterpolateOrbit(self):
        ans = [4.5,5.5,6.5]
        sv = self.linearOrbit.interpolateOrbit(datetime.datetime(year=2010,month=1,day=1,hour=0,minute=3,second=30),method='legendre')
        pos = sv.getPosition()
        for i in range(3):
            self.assertAlmostEquals(pos[i],ans[i],5)

        ans = [2.225,3.225,4.225]
        sv = self.quadOrbit.interpolateOrbit(datetime.datetime(year=2010,month=1,day=1,hour=0,minute=3,second=30),method='legendre')
        pos = sv.getPosition()
        for i in range(3):
            self.assertAlmostEquals(pos[i],ans[i],5)

    def testInterpolateOrbitOutOfBounds(self):
        dt = datetime.datetime(year=2010,month=1,day=2)
        self.assertRaises(ValueError,self.linearOrbit.interpolateOrbit,dt)

if __name__ == "__main__":
    unittest.main()
