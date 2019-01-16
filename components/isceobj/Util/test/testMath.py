import unittest
from isceobj.Util.mathModule import MathModule as MM

class MathModuleTest(unittest.TestCase):

    def setUp(self):
        self.V = [1,2,3]
        self.M = [[1,2,3],
                  [4,5,6],
                  [7,8,9]]
        self.N = [[1,2,3],
                  [1,2,3],
                  [1,2,3]]

    def tearDown(self):
        pass

    def testMultiplyMatrices(self):
        ans = [[6,12,18],
               [15,30,45],
               [24,48,72]]
        mM = MM.multiplyMatrices(self.M,self.N)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEquals(mM[i][j],ans[i][j],5)

    def testMatrixTranspose(self):
        ans = [[1,4,7],
               [2,5,8],
               [3,6,9]]
        mT = MM.matrixTranspose(self.M)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEquals(mT[i][j],ans[i][j],5)

    def testMatrixVectorProduct(self):
        ans = [14,32,50]
        mV = MM.matrixVectorProduct(self.M,self.V)
        for i in range(3):
            self.assertAlmostEquals(mV[i],ans[i],5)

    def testMean(self):
        ans = 2
        mean = MM.mean(self.V)
        self.assertAlmostEquals(mean,ans)

if __name__ == "__main__":
    unittest.main()
