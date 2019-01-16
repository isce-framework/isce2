import datetime
import unittest
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil

class DateTimeUtilTest(unittest.TestCase):

    def setUp(self):
        self.dt1 = datetime.datetime(year=2004,month=3,day=15,hour=12,minute=30,second=0)
        self.dt2 = datetime.datetime(year=2004,month=3,day=15,hour=12,minute=59,second=15)


    def tearDown(self):
        pass

    def testTimeDeltaToSeconds(self):
        ans = 29*60.0+15
        td = self.dt2-self.dt1
        numSeconds = DateTimeUtil.timeDeltaToSeconds(td)
        self.assertAlmostEquals(numSeconds,ans,5)


    def testSecondsSinceMidnight(self):
        ans = 86400.0/2 + 30.0*60
        numSeconds = DateTimeUtil.secondsSinceMidnight(self.dt1)
        self.assertAlmostEquals(numSeconds,ans,5)

    def testDateTimeToDecimalYear(self):
        ans = 2004.2053388
        decimalYear = DateTimeUtil.dateTimeToDecimalYear(self.dt1)
        self.assertAlmostEquals(decimalYear,ans,5)

if __name__ == "__main__":
    unittest.main()
