import unittest
from isceobj.Location.Offset import OffsetField, Offset

class OffsetTest(unittest.TestCase):

    def setUp(self):
        self.offsetField = OffsetField()
	for i in range(10):
            snr = 1.0
	    if (i == 5):
                snr = 0.3
	    elif (i == 8):
		snr = 0.1
            offset = Offset(x=i,y=i,dx=1,dy=2,snr=snr)
	    self.offsetField.addOffset(offset)

    def tearDown(self):
        pass

    def testCull(self):
        """
        Test that culling offsets below a given signal-to-noise
	works.
	"""
        culledOffsetField = self.offsetField.cull(1.0)
	i = 0
        for offset in culledOffsetField:
            if (offset.getSignalToNoise() < 1.0):
                self.fail()
	    i = i+1
	self.assertEquals(i,8)

    def testNaN(self):
       """
       Test that NaN signal-to-noise values are converted to 0.0.
       """
       nanOffset = Offset(x=4,y=5,dx=8,dy=9,snr='nan')
       self.assertAlmostEquals(nanOffset.getSignalToNoise(),0.0,5)

if __name__ == "__main__":
    unittest.main()
