import unittest
####
#### THIS IS NOT A WORKING TEST JUST A PLACE HOLDER
####
####

class ODRTest(unittest.TestCase):
    
    time = datetime.datetime(year=2004,month=3,day=1,hour=12,minute=3,second=2)
    arclist = Arclist(file=ARCLIST)
    arclist.parse()
    file = arclist.getOrbitFile(time)
    file = os.path.join(ENVISAT, file)
    odr = ODR(file=file)
    odr.parseHeader()
    for sv in odr._ephemeris:
        print(sv)
        pass
