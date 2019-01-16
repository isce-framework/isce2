import os
import logging
import datetime
from isceobj.Orbit.Orbit import Orbit
from isceobj.Orbit.Orbit import StateVector
from isceobj.Util.decorators import type_check, logged, pickled

class PRC(object):
    """A class to parse orbit data from D-PAF"""

    logging_name = "isce.orbit.PRC.PRC"
    @logged
    def __init__(self, file=None):
        self.filename = file
        self.firstEpoch = 0
        self.lastEpoch = 0
        self.tdtOffset = 0
        self.orbit = Orbit()
        self.orbit.configure()
        self.orbit.setOrbitQuality('Precise')
        self.orbit.setOrbitSource('PRC')
        return None

    def getOrbit(self):
        return self.orbit

    def parse(self):
        #People still seem to be using the old .Z format
        #Adding support for it - PSA
        if os.path.splitext(self.filename)[1] == '.Z':
            from subprocess import Popen, PIPE
            fp =  Popen(["zcat", self.filename], stdout=PIPE).stdout
        else:
             fp = open(self.filename,'r')
        data = fp.read()
        fp.close()

        numLines = int(len(data)/130)
        for i in range(numLines):
            line = data[i*130:(i+1)*130]
            self.__parseLine(line)

    def __parseLine(self,line):
        """Parse a line from a PRC orbit file"""
        referenceFrame = line[0:6].decode('utf-8')
        if (referenceFrame == 'STATE '):
            self.__parseStateLine(line)
        if (referenceFrame == 'STTERR'):
            self.__parseTerrestrialLine(line)

    def __parseTerrestrialLine(self,line):
        j2000Day = float(line[14:20])/10.0 +  0.5
        tdt = float(line[20:31])/1e6
        x = float(line[31:43])/1e3
        y = float(line[43:55])/1e3
        z = float(line[55:67])/1e3
        vx = float(line[67:78])/1e6
        vy = float(line[78:89])/1e6
        vz = float(line[89:100])/1e6
        quality = line[127]

        tdt = tdt - self.tdtOffset
        dt = self.__j2000ToDatetime(j2000Day,tdt)

        sv = StateVector()
        sv.configure()
        sv.setTime(dt)
        sv.setPosition([x,y,z])
        sv.setVelocity([vx,vy,vz])
        self.orbit.addStateVector(sv)

    def __parseStateLine(self,line):
        self.firstEpoch = self.__j2000ToDatetime(float(line[6:12])/10.0,0.0)
        self.lastEpoch = self.__j2000ToDatetime(float(line[12:18])/10.0,0.0)
        self.tdtOffset = float(line[47:52])
        self.tdtOffset = self.tdtOffset/1e3

    def __j2000ToDatetime(self,j2000Day,tdt):
        """Convert the number of days since 1 Jan. 2000 to a datetime object"""
        j2000 = datetime.datetime(year=2000,month=1,day=1)
        dt = j2000 + datetime.timedelta(days=j2000Day,seconds=tdt)
        return dt
    pass

@pickled
class Arclist(object):
    """A class for parsing the old ROI_PAC PRC arclist file"""

    logging_name = 'isce.Orbit.PRC.Arclist'

    @logged
    def __init__(self, file=None):
        self.filename = file
        self.arclist = []
        return None

    def parse(self):
        fp = open(self.filename,'r')

        for line in fp.readlines():
            data = line.split()
            start = float(data[1])/10.0
            end = float(data[2])/10.0
            arc = Arc()
            arc.filename = data[0]
            arc.setStart(self.__j2000ToDatetime(start, 86400.0/2.0))
            arc.setStop(self.__j2000ToDatetime(end,86400.0/2.0))
            self.arclist.append(arc)

    def getArc(self,time):
        """Given a datetime object, determine the first arc number that contains precise ephemeris"""
        inRange = []
        # Make a list containing all of the
        # arcs that span <code>time</code>
        for arc in self.arclist:
            if (arc.inRange(time)):
                inRange.append(arc)

        if (len(inRange) == 0):
            self.logger.error("No valid arcs found spanning %s" % (time))
        if (len(inRange) > 0):
            self.logger.info("%s valid arcs found spanning %s" % (len(inRange),time))

        return inRange[0].filename

    def getOrbitFile(self,time):
        filename = self.getArc(time)
        return filename

    def __j2000ToDatetime(self,j2000Day,tdt):
        """Convert the number of days since 1 Jan. 2000 to a datetime object"""
        j2000 = datetime.datetime(year=2000,month=1,day=1)
        dt = j2000 + datetime.timedelta(days=j2000Day,seconds=tdt)
        return dt

class Arc(object):
    """A class representing an orbital arc segment"""

    def __init__(self):
        self.filename = None
        self._start = None
        self._stop = None

    def getStart(self):
        return self._start

    @type_check(datetime.datetime)
    def setStart(self,start):
        self._start = start

    def getStop(self):
        return self._stop

    @type_check(datetime.datetime)
    def setStop(self,stop):
        self._stop = stop

    def inRange(self, time):
        """Determine whether a time stamp lies within the
        start and stop times"""
        return self._start <= time <= self._stop


    start = property(fget=getStart,fset=setStart)
    stop = property(fget=getStop,fset=setStop)

    pass
