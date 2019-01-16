
import re
import datetime
from isceobj.Orbit.Orbit import Orbit,StateVector

class PDS(object):

    def __init__(self,file=None):
        self.filename = file
        self.firstEpoch = 0
        self.lastEpoch = 0
        self.orbit = Orbit()
        self.orbit.configure()
        self.orbit.setOrbitSource('PDS')

    def getOrbit(self):
        return self.orbit

    def parse(self):
        fp = open(self.filename,'r')
        for line in fp.readlines():
            if (line[0].isdigit()):
                self.__parseStateVectorLine(line)
            else:
                self.__parseRecordLine(line)
        fp.close()


    def __parseRecordLine(self,line):
        line = line.strip()
        if (line.startswith('START_TIME')):
            values = line.split('=')
            values[1] = values[1].strip('"')
            dateTime = values[1].split()
            self.firstEpoch = self.__parseDateTimeString(dateTime[0],dateTime[1])
        elif (line.startswith('STOP_TIME')):
            values = line.split('=')
            values[1] = values[1].strip('"')
            dateTime = values[1].split()
            self.lastEpoch = self.__parseDateTimeString(dateTime[0],dateTime[1])
        elif (line.startswith('LEAP_UTC')):
            pass
        elif (line.startswith('LEAP_SIGN')):
            pass
        elif (line.startswith('RECORD_SIZE')):
            pass
        elif (line.startswith('NUM_REC')):
            pass

    def __parseStateVectorLine(self,line):
        date = line[0:11]
        time = line[12:27]
        x = float(line[44:56])
        y = float(line[57:69])
        z = float(line[70:82])
        vx = float(line[83:95])
        vy = float(line[96:108])
        vz = float(line[109:121])

        dt = self.__parseDateTimeString(date,time)
        
        sv = StateVector()
        sv.configure()
        sv.setTime(dt)
        sv.setPosition([x,y,z])
        sv.setVelocity([vx,vy,vz])
        self.orbit.addStateVector(sv)
    
    def __parseDateTimeString(self,date,time):
        """
        Fix idiosyncrasies in the date and time strings
        """
        time = time.replace('-','0') # For some reason, there are occasionally - signs where there should be zeros
        dt = datetime.datetime.strptime(date + ' ' + time,'%d-%b-%Y %H:%M:%S.%f')
        return dt
