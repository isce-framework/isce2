#Author: Cunren Liang, 2015-

def createSwath():
    from .Swath import Swath
    return Swath()

def createFrame():
    from .Frame import Frame
    return Frame()

def createTrack():
    from .Track import Track
    return Track()


def createALOS2(name=None):
    from .ALOS2 import ALOS2
    return ALOS2()


SENSORS = {
             'ALOS2' : createALOS2,
          }

def getFactoriesInfo():
    """
    Returns a dictionary with information on how to create an object Sensor from its factory
    """
    return  {'MultiModeSensor':
                     {'args':
                           {
                            'sensor':{'value':list(SENSORS.keys()),'type':'str','optional':False}
                            },
                     'factory':'createSensor'
                     }
              }



def createSensor(sensor='', name=None):
    
    try:
        cls = SENSORS[str(sensor).upper()]
        try:
            instance = cls(name)
        except AttributeError:
            raise TypeError("'sensor name'=%s  cannot be interpreted" %
                            str(sensor))
        pass
    except:
        print("Sensor type not recognized. Valid Sensor types:\n",
              SENSORS.keys())
        instance = None
        pass
    return instance
