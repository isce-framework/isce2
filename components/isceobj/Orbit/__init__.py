def createOrbit(name=None):
    from .Orbit import Orbit
    return Orbit()
def getFactoriesInfo():
    """
    Returns a dictionary with information on how to create an object Sensor from its factory
    """
    return  {'Orbit':
                     {
                     'factory':'createOrbit'
                     }
              }    
