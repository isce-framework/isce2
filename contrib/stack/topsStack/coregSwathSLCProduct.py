import isce
import isceobj
from iscesys.Component.Component import Component
from isceobj.Sensor.TOPS.TOPSSwathSLCProduct import TOPSSwathSLCProduct

REFERENCE = Component.Facility('reference',
             public_name = 'reference',
             module = 'isceobj.Sensor.TOPS',
             factory = 'createTOPSSwathSLCProduct',
             args=(),
             mandatory = True,
             doc = 'reference of the stack to be coregistered to')

SOURCE = Component.Facility('source',
             public_name = 'source',
             module = 'isceobj.Sensor.TOPS',
             factory = 'createTOPSSwathSLCProduct',
             args=(),
             mandatory = True,
             doc = 'original source of the image before coregistration')


###############
'''

adding reference and source to TOPSSwathSLCProduct and name the new instance coregSwathSLCProduct. 
This way we can store the source(before coregistration) and the refernce (stack master) images.

'''
class coregSwathSLCProduct(TOPSSwathSLCProduct):

    
    facility_list = TOPSSwathSLCProduct.facility_list + (REFERENCE, SOURCE)
    
    def __init__(self,name=''):
        super(coregSwathSLCProduct, self).__init__(name=name)
        return None      


