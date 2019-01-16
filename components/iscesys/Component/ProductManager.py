#!/usr/bin/env python3
import isce
from iscesys.Component.Configurable import Configurable

INSTANCE = Configurable.Facility('_instance',
                                       public_name='instance',
                                       factory='default',
                                       mandatory=True,
                                       private=True,
                                       doc='Container facility for object to load or dump')

class ProductManager(Configurable):
    facility_list = (
                       INSTANCE,
                     )
    family = 'productmanager'
    def __init__(self,family='', name=''):
        super(ProductManager, self).__init__(family if family else  self.__class__.family, name=name)
        
    def dumpProduct(self,obj,filename):
        self._instance = obj
        self.dump(filename)
        
    def loadProduct(self,filename):
        self.load(filename)
        return self._instance
        
    
