#/usr/bin/env python3
import isce
from .ProductManager import ProductManager as PM
__prdManager = PM('productmanager_name')
__prdManager.configure()

def dump(obj,filename):
    __prdManager.dumpProduct(obj,filename)

def load(filename):
    return __prdManager.loadProduct(filename)
