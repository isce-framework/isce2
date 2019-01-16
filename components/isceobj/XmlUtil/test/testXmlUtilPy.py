#!/usr/bin/env python3
from __future__ import print_function
import sys
import xml.etree.ElementTree as ET 
from isceobj.XmlUtil.XmlUtil import XmlUtil
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()

def main():
    
    obj = XmlUtil()
    tree = obj.readFile('test1.xml')
    root = tree.getroot()
    print(len(root))
    ET.dump(root)
    #obj.indent(root)
    obj.writeFile('test2.xml',root)
    obj.writeFile('test3.xml',tree)
    '''
    print(root.findall('name'))
    ET.dump(tree)
    obj.createDictionary(tree)
    '''
    

if __name__ == "__main__":
    sys.exit(main())
