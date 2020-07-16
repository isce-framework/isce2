#!/usr/bin/env python3
import isce
import sys
from isceobj.Util.test import testInterpolator as ti
from isceobj.Util.PolyFactory import createPoly
def main():
    from iscesys.Parsers.FileParserFactory import createFileParser
    from isceobj import createImage
    parser = createFileParser('xml')
    #get the properties from the file init file
    prop, fac, misc = parser.parse(sys.argv[1])
    #this dictionary has an initial dummy key whose value is the dictionary with all the properties

    image  = createImage()
    image.init(prop,fac,misc)
    
    #create the params
    azOrder = 2
    rgOrder = 3
    cnt = 0.0
    params = [[0 for x in range(rgOrder+1)] for x in range(azOrder+1)]
    paramsaz = [0 for x in range(azOrder+1)]
    for  i in range(azOrder + 1):
        paramsaz[i] = cnt
        for j in range(rgOrder + 1):
            params[i][j] =  cnt
            cnt = cnt+1
    #create a 2d accessor 
    p2d = createPoly('2d',name='test')
    p2d.initPoly(rgOrder,azOrder, coeffs = params,image=image)

    #create a 1d accessor for azimuth poly (direction = 'y')
    p1d = createPoly('1d',name='test')
    p1d.initPoly(azOrder, coeffs = paramsaz,image=image,direction='y')

    #call the test
    p2d.dump('p2d.xml')
    p1d.dump('p1d.xml')

    ti.testInterpolator(p2d._accessor,p1d._accessor)
    
    p2dNew = createPoly('2d',name='test')
    #create a 1d accessor for azimuth poly (direction = 'y')
    p1dNew = createPoly('1d',name='test')
    #call the test
    p2dNew.load('p2d.xml')
    p1dNew.load('p1d.xml')
    ti.testInterpolator(p2dNew._accessor,p1dNew._accessor)


if __name__ == '__main__':
    sys.exit(main())
