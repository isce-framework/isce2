#!/usr/bin/env python3
def createPlanet(pname,name=''):
    from isceobj.Planet.Planet import Planet
    return Planet(name=name,pname=pname)
