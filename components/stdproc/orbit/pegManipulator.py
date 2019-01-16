#!/usr/bin/env python3
from isceobj.Location.Peg import Peg


def averagePeg(pegList, planet):
    '''Computes the average of a given list of pegpoints.'''
    nPeg = len(pegList)
    elp = planet.get_elp()
    avgPeg = Peg()
    for attribute in ['latitude', 'longitude', 'heading']:
        setattr(avgPeg, attribute, sum([getattr(pegPt, attribute) for pegPt in pegList])/(1.0*nPeg))

    
    avgPeg.updateRadiusOfCurvature(elp)
    
    return avgPeg 


def medianPeg(pegList, planet):
    '''Computes the median of a given list of pegpoints.'''
    import numpy
    elp = planet.get_elp()
    medPeg = Peg()
    nPeg = len(peglist)
    for attribute in ['latitude', 'longitude', 'heading']:
        setattr(medPeg, attribute,numpy.median([getattr(pegPt, attribute) for pegPt in pegList]))

    medPeg.updateRadiusOfCurvature(elp)
    
    return medPeg 


