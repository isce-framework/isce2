#!/usr/bin/env python3

import os
import numpy as np
import datetime
from isceobj.Orbit.Orbit import StateVector, Orbit


class ECI2ECR(object):
    '''
    Class for converting Inertial orbits to ECEF orbits using GAST.
    Reference: "Digital Surface Modelling in Developing Countries Using Spaceborne SAR Techniques" by Earl Peter Fitz-Gerald Edwards, 2005.
    '''

    LengthOfDayFactor = 1.002737822
    Omega = 2 * LengthOfDayFactor * np.pi/86400.0

    def __init__(self, orbit, GAST=None, epoch=None):
        '''
        GAST should be provided in mean hour angle in degrees.
        '''

        if GAST is None:
            raise Exception('GAST value needs to be provided for conversion.')

        self.referenceGAST = np.radians(GAST)
        self.referenceEpoch = epoch
        self.orbit = orbit


    def convert(self):
        '''
        Convert ECI orbit to ECEF orbit.
        '''

        ECROrbit = Orbit()
        ECROrbit.configure()

        for sv in self.orbit:
            svtime = sv.getTime()
            position = sv.getPosition()
            velocity = sv.getVelocity()

            ####Compute GMST from GAST - Eq 5.13
            dtiff = (svtime -  self.referenceEpoch).total_seconds()
            theta = self.referenceGAST + self.Omega * dtiff

   
            costh = np.cos(theta)
            sinth = np.sin(theta)

            ###Position transformation
            A = np.zeros((3,3))
            A[0,0] = costh
            A[0,1] = sinth
            A[1,0] = -sinth
            A[1,1] = costh
            A[2,2] = 1

            ###Velocity transformation
            Adot = np.zeros((3,3))
            Adot[0,0] = -self.Omega * sinth
            Adot[0,1] = self.Omega * costh
            Adot[1,0] = -self.Omega * costh
            Adot[1,1] = -self.Omega * sinth

            
            ###Compute ECR state vector
            newPos = np.dot(A, position)
            newVel = np.dot(Adot, position) + np.dot(A, velocity)

            ####Create state vector object
            newsv = StateVector()
            newsv.setTime(svtime)
            newsv.setPosition(newPos.tolist())
            newsv.setVelocity(newVel.tolist())

            ###Append to orbit
            ECROrbit.addStateVector(newsv)

        ECROrbit.setOrbitSource( 'Sidereal angle conversion')
        ECROrbit.setOrbitQuality( self.orbit.getOrbitQuality() )
        return ECROrbit
