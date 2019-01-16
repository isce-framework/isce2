#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2010 California Institute of Technology. ALL RIGHTS RESERVED.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# United States Government Sponsorship acknowledged. This software is subject to
# U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
# (No [Export] License Required except when exporting to an embargoed country,
# end user, or in support of a prohibited end use). By downloading this software,
# the user agrees to comply with all applicable U.S. export laws and regulations.
# The user has the responsibility to obtain export licenses, or other export
# authority as may be required before exporting this software to any 'EAR99'
# embargoed foreign country or citizen of those countries.
#
# Author: Walter Szeliga
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




from iscesys.Component.Component import Component
PRF = Component.Parameter('prf',
        public_name='prf',
        default=None,
        type=float,
        mandatory=True,
        doc = 'The pulse repetition frequency [Hz]')

AMBIGUITY = Component.Parameter('ambiguity',
        public_name='ambiguity',
        default=0,
        type=float,
        mandatory=False,
        doc = 'The integer ambiguity of the Doppler centroid')

FRACTIONAL_CENTROID = Component.Parameter('fractionalCentroid',
        public_name='fractionalCentroid',
        default=0,
        type=float,
        mandatory=False,
        intent='output',
        doc = 'The fractional part of the Doppler centroid [Hz/PRF]')

LINEAR_TERM  = Component.Parameter('linearTerm',
        public_name='linearTerm',
        default=0,
        type=float,
        mandatory=False,
        intent='output',
        doc = 'The linear term in the Doppler vs. range polynomical [Hz/PRF]')

QUADRATIC_TERM = Component.Parameter('quadraticTerm',
        public_name='quadraticTerm',
        default=0,
        type=float,
        mandatory=False,
        intent='output',
        doc = 'Quadratic Term')

CUBIC_TERM = Component.Parameter('cubicTerm',
        public_name='cubicTerm',
        default=0,
        type=float,
        mandatory=False,
        intent='output',
        doc = 'cubicTerm The cubic term in the Doppler vs. range polynomical [Hz/PRF]')

COEFS = Component.Parameter('coefs',
        public_name='coefs',
        default=[],
        container=list,
        type=float,
        mandatory=False,
        intent='output',
        doc = 'List of the doppler coefficients')

class Doppler(Component):

    family = 'doppler'

    parameter_list = (
                      PRF,
                      AMBIGUITY,
                      FRACTIONAL_CENTROID,
                      LINEAR_TERM,
                      QUADRATIC_TERM,
                      CUBIC_TERM,
                      COEFS
                     )

    def __init__(self,family=None,name=None,prf=0):
        super(Doppler, self).__init__(
            family=family if family else  self.__class__.family, name=name)
        """A class to hold Doppler polynomial coefficients.

        @note The polynomial is expected to be referenced to range bin.

        @param prf The pulse repetition frequency [Hz]
        @param ambigutiy The integer ambiguity of the Doppler centroid
        @param fractionalCentroid The fractional part of the Doppler centroid
        [Hz/PRF]
        @param linearTerm The linear term in the Doppler vs. range polynomical
        [Hz/PRF]
        @param quadraticTerm The quadratic term in the Doppler vs. range
        polynomical [Hz/PRF]
        @param cubicTerm The cubic term in the Doppler vs. range polynomical
        [Hz/PRF]
        """
        self.prf = prf
        self.numCoefs = 4
        return

    def getDopplerCoefficients(self,inHz=False):
        """Get the Doppler polynomial coefficients as a function of range,
        optionally scaled by the PRF.

        @param inHz (\a boolean) True if the returned coefficients should
        have units of Hz, False if the "units" should be Hz/PRF
        @return the Doppler polynomial coefficients as a function of range.
        """

        coef = [self.ambiguity+self.fractionalCentroid]
        coef += self.coefs[1:]

        if inHz:
            coef = [x*self.prf for x in coef]

        return coef

    def setDopplerCoefficients(self, coef, ambiguity=0, inHz=False):
        """Set the Doppler polynomial coefficients as a function of range.

        @param coef a list containing the cubic polynomial Doppler
        coefficients as a function of range
        @param ambiguity (\a int) the absolute Doppler ambiguity
        @param inHz (\a boolean) True if the Doppler coefficients have units
        of Hz, False if the "units" are Hz/PRF
        """
        self.coefs = coef   #for code that handles higher order polynomials
                            #while continuing to support code that uses the quadratic
        self.numCoefs = len(coef)

        if inHz and (self.prf != 0.0):
            coef = [x/self.prf for x in coef]
            self.coefs = [x/self.prf for x in self.coefs]

        self.fractionalCentroid = coef[0] - self.ambiguity
        self.linearTerm = coef[1]
        self.quadraticTerm = coef[2]
        self.cubicTerm = coef[3]

    def average(self, *others):
        """Average my Doppler with other Doppler objects"""
        from operator import truediv
        n = 1 + len(others)
        prfSum = self.prf
        coefSum = self.getDopplerCoefficients(inHz=True)
        for e in others:
            prfSum += e.prf
            otherCoef = e.getDopplerCoefficients(inHz=True)
            for i in range(self.numCoefs): coefSum[i] += otherCoef[i]

        prf = truediv(prfSum, n)
        coef = [truediv(coefSum[i], n) for i in range(self.numCoefs)]
        averageDoppler = self.__class__(prf=prf)
        averageDoppler.setDopplerCoefficients(coef, inHz=True)

        return averageDoppler

    def evaluate(self, rangeBin=0, inHz=False):
        """Calculate the Doppler in a particular range bin by evaluating the
        Doppler polynomial."""
        dop = (
            (self.ambiguity + self.fractionalCentroid) +
            self.linearTerm*rangeBin +
            self.quadraticTerm*rangeBin**2 + self.cubicTerm*rangeBin**3
            )

        if inHz:
            dop = dop*self.prf

        return dop

    ## An obvious overload?
    def __call__(self, rangeBin=0, inHz=False):
        return self.evaluate(rangeBin=rangeBin, inHz=inHz)

    ## Convert to a standard numpy.poly1d object
    def poly1d(self, inHz=False):
        from numpy import poly1d, array
        if inHz:
            factor = 1./self.prf
            variable = 'Hz'
        else:
            factor = 1.
            variable = 'PRF'

        return poly1d(array([
                    self.cubicTerm,
                    self.quadraticTerm,
                    self.linearTerm,
                    (self.ambiguity + self.fractionalCentroid)
                    ]) * factor, variable=variable)

    def __getstate__(self):
        d = dict(self.__dict__)
        return d

    def __setstate__(self,d):
        self.__dict__.update(d)

        #For backwards compatibility with old PICKLE files that do not
        #contain the coefs attribute and contain named coefficients only.
        if not hasattr(self, 'coefs'):
            coef = [self.ambiguity+self.fractionalCentroid,
                    self.linearTerm,
                    self.quadraticTerm,
                    self.cubicTerm]
            self.coefs  = coef
        return

    def __str__(self):
        retstr = "PRF: %s\n"
        retlst = (self.prf,)
        retstr += "Ambiguity: %s\n"
        retlst += (self.ambiguity,)
        retstr += "Centroid: %s\n"
        retlst += (self.fractionalCentroid,)
        retstr += "Linear Term: %s\n"
        retlst += (self.linearTerm,)
        retstr += "Quadratic Term: %s\n"
        retlst += (self.quadraticTerm,)
        retstr += "Cubic Term: %s\n"
        retlst += (self.cubicTerm,)
        retstr += "All coefficients: %r\n"
        retlst += (self.coefs,)
        return retstr % retlst
