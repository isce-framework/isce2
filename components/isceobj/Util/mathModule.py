#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                                  Giangi Sacco
#                        NASA Jet Propulsion Laboratory
#                      California Institute of Technology
#                      (C) 2009-2010  All Rights Reserved
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from __future__ import print_function
import sys
import os
import getopt
import math
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()


class MathModule:

    @staticmethod
    def is_power2(num):
        '''
        http://code.activestate.com/recipes/577514-chek-if-a-number-is-a-power-of-two/
        '''
        return  num != 0 and ((num & (num-1)) == 0)

    @staticmethod
    def nint(x):
        """nint(x) returns nearest integer value to x.  Ambiguity resolution:  nint(+0.5) = 1, nint(-0.5) = -1."""
        return int(x+math.copysign(0.5,x))

    @staticmethod
    def multiplyMatrices(mat1,mat2):
        row1 = len(mat1)
        col1 = len(mat1[0])
        row2 = len(mat2)
        col2 = len(mat2[0])
        if not (col1 == row2):
            print("Error. Number of columns in first matrix has to match the number of rows in second matrix")
            raise Exception
        retMat = [[0 for i in range(col2)] for j in range(row1)]
        for i in range(row1):
            for j in range(col2):
                for k in range(col1):
                    retMat[i][j] += mat1[i][k]*mat2[k][j]
        return retMat

    @staticmethod
    def invertMatrix(mat):

        a11 = mat[0][0]
        a12 = mat[0][1]
        a13 = mat[0][2]
        a21 = mat[1][0]
        a22 = mat[1][1]
        a23 = mat[1][2]
        a31 = mat[2][0]
        a32 = mat[2][1]
        a33 = mat[2][2]

        det = a11*(a22*a33 - a32*a23)+a21*(a32*a13 - a12*a33)+a31*(a12*a23 - a22*a13)
        matI = [[0 for i in range(3)] for j in range(3)]
        matI[0][0] = 1/float(det)*(a22*a33-a23*a32)
        matI[0][1] = 1/float(det)*(a13*a32-a12*a33)
        matI[0][2] = 1/float(det)*(a12*a23-a13*a22)
        matI[1][0] = 1/float(det)*(a23*a31-a21*a33)
        matI[1][1] = 1/float(det)*(a11*a33-a13*a31)
        matI[1][2] = 1/float(det)*(a13*a21-a11*a23)
        matI[2][0] = 1/float(det)*(a21*a32-a22*a31)
        matI[2][1] = 1/float(det)*(a12*a31-a11*a32)
        matI[2][2] = 1/float(det)*(a11*a22-a12*a21)
        return matI

    @staticmethod
    def matrixTranspose(mat):
        """Calculate the transpose of a matrix"""
        row = len(mat)
        col = len(mat[0])

        retMat = [[0 for i in range(row)] for j in range(col)]
        for i in range(row):
            for j in range(col):
                retMat[i][j] = mat[j][i]

        return retMat

    @staticmethod
    def matrixVectorProduct(mat,vec):
        """Calculate the matrix-vector product mat*vec"""
        row1 = len(mat)
        col1 = len(mat[0])
        row2 = len(vec)

        if not (col1 == row2):
            print("Error. Number of columns in first matrix has to match the number of rows in the vector")
            raise Exception
        retVec = [0 for i in range(row1)]
        for i in range(row1):
            for k in range(col1):
                retVec[i] += mat[i][k]*vec[k]

        return retVec

    @staticmethod
    def crossProduct(v1,v2):
        if (not len(v1) == len(v2)) or (not len(v1) == 3)  :
            print ("Error in crossProduct. The two vectors need to have same size = 3.")
            raise Exception
        v =[0,0,0]
        v[0] = v1[1]*v2[2] - v1[2]*v2[1]
        v[1] = v1[2]*v2[0] - v1[0]*v2[2]
        v[2] = v1[0]*v2[1] - v1[1]*v2[0]
        return v

    @staticmethod
    def normalizeVector(v1):
        norm = MathModule.norm(v1);
        vret = [0]*len(v1)
        for i in range(0,len(v1)):
            vret[i] = v1[i]/norm
        return vret

    @staticmethod
    def norm(v1):
        sum = 0
        for i in range(0,len(v1)):
            sum = sum + v1[i]*v1[i]
        return math.sqrt(sum)


    @staticmethod
    def dotProduct(v1,v2):
        if (not len(v1) == len(v2)):
            print("Error in crossProduct. The two vectors need to have same size.")
            raise Exception
        sum = 0
        for i in range(0,len(v1)):
            sum = sum + v1[i]*v2[i]
        return sum


    @staticmethod
    def median( list):
        list.sort()
        median = 0
        length = len(list)
        if(not length == 0):
            if((length%2) == 0):

                median = (list[length/2] + list[length/2 - 1])/2.0

            else:

                median = list[int(length/2)]

        return median


    @staticmethod
    def mean(list):
        return sum(list)/len(list)

    @staticmethod
    def linearFit(x,y):
        """
        Fit a line

        @param x a list of numbers representing the abscissa values
        @param y a list of number representing the ordinate values
        @return a tuple consisting of the intercept, slope, and standard deviation
        """
#        if len(x) == 0:
#            import pdb
#            pdb.set_trace()
        avgX = sum(x) / len(x)
        avgY = sum(y) / len(x)

        slopeNum = 0.0
        slopeDenom = 0.0
        for i in range(len(x)):
            slopeNum   += (x[i]-avgX)*(y[i]-avgY)
            slopeDenom += (x[i]-avgX)*(x[i]-avgX)

        slope = slopeNum / slopeDenom
        intercept = avgY - slope * avgX

        sumErr = 0.0
        for i in range(len(x)):
            sumErr += (y[i]-(intercept+slope*x[i]))**2;

        stdDev = math.sqrt( sumErr / len(x) )

        return intercept, slope, stdDev

    @staticmethod
    def quadraticFit(x,y):
        """
        Fit a parabola

        @param x a list of numbers representing the abscissa values
        @param y a list of number representing the ordinate values
        @return a tuple consisting of the constant, linear, and quadratic polynomial coefficients
        """
        sumX = [0,0,0,0,0]
        sumYX = [0,0,0]

        for i in range(len(x)):
            sumX[0]  += 1.0
            sumX[1]  += x[i]
            sumX[2]  += x[i]**2
            sumX[3]  += x[i]**3
            sumX[4]  += x[i]**4
            sumYX[0] += y[i]
            sumYX[1] += y[i] * x[i]
            sumYX[2] += y[i] * x[i] * x[i]

        A = [[sumX[0], sumX[1], sumX[2]],
             [sumX[1], sumX[2], sumX[3]],
             [sumX[2], sumX[3], sumX[4]]]

        inversed = MathModule.invertMatrix(A)

        a = inversed[0][0] * sumYX[0] + inversed[1][0] * sumYX[1] + inversed[2][0] * sumYX[2]
        b = inversed[0][1] * sumYX[0] + inversed[1][1] * sumYX[1] + inversed[2][1] * sumYX[2]
        c = inversed[0][2] * sumYX[0] + inversed[1][2] * sumYX[1] + inversed[2][2] * sumYX[2]

        return a, b, c

    def __init__(self):
        return


# end class

is_power2 = MathModule().is_power2
nint = MathModule().nint
