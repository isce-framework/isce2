#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2019 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Yang Lei
#
# Note: this is based on the MATLAB code, "auto-RIFT", written by Alex Gardner,
#       and has been translated to Python and further optimized.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



import pdb
import subprocess
import re
import string
import sys



class autoRIFT:
    '''
    Class for mapping regular geographic grid on radar imagery.
    '''
    
    
    
    def preprocess_filt_wal(self):
        '''
        Do the pre processing using wallis filter (10 min vs 15 min in Matlab).
        '''
        import cv2
        import numpy as np
#        import scipy.io as sio
        
        
        if self.zeroMask is not None:
            self.zeroMask = (self.I1 == 0)
        
        kernel = np.ones((self.WallisFilterWidth,self.WallisFilterWidth), dtype=np.float32)
        
        m = cv2.filter2D(self.I1,-1,kernel,borderType=cv2.BORDER_CONSTANT)/np.sum(kernel)
        
        m2 = (self.I1)**2
        
        m2 = cv2.filter2D(m2,-1,kernel,borderType=cv2.BORDER_CONSTANT)/np.sum(kernel)
    
        s = np.sqrt(m2 - m**2) * np.sqrt(np.sum(kernel)/(np.sum(kernel)-1.0))
    
        self.I1 = (self.I1 - m) / s
        
#        pdb.set_trace()

        m = cv2.filter2D(self.I2,-1,kernel,borderType=cv2.BORDER_CONSTANT)/np.sum(kernel)
        
        m2 = (self.I2)**2
        
        m2 = cv2.filter2D(m2,-1,kernel,borderType=cv2.BORDER_CONSTANT)/np.sum(kernel)
        
        s = np.sqrt(m2 - m**2) * np.sqrt(np.sum(kernel)/(np.sum(kernel)-1.0))
        
        self.I2 = (self.I2 - m) / s
    
    
    
#    ####   obsolete definition of "preprocess_filt_hps"
#    def preprocess_filt_hps(self):
#        '''
#        Do the pre processing using (orig - low-pass filter) = high-pass filter filter (3.9/5.3 min).
#        '''
#        import cv2
#        import numpy as np
#
#        if self.zeroMask is not None:
#            self.zeroMask = (self.I1 == 0)
#
#        kernel = np.ones((self.WallisFilterWidth,self.WallisFilterWidth), dtype=np.float32)
#
#        lp = cv2.filter2D(self.I1,-1,kernel,borderType=cv2.BORDER_CONSTANT)/np.sum(kernel)
#
#        self.I1 = (self.I1 - lp)
#
#        lp = cv2.filter2D(self.I2,-1,kernel,borderType=cv2.BORDER_CONSTANT)/np.sum(kernel)
#
#        self.I2 = (self.I2 - lp)

    
    def preprocess_filt_hps(self):
        '''
        Do the pre processing using (orig - low-pass filter) = high-pass filter filter (3.9/5.3 min).
        '''
        import cv2
        import numpy as np
        

        if self.zeroMask is not None:
            self.zeroMask = (self.I1 == 0)

        kernel = -np.ones((self.WallisFilterWidth,self.WallisFilterWidth), dtype=np.float32)

        kernel[int((self.WallisFilterWidth-1)/2),int((self.WallisFilterWidth-1)/2)] = kernel.size - 1

        kernel = kernel / kernel.size

#        pdb.set_trace()

        self.I1 = cv2.filter2D(self.I1,-1,kernel,borderType=cv2.BORDER_CONSTANT)

        self.I2 = cv2.filter2D(self.I2,-1,kernel,borderType=cv2.BORDER_CONSTANT)


    
    def preprocess_db(self):
        '''
        Do the pre processing using db scale (4 min).
        '''
        import cv2
        import numpy as np
        
        
        if self.zeroMask is not None:
            self.zeroMask = (self.I1 == 0)
        
#        pdb.set_trace()

        self.I1 = 20.0 * np.log10(self.I1)
        
        self.I2 = 20.0 * np.log10(self.I2)
        
        
        
        
    def preprocess_filt_sob(self):
        '''
        Do the pre processing using sobel filter (4.5/5.8 min).
        '''
        import cv2
        import numpy as np
        
        
        
        if self.zeroMask is not None:
            self.zeroMask = (self.I1 == 0)
        
        sobelx = cv2.getDerivKernels(1,0,self.WallisFilterWidth)
        
        kernelx = np.outer(sobelx[0],sobelx[1])
        
        sobely = cv2.getDerivKernels(0,1,self.WallisFilterWidth)
        
        kernely = np.outer(sobely[0],sobely[1])
        
        kernel = kernelx + kernely
        
        self.I1 = cv2.filter2D(self.I1,-1,kernel,borderType=cv2.BORDER_CONSTANT)
        
        self.I2 = cv2.filter2D(self.I2,-1,kernel,borderType=cv2.BORDER_CONSTANT)
        
        
        

        

    
    

    def preprocess_filt_lap(self):
        '''
        Do the pre processing using Laplacian filter (2.5 min / 4 min).
        '''
        import cv2
        import numpy as np
        
        
        if self.zeroMask is not None:
            self.zeroMask = (self.I1 == 0)

        self.I1 = 20.0 * np.log10(self.I1)
        self.I1 = cv2.Laplacian(self.I1,-1,ksize=self.WallisFilterWidth,borderType=cv2.BORDER_CONSTANT)

        self.I2 = 20.0 * np.log10(self.I2)
        self.I2 = cv2.Laplacian(self.I2,-1,ksize=self.WallisFilterWidth,borderType=cv2.BORDER_CONSTANT)
    





        
        
        
    def uniform_data_type(self):
        
        import numpy as np
        
        if self.DataType == 0:
#            validData = np.logical_not(np.isnan(self.I1))
            validData = np.isfinite(self.I1)
            S1 = np.std(self.I1[validData])*np.sqrt(self.I1[validData].size/(self.I1[validData].size-1.0))
            M1 = np.mean(self.I1[validData])
            self.I1 = (self.I1 - (M1 - 3*S1)) / (6*S1) * (2**8 - 0)
            
#            self.I1[np.logical_not(np.isfinite(self.I1))] = 0
            self.I1 = np.round(np.clip(self.I1, 0, 255)).astype(np.uint8)

#            validData = np.logical_not(np.isnan(self.I2))
            validData = np.isfinite(self.I2)
            S2 = np.std(self.I2[validData])*np.sqrt(self.I2[validData].size/(self.I2[validData].size-1.0))
            M2 = np.mean(self.I2[validData])
            self.I2 = (self.I2 - (M2 - 3*S2)) / (6*S2) * (2**8 - 0)
            
#            self.I2[np.logical_not(np.isfinite(self.I2))] = 0
            self.I2 = np.round(np.clip(self.I2, 0, 255)).astype(np.uint8)
            
            if self.zeroMask is not None:
                self.I1[self.zeroMask] = 0
                self.I2[self.zeroMask] = 0
                self.zeroMask = None
        
        elif self.DataType == 1:
            self.I1[np.logical_not(np.isfinite(self.I1))] = 0
            self.I2[np.logical_not(np.isfinite(self.I2))] = 0
            self.I1 = self.I1.astype(np.float32)
            self.I2 = self.I2.astype(np.float32)
            
            if self.zeroMask is not None:
                self.I1[self.zeroMask] = 0
                self.I2[self.zeroMask] = 0
                self.zeroMask = None

        else:
            sys.exit('invalid data type for the image pair which must be unsigned integer 8 or 32-bit float')


    def autorift(self):
        '''
        Do the actual processing.
        '''
        import numpy as np
        import cv2
        from scipy import ndimage


        ChipSizeUniX = np.unique(np.append(np.unique(self.ChipSizeMinX), np.unique(self.ChipSizeMaxX)))
        ChipSizeUniX = np.delete(ChipSizeUniX,np.where(ChipSizeUniX == 0)[0])

        if np.any(np.mod(ChipSizeUniX,self.ChipSize0X) != 0):
            sys.exit('chip sizes must be even integers of ChipSize0')

        ChipRangeX = self.ChipSize0X * np.array([1,2,4,8,16,32,64],np.float32)
#        ChipRangeX = ChipRangeX[ChipRangeX < (2**8 - 1)]
        if np.max(ChipSizeUniX) > np.max(ChipRangeX):
            sys.exit('max each chip size is out of range')

        ChipSizeUniX = ChipRangeX[(ChipRangeX >= np.min(ChipSizeUniX)) & (ChipRangeX <= np.max(ChipSizeUniX))]

        maxScale = np.max(ChipSizeUniX) / self.ChipSize0X

        if (np.mod(self.xGrid.shape[0],maxScale) != 0)|(np.mod(self.xGrid.shape[1],maxScale) != 0):
            message = 'xgrid and ygrid have an incorect size ' + str(self.xGrid.shape) + ' for nested search, they must have dimensions that an interger multiple of ' + str(maxScale)
            sys.exit(message)
        
        self.xGrid = self.xGrid.astype(np.float32)
        self.yGrid = self.yGrid.astype(np.float32)
        
        if np.size(self.Dx0) == 1:
            self.Dx0 = np.ones(self.xGrid.shape, np.float32) * np.round(self.Dx0)
        else:
            self.Dx0 = self.Dx0.astype(np.float32)
        if np.size(self.Dy0) == 1:
            self.Dy0 = np.ones(self.xGrid.shape, np.float32) * np.round(self.Dy0)
        else:
            self.Dy0 = self.Dy0.astype(np.float32)
        if np.size(self.SearchLimitX) == 1:
            self.SearchLimitX = np.ones(self.xGrid.shape, np.float32) * np.round(self.SearchLimitX)
        else:
            self.SearchLimitX = self.SearchLimitX.astype(np.float32)
        if np.size(self.SearchLimitY) == 1:
            self.SearchLimitY = np.ones(self.xGrid.shape, np.float32) * np.round(self.SearchLimitY)
        else:
            self.SearchLimitY = self.SearchLimitY.astype(np.float32)
        if np.size(self.ChipSizeMinX) == 1:
            self.ChipSizeMinX = np.ones(self.xGrid.shape, np.float32) * np.round(self.ChipSizeMinX)
        else:
            self.ChipSizeMinX = self.ChipSizeMinX.astype(np.float32)
        if np.size(self.ChipSizeMaxX) == 1:
            self.ChipSizeMaxX = np.ones(self.xGrid.shape, np.float32) * np.round(self.ChipSizeMaxX)
        else:
            self.ChipSizeMaxX = self.ChipSizeMaxX.astype(np.float32)

        ChipSizeX = np.zeros(self.xGrid.shape, np.float32)
        InterpMask = np.zeros(self.xGrid.shape, np.bool)
        Dx = np.empty(self.xGrid.shape, dtype=np.float32)
        Dx.fill(np.nan)
        Dy = np.empty(self.xGrid.shape, dtype=np.float32)
        Dy.fill(np.nan)

        Flag = 3

        for i in range(ChipSizeUniX.__len__()):
            
            # Nested grid setup: chip size being ChipSize0X no need to resize, otherwise has to resize the arrays
            if self.ChipSize0X != ChipSizeUniX[i]:
                Scale = self.ChipSize0X / ChipSizeUniX[i]
                dstShape = (int(self.xGrid.shape[0]*Scale),int(self.xGrid.shape[1]*Scale))
                xGrid0 = cv2.resize(self.xGrid.astype(np.float32),dstShape[::-1],interpolation=cv2.INTER_AREA)
                yGrid0 = cv2.resize(self.yGrid.astype(np.float32),dstShape[::-1],interpolation=cv2.INTER_AREA)
                
                if np.mod(ChipSizeUniX[i],2) == 0:
                    xGrid0 = np.round(xGrid0+0.5)-0.5
                    yGrid0 = np.round(yGrid0+0.5)-0.5
                else:
                    xGrid0 = np.round(xGrid0)
                    yGrid0 = np.round(yGrid0)

                M0 = (ChipSizeX == 0) & (self.ChipSizeMinX <= ChipSizeUniX[i]) & (self.ChipSizeMaxX >= ChipSizeUniX[i])
                M0 = colfilt(M0.copy(), (int(1/Scale*6), int(1/Scale*6)), 0)
                M0 = cv2.resize(np.logical_not(M0).astype(np.uint8),dstShape[::-1],interpolation=cv2.INTER_NEAREST).astype(np.bool)

                SearchLimitX0 = colfilt(self.SearchLimitX.copy(), (int(1/Scale), int(1/Scale)), 0) + colfilt(self.Dx0.copy(), (int(1/Scale), int(1/Scale)), 4)
                SearchLimitY0 = colfilt(self.SearchLimitY.copy(), (int(1/Scale), int(1/Scale)), 0) + colfilt(self.Dy0.copy(), (int(1/Scale), int(1/Scale)), 4)
                Dx00 = colfilt(self.Dx0.copy(), (int(1/Scale), int(1/Scale)), 2)
                Dy00 = colfilt(self.Dy0.copy(), (int(1/Scale), int(1/Scale)), 2)

                SearchLimitX0 = np.ceil(cv2.resize(SearchLimitX0,dstShape[::-1]))
                SearchLimitY0 = np.ceil(cv2.resize(SearchLimitY0,dstShape[::-1]))
                SearchLimitX0[M0] = 0
                SearchLimitY0[M0] = 0
                Dx00 = np.round(cv2.resize(Dx00,dstShape[::-1],interpolation=cv2.INTER_NEAREST))
                Dy00 = np.round(cv2.resize(Dy00,dstShape[::-1],interpolation=cv2.INTER_NEAREST))
#                pdb.set_trace()
            else:
                SearchLimitX0 = self.SearchLimitX.copy()
                SearchLimitY0 = self.SearchLimitY.copy()
                Dx00 = self.Dx0.copy()
                Dy00 = self.Dy0.copy()
                xGrid0 = self.xGrid.copy()
                yGrid0 = self.yGrid.copy()
#                M0 = (ChipSizeX == 0) & (self.ChipSizeMinX <= ChipSizeUniX[i]) & (self.ChipSizeMaxX >= ChipSizeUniX[i])
#                SearchLimitX0[np.logical_not(M0)] = 0
#                SearchLimitY0[np.logical_not(M0)] = 0

            if np.logical_not(np.any(SearchLimitX0 != 0)):
                continue

            idxZero = (SearchLimitX0 <= 0) | (SearchLimitY0 <= 0)
            SearchLimitX0[idxZero] = 0
            SearchLimitY0[idxZero] = 0
            SearchLimitX0[(np.logical_not(idxZero)) & (SearchLimitX0 < self.minSearch)] = self.minSearch
            SearchLimitY0[(np.logical_not(idxZero)) & (SearchLimitY0 < self.minSearch)] = self.minSearch

            if ((xGrid0.shape[0] - 2)/self.sparseSearchSampleRate < 5) | ((xGrid0.shape[1] - 2)/self.sparseSearchSampleRate < 5):
                Flag = 2
                return Flag
        
            # Setup for coarse search: sparse sampling / resize
            rIdxC = slice(self.sparseSearchSampleRate-1,xGrid0.shape[0],self.sparseSearchSampleRate)
            cIdxC = slice(self.sparseSearchSampleRate-1,xGrid0.shape[1],self.sparseSearchSampleRate)
            xGrid0C = xGrid0[rIdxC,cIdxC]
            yGrid0C = yGrid0[rIdxC,cIdxC]
            
#            pdb.set_trace()

            if np.remainder(self.sparseSearchSampleRate,2) == 0:
                filtWidth = self.sparseSearchSampleRate + 1
            else:
                filtWidth = self.sparseSearchSampleRate

            SearchLimitX0C = colfilt(SearchLimitX0.copy(), (int(filtWidth), int(filtWidth)), 0)
            SearchLimitY0C = colfilt(SearchLimitY0.copy(), (int(filtWidth), int(filtWidth)), 0)
            SearchLimitX0C = SearchLimitX0C[rIdxC,cIdxC]
            SearchLimitY0C = SearchLimitY0C[rIdxC,cIdxC]

            Dx0C = Dx00[rIdxC,cIdxC]
            Dy0C = Dy00[rIdxC,cIdxC]

            # Coarse search
            SubPixFlag = False
            ChipSizeXC = ChipSizeUniX[i]
            ChipSizeYC = np.float32(np.round(ChipSizeXC*self.ScaleChipSizeY/2)*2)
            
            if type(self.OverSampleRatio) is dict:
                overSampleRatio = self.OverSampleRatio[ChipSizeUniX[i]]
            else:
                overSampleRatio = self.OverSampleRatio
            
#            pdb.set_trace()

            if self.I1.dtype == np.uint8:
                DxC, DyC = arImgDisp_u(self.I2.copy(), self.I1.copy(), xGrid0C.copy(), yGrid0C.copy(), ChipSizeXC, ChipSizeYC, SearchLimitX0C.copy(), SearchLimitY0C.copy(), Dx0C.copy(), Dy0C.copy(), SubPixFlag, overSampleRatio)
            elif self.I1.dtype == np.float32:
                DxC, DyC = arImgDisp_s(self.I2.copy(), self.I1.copy(), xGrid0C.copy(), yGrid0C.copy(), ChipSizeXC, ChipSizeYC, SearchLimitX0C.copy(), SearchLimitY0C.copy(), Dx0C.copy(), Dy0C.copy(), SubPixFlag, overSampleRatio)
            else:
                sys.exit('invalid data type for the image pair which must be unsigned integer 8 or 32-bit float')
            
#            pdb.set_trace()

            # M0C is the mask for reliable estimates after coarse search, MC is the mask after disparity filtering, MC2 is the mask after area closing for fine search
            M0C = np.logical_not(np.isnan(DxC))

            DispFiltC = DISP_FILT()
            if ChipSizeUniX[i] == ChipSizeUniX[0]:
                DispFiltC.FracValid = self.FracValid
                DispFiltC.FracSearch = self.FracSearch
                DispFiltC.FiltWidth = self.FiltWidth
                DispFiltC.Iter = self.Iter
                DispFiltC.MadScalar = self.MadScalar
            DispFiltC.Iter = DispFiltC.Iter - 1
            MC = DispFiltC.filtDisp(DxC.copy(), DyC.copy(), SearchLimitX0C.copy(), SearchLimitY0C.copy(), M0C.copy(), overSampleRatio)

            MC[np.logical_not(M0C)] = False
    
            ROIC = (SearchLimitX0C > 0)
            CoarseCorValidFac = np.sum(MC[ROIC]) / np.sum(M0C[ROIC])
            if (CoarseCorValidFac < self.CoarseCorCutoff):
                continue
            
            MC2 = ndimage.distance_transform_edt(np.logical_not(MC)) < self.BuffDistanceC
            dstShape = (int(MC2.shape[0]*self.sparseSearchSampleRate),int(MC2.shape[1]*self.sparseSearchSampleRate))

            MC2 = cv2.resize(MC2.astype(np.uint8),dstShape[::-1],interpolation=cv2.INTER_NEAREST).astype(np.bool)
#            pdb.set_trace()
            if np.logical_not(np.all(MC2.shape == SearchLimitX0.shape)):
                rowAdd = SearchLimitX0.shape[0] - MC2.shape[0]
                colAdd = SearchLimitX0.shape[1] - MC2.shape[1]
                if rowAdd>0:
                    MC2 = np.append(MC2,MC2[-rowAdd:,:],axis=0)
                if colAdd>0:
                    MC2 = np.append(MC2,MC2[:,-colAdd:],axis=1)

            SearchLimitX0[np.logical_not(MC2)] = 0
            SearchLimitY0[np.logical_not(MC2)] = 0

            # Fine Search
            SubPixFlag = True
            ChipSizeXF = ChipSizeUniX[i]
            ChipSizeYF = np.float32(np.round(ChipSizeXF*self.ScaleChipSizeY/2)*2)
#            pdb.set_trace()
            if self.I1.dtype == np.uint8:
                DxF, DyF = arImgDisp_u(self.I2.copy(), self.I1.copy(), xGrid0.copy(), yGrid0.copy(), ChipSizeXF, ChipSizeYF, SearchLimitX0.copy(), SearchLimitY0.copy(), Dx00.copy(), Dy00.copy(), SubPixFlag, overSampleRatio)
            elif self.I1.dtype == np.float32:
                DxF, DyF = arImgDisp_s(self.I2.copy(), self.I1.copy(), xGrid0.copy(), yGrid0.copy(), ChipSizeXF, ChipSizeYF, SearchLimitX0.copy(), SearchLimitY0.copy(), Dx00.copy(), Dy00.copy(), SubPixFlag, overSampleRatio)
            else:
                sys.exit('invalid data type for the image pair which must be unsigned integer 8 or 32-bit float')

#            pdb.set_trace()

            DispFiltF = DISP_FILT()
            if ChipSizeUniX[i] == ChipSizeUniX[0]:
                DispFiltF.FracValid = self.FracValid
                DispFiltF.FracSearch = self.FracSearch
                DispFiltF.FiltWidth = self.FiltWidth
                DispFiltF.Iter = self.Iter
                DispFiltF.MadScalar = self.MadScalar

            
            M0 = DispFiltF.filtDisp(DxF.copy(), DyF.copy(), SearchLimitX0.copy(), SearchLimitY0.copy(), np.logical_not(np.isnan(DxF)), overSampleRatio)
#            pdb.set_trace()
            DxF[np.logical_not(M0)] = np.nan
            DyF[np.logical_not(M0)] = np.nan
            
            # Light interpolation with median filtered values: DxFM (filtered) and DxF (unfiltered)
            DxFM = colfilt(DxF.copy(), (self.fillFiltWidth, self.fillFiltWidth), 3)
            DyFM = colfilt(DyF.copy(), (self.fillFiltWidth, self.fillFiltWidth), 3)
            
            # M0 is mask for original valid estimates, MF is mask for filled ones, MM is mask where filtered ones exist for filling
            MF = np.zeros(M0.shape, dtype=np.bool)
            MM = np.logical_not(np.isnan(DxFM))

            for j in range(3):
                foo = MF | M0   # initial valid estimates
                foo1 = (cv2.filter2D(foo.astype(np.float32),-1,np.ones((3,3)),borderType=cv2.BORDER_CONSTANT) >= 6) | foo     # 1st area closing followed by the 2nd (part of the next line calling OpenCV)
#                pdb.set_trace()
                fillIdx = np.logical_not(bwareaopen(np.logical_not(foo1).astype(np.uint8), 5)) & np.logical_not(foo) & MM
                MF[fillIdx] = True
                DxF[fillIdx] = DxFM[fillIdx]
                DyF[fillIdx] = DyFM[fillIdx]
            
            # Below is for replacing the valid estimates with the bicubic filtered values for robust and accurate estimation
            if self.ChipSize0X == ChipSizeUniX[i]:
                Dx = DxF
                Dy = DyF
                ChipSizeX[M0|MF] = ChipSizeUniX[i]
                InterpMask[MF] = True
#                pdb.set_trace()
            else:
#                pdb.set_trace()
                Scale = ChipSizeUniX[i] / self.ChipSize0X
                dstShape = (int(Dx.shape[0]/Scale),int(Dx.shape[1]/Scale))
                
                # DxF0 (filtered) / Dx (unfiltered) is the result from earlier iterations, DxFM (filtered) / DxF (unfiltered) is that of the current iteration
                # first colfilt nans within 2-by-2 area (otherwise 1 nan will contaminate all 4 points)
                DxF0 = colfilt(Dx.copy(),(int(Scale+1),int(Scale+1)),2)
                # then resize to half size using area (similar to averaging) to match the current iteration
                DxF0 = cv2.resize(DxF0,dstShape[::-1],interpolation=cv2.INTER_AREA)
                DyF0 = colfilt(Dy.copy(),(int(Scale+1),int(Scale+1)),2)
                DyF0 = cv2.resize(DyF0,dstShape[::-1],interpolation=cv2.INTER_AREA)
                
                # Note this DxFM is almost the same as DxFM (same variable) in the light interpolation (only slightly better); however, only small portion of it will be used later at locations specified by M0 and MF that are determined in the light interpolation. So even without the following two lines, the final Dx and Dy result is still the same.
                # to fill out all of the missing values in DxF
                DxFM = colfilt(DxF.copy(), (5,5), 3)
                DyFM = colfilt(DyF.copy(), (5,5), 3)
                
                # fill the current-iteration result with previously determined reliable estimates that are not searched in the current iteration
                idx = np.isnan(DxF) & np.logical_not(np.isnan(DxF0))
                DxFM[idx] = DxF0[idx]
                DyFM[idx] = DyF0[idx]
                
                # Strong interpolation: use filtered estimates wherever the unfiltered estimates do not exist
                idx = np.isnan(DxF) & np.logical_not(np.isnan(DxFM))
                DxF[idx] = DxFM[idx]
                DyF[idx] = DyFM[idx]
                
                dstShape = (Dx.shape[0],Dx.shape[1])
                DxF = cv2.resize(DxF,dstShape[::-1],interpolation=cv2.INTER_CUBIC)
                DyF = cv2.resize(DyF,dstShape[::-1],interpolation=cv2.INTER_CUBIC)
                MF = cv2.resize(MF.astype(np.uint8),dstShape[::-1],interpolation=cv2.INTER_NEAREST).astype(np.bool)
                M0 = cv2.resize(M0.astype(np.uint8),dstShape[::-1],interpolation=cv2.INTER_NEAREST).astype(np.bool)
                
                idxRaw = M0 & (ChipSizeX == 0)
                idxFill = MF & (ChipSizeX == 0)
                ChipSizeX[idxRaw | idxFill] = ChipSizeUniX[i]
                InterpMask[idxFill] = True
                Dx[idxRaw | idxFill] = DxF[idxRaw | idxFill]
                Dy[idxRaw | idxFill] = DyF[idxRaw | idxFill]
                
        Flag = 1
        ChipSizeY = np.round(ChipSizeX * self.ScaleChipSizeY /2) * 2
        self.Dx = Dx
        self.Dy = Dy
        self.InterpMask = InterpMask
        self.Flag = Flag
        self.ChipSizeX = ChipSizeX
        self.ChipSizeY = ChipSizeY




    



    def runAutorift(self):
        '''
        quick processing routine which calls autorift main function (user can define their own way by mimicing the workflow here).
        '''
        import numpy as np
        
        
        # truncate the grid to fit the nested grid
        if np.size(self.ChipSizeMaxX) == 1:
            chopFactor = self.ChipSizeMaxX / self.ChipSize0X
        else:
            chopFactor = np.max(self.ChipSizeMaxX) / self.ChipSize0X
        rlim = int(np.floor(self.xGrid.shape[0] / chopFactor) * chopFactor)
        clim = int(np.floor(self.xGrid.shape[1] / chopFactor) * chopFactor)
        self.origSize = self.xGrid.shape
#        pdb.set_trace()
        self.xGrid = np.round(self.xGrid[0:rlim,0:clim]) + 0.5
        self.yGrid = np.round(self.yGrid[0:rlim,0:clim]) + 0.5
        
        # truncate the initial offset as well if they exist
        if np.size(self.Dx0) != 1:
            self.Dx0 = self.Dx0[0:rlim,0:clim]
            self.Dy0 = self.Dy0[0:rlim,0:clim]
        
        # truncate the search limits as well if they exist
        if np.size(self.SearchLimitX) != 1:
            self.SearchLimitX = self.SearchLimitX[0:rlim,0:clim]
            self.SearchLimitY = self.SearchLimitY[0:rlim,0:clim]
                
        # truncate the chip sizes as well if they exist
        if np.size(self.ChipSizeMaxX) != 1:
            self.ChipSizeMaxX = self.ChipSizeMaxX[0:rlim,0:clim]
            self.ChipSizeMinX = self.ChipSizeMinX[0:rlim,0:clim]
        
        # call autoRIFT main function
        self.autorift()
        
        
    
    

    def __init__(self):
        
        super(autoRIFT, self).__init__()
        
        ##Input related parameters
        self.I1 = None
        self.I2 = None
        self.xGrid = None
        self.yGrid = None
        self.Dx0 = 0
        self.Dy0 = 0
        self.origSize = None
        self.zeroMask = None

        ##Output file
        self.Dx = None
        self.Dy = None
        self.InterpMask = None
        self.Flag = None
        self.ChipSizeX = None
        self.ChipSizeY = None

        ##Parameter list
        self.WallisFilterWidth = 21
        self.ChipSizeMinX = 32
        self.ChipSizeMaxX = 64
        self.ChipSize0X = 32
        self.ScaleChipSizeY = 1
        self.SearchLimitX = 25
        self.SearchLimitY = 25
        self.SkipSampleX = 32
        self.SkipSampleY = 32
        self.fillFiltWidth = 3
        self.minSearch = 6
        self.sparseSearchSampleRate = 4
        self.FracValid = 8/25
        self.FracSearch = 0.25
        self.FiltWidth = 5
        self.Iter = 3
        self.MadScalar = 4
        self.BuffDistanceC = 8
        self.CoarseCorCutoff = 0.01
        self.OverSampleRatio = 16
        self.DataType = 0






class AUTO_RIFT_CORE:
    def __init__(self):
        ##Pointer to C
        self._autoriftcore = None



def arImgDisp_u(I1, I2, xGrid, yGrid, ChipSizeX, ChipSizeY, SearchLimitX, SearchLimitY, Dx0, Dy0, SubPixFlag, oversample):

    import numpy as np
    from . import autoriftcore
    
    core = AUTO_RIFT_CORE()
    if core._autoriftcore is not None:
        autoriftcore.destroyAutoRiftCore_Py(core._autoriftcore)

    core._autoriftcore = autoriftcore.createAutoRiftCore_Py()
    
    
#    if np.size(I1) == 1:
#        if np.logical_not(isinstance(I1,np.uint8) & isinstance(I2,np.uint8)):
#            sys.exit('input images must be uint8')
#    else:
#        if np.logical_not((I1.dtype == np.uint8) & (I2.dtype == np.uint8)):
#            sys.exit('input images must be uint8')

    if np.size(SearchLimitX) == 1:
        if np.logical_not(isinstance(SearchLimitX,np.float32) & isinstance(SearchLimitY,np.float32)):
            sys.exit('SearchLimit must be float')
    else:
        if np.logical_not((SearchLimitX.dtype == np.float32) & (SearchLimitY.dtype == np.float32)):
            sys.exit('SearchLimit must be float')

    if np.size(Dx0) == 1:
        if np.logical_not(isinstance(Dx0,np.float32) & isinstance(Dy0,np.float32)):
            sys.exit('Search offsets must be float')
    else:
        if np.logical_not((Dx0.dtype == np.float32) & (Dy0.dtype == np.float32)):
            sys.exit('Search offsets must be float')

    if np.size(ChipSizeX) == 1:
        if np.logical_not(isinstance(ChipSizeX,np.float32) & isinstance(ChipSizeY,np.float32)):
            sys.exit('ChipSize must be float')
    else:
        if np.logical_not((ChipSizeX.dtype == np.float32) & (ChipSizeY.dtype == np.float32)):
            sys.exit('ChipSize must be float')


    
    if np.any(np.mod(ChipSizeX,2) != 0) | np.any(np.mod(ChipSizeY,2) != 0):
#        if np.any(np.mod(xGrid-0.5,1) == 0) & np.any(np.mod(yGrid-0.5,1) == 0):
#            sys.exit('for an even chip size ImgDisp returns displacements centered at pixel boundaries so xGrid and yGrid must an inter - 1/2 [example: if you want the velocity centered between pixel (1,1) and (2,2) then specify a grid center of (1.5, 1.5)]')
#        else:
#            xGrid = np.ceil(xGrid)
#            yGrid = np.ceil(yGrid)
        sys.exit('it is better to have ChipSize = even number')
    
    if np.any(np.mod(SearchLimitX,1) != 0) | np.any(np.mod(SearchLimitY,1) != 0):
        sys.exit('SearchLimit must be an integar value')
    
    if np.any(SearchLimitX < 0) | np.any(SearchLimitY < 0):
        sys.exit('SearchLimit cannot be negative')

    if np.any(np.mod(ChipSizeX,4) != 0) | np.any(np.mod(ChipSizeY,4) != 0):
        sys.exit('ChipSize should be evenly divisible by 4')

    if np.size(Dx0) == 1:
        Dx0 = np.ones(xGrid.shape, dtype=np.float32) * Dx0

    if np.size(Dy0) == 1:
        Dy0 = np.ones(xGrid.shape, dtype=np.float32) * Dy0

    if np.size(SearchLimitX) == 1:
        SearchLimitX = np.ones(xGrid.shape, dtype=np.float32) * SearchLimitX
    
    if np.size(SearchLimitY) == 1:
        SearchLimitY = np.ones(xGrid.shape, dtype=np.float32) * SearchLimitY

    if np.size(ChipSizeX) == 1:
        ChipSizeX = np.ones(xGrid.shape, dtype=np.float32) * ChipSizeX
    
    if np.size(ChipSizeY) == 1:
        ChipSizeY = np.ones(xGrid.shape, dtype=np.float32) * ChipSizeY

    # convert from cartesian X-Y to matrix X-Y: X no change, Y from up being positive to down being positive
    Dy0 = -Dy0

    SLx_max = np.max(SearchLimitX + np.abs(Dx0))
    Px = np.int(np.max(ChipSizeX)/2 + SLx_max + 2)
    SLy_max = np.max(SearchLimitY + np.abs(Dy0))
    Py = np.int(np.max(ChipSizeY)/2 + SLy_max + 2)

    I1 = np.lib.pad(I1,((Py,Py),(Px,Px)),'constant')
    I2 = np.lib.pad(I2,((Py,Py),(Px,Px)),'constant')

    # adjust center location by the padarray size and 0.5 is added because we need to extract the chip centered at X+1 with -chipsize/2:chipsize/2-1, which equivalently centers at X+0.5 (X is the original grid point location). So for even chipsize, always returns offset estimates at (X+0.5).
    xGrid += (Px + 0.5)
    yGrid += (Py + 0.5)

    Dx = np.empty(xGrid.shape,dtype=np.float32)
    Dx.fill(np.nan)
    Dy = Dx.copy()


    for jj in range(xGrid.shape[1]):
        if np.all(SearchLimitX[:,jj] == 0) & np.all(SearchLimitY[:,jj] == 0):
            continue
        Dx1 = Dx[:,jj]
        Dy1 = Dy[:,jj]
        for ii in range(xGrid.shape[0]):
            if (SearchLimitX[ii,jj] == 0) & (SearchLimitY[ii,jj] == 0):
                continue
            
            # remember motion terms Dx and Dy correspond to I1 relative to I2 (reference)
            clx = np.floor(ChipSizeX[ii,jj]/2)
            ChipRangeX = slice(int(-clx - Dx0[ii,jj] + xGrid[ii,jj]) , int(clx - Dx0[ii,jj] + xGrid[ii,jj]))
            cly = np.floor(ChipSizeY[ii,jj]/2)
            ChipRangeY = slice(int(-cly - Dy0[ii,jj] + yGrid[ii,jj]) , int(cly - Dy0[ii,jj] + yGrid[ii,jj]))
            ChipI = I2[ChipRangeY,ChipRangeX]

            SearchRangeX = slice(int(-clx - SearchLimitX[ii,jj] + xGrid[ii,jj]) , int(clx + SearchLimitX[ii,jj] - 1 + xGrid[ii,jj]))
            SearchRangeY = slice(int(-cly - SearchLimitY[ii,jj] + yGrid[ii,jj]) , int(cly + SearchLimitY[ii,jj] - 1 + yGrid[ii,jj]))
            RefI = I1[SearchRangeY,SearchRangeX]
            
            minChipI = np.min(ChipI)
            if minChipI < 0:
                ChipI = ChipI - minChipI
            if np.all(ChipI == ChipI[0,0]):
                continue
                    
            minRefI = np.min(RefI)
            if minRefI < 0:
                RefI = RefI - minRefI
            if np.all(RefI == RefI[0,0]):
                continue
            

            if SubPixFlag:
                # call C++
                Dx1[ii], Dy1[ii] = np.float32(autoriftcore.arSubPixDisp_u_Py(core._autoriftcore,ChipI.shape[1],ChipI.shape[0],ChipI.ravel(),RefI.shape[1],RefI.shape[0],RefI.ravel(),oversample))
#                   # call Python
#                   Dx1[ii], Dy1[ii] = arSubPixDisp(ChipI,RefI)
            else:
                # call C++
                Dx1[ii], Dy1[ii] = np.float32(autoriftcore.arPixDisp_u_Py(core._autoriftcore,ChipI.shape[1],ChipI.shape[0],ChipI.ravel(),RefI.shape[1],RefI.shape[0],RefI.ravel()))
#                   # call Python
#                   Dx1[ii], Dy1[ii] = arPixDisp(ChipI,RefI)


    # add back 1) I1 (RefI) relative to I2 (ChipI) initial offset Dx0 and Dy0, and
    #          2) RefI relative to ChipI has a left/top boundary offset of -SearchLimitX and -SearchLimitY
    idx = np.logical_not(np.isnan(Dx))
    Dx[idx] += (Dx0[idx] - SearchLimitX[idx])
    Dy[idx] += (Dy0[idx] - SearchLimitY[idx])
    
    # convert from matrix X-Y to cartesian X-Y: X no change, Y from down being positive to up being positive
    Dy = -Dy
    
    autoriftcore.destroyAutoRiftCore_Py(core._autoriftcore)
    core._autoriftcore = None
    
    return Dx, Dy






def arImgDisp_s(I1, I2, xGrid, yGrid, ChipSizeX, ChipSizeY, SearchLimitX, SearchLimitY, Dx0, Dy0, SubPixFlag, oversample):
    
    import numpy as np
    from . import autoriftcore
    
    core = AUTO_RIFT_CORE()
    if core._autoriftcore is not None:
        autoriftcore.destroyAutoRiftCore_Py(core._autoriftcore)
    
    core._autoriftcore = autoriftcore.createAutoRiftCore_Py()
    
    
#    if np.size(I1) == 1:
#        if np.logical_not(isinstance(I1,np.uint8) & isinstance(I2,np.uint8)):
#            sys.exit('input images must be uint8')
#    else:
#        if np.logical_not((I1.dtype == np.uint8) & (I2.dtype == np.uint8)):
#            sys.exit('input images must be uint8')

    if np.size(SearchLimitX) == 1:
        if np.logical_not(isinstance(SearchLimitX,np.float32) & isinstance(SearchLimitY,np.float32)):
            sys.exit('SearchLimit must be float')
    else:
        if np.logical_not((SearchLimitX.dtype == np.float32) & (SearchLimitY.dtype == np.float32)):
            sys.exit('SearchLimit must be float')

    if np.size(Dx0) == 1:
        if np.logical_not(isinstance(Dx0,np.float32) & isinstance(Dy0,np.float32)):
            sys.exit('Search offsets must be float')
    else:
        if np.logical_not((Dx0.dtype == np.float32) & (Dy0.dtype == np.float32)):
            sys.exit('Search offsets must be float')

    if np.size(ChipSizeX) == 1:
        if np.logical_not(isinstance(ChipSizeX,np.float32) & isinstance(ChipSizeY,np.float32)):
            sys.exit('ChipSize must be float')
    else:
        if np.logical_not((ChipSizeX.dtype == np.float32) & (ChipSizeY.dtype == np.float32)):
            sys.exit('ChipSize must be float')



    if np.any(np.mod(ChipSizeX,2) != 0) | np.any(np.mod(ChipSizeY,2) != 0):
#        if np.any(np.mod(xGrid-0.5,1) == 0) & np.any(np.mod(yGrid-0.5,1) == 0):
#            sys.exit('for an even chip size ImgDisp returns displacements centered at pixel boundaries so xGrid and yGrid must an inter - 1/2 [example: if you want the velocity centered between pixel (1,1) and (2,2) then specify a grid center of (1.5, 1.5)]')
#        else:
#            xGrid = np.ceil(xGrid)
#            yGrid = np.ceil(yGrid)
        sys.exit('it is better to have ChipSize = even number')
    
    if np.any(np.mod(SearchLimitX,1) != 0) | np.any(np.mod(SearchLimitY,1) != 0):
        sys.exit('SearchLimit must be an integar value')

    if np.any(SearchLimitX < 0) | np.any(SearchLimitY < 0):
        sys.exit('SearchLimit cannot be negative')
    
    if np.any(np.mod(ChipSizeX,4) != 0) | np.any(np.mod(ChipSizeY,4) != 0):
        sys.exit('ChipSize should be evenly divisible by 4')
    
    if np.size(Dx0) == 1:
        Dx0 = np.ones(xGrid.shape, dtype=np.float32) * Dx0
    
    if np.size(Dy0) == 1:
        Dy0 = np.ones(xGrid.shape, dtype=np.float32) * Dy0
    
    if np.size(SearchLimitX) == 1:
        SearchLimitX = np.ones(xGrid.shape, dtype=np.float32) * SearchLimitX
    
    if np.size(SearchLimitY) == 1:
        SearchLimitY = np.ones(xGrid.shape, dtype=np.float32) * SearchLimitY

    if np.size(ChipSizeX) == 1:
        ChipSizeX = np.ones(xGrid.shape, dtype=np.float32) * ChipSizeX
    
    if np.size(ChipSizeY) == 1:
        ChipSizeY = np.ones(xGrid.shape, dtype=np.float32) * ChipSizeY

    # convert from cartesian X-Y to matrix X-Y: X no change, Y from up being positive to down being positive
    Dy0 = -Dy0
    
    SLx_max = np.max(SearchLimitX + np.abs(Dx0))
    Px = np.int(np.max(ChipSizeX)/2 + SLx_max + 2)
    SLy_max = np.max(SearchLimitY + np.abs(Dy0))
    Py = np.int(np.max(ChipSizeY)/2 + SLy_max + 2)
    
    I1 = np.lib.pad(I1,((Py,Py),(Px,Px)),'constant')
    I2 = np.lib.pad(I2,((Py,Py),(Px,Px)),'constant')
    
    # adjust center location by the padarray size and 0.5 is added because we need to extract the chip centered at X+1 with -chipsize/2:chipsize/2-1, which equivalently centers at X+0.5 (X is the original grid point location). So for even chipsize, always returns offset estimates at (X+0.5).
    xGrid += (Px + 0.5)
    yGrid += (Py + 0.5)
    
    Dx = np.empty(xGrid.shape,dtype=np.float32)
    Dx.fill(np.nan)
    Dy = Dx.copy()
    
    
    for jj in range(xGrid.shape[1]):
        if np.all(SearchLimitX[:,jj] == 0) & np.all(SearchLimitY[:,jj] == 0):
            continue
        Dx1 = Dx[:,jj]
        Dy1 = Dy[:,jj]
        for ii in range(xGrid.shape[0]):
            if (SearchLimitX[ii,jj] == 0) & (SearchLimitY[ii,jj] == 0):
                continue
        
            # remember motion terms Dx and Dy correspond to I1 relative to I2 (reference)
            clx = np.floor(ChipSizeX[ii,jj]/2)
            ChipRangeX = slice(int(-clx - Dx0[ii,jj] + xGrid[ii,jj]) , int(clx - Dx0[ii,jj] + xGrid[ii,jj]))
            cly = np.floor(ChipSizeY[ii,jj]/2)
            ChipRangeY = slice(int(-cly - Dy0[ii,jj] + yGrid[ii,jj]) , int(cly - Dy0[ii,jj] + yGrid[ii,jj]))
            ChipI = I2[ChipRangeY,ChipRangeX]
            
            SearchRangeX = slice(int(-clx - SearchLimitX[ii,jj] + xGrid[ii,jj]) , int(clx + SearchLimitX[ii,jj] - 1 + xGrid[ii,jj]))
            SearchRangeY = slice(int(-cly - SearchLimitY[ii,jj] + yGrid[ii,jj]) , int(cly + SearchLimitY[ii,jj] - 1 + yGrid[ii,jj]))
            RefI = I1[SearchRangeY,SearchRangeX]
            
            minChipI = np.min(ChipI)
            if minChipI < 0:
                ChipI = ChipI - minChipI
            if np.all(ChipI == ChipI[0,0]):
                continue

            minRefI = np.min(RefI)
            if minRefI < 0:
                RefI = RefI - minRefI
            if np.all(RefI == RefI[0,0]):
                continue
            

            if SubPixFlag:
                # call C++
                Dx1[ii], Dy1[ii] = np.float32(autoriftcore.arSubPixDisp_s_Py(core._autoriftcore,ChipI.shape[1],ChipI.shape[0],ChipI.ravel(),RefI.shape[1],RefI.shape[0],RefI.ravel(), oversample))
#                   # call Python
#                   Dx1[ii], Dy1[ii] = arSubPixDisp(ChipI,RefI)
            else:
                # call C++
                Dx1[ii], Dy1[ii] = np.float32(autoriftcore.arPixDisp_s_Py(core._autoriftcore,ChipI.shape[1],ChipI.shape[0],ChipI.ravel(),RefI.shape[1],RefI.shape[0],RefI.ravel()))
#                   # call Python
#                   Dx1[ii], Dy1[ii] = arPixDisp(ChipI,RefI)


    # add back 1) I1 (RefI) relative to I2 (ChipI) initial offset Dx0 and Dy0, and
    #          2) RefI relative to ChipI has a left/top boundary offset of -SearchLimitX and -SearchLimitY
    idx = np.logical_not(np.isnan(Dx))
    Dx[idx] += (Dx0[idx] - SearchLimitX[idx])
    Dy[idx] += (Dy0[idx] - SearchLimitY[idx])
    
    # convert from matrix X-Y to cartesian X-Y: X no change, Y from down being positive to up being positive
    Dy = -Dy
    
    autoriftcore.destroyAutoRiftCore_Py(core._autoriftcore)
    core._autoriftcore = None
    
    return Dx, Dy




def colfilt(A, kernelSize, option):
    
    from skimage.util import view_as_windows as viewW
    import numpy as np
    
    A = np.lib.pad(A,((int((kernelSize[0]-1)/2),int((kernelSize[0]-1)/2)),(int((kernelSize[1]-1)/2),int((kernelSize[1]-1)/2))),mode='constant',constant_values=np.nan)
    
    B = viewW(A, kernelSize).reshape(-1,kernelSize[0]*kernelSize[1]).T[:,::1]
    
    output_size = (A.shape[0]-kernelSize[0]+1,A.shape[1]-kernelSize[1]+1)
    C = np.zeros(output_size,dtype=A.dtype)
    if option == 0:#    max
        C = np.nanmax(B,axis=0).reshape(output_size)
    elif option == 1:#  min
        C = np.nanmin(B,axis=0).reshape(output_size)
    elif option == 2:#  mean
        C = np.nanmean(B,axis=0).reshape(output_size)
    elif option == 3:#  median
        C = np.nanmedian(B,axis=0).reshape(output_size)
    elif option == 4:#  range
        C = np.nanmax(B,axis=0).reshape(output_size) - np.nanmin(B,axis=0).reshape(output_size)
    elif option == 6:#  MAD (Median Absolute Deviation)
        m = B.shape[0]
        D = np.abs(B - np.dot(np.ones((m,1),dtype=A.dtype), np.array([np.nanmedian(B,axis=0)])))
        C = np.nanmedian(D,axis=0).reshape(output_size)
    elif option[0] == 5:#  displacement distance count with option[1] being the threshold
        m = B.shape[0]
        c = int(np.round((m + 1) / 2)-1)
#        c = 0
        D = np.abs(B - np.dot(np.ones((m,1),dtype=A.dtype), np.array([B[c,:]])))
        C = np.sum(D<option[1],axis=0).reshape(output_size)
    else:
        sys.exit('invalid option for columnwise neighborhood filtering')

    C = C.astype(A.dtype)
    
    return C




class DISP_FILT:
    
    def __init__(self):
        ##filter parameters; try different parameters to decide how much fine-resolution estimates we keep, which can make the final images smoother
        
        self.FracValid = 8/25
        self.FracSearch = 0.25
        self.FiltWidth = 5
        self.Iter = 3
        self.MadScalar = 4
    
    def filtDisp(self, Dx, Dy, SearchLimitX, SearchLimitY, M, OverSampleRatio):
        
        import numpy as np
        
        dToleranceX = self.FracValid * self.FiltWidth**2
        dToleranceY = self.FracValid * self.FiltWidth**2
#        pdb.set_trace()
        Dx = Dx / SearchLimitX
        Dy = Dy / SearchLimitY
        
        DxMadmin = np.ones(Dx.shape) / OverSampleRatio / SearchLimitX * 2;
        DyMadmin = np.ones(Dy.shape) / OverSampleRatio / SearchLimitY * 2;
        
        
        
        for i in range(self.Iter):
            Dx[np.logical_not(M)] = np.nan
            Dy[np.logical_not(M)] = np.nan
            M = (colfilt(Dx.copy(), (self.FiltWidth, self.FiltWidth), (5,self.FracSearch)) >= dToleranceX) & (colfilt(Dy.copy(), (self.FiltWidth, self.FiltWidth), (5,self.FracSearch)) >= dToleranceY)
        
#        if self.Iter == 3:
#            pdb.set_trace()

        for i in range(np.max([self.Iter-1,1])):
            Dx[np.logical_not(M)] = np.nan
            Dy[np.logical_not(M)] = np.nan
            
            DxMad = colfilt(Dx.copy(), (self.FiltWidth, self.FiltWidth), 6)
            DyMad = colfilt(Dy.copy(), (self.FiltWidth, self.FiltWidth), 6)
        
            DxM = colfilt(Dx.copy(), (self.FiltWidth, self.FiltWidth), 3)
            DyM = colfilt(Dy.copy(), (self.FiltWidth, self.FiltWidth), 3)
            
            M = (np.abs(Dx - DxM) <= np.maximum(self.MadScalar * DxMad, DxMadmin)) & (np.abs(Dy - DyM) <= np.maximum(self.MadScalar * DyMad, DyMadmin)) & M
        
        return M




def bwareaopen(image,size1):
    
    import numpy as np
    from skimage import measure
    
    # now identify the objects and remove those above a threshold
    labels, N = measure.label(image,connectivity=2,return_num=True)
    label_size = [(labels == label).sum() for label in range(N + 1)]
    
    # now remove the labels
    for label,size in enumerate(label_size):
        if size < size1:
            image[labels == label] = 0

    return image




