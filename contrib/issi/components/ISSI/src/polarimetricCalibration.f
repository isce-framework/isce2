
c23456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012
c> Calculate the polarimetric scattering matrix using the distortion
c> parameters
c>
c> @param hhInFile
c> @param transmission
c> @param reception
c> @param samples
c> @param lines
c>
c> @see http://earth.esa.int/pcs/alos/palsar/articles/Calibration_palsar_products_v13.pdf
c> @see http://earth.esa.int/workshops/polinsar2009/participants/493/paper_493_s3_3shim.pdf
c>
c> The polarimetric calibration is performed using the following
c> formula:
c>
c> \f[O = R * F * S * F * T = R * \hat{O} * T \f]
c>
c> where \f[\hat{O} = F * S * F = R^{-1} * O * T^{-1} \f]
c>
c> and
c>
c> \f[O\f] is the measured (uncalibrated) scattering matrix that includes polarimetric distortions,
c> \f[R\f] is the reception distortion matrix which includes the effects of x-talk and channel imbalance
c> \f[F\f] is the Faraday Rotation matrix
c> \f[S\f] is the true scattering matrix
c> \f[T\f] is the transmission distortion matrix which includes the effects of x-talk and channel imbalance
c> \f[\hat{O}\f] is the measured matrix with x-talk and channel imbalance corrected
c>
c> The matrix capO  has the following entries:
c> (1,1) = hh
c> (1,2) = hv
c> (2,1) = vh
c> (2,2) = vv
c23456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012

      subroutine polarimetricCalibration(hhInFileName,hvInFileName,
     +                                   vhInFileName,vvInFileName,
     +                                   hhOutFileName,hvOutFileName,
     +                                   vhOutFileName,vvOutFileName,
     +                                   transmission,reception,samples,lines)

      use iso_c_binding
      implicit none

ccccc declare parameters

      integer*4      maxSamples
      parameter     (maxSamples = 30000)


ccccc delare derived types

      type, BIND(C) :: distortion_type
       complex(C_FLOAT_COMPLEX) :: crossTalk1
       complex(C_FLOAT_COMPLEX) :: crossTalk2
       complex(C_FLOAT_COMPLEX) :: channelImbalance
      end type
      type(distortion_type) :: transmission, reception

ccccc declare scalars

      integer*4      hhInFile
      integer*4      hhOutFile
      integer*4      hvInFile
      integer*4      hvOutFile
      integer*4      lineCnt
      integer*4      lines
      integer*4      sampleCnt
      integer*4      samples
      integer*4      vhInFile
      integer*4      vhOutFile
      integer*4      vvInFile
      integer*4      vvOutFile

      character*255 hhInFileName
      character*255 hhOutFileName
      character*255 hvInFileName
      character*255 hvOutFileName
      character*255 vhInFileName
      character*255 vhOutFileName
      character*255 vvInFileName
      character*255 vvOutFileName

ccccc declare arrays

      complex*8      hhInLine(maxSamples)
      complex*8      hhOutLine(maxSamples)
      complex*8      hvInLine(maxSamples)
      complex*8      hvOutLine(maxSamples)
      complex*8      vhInLine(maxSamples)
      complex*8      vhOutLine(maxSamples)
      complex*8      vvInLine(maxSamples)
      complex*8      vvOutLine(maxSamples)

      complex*8      capO(2,2)
      complex*8      capOhat(2,2)
      complex*8      capR(2,2)
      complex*8      capRinv(2,2)
      complex*8      capT(2,2)
      complex*8      capTinv(2,2)
      complex*8      tmpCmplxMat(2,2)


ccccc initialize

      hhInFile = 23
      hvInFile = 24
      vhInFile = 25
      vvInFile = 26

      hhOutFile = 27
      hvOutFile = 28
      vhOutFile = 29
      vvOutFile = 30

      capR(1,1) = cmplx(1.0,0.0)
      capR(1,2) = reception%CrossTalk1
      capR(2,1) = reception%CrossTalk2
      capR(2,2) = reception%ChannelImbalance

      capT(1,1) = cmplx(1.0,0.0)
      capT(1,2) = transmission%CrossTalk1
      capT(2,1) = transmission%CrossTalk2
      capT(2,2) = transmission%ChannelImbalance


ccccc check for errors

      if (samples .gt. maxSamples) then
         write (*,*) '***** ERROR - samples greater than maxSamples: ' , samples , maxSamples
         stop
      endif


ccccc initialize reception distortion matrix and transmission distortion matrix

      call twoByTwoCmplxMatInv(capR, capRinv)
      call twoByTwoCmplxMatInv(capT, capTinv)


ccccc open files

      open (unit=hhInFile, file=hhInFileName, status='old', access='direct', recl=8*samples)
      open (unit=hvInFile, file=hvInFileName, status='old', access='direct', recl=8*samples)
      open (unit=vhInFile, file=vhInFileName, status='old', access='direct', recl=8*samples)
      open (unit=vvInFile, file=vvInFileName, status='old', access='direct', recl=8*samples)

      open (unit=hhOutFile, file=hhOutFileName, status='replace', access='direct', recl=8*samples)
      open (unit=hvOutFile, file=hvOutFileName, status='replace', access='direct', recl=8*samples)
      open (unit=vhOutFile, file=vhOutFileName, status='replace', access='direct', recl=8*samples)
      open (unit=vvOutFile, file=vvOutFileName, status='replace', access='direct', recl=8*samples)


ccccc determine capOhat = capRinv * capO * capTinv

      do lineCnt = 1 , lines                                                            ! begin loop over lines

         read  (hhInFile, rec=lineCnt) (hhInLine(sampleCnt), sampleCnt = 1 , samples)   ! read a line of data
         read  (hvInFile, rec=lineCnt) (hvInLine(sampleCnt), sampleCnt = 1 , samples)
         read  (vhInFile, rec=lineCnt) (vhInLine(sampleCnt), sampleCnt = 1 , samples)
         read  (vvInFile, rec=lineCnt) (vvInLine(sampleCnt), sampleCnt = 1 , samples)

         do sampleCnt = 1 , samples                                                     ! begin loop over samples

            capO(1,1) = hhInLine(sampleCnt)
            capO(1,2) = hvInLine(sampleCnt)
            capO(2,1) = vhInLine(sampleCnt)
            capO(2,2) = vvInLine(sampleCnt)

            call twoByTwoCmplxMatMlt(capRinv, capO, tmpCmplxMat)
            call twoByTwoCmplxMatMlt(tmpCmplxMat, capTinv, capOhat)

            hhOutLine(sampleCnt) = capOhat(1,1)
            hvOutLine(sampleCnt) = capOhat(1,2)
            vhOutLine(sampleCnt) = capOhat(2,1)
            vvOutLine(sampleCnt) = capOhat(2,2)

         enddo                                                                          ! end loop over samples

         write (hhOutFile, rec=lineCnt) (hhOutLine(sampleCnt), sampleCnt = 1 , samples) ! write a line of data
         write (hvOutFile, rec=lineCnt) (hvOutLine(sampleCnt), sampleCnt = 1 , samples)
         write (vhOutFile, rec=lineCnt) (vhOutLine(sampleCnt), sampleCnt = 1 , samples)
         write (vvOutFile, rec=lineCnt) (vvOutLine(sampleCnt), sampleCnt = 1 , samples)

      enddo                                                                             ! end loop over lines


ccccc close files

      close (hhInFile)
      close (hvInFile)
      close (vhInFile)
      close (vvInFile)

      close (hhOutFile)
      close (hvOutFile)
      close (vhOutFile)
      close (vvOutFile)

      return

      end

c23456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012

      subroutine twoByTwoCmplxMatInv(inMat, outMat)

      implicit none

      complex*8 inMat(2,2), outMat(2,2), inMatDet

      inMatDet = inMat(1,1) * inMat(2,2) - inMat(1,2) * inMat(2,1)

      outMat(1,1) = +inMat(2,2) / inMatDet
      outMat(1,2) = -inMat(1,2) / inMatDet
      outMat(2,1) = -inMat(2,1) / inMatDet
      outMat(2,2) = +inMat(1,1) / inMatDet

      return

      end

c23456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012

      subroutine twoByTwoCmplxMatMlt(inMat1, inMat2, outMat)

      implicit none

      complex*8 inMat1(2,2), inMat2(2,2), outMat(2,2)

      outMat(1,1) = inMat1(1,1) * inMat2(1,1) + inMat1(1,2) * inMat2(2,1)
      outMat(1,2) = inMat1(1,1) * inMat2(1,2) + inMat1(1,2) * inMat2(2,2)
      outMat(2,1) = inMat1(2,1) * inMat2(1,1) + inMat1(2,2) * inMat2(2,1)
      outMat(2,2) = inMat1(2,1) * inMat2(1,2) + inMat1(2,2) * inMat2(2,2)

      return

      end

c23456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012

