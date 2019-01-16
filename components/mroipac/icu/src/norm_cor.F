!c****************************************************************

      subroutine norm_cor(intb, ampb, pslope, sline, eline, 
     +     ssamp, esamp, winsz, ncorr)

!c****************************************************************
!c**     
!c**   FILE NAME: norm_cor.f
!c**     
!c**   DATE WRITTEN: 5-Mar-98
!c**     
!c**   PROGRAMMER: Charles Werner and Paul Rosen
!c**     
!c**   FUNCTIONAL DESCRIPTION: Calculate correlation, correcting 
!c**   for the local phase slope calculated using subroutine ph_slope.
!c**   The correlation estimate is obtained by a weighted summation over
!c**   a rectangular window.
!c**     
!c**   ROUTINES CALLED:
!c**     
!c**   NOTES: 
!c**     
!c**   UPDATE LOG:
!c**
!c**   Date Changed        Reason Changed        
!c**   ------------       ----------------
!c**   1-Nov-98  v1.1 Corrected calculation of weighting function 
!c**     
!c*****************************************************************

      use icuState
      implicit none


!c     INPUT VARIABLES:
 

      complex intb(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1) !input interferogram
      complex ampb(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1) !amplitude of the SLC data in packed complex format
      complex pslope(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)	!phase gradient of interf. in packed complex format
      integer*4 sline,eline		!starting and ending line with valid data
      integer*4 ssamp,esamp		!starting and ending sample with valid data
      integer*4 winsz
	
!c     OUTPUT VARIABLES:

      real*4 ncorr(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)

!c     LOCAL VARIABLES:

      integer*4 i,j,k,n

      complex xp,ex
      real*4 pwr1,pwr2,pwrgm
      real*4 azph,ph
      real*4 w1,s1 
      real*4 r_slp				!range phase slope
      real*4 az_slp				!azimuth phase slope

      real*4  wf(0:WIN_MAX-1, 0:WIN_MAX-1) !weighting function window 


!c     PROCESSING STEPS:

      s1=0.0				!initialize sum of weights

      do k = 0 , winsz - 1      	!generate patch weighting
        do j = 0 , winsz - 1
          w1 = (k - winsz/2)**2 + (j - winsz/2)**2
          wf(k,j) = exp(-w1/((winsz/2.0)))
          s1 = s1 + wf(k,j)
          write(*,'(1x,i4,2x,i4,2x,f10.5)') k,j,wf(k,j)
        end do
      end do

      do k = 0, winsz - 1         
        do j = 0, winsz - 1
           wf(k,j) = wf(k,j)/s1		!normalize weights such that sum of weights = 1.0
        end do
      end do

c$doacross local(i,j,k,xp,az_slp,r_slp,pwr1,pwr2,azph,n,ph,ex,
c$&              w1,pwrgm),
c$&        share(sline,eline,ssamp,esamp,winsz,pslope,wf,intb,ampb,ncorr)
      do i = sline + winsz/2, eline - winsz/2 - 1	!azimuth loop -- trim edges
         do j = ssamp + winsz/2, esamp - winsz/2 - 1	!range loop -- trim edges

          xp = cmplx(0.0,0.0)			!weighted and deramped sum
          pwr1 = 0.0				!sum of powers image-1
          pwr2 = 0.0				!sum of powers image-2
          az_slp = aimag(pslope(j,i))		!azimuth phase slope
          r_slp = real(pslope(j,i))		!range phase slope
 
          do k = -winsz/2, winsz/2		!scan in azimuth over the estimation region
            azph = k*az_slp			!azimuth phase shift
            do n = -winsz/2, winsz/2		!scan in range over the estimation region
              w1 = wf(n+winsz/2,k+winsz/2)	!weight factor
              ph = n*r_slp + azph		!range phase shift + azimuth phase shift
              ex = cmplx(cos(ph),-sin(ph))	!phase rotation vector
              xp = xp + w1*ex*intb(j+n,i+k) 		!sum deramped interf. samples
              pwr1 = pwr1 + w1*real(ampb(j+n,i+k))**2	!sum of intensity images
              pwr2 = pwr2 + w1*aimag(ampb(j+n,i+k))**2            
            end do
          end do
 
          pwrgm = sqrt(pwr1*pwr2)		!geometric mean of image intensities
          if(pwrgm .gt. 0.0)then
             ncorr(j,i) = cabs(xp)/pwrgm 	!normalized correlation
          endif

        end do

      end do
      return
      end 

 
