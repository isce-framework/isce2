!c****************************************************************

      subroutine std_cor(intb, ampb, sline, eline, ssamp, esamp, winsz, ncorr)

!c****************************************************************
!c**     
!c**   FILE NAME: std_cor.f
!c**     
!c**   DATE WRITTEN: 5-Mar-98
!c**     
!c**   PROGRAMMER: Charles Werner and Paul Rosen
!c**     
!c**   FUNCTIONAL DESCRIPTION:  Calculate standard correlation. The
!c**   correlation estimate is obtained by a weighted summation over 
!c**   a rectangular window. No correction for local slope.
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
!c**   1-Nov-98  v1.1 Changed indexing on filter loop
!c**   1-Nov-98  v1.1 Moved increments of l and m to end of loops, rather than at start
!c**
!c*****************************************************************

      use icuState
      implicit none


!c     INPUT VARIABLES:

      complex*8 intb(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)	!input interferogram 
      complex*8 ampb(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)	!!amplitude of the SLC data in packed complex format
      integer*4 sline,eline		!starting and ending line with valid data
      integer*4 ssamp,esamp		!starting and ending sample with valid data
      integer*4 winsz
	
!c     OUTPUT VARIABLES:

      real*4 ncorr(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)	!correlation data array

!c     LOCAL VARIABLES:

      real*4  wf(0:WIN_MAX-1, 0:WIN_MAX-1) !weighting function window 
      complex*8 xp 
      real*4 pwr1,pwr2,pwrgm
      real*4 w1,s1 
      integer*4 i,j,k,l,n,m

!c     PROCESSING STEPS:

      s1=0.0				!initialize sum of weights

      do k = 0 , winsz - 1      	!generate patch weighting
        do j = 0 , winsz - 1
           w1 = (k - winsz/2)**2 + (j - winsz/2)**2
           wf(k,j) = exp(-w1/((winsz/2.0)))
           s1 = s1 + wf(k,j)
!c           write(*,'(i5 i5 10.5f)')k,j,wf(k,j)
        end do
      end do

      do k = 0, winsz - 1         
        do j = 0, winsz - 1
           wf(k,j) = wf(k,j)/s1			!normalize weights such that sum of weights = 1.0
        end do
      end do

c$doacross local(i,j,k,l,m,n,xp,pwr1,pwr2,w1,pwrgm),
c$&        share(sline,eline,ssamp,esamp,winsz,wf,intb,ampb,ncorr)
      do i = sline + winsz/2, eline - winsz/2 - 1	!azimuth loop -- trim edges
         do j = ssamp + winsz/2, esamp - winsz/2 - 1	!range loop -- trim edges
 
             pwr1 = 0.0				!sum of powers image-1
             pwr2 = 0.0				!sum of powers image-2
             xp = cmplx(0.0, 0.0)		!weighted sum of interferogram samples
             m=0
 
             do k = i-winsz/2, i-winsz/2 + winsz-1
                l = 0
                do n = j-winsz/2, j-winsz/2 + winsz-1
                    w1 = wf(m,l)			!weighting factor
                    pwr1 = pwr1 + w1*real(ampb(n,k))**2
                    pwr2 = pwr2 + w1*aimag(ampb(n,k))**2
                    xp = xp + w1*intb(n,k) 	!weight interferogram
                    l = l+1
                end do
                m = m+1
             end do

             pwrgm = sqrt(pwr1*pwr2)		!geometric mean of image intensities
             if(pwrgm .gt. 0.0)then
                ncorr(j,i) = cabs(xp)/pwrgm !normalized correlation
             endif

         end do

      end do

      return
      end 

 
