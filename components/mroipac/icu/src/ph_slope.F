!c****************************************************************

      subroutine ph_slope(intbf, pslope)

!c****************************************************************
!c**     
!c**   FILE NAME: ph_slope.f
!c**     
!c**   DATE WRITTEN: 19-Jan-98
!c**     
!c**   PROGRAMMER: Charles Werner
!c**     
!c**   FUNCTIONAL DESCRIPTION:  Calculate phase gradient using averages of weighted 
!c**   phase differences. This is the same algorithm used for calculation of the doppler
!c**   centroid developed by Madsen, now applied to determination of phase gradients. We
!c**   are estimating the a smoothed version of the gradient by averaging the 
!c**   estimate over a region assuming that it is stationary.  Weighting of the
!c**   differences is performed to emphasize the estimates close to the desired point.
!c**
!c**   NOTES:
!c**     
!c**   ROUTINES CALLED:
!c**     
!c**   UPDATE LOG:
!c**
!c**   Date Changed        Reason Changed                  
!c**   ------------       ----------------              
!c**   19-Jan-98 v1.0      created 
!c**   5-Mar-98  v1.1      update of data structures
!c**   1-Nov-98  v1.2      Corrected calculation of weighting function 
!c**   1-Nov-98  v1.2      Changed indexing on filter loop
!c**   1-Nov-98  v1.2      Moved increments of l and m to end of loops, 
!c**                       rather than at start
!c*****************************************************************

      use icuState
      implicit none


!c     INPUT VARIABLES:

      complex*8 intbf(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)	!input interferogram array
	
!c     OUTPUT VARIABLES:

      complex*8 pslope(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)	!output slope in packed complex format (range slope, azimuth slope)

!c     LOCAL VARIABLES:

      integer*4 i,j,k,l,n,m
      real*4  wf(0:WIN_MAX-1, 0:WIN_MAX-1) 	!weighting function window 

      complex*8 sr,saz
      real*4  phr,phaz
      real*4  w1,s1 
      integer*4 winsz				!size of estimation window

!c     PROCESSING STEPS:

      s1=0.0					!initialize sum of weights
      winsz=infp%i_cc_winsz			!width of window to estimate phase gradient

      do k = 0 , winsz - 1      		!generate patch weighting
        do j = 0 , winsz - 1
          w1 = (k - winsz/2)**2 + (j - winsz/2)**2
          wf(k,j) = exp(-w1/((winsz/2.0)))
          s1 = s1 + wf(k,j)			!sum weights to calculate normalization
        end do
      end do

      do k = 0, winsz - 1         
        do j = 0, winsz - 1
           wf(k,j) = wf(k,j)/s1		!normalize weights such that sum of weights = 1.0
        end do
      end do

      do i = 0, infp%i_azbufsize-1
        do j=0, infp%i_rsamps-1		!init. output
          pslope(j,i) = cmplx(0.,0.)
        end do
      end do    

c$doacross local(i,j,k,m,l,sr,saz,n,w1,phr,phaz),
c$&        share(infp,winsz,pslope,wf,intbf)
      do i = infp%i_sline+winsz/2+1, infp%i_eline - winsz/2 	!azimuth loop -- trim edges
        do j = infp%i_ssamp+winsz/2+1, infp%i_esamp- winsz/2	!range loop -- trim edges
 
          sr = cmplx(0.0, 0.0)		!init. sum of differences in range and azimuth
          saz = cmplx(0.0, 0.0)
          m=0
 
          do k = i-winsz/2, i+winsz/2
            l = 0
            do n = j-winsz/2, j+winsz/2	!average first differences in range and azimuth
              w1 = wf(m,l)
              sr =  sr +  w1 * intbf(n,k) * conjg(intbf(n-1, k))
              saz = saz + w1 * intbf(n,k) * conjg(intbf(n, k-1))
              l = l+1
            end do
            m = m+1
          end do
             
          if(real(sr) .ne. 0.0)then
            phr = atan2(aimag(sr), real(sr))
          endif

          if(real(saz) .ne. 0.0)then
            phaz = atan2(aimag(saz), real(saz))
          end if

          pslope(j,i) = cmplx(phr, phaz)
        end do
      end do
      return
      end 

 






