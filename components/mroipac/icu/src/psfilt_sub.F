!c****************************************************************

      subroutine psfilt(intb, ampb, intb_filt, sline, eline, ssamp, esamp, alpha)

!c****************************************************************
!c**     
!c**   FILE NAME: psfilt_sub.f
!c**     
!c**   DATE WRITTEN: 19-Jan-98
!c**     
!c**   PROGRAMMER: Charles Werner and Paul Rosen
!c**     
!c**   FUNCTIONAL DESCRIPTION: This routine performs adaptive
!c**   power spectral filtering.
!c**     
!c**   ROUTINES CALLED: spec_wgt
!c**     
!c**   NOTES: 
!c**     
!c**   UPDATE LOG:
!c**
!c**   Date Changed        Reason Changed                  CR # and Version #
!c**   ------------       ----------------                 -----------------
!c**   v1.1		 corrected error in range width processing
!c**   v1.2     4-Mar-98  added parameters for starting and ending lines and pixels
!c**     
!c*****************************************************************
      use icuState
      implicit none

!c     INPUT VARIABLES:

      complex*8 intb(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)
      complex*8 ampb(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)
      integer*4 ssamp, esamp		!starting and ending valid interf. sample
      integer*4 sline, eline		!starting and ending valid line in the interf.		
      real*4  alpha			!power spectral filter exponent

!c     OUTPUT VARIABLES:

      complex*8 intb_filt(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)

!c     LOCAL VARIABLES:

      complex*8 patch(0:NFFT-1, 0:NFFT-1)

      real*4  pwr
      real*4  rw, azw
      real*4  wf(0:NFFT-1,0:NFFT-1)
      real*4 patchmagin(0:NFFT-1, 0:NFFT-1)
      real*4 patchmagout(0:NFFT-1, 0:NFFT-1)

      integer*4 i, j, i1, j1, ip, jp

!c     PROCESSING STEPS:

      do i = 0 ,  NFFT-1        !output patch weighting 
        do j = 0 , NFFT-1
           azw = 1.0 - abs(2.0*float(i-NFFT/2)/(NFFT+1))
           rw  = 1.0 - abs(2.0*float(j-NFFT/2)/(NFFT+1))
           wf(i,j) = azw*rw/float(NFFT*NFFT)
        end do
      end do

      do i = sline, eline
         do j = ssamp, esamp
            intb_filt(j,i) = cmplx(0.,0.)
         end do
      enddo      

c$doacross local(i,j,i1,j1,jp,ip,pwr,patch,patchmagin,patchmagout),
c$&        share(intb,sline,eline,ssamp,esamp,alpha,wf,intb_filt,ampb)      
       do i=sline, eline-NFFT, STEP
         do j=ssamp, esamp-NFFT, STEP	!corrected error 2-Feb-98 clw

            do i1 = 0, NFFT-1   	!normalize input data, do not change the input data 
               do j1 = 0, NFFT-1
                  jp = j+j1
                  ip = i+i1
                  pwr = real(ampb(jp,ip))*aimag(ampb(jp,ip))
                  patch(j1,i1) = cmplx(0.,0.)
                  if (pwr .gt. 0.0)then
                     patch(j1,i1) = intb(jp,ip)/pwr
                  endif
               end do
            end do
         
            call cfft2d(NFFT,NFFT,patch,NFFT,1)
            call spec_wgt(patch, patchmagin, patchmagout, alpha, NFFT)
            call cfft2d(NFFT,NFFT,patch,NFFT, -1)
            do i1=0, NFFT-1 
               do j1=0, NFFT-1
                  intb_filt(j+j1,i+i1) = intb_filt(j+j1,i+i1) + wf(j1,i1)*patch(j1,i1)
               end do
            end do

         end do
      end do

      return

      end 


!c****************************************************************

      subroutine spec_wgt(patch, patchmagin, patchmagout, alpha, n) 

!c****************************************************************
!c**     
!c**   FILE NAME: psfilt_sub.f
!c**     
!c**   DATE WRITTEN:19-Jan-98
!c**     
!c**   PROGRAMMER: Charles Werner, Paul Rosen
!c**     
!c**   FUNCTIONAL DESCRIPTION: weights the power spectrum of
!c**   a small image patch
!c**     
!c**   ROUTINES CALLED:
!c**     
!c**   NOTES: 
!c**     
!c**   UPDATE LOG:
!c**
!c**   Date Changed        Reason Changed                  CR # and Version #
!c**   ------------       ----------------                 -----------------
!c**    20-Jun-98         changed calculation of PSD
!c**                      intensity from cabs(patch(i,j)) to 
!c**		         real(patch(i,j)**2 + aimag(patch(i,j))**2
!c**    13-Nov-98         modified exponent alpha s.t. to conform to definition
!c**                      |psd|** alpha  by dividing alpha by 2. (clw)
!c**
!c*****************************************************************

      use icuState
      implicit none

!c     INPUT VARIABLES:

      integer*4 n
      complex patch(0:n-1,0:n-1)
      real*4 patchmagin(0:n-1,0:n-1)
      real*4 patchmagout(0:n-1,0:n-1)
      real*4 alpha
	

!c     LOCAL VARIABLES:

      integer*4  i, j, k, l, m, nn
      real*4 alpha2

!c     PROCESSING STEPS:

      do i=0,  n-1
         do j=0,  n -1
            patchmagin(i,j) = (real(patch(i,j)))**2 + (aimag(patch(i,j)))**2 
            patchmagout(i,j) = 0.0
         end do
      end do

      do i= 0, n-1
         do j = 0,  n-1
            do  k = -PSD_WIN/2, PSD_WIN/2
               m = mod(((i+k)+n),n)
               do l = -PSD_WIN/2,  PSD_WIN/2
                 nn = mod(((j+l)+n),n)
                 patchmagout(i,j) = patchmagout(i,j) + patchmagin(m,nn) 
               end do
             end do
          end do
       end do

       alpha2= alpha/2.			! filter must be in terms of amplitude, equivalent to square root! 

       do i=0,  n-1
         do j=0,  n -1
            patch(i,j) = patchmagout(i,j)**alpha2 * patch(i,j)
         end do
       end do

       return
       end 
      
