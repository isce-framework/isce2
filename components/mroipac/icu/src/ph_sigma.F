!c****************************************************************

      subroutine ph_sigma(intb, pslope, sline, eline, ssamp, esamp, winsz, sigma, sigma_cc)

!c****************************************************************
!c**     
!c**   FILE NAME: ph_sigma.f
!c**     
!c**   DATE WRITTEN: 5-Mar-98
!c**     
!c**   PROGRAMMER: Charles Werner
!c**     
!c**   FUNCTIONAL DESCRIPTION: Calculate phase standard deviation 
!c**   and effective correlation using phase gradient data.
!c**
!c**   NOTES: none
!c**     
!c**   ROUTINES CALLED: none
!c**     
!c**   UPDATE LOG:
!c**
!c**   Date Changed        Reason Changed     
!c**   ------------       ----------------
!c**   1-Nov-98  v1.1 Corrected calculation of weighting function 
!c**   1-Nov-98  v1.1 Corrected allocation of size of array sw to be fixed
!c**                  rather than a non-standard variable size
!c**
!c*****************************************************************

      use icuState
      implicit none


      real*4 NLKS
      parameter(NLKS=3)		!looks used for estimation of correlation

!c     INPUT VARIABLES:

      complex*8 intb(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)	!complex interferogram
      complex*8 pslope(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)	!phase gradients in packed format
      integer*4 sline,eline		!starting and ending line with valid data
      integer*4 ssamp,esamp		!starting and ending sample with valid data
      integer*4 winsz			!size of averaging window
	
!c     OUTPUT VARIABLES:

      real*4 sigma(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)	!phase standard deviation
      real*4 sigma_cc(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)	!effective correlation coefficient

!c     LOCAL VARIABLES:

      complex sw(0:WIN_MAX-1,0:WIN_MAX-1)	!complex deramped values
      real*4 wf(0:WIN_MAX-1,0:WIN_MAX-1) 	!weighting function window 

      complex*8 xp		!complex sum interf. window
      complex*8 ex		!complex exponential to deramp window
      complex*8 sw1		!average phase shifted to 0.0 deg.

      real*4 azph		!azimuth phase ramp
      real*4 ph			!net range and azimuth phase ramp
      real*4 s1 		!sum of weights over window
      real*4 ph_av		!average phase over the window
      real*4 ph2		!sum of squares for the phase data
      real*4 ps			!phase
      real*4 wt			!weighting factor
      real*4 xpm		!magnitude of the sum of samples in the region
      real*4 r_slp		!range phase slope
      real*4 az_slp		!azimuth phase slope
      real*4 var		!phase varienc

      integer*4 i,j,k,n		!loop indices

c     PROCESSING STEPS:

      s1=0.0			!initialize sum of weights

      do k = 0 , winsz - 1      !generate window weighting function
        do j = 0 , winsz - 1
           wt = (k - winsz/2)**2 + (j - winsz/2)**2
           wf(k,j) = exp(-wt/((winsz/2.0)))
           s1 = s1 + wf(k,j)
        end do
      end do

      do k = 0, winsz - 1         
        do j = 0, winsz - 1
           wf(k,j) = wf(k,j)/s1			!normalize weights to sum to 1.0
        end do
      end do

c$doacross local(i,j,k,xp,az_slp,r_slp,azph,n,ph,ex,sw,xpm,
c$&              ph_av,ph2,wt,sw1,ps,var),
c$&        share(sline,eline,ssamp,esamp,winsz,pslope,wf,intb,sigma,sigma_cc)
      do i = sline + winsz/2, eline - winsz/2 - 1	!azimuth loop -- trim edges
         do j = ssamp + winsz/2, esamp - winsz/2 - 1	!range loop -- trim edges

          xp = cmplx(0.0,0.0)			!weighted and deramped sum
          az_slp = aimag(pslope(j,i))		!azimuth phase slope
          r_slp = real(pslope(j,i))		!range phase slope
 
          do k = -winsz/2, winsz/2		!scan in azimuth over the estimation region
            azph = k*az_slp			!azimuth phase shift
            do n = -winsz/2, winsz/2		!scan in range over the estimation region
              ph = n*r_slp + azph		!range phase shift + azimuth phase shift
              ex = cmplx(cos(ph),-sin(ph))	!phase rotation vector
              sw(n+winsz/2,k+winsz/2) = ex*intb(j+n,i+k) !save deramped interf. samples
              xp = xp + sw(n+winsz/2, k+winsz/2) 	 !sum the samples
            end do
          end do

          xpm = cabs(xp)			!magnitude of sum of deramped samples

          if (xpm .gt. 0.0) then		!check if non-zero data

            xp = conjg(xp/xpm) 			!conjugate and normalize to unit magnitude 
            ph_av = 0.0				!initialize sum of phases 
            ph2 = 0.0				!initialize sum of squared phase values

            do k= 0, winsz-1 			!evaluate over the 2-D window
              do n= 0, winsz-1 
                wt = wf(n,k)			!window weighting function   
                sw1 = sw(n,k)*xp		!remove phase offset
                if(real(sw1) .ne. 0.0) then	!check if non-zero data
                  ps =  atan2(aimag(sw1),real(sw1))	!evaluate phase from complex  
                  ph_av = ph_av + wt*ps		!sum up weighted phase values
                  ph2 = ph2 + wt*(ps * ps)	!sum up weighted squares of phase
                end if
              end do
            end do
            
            var = ph2 - ph_av*ph_av		!phase variance
            sigma(j,i) = sqrt(var) 		!phase standard deviation sigma
            sigma_cc(j,i) = 1./sqrt(2.*NLKS*var + 1.)	!effective correlation

          else 

            sigma(j,i) = 0.0			!case for no data
            sigma_cc(j,i) = 0.0

          end if
	end do
      end do  
      return
      end 

 
