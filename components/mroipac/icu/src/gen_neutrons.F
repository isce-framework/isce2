!c**************************************************************************************

      subroutine gen_neutrons(flag, intb_filt, ampb, cor, pslope, nr_start, nr_end,
     $                        naz_start, naz_end, neutypes, neuthres)

!c**************************************************************************************
!c**     
!c**   FILE NAME: gen_neutrons.f
!c**     
!c**   DATE WRITTEN: 25-Aug-97
!c**     
!c**   PROGRAMMERS: Charles Werner, Scott Hensley
!c**     
!c**   FUNCTIONAL DESCRIPTION: Subroutine to calculate neutrons from
!c**   intensity, correlation, and phase gradients. Several different
!c**   algorithms are used to select neutrons, including the TOPSAR range
!c**   phase gradient, the range phase gradient estimated from the filtered
!c**   interferogram, the local intensity and correlation, and finally, the
!c**   second derivative of the range phase.
!c**
!c**
!c**   Algorithm 1 Range phase gradient
!c**   ****************************************
!c**   
!c**   This algorithm uses the range phase gradient determined by 
!c**   averaging the first finite differences of interferometric phase.
!c**   Averaging is carried out over a 5x5 moving patch with exponential
!c**   weighting of the differences. The weighting function falls off
!c**   as the inverse square of the distance from the central sample and
!c**   is estimated in both the range and azimuth directions.
!c**   
!c**   A threshold on the magnitude of the phase gradient is used to decide
!c**   if a neutron should be placed at a particular location.
!c**   
!c**   Algorithm 2 Intensity threshold neutrons
!c**   ****************************************
!c**   
!c**   Layover in SAR images is accompanied by relatively radar high backscatter
!c**   and low interferometric correlation. Since these areas should not be unwrapped  
!c**   or traversed by the unwrapped, a large number of neutrons, in the 
!c**   presense of charges, will create a thicket of branch cuts that 
!c**   will exclude the region. All points classified
!c**   as layover and subsequently marked by neutrons must pass two tests.  
!c**   The first test checks the scene intensity. Due to the
!c**   variability in scene reflectance, an adaptive scheme for detection
!c**   of layover has been implemented that compares a particular pixels value
!c**   with the average intensity over the scene. For a point to be classified
!c**   as layover, it is necessary that the  local intensity exceed the scene 
!c**   average by a  specified number of multiples of the image 
!c**   intensity standard deviation.  The number of
!c**   multiples is in the range of 1.5 to 2.5 depending on the image SNR, 
!c**   and average change density.
!c** 
!c**   Another characteristic of layover is that the correlation is low. In
!c**   order to differentialte between bright targets that are not layover and
!c**   layover, a second test is implemented that checks that the correlation
!c**   is below a threshold, typically .7. Points that pass both tests are
!c**   marked.
!c**     
!c**   ROUTINES CALLED:
!c**     
!c**   NOTES:  Neutron algorthm flags:
!c**              
!c**    smoothed range gradient:                1
!c**    intensity with correlation:             2
!c**
!c**     
!c**   UPDATE LOG:
!c**
!c**   Date Changed        Reason Changed                  CR # and Version #
!c**   ------------       ----------------                 -----------------
!c**    10-Jul-97            Created                          v1.0
!c**    27-Aug-97            updated for sizes.inc            v1.1
!c**
!c*********************************************************************************************
      use icuState
      implicit none

!c     INPUT VARIABLES:

      complex*8 intb_filt(0:infp%i_rsamps-1,0:infp%i_azbufsize-1) !amplitude data
      complex*8 ampb(0:infp%i_rsamps-1,0:infp%i_azbufsize-1) 	!amplitude data
      real*4  cor(0:infp%i_rsamps-1,0:infp%i_azbufsize-1) 	!correlation (either normalized or unnormalized) 
      complex*8 pslope(0:infp%i_rsamps-1,0:infp%i_azbufsize-1) 	!phase slope data
      integer*4 nr_start, nr_end		!start and ending range sample
      integer*4 naz_start, naz_end		!start and ending azimuth sample
      integer*4 neutypes(MAXNEUTYPES)		!array with flags to select different ways of determining neutrons
      real*4  neuthres(MAXNEUTYPES,MAXTHRES)	!thresholds for each type of neutron

!c     OUTPUT VARIABLES:

      integer*1  flag(0:infp%i_rsamps-1,0:infp%i_azbufsize-1) 	!flag array to receive neutrons 

!c     LOCAL VARIABLES:

      integer i,j 		!indices
      integer*4 nv		!total neutron counter
      integer*4 ns		!number of points used to estimate intensity and variance

      real*8 sum1,sum2		!sum of image intensity, sum of squares
      real*4 var		!variance of the intensity
      real*4 thr_pwr2		!intensity threshold
      real*4 sigma		!standard deviation of the intensity
      real*4 av			!average scene intensity
      real*4 dph		!phase step
      real*4 pwr		!intensity

!c     PROCESSING STEPS:

      nv = 0			!initialize neutron counter
     
      if(neutypes(1) .eq. 1) then
c$doacross local(i,j,dph),
c$&        share(nr_start,nr_end,naz_start,naz_end,flag,pslope,neuthres), reduction(nv)
        do j=naz_start+1, naz_end		!loop over lines in azimuth, then range
          do i=nr_start+1, nr_end
            dph = abs(real(pslope(i,j)))
            if (dph .gt. neuthres(1,1) ) then 	!nominal threshold = .25 *PI
              nv = nv+1          		!increment local neutron count
              flag(i,j) = IOR(flag(i,j),NEUTRON)		!set the neutron flag
            end if
          end do
        end do
!c        write(6,'(1x,a,i7)')'GEN_NEUTRONS: phase gradient neutrons: ',nv
      end if

      if(neutypes(2) .eq. 1) then	!intensity neutrons
        nv = 0			        !initialize local neutron counter
        sum1 = 0.0			!sum of intensities
        sum2 = 0.0			!sum of squared intensities
        ns = 0				!initialize number of points in the sums		
 
c$doacross local(i,j,pwr),
c$&        share(nr_start,nr_end,naz_start,naz_end,ampb), reduction(sum1,sum2,ns)
        do j=naz_start+16, naz_end-16, 4 
          do i=nr_start+32, nr_end-32, 4	!evaluate mean and variance of the intensity			 
             pwr = (real(ampb(i,j)))**2
             sum1 = sum1 + pwr
             sum2 = sum2 + pwr**2
             ns = ns + 1
          end do
        end do

        av = sum1/float(ns)
        var = sum2/float(ns)
        sigma = sqrt(var - av*av)		!standard deviation
        thr_pwr2 = av + neuthres(2,1)*sigma	!intensity threshold

!c        write(6,'(1x,a,1pg12.5)')'GEN_NEUTRONS: average image intensity:            ',av
!c        write(6,'(1x,a,1pg12.5)')'GEN_NEUTRONS: image intensity standard deviation: ',sigma

c$doacross local(i,j),
c$&        share(nr_start,nr_end,naz_start,naz_end,flag,ampb,cor
c$&         ,thr_pwr2),reduction(nv)
        do j = naz_start, naz_end 
          do i = nr_start, nr_end	

!  if(((real(ampb(i,j)))**2 .gt. thr_pwr2) .and. (cor(i,j) .lt. neuthres(2,2))) then

            if(((real(ampb(i,j)))**2 .gt. thr_pwr2) .and. ((cor(i,j) .lt. neuthres(2,2)))) then
              flag(i,j) = IOR(flag(i,j),NEUTRON)		!set the neutron flag
              nv = nv + 1
            end if
          end do
        end do
 !c       write(6,'(1x,a,i7)')'GEN_NEUTRONS: image intensity neutrons: ',nv
      end if   
      return
      end
