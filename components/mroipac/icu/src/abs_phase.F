!c***************************************************************************
      subroutine abs_phase(r_unw, c_ampb, r_amp, b_all_unwrap,r_bphase,
     $          r_bamp, i_complist, i_patch)

      use icuState  
      implicit none


!C   INPUT VARIABLES:

      real*4 r_xofr(0:infp%i_rsamps-1)	!azimuth offsets (pixels) for the bootstrap phase
      complex*8 c_ampb(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)		!amplitude of the two SLCs used to form the interferogram
      real*4 r_unw(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)		!unwrapped phase
      integer*1 b_all_unwrap(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1) !flag array marking all samples unwrapped in the patch
      real*4 r_bphase(0:infp%i_rsamps-1,0:NBL-1)             !bootstrap phase data
      real*4 r_bamp(0:infp%i_rsamps-1,0:NBL-1)		!bootstrap amplitude data
      integer*4 i_complist(0:1,MAXCCOMP)
      integer*4 i_patch, ml, ib, i_amb


!c   OUTPUT VARIABLES:

      real*4 r_amp(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)		!image amplitude (0 where not unwrapped)c

!c   LOCAL VARIABLES:
         
      integer*4 i_ccc           !connected component counter
      integer*4 ir, ia, ind, i_bcnt
      integer*4 i_abs_phase, i_printfreq, i_abs_phase_index
      integer*4 i_nobind, i_noblist(MAXCCOMP), i_numcc

!c     external rt_cc

!c     DATA STATEMENTS:

      data i_abs_phase_index /0/

                
!c     PROCESSING STEPS:

      i_numcc = 0
      i_nobind = 0



      !!Identify number of components to be bootstrapped
      do ir = 1, MAXCCOMP
        if(i_complist(0,ir).gt.0) then
            i_numcc = i_numcc+1
            if(i_complist(1,ir) .eq. NOBOOT) then
                print *, 'Adding component to non-boot list: ', ir
                i_nobind = i_nobind + 1
                i_noblist(i_nobind) = ir
            end if
        end if
      end do
      print *, 'Total number of connected components: ', i_numcc
      print *, 'Total components that need booting: ', i_nobind


!c  Establish amplitudes from unwrapped bootstrapped data
      do ia = unwp%i_sunw, unwp%i_eunw !initialize unwrap data arrays
         do ir = unwp%i_spixel, unwp%i_epixel
            r_amp(ir,ia) = 0.
            ind = b_all_unwrap(ir,ia)
            if(ind .ne. 0)then
               if(i_complist(1,ind) .eq. BOOT) then
                  r_amp(ir,ia) = sqrt(real(c_ampb(ir,ia))*aimag(c_ampb(ir,ia)))
               end if
            end if
         end do
      end do
      print *, 'Done fixing boot-strapped components'

!c Loop over non-bootstrapped connected components to find absolute phase

      do ib=1, i_nobind
       
        !!This part can be augmented to determine i_amb from external DEM / deformation model
        !!For now set this to zero
        i_amb = 0

        do ia = unwp%i_sunw, unwp%i_eunw
            do ir = unwp%i_spixel, unwp%i_epixel
                if (b_all_unwrap(ir,ia) .eq. i_noblist(ib)) then
                    r_amp(ir,ia) = sqrt(real(c_ampb(ir,ia))
     $                               *aimag(c_ampb(ir,ia)))
                    r_unw(ir,ia) = r_unw(ir,ia) + i_amb * TWO_PI_SP
                endif
            end do
        end do

        print *, 'Done fixing non-bootstrapped component', ib, i_noblist(ib)

        !c Update bootstrap phase with the ambiguity for this component
        do ind=0, NBL-1
            ia=infp%i_azbufsize - unwp%i_ovloff - NBL/2 + ind
            do ir = unwp%i_spixel, unwp%i_epixel
                if ((r_bamp(ir,ind) .ne. 0) .and.
     $               (b_all_unwrap(ir,ia) .eq. i_noblist(ib))) then
                    r_bphase(ir,ind) = r_bphase(ir,ind) + i_amb * TWO_PI_SP
                end if
            end do
        end do
      end do !non-bootstrapped components


!c* set the bootstrap phase

      i_bcnt = 0                !initialize counter of bootstrap points
      do ind = 0, NBL-1
         do ir = unwp%i_spixel, unwp%i_epixel
            ia = infp%i_azbufsize - unwp%i_ovloff - NBL/2 + ind
            r_bamp(ir,ind)   = r_amp(ir,ia)
            r_bphase(ir,ind) = r_unw(ir,ia)
            if(r_amp(ir,ia) .ne. 0) then
               i_bcnt = i_bcnt + 1
            end if
         end do
      end do
      print *, 'Done with bootstrap counter for patch: ', i_patch

      if(i_bcnt .eq. 0) then
        print *, 'WARNING: UNW_RT: no points for phase bootstap of next patch' 
      else 
        print *, 'UNW_RT: phase bootstrap points for next patch: ',i_bcnt	
      end if
      

      end

