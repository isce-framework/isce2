!c***************************************************************************
      subroutine unw_rt( c_intb, c_ampb, c_pslope, r_corr,  r_xofr,
     $     r_unw, b_PatchTrees, b_all_unwrap, r_bphase, r_bamp,
     $     i_complist, i_unw_tot)
      
      use icuState
      implicit none


!C   INPUT VARIABLES:

      complex*8 c_intb(0:infp%i_rsamps-1,0:infp%i_azbufsize-1)          !complex interferogram
      complex*8 c_ampb(0:infp%i_rsamps-1,0:infp%i_azbufsize-1)          !amplitude data of the two images
      complex*8 c_pslope(0:infp%i_rsamps-1,0:infp%i_azbufsize-1)                !phase gradiant
      real*4 r_corr(0:infp%i_rsamps-1,0:infp%i_azbufsize-1)             !interferometric correlation

      real*4 r_xofr(0:infp%i_rsamps-1)  !azimuth offsets (pixels) for the bootstrap phase

!c   OUTPUT VARIABLES:

      real*4 r_unw(0:infp%i_rsamps-1,0:infp%i_azbufsize-1)           !unwrapped phase
      integer*1 b_PatchTrees(0:infp%i_rsamps-1,0:infp%i_azbufsize-1) !unwrapping flag array
      integer*1 b_all_unwrap(0:infp%i_rsamps-1,0:infp%i_azbufsize-1) !flag array marking all samples unwrapped in the patch
      real*4 r_bphase(0:infp%i_rsamps-1,0:NBL-1)             !bootstrap phase data
      real*4 r_bamp(0:infp%i_rsamps-1,0:NBL-1)          !bootstrap amplitude data
      integer*4 i_complist(0:1,MAXCCOMP)
      integer*4 i_unw_tot       !total number of unwrapped pixels
      
!c   LOCAL VARIABLES:
         
      real*4, dimension(:,:), allocatable :: r_phase
      real*4, dimension(:,:), allocatable :: r_area
      
      integer*4, dimension(0:infp%i_rsamps-1):: i_azboot                        !azimuth positions of bootstrap phase values in the current patch
!c      integer*2, dimension(0:infp%i_rsamps*infp%i_azbufsize/MEM_RES_FACTOR-1) :: i_iz,i_jz    !lists of residues in the interferogram
      integer*4, dimension(:), allocatable :: i_iz,i_jz         !lists of residues in the interferogram

      real*4 r_dphase           !number of multiples of 2pi between unwrapped phase and bootstrap phase values
      real*4 r_corr_ave         !average correlation in a connected component
      real*4 r_corrthres        !current correlation threshold
      real*4 r_sumph            !sum of phase values along the bootstrap line 
      real*4 r_sumph2           !sum of square of phases along the bootstrap line
      real*4 r_phvar            !variance of the phases along the bootstrap line
      real*4 r_area_max         !maximum patch fraction unwrapped by any seed
      real*4 r_area_tot         !total fraction of the patch unwrapped
      
      integer*4 i_res_chg       !residual charge
      integer*4 i_num_seeds     !total number of seeds used in the unwrap in range
      integer*4 j_num_seeds     !total number of seeds used in the unwrap in lines
      integer*4 i_num_seeds_found !total number of seeds not masked
      integer*4 i_seed_cntr     !counter of the number of range seeds used in the connected comp unwrap
      integer*4 j_seed_cntr     !counter of the number of line seeds used in the connected comp unwrap
      integer*4, dimension(infp%i_rsamps) :: i_seed_index !cross track pixel locations for seeds used in connected comp. unwrap
      integer*4, dimension(infp%i_azbufsize) :: j_seed_index   !line locations for seeds used in connected comp. unwrap
      integer*4, dimension(:,:), allocatable :: i_seed_flag   !flag to indicate if a particular range seed has been visited
      integer*4 i_seed_max      !range sample number of the seed that unwrapped the most of the patch
      integer*4 j_seed_max      !line         number of the seed that unwrapped the most of the patch
      integer*4 i_unw_cntr      !number of unwrapped pixels in a connected component
      integer*4 i_cnt           !number of phase values available for bootstrap for a connected comp.
      integer*4 ir,ia           !loop indices
      integer*4 i_nres          !number of residues in the patch
      integer*4 i,j,ibl         !loop indices
      integer*4 i_mxrng         !array size in range (across)
      integer*4 i_mxnaz         !array size in azimuth (down)
      integer*4 i_azbpt         !azimuth position to extract bootstrap phase
      integer*4 i_ccc           !connected component counter
      
!      reference to undefined (and unused) external rt_cc
!      caused g95 link failure.
!      integer*4 rt_cc          !external subroutine

      integer*4 i_unwrap_rt, i_printfreq, i_unwrap_rt_index

!      external rt_cc

!c     DATA STATEMENTS:

      data i_unwrap_rt_index /0/

                
!c     PROCESSING STEPS:


      ALLOCATE ( r_phase(0:infp%i_rsamps-1,0:infp%i_azbufsize-1) )
      ALLOCATE ( r_area (0:infp%i_rsamps-1,0:infp%i_azbufsize-1) )
      ALLOCATE ( i_iz   (0:infp%i_rsamps*infp%i_azbufsize-1) )
      ALLOCATE ( i_jz   (0:infp%i_rsamps*infp%i_azbufsize-1) )
      ALLOCATE ( i_seed_flag (infp%i_rsamps,infp%i_azbufsize) )

      do ia = 0, infp%i_azbufsize - 1
        do ir = 0, infp%i_rsamps - 1
          b_PatchTrees(ir,ia) = 0
          b_all_unwrap(ir,ia) = 0
          r_unw(ir,ia) = 0.0
          r_phase(ir,ia) = 0.0                  
        end do
      end do

!c initialize connected component list to 0

      do ia = 1, MAXCCOMP
         i_complist(0,ia) = 0
         i_complist(1,ia) = NOBOOT
      end do

      unwp%i_sunw = infp%i_sline + NFFT/8       !starting output line
      unwp%i_eunw = infp%i_eline - NFFT/8       !ending unwrap line
      unwp%i_spixel = infp%i_ssamp + NFFT/8     !first valid pixel unwrapped (0 base index)
      unwp%i_epixel = infp%i_esamp - NFFT/8     !last valid pixel unwrapped  (0 base index)

      print *,'INIT_UNW_RT: starting unwrap line:  ',unwp%i_sunw+1, '  ending unwrap line:  ',unwp%i_eunw+1 
      print *,'INIT_UNW_RT: starting unwrap pixel: ',unwp%i_spixel+1,'  ending unwrap pixel: ',unwp%i_epixel+1 
   
      do ir = unwp%i_spixel, unwp%i_epixel      !calculate azimuth positions of bootstrap phase line
        i_azboot(ir) = unwp%i_ovloff - nint(r_xofr(ir))
        if (i_azboot(ir) .lt. 0)then
          print *, 'WARNING: UNW_RT: phase bootstrap line does not lie within patch'
          i_azboot(ir) = 0
        end if
      end do  
        
      call gen_neutrons(b_PatchTrees, c_intb, c_ampb, r_corr, c_pslope,
     $  infp%i_ssamp, infp%i_esamp, infp%i_sline, infp%i_eline, unwp%i_neutypes, unwp%r_neuthres)
          
      call residues(c_intb, b_PatchTrees, r_phase, infp%i_ssamp, infp%i_esamp,   
     $  infp%i_sline, infp%i_eline, i_iz, i_jz, i_nres)

      print *, 'UNW_RT: number of residues: ',i_nres

      if(unwp%i_tree_type .eq. TREE_GZW) then
        print *, 'UNW_RT: GZW trees for unwrapping'
        call rt(b_PatchTrees, i_iz, i_jz, i_nres, unwp%i_spixel, unwp%i_epixel, 
     $   unwp%i_sunw, unwp%i_eunw, unwp%i_tree_sets, i_res_chg)
      else if(unwp%i_tree_type .eq. TREE_CC) then
        print *, 'UNW_RT: CC trees for unwrapping'
      
        i_mxrng = infp%i_rsamps
        i_mxnaz = infp%i_azbufsize

!c        write(6,'(1x,a,i10)')'ERROR: UNW_RT: tree type not implemented: ',unwp%i_tree_type
        stop
       else 
!c        write(6,'(1x,a,i10)')'ERROR: UNW_RT: invalid tree type: ',unwp%i_tree_type
        stop
      end if

!c      write(6,'(1x,a,i10)')'UNW_RT: patch residual charge: ',i_res_chg

      r_phvar =  unwp%r_phvarmax                !initialize phase variance             
      r_corrthres = unwp%r_ccthr_min            !initial correlation threshold

      do while ((r_corrthres .le. unwp%r_ccthr_max) .and. (r_phvar .ge. unwp%r_phvarmax))

 !c       write(6,'(/1x,a,f8.6)')'UNW_RT: correlation threshold: ',r_corrthres
!c        if (r_corrthres .ne. unwp%r_ccthr_min) write(6,'(1x,a,f10.6)')'unwrapped phase variance: ',r_phvar 

        do ia = unwp%i_sunw, unwp%i_eunw        !initialize unwrap data arrays
          do ir = unwp%i_spixel, unwp%i_epixel
            b_all_unwrap(ir,ia) = 0             !unmark all regions unwrapped
            r_unw(ir,ia) = 0.
            b_PatchTrees(ir,ia) = IAND(b_PatchTrees(ir,ia),NOT(LAWN))

!c    Mark low correlation regions

            if(r_corr(ir,ia) .lt. r_corrthres) then
              b_PatchTrees(ir,ia) = IOR(b_PatchTrees(ir,ia),LCORR )
            else
              b_PatchTrees(ir,ia) = IAND(b_PatchTrees(ir,ia),NOT(LCORR))
            end if

          end do       
        end do
        i_num_seeds = (unwp%i_epixel - unwp%i_spixel - unwp%i_minbootpts)/unwp%i_minbootpts
        j_num_seeds = max((unwp%i_eunw - unwp%i_sunw - unwp%i_minbootlns)/unwp%i_minbootlns,1)
        i_num_seeds_found = 0
        write(*,*) 'Number of Range Seeds ',i_num_seeds
        write(*,*) 'Number of Line  Seeds ',j_num_seeds
        write(*,*) 'Total Seeds           ',i_num_seeds*j_num_seeds

!c        do i = 1 , 100
!c           write(*,*) i, i_iz(i), i_jz(i), b_PatchTrees(i_iz(i), i_jz(i))
!c        end do
        do i=1,i_num_seeds
           i_seed_index(i) = unwp%i_spixel + unwp%i_minbootpts/2 
     &                       + (i-1)*unwp%i_minbootpts !seed on CUT or LCORR?
           do j=1,j_num_seeds
              r_area(i,j) = 0.  !initialize fraction of patch unwrapped
              i_seed_flag(i,j) = 0          
              j_seed_index(j) = min(unwp%i_sunw + unwp%i_minbootlns/2 
     &                          + (j-1)*unwp%i_minbootlns,
     &              unwp%i_eunw - unwp%i_minbootlns/2) !seed on CUT or LCORR?

              if ((IAND(b_PatchTrees(i_seed_index(i), j_seed_index(j)), CUT) .eq. CUT) .or. (IAND(b_PatchTrees(i_seed_index(i),j_seed_index(j)), LCORR) .eq. LCORR)) then
                i_seed_flag(i,j)=1
              else
                  i_num_seeds_found = i_num_seeds_found + 1
              end if
           end do
        end do
      
        !c Raise coherence threshold if no seed is unwrappable
        if (i_num_seeds_found .eq. 0) then
            print *, 'No seeds in unwrappable area; raising threshold'
            r_corrthres = r_corrthres + unwp%r_ccthr_incr
            goto 150
        end if
!c     Unwrap phase in each connected component of the image and
!c     maintain absolute phase in each connected component by phase
!c     bootstrap
        
        r_area_max = 0.0        !initialize area of largest connected comp
        i_seed_max = 1          !range sample number of seed for largest connected comp.
        j_seed_max = 1          !range sample number of seed for largest connected comp.
        i_unw_tot = 0           !initialize counter of all connected comp. samples
        i_ccc = 0

        do i_seed_cntr = 1, i_num_seeds                 !unwrap all seeds                 
           do j_seed_cntr = 1, j_num_seeds !unwrap all seeds                 

              if(i_seed_flag(i_seed_cntr,j_seed_cntr) .ne. 0) then
                 goto 100       !test if seed location already unwrapped
              end if
          
!c    unwrap from current seed at (i_seed_index(i_seed_cntr),j_seed_index(j_seed_cntr))

              do ia = unwp%i_sunw, unwp%i_eunw !remove any old LAWN, test if any points unwrapped
                 do ir = unwp%i_spixel, unwp%i_epixel
                    if(IAND(b_PatchTrees(ir,ia), LAWN) .eq. LAWN) then 
                       b_PatchTrees(ir,ia) = IAND(b_PatchTrees(ir,ia),NOT(LAWN))
                    end if
                 end do
              end do
              
              call grass(r_phase,i_seed_index(i_seed_cntr), j_seed_index(j_seed_cntr), 
     $             b_PatchTrees,unwp%i_spixel, unwp%i_epixel, unwp%i_sunw,
     $             unwp%i_eunw, r_unw, i_unw_cntr)
              
              r_area(i_seed_cntr,j_seed_cntr) = float(i_unw_cntr)/
     $             float((unwp%i_epixel - unwp%i_spixel + 1)*(unwp%i_eunw -
     $             unwp%i_sunw + 1))

              
              if (r_area(i_seed_cntr,j_seed_cntr) .lt. 0.1/MAXCCOMP) then
                 goto 100       !test if connected component is large enough
              end if
              print *, 'seed:',i_seed_cntr,j_seed_cntr
              print *, 'range:',i_seed_index(i_seed_cntr)
              print *, 'azimuth:',j_seed_index(j_seed_cntr)
              print *, 'points unwrapped: ',i_unw_cntr
              print *, 'area fraction: ',r_area(i_seed_cntr,j_seed_cntr)
              i_ccc = i_ccc + 1
              i_complist(0,i_ccc) = i_unw_cntr
              r_corr_ave = 0.0  !determine average correlation, init sum of corr.
              do ia = unwp%i_sunw, unwp%i_eunw !test if points are unwrapped before averaging
                 do ir = unwp%i_spixel, unwp%i_epixel
                    if(IAND(b_PatchTrees(ir,ia),LAWN) .eq. LAWN) then  
                       r_corr_ave = r_corr_ave + r_corr(ir,ia)
                    end if
                 end do
              end do
              r_corr_ave = r_corr_ave/float(i_unw_cntr)
              print *, 'average correlation of connected component: ',r_corr_ave
              
!c     mark all seed locations that have been unwrapped in this connected comp.

              do i=1, i_num_seeds                       
                 do j=1, j_num_seeds                    
                    if(IAND(b_PatchTrees(i_seed_index(i), j_seed_index(j)), LAWN) .eq. LAWN)then 
                       i_seed_flag(i,j) = 1
                    end if
                 end do
              end do

!c     evaluate phase variance along the phase bootstrap region for this
!c     connected component
              
              r_sumph = 0.      !init sum of phases and sum of squares
              r_sumph2 = 0.
              i_cnt = 0
              
              do ibl = 0, NBL-1        
                 do ir = unwp%i_spixel, unwp%i_epixel
                    i_azbpt = i_azboot(ir) - NBL/2 + ibl !azimuth position of bootstrap data
                    if((IAND(b_PatchTrees(ir,i_azbpt), LAWN) .eq. LAWN)        !test if bootstrap unwrapped, and unwrapped
     1                    .and.(r_bamp(ir,ibl) .gt. 0.0)) then
                       i_cnt = i_cnt +1 !count up phase bootstrap points
                       r_sumph  = r_sumph  + (r_unw(ir,i_azbpt) - r_bphase(ir,ibl) - r_xofr(ir))
                       r_sumph2 = r_sumph2 + (r_unw(ir,i_azbpt) - r_bphase(ir,ibl) - r_xofr(ir))**2
                    end if
                 end do 
              end do

!c     Test to see if any points on the bootstrap phase unwrap line were
!c     unwrapped in the current patch
!c     and evaluate the phase difference and variance for these points.
!c     If none are in common, unmark
!c     all points unwrapped and then try a new seed point for unwrapping
     
              if(i_cnt .ge. unwp%i_minbootpts) then !test for overlap on the bootstrap line
                 i_complist(1,i_ccc) = BOOT
                 r_sumph = r_sumph/float(i_cnt) 
                 r_phvar = r_sumph2/float(i_cnt) - r_sumph**2
                 r_dphase = nint(r_sumph/TWO_PI_SP)*TWO_PI_SP
               
                 print *, 'Number of bootstrap points: ', i_cnt
                 print *, 'Adjustment phase: ', r_dphase
                 print *, 'Phase difference: ', r_sumph
                 print *, 'Cycles difference: ', r_dphase/TWO_PI_SP
                 
              else              !insufficient overlap so go on to next seed          
                print *, 'WARNING: UNW_RT: insufficient points for phase bootstrap: ', i_cnt
                 i_complist(1,i_ccc) = NOBOOT
              end if                     
     
!c     check phase variance, if too high, then unmark all the components and try again
     
              if(i_complist(1,i_ccc) .eq. BOOT .and. r_phvar .lt. unwp%r_phvarmax) then

                 if(r_area(i_seed_cntr,j_seed_cntr) .gt. r_area_max)then
                    r_area_max = r_area(i_seed_cntr,j_seed_cntr)
                    i_seed_max = i_seed_index(i_seed_cntr)
                    j_seed_max = j_seed_index(j_seed_cntr)
                 end if
 
                 do ia = unwp%i_sunw, unwp%i_eunw
                    do ir =  unwp%i_spixel, unwp%i_epixel
                       if(IAND(b_PatchTrees(ir,ia), LAWN) .eq. LAWN)   !check if unwrapped
     &                 then 
                          b_all_unwrap(ir,ia) = i_ccc              !mark as unwrapped in map of all unwrapped connected comp.
                          i_unw_tot = i_unw_tot+1                  !incr. counter of unwrapped connected comp. samples
                          r_unw(ir,ia) = r_unw(ir,ia) - r_dphase   !unwrap phase
                       end if
                    end do
                 end do

              elseif(i_complist(1,i_ccc) .eq. BOOT 
     &               .and. r_phvar .ge. unwp%r_phvarmax)then        !phase variance too high
                                
                 print *,
     $                 'WARNING: UNW_RT: phase variance above threshold: unmarking all components'
                 goto 150       
                
              else

                 r_phvar = 0.
                 do ia = unwp%i_sunw, unwp%i_eunw
                    do ir =  unwp%i_spixel, unwp%i_epixel
                       if(IAND(b_PatchTrees(ir,ia), LAWN) .eq. LAWN)   !check if unwrapped
     &                 then
                          b_all_unwrap(ir,ia) = i_ccc    !mark as unwrapped in map of all unwrapped connected comp.
                          i_unw_tot = i_unw_tot+1        !incr. counter of unwrapped connected comp. samples
                       end if
                    end do
                 end do
              
              endif             
               
 100          continue
           end do               !end of loop over seed locations
        end do
 150    continue
        r_corrthres = r_corrthres + unwp%r_ccthr_incr
      end do                    !loop over correlation thresholds

 200  if(r_corrthres .le. unwp%r_ccthr_max) then

        r_area_tot = real(i_unw_tot) / real((unwp%i_epixel - unwp%i_spixel + 1) * (unwp%i_eunw - unwp%i_sunw + 1))
        print *, 'UNW_RT: points unwrapped:',i_unw_tot,'  area fraction:',r_area_tot       

        do ia = unwp%i_sunw, unwp%i_eunw        !update the b_PatchTrees LAWN flag, clean up output arrays
          do ir = unwp%i_spixel, unwp%i_epixel
            if(b_all_unwrap(ir,ia) .ne. 0)then  !use i_all_unwrap flags to update LAWN flag in b_PatchTrees
              b_PatchTrees(ir,ia) =  IOR(b_PatchTrees(ir,ia),LAWN)
            else
              r_unw(ir,ia) = 0.0                !delete any connected comp. regions where bootstrap failed
            end if                   
          end do
        end do

      endif                                      !no consistant unwrapping possible for this patch

      do ibl = 0, NBL-1
        do ir = 0, infp%i_rsamps-1         !clear bootstrap phase data 
            r_bphase(ir,ibl) = 0.0
            r_bamp(ir,ibl) = 0.0                !mark as not unwrapped
        end do
      end do


      do ibl = 0, NBL-1
        ia = infp%i_azbufsize - unwp%i_ovloff - NBL/2 + ibl
        do ir=0, infp%i_rsamps-1
            if (b_all_unwrap(ir,ia) .ne. 0) then
                r_bphase(ir,ibl) = r_unw(ir,ia)
                r_bamp(ir,ibl) = sqrt(real(c_ampb(ir,ia))*aimag(c_ampb(ir,ia)))
            endif
        end do
      end do

      DEALLOCATE ( r_phase )
      DEALLOCATE ( r_area )
      DEALLOCATE ( i_iz   )
      DEALLOCATE ( i_jz   )
      DEALLOCATE ( i_seed_flag )

      return
      end    
