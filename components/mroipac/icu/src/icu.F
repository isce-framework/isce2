!c****************************************************************

       subroutine icu(intAcc,ampAcc,filtAcc,corrAcc,gccAcc,phsigcorrAcc,unwAcc,conncompAcc) 

      use icuState
      implicit none

!c    PARAMETER STATEMENTS:

      integer*8 intAcc,ampAcc,corrAcc
      integer*8 filtAcc,gccAcc,phsigcorrAcc
      integer*8 unwAcc,conncompAcc

!c**************Local Variable Definitions ******************* 
      complex*8 patch(0:NFFT-1, 0:NFFT-1)               !used for initialization of FFT 


      complex*8, dimension(:,:), allocatable :: c_intb
      complex*8, dimension(:,:), allocatable :: c_ampb
      complex*8, dimension(:,:), allocatable :: c_intb_filt
      complex*8, dimension(:,:), allocatable :: c_pslope

      real*4, dimension(:,:,:), allocatable :: r_cc
      real*4, dimension(:,:), allocatable :: r_sigma
      real*4, dimension(:,:), allocatable :: r_unw
      real*4, dimension(:,:), allocatable :: r_amp
      real*4, dimension(:,:), allocatable :: r_bphase
      real*4, dimension(:,:), allocatable :: r_bamp
      real*4, dimension(:), allocatable  :: r_xofr

      integer*4 i_complist(0:1,MAXCCOMP)
      integer*4 i_azskip        !number of lines to increment to the start of the next patch
      integer*4 i_unw_tot       !total number of unwrapped pixels
      integer*4 i_sl, i_el      !starting and last output line/patch (0 based arrays)
      integer*4 i_patch         !patch number (starts with 1)
      integer*4 i_numpatch      !number of patches
      integer*4 i_azovlp        !overlap between patches in lines
      integer*4 i_bcnt          !number of points available for bootstrap of the phase
      integer*4 ia              !loop index for azimuth line
      integer*4 j               !starting line of the current patch in the interferogram
      integer*4 i, l         !loop indices
      integer*4 b1,b2        !band indices

      integer*1, dimension(:,:), allocatable ::  b_PatchTrees
      integer*1, dimension(:,:), allocatable ::  b_all_unwrap
      

      write(*,'(/1x,a/)') '<< PS filtering >>'

      !c Array allocation
      ALLOCATE( c_intb(0:infp%i_rsamps-1,0:i_azbuf-1) )
      ALLOCATE( c_ampb(0:infp%i_rsamps-1,0:i_azbuf-1) )
      ALLOCATE( c_intb_filt(0:infp%i_rsamps-1,0:i_azbuf-1) )
      ALLOCATE( c_pslope(0:infp%i_rsamps-1,0:i_azbuf-1) )

      ALLOCATE( r_cc(0:infp%i_rsamps-1,0:i_azbuf-1,3) )
      ALLOCATE( r_sigma(0:infp%i_rsamps-1,0:i_azbuf-1) )
      ALLOCATE( r_unw(0:infp%i_rsamps-1,0:i_azbuf-1) )
      ALLOCATE( r_amp(0:infp%i_rsamps-1,0:i_azbuf-1) )
      ALLOCATE( r_bphase(0:infp%i_rsamps-1,0:NBL-1) )
      ALLOCATE( r_bamp(0:infp%i_rsamps-1,0:NBL-1) )
      ALLOCATE( r_xofr(0:infp%i_rsamps-1) )
      ALLOCATE( b_PatchTrees(0:infp%i_rsamps-1,0:i_azbuf-1) )
      ALLOCATE( b_all_unwrap(0:infp%i_rsamps-1,0:i_azbuf-1) )

      write(*,'(/1x,a,i6,a,i6)') 'interferogram width:',infp%i_rsamps,'  number of lines/patch:',i_azbuf
      write(*,'(1x,a,i6,a,i6)') 'start line:   ',i_strtline,  '  number of lines:  ',i_numlines
      write(*,'(1x,a,i6,a,i6)') 'start sample: ',infp%i_ssamp,'  end sample:       ',infp%i_esamp
        
      if(infp%i_esamp .gt. infp%i_rsamps) then
        write(*,'(1x,a,1x,i5,1x,a,1x,i5/)') 'ERROR: specified far edge of valid data exceeds specified width: ',
     $        infp%i_esamp,'greater than',infp%i_rsamps
        stop
      end if

      if(infp%i_ssamp .lt. 1) then
        write(*,'(1x,a,1x,i5/)') 'ERROR: specified near edge of valid data less than 1: ',infp%i_ssamp
        stop
      end if

      infp%i_ssamp = infp%i_ssamp - 1                   !adjust bounds for 0 base array indices
      infp%i_esamp = infp%i_esamp - 1

      unwp%i_spixel = infp%i_ssamp
      unwp%i_epixel = infp%i_esamp

      if(infp%i_cc_winsz .gt. WIN_MAX)then
        write(*,'(1x,a,x,i5,x,a,x,i5/)') 'ERROR: corr. estimation box size exceeds limit: ',
     $   infp%i_cc_winsz,'greater than',WIN_MAX
         stop
      end if

      if(infp%i_phs_winsz .gt. WIN_MAX)then
        write(*,'(1x,a,1x,i5,1x,a,1x,i5/)') 'ERROR: phase std. dev. estimation box exceeds limit: ',
     $   infp%i_phs_winsz,'greater than',WIN_MAX
         stop
      end if

!c     initialize debug data structure and output 

      call cfft2d(NFFT,NFFT,patch,NFFT,0)       !initialize FFT

      if(i_unwrap_flag .eq. 1)then

        write(*,'(/1x,a/)') '<< Unwrapping with icu, random trees 3-Nov-98 CW/PAR/SH >>'

        if(unwp%i_tree_type .eq. TREE_GZW) then
          write(*,'(1x,a)') 'Branch Cut Tree Type:             GZW'
        else if (unwp%i_tree_type .eq. TREE_CC) then
          write(*,'(1x,a)') 'Branch Cut Tree Type:              CC'
        endif
        write(*,'(1x,a,i8)') 'number of realizations of the trees:         ',unwp%i_tree_sets
        write(*,'(1x,a,f8.4)') 'minimum unwrap correlation threshold:        ',unwp%r_ccthr_min 
        write(*,'(1x,a,f8.4)') 'maximum unwrap correlation threshold:        ',unwp%r_ccthr_max
        write(*,'(1x,a,f8.4)') 'bootstrap phase variance threshold:          ',unwp%r_phvarmax
        write(*,'(1x,a,i8/) ') 'min. points overlap for the bootstrap:       ',unwp%i_minbootpts
        write(*,'(1x,a,i8/) ') 'this is also seed spacing in range:          ',unwp%i_minbootpts
        write(*,'(1x,a,i8/) ') 'line seed spacing:                           ',unwp%i_minbootlns
        write(*,'(1x,a,i8)') 'phase gradient neutron flag:                 ',unwp%i_neutypes(1)           
        write(*,'(1x,a,f8.4)') 'phase gradient neutron threshold (radians):  ',unwp%r_neuthres(1,1)
        write(*,'(1x,a,i8)') 'intensity neutron flag:                      ',unwp%i_neutypes(2)           
        write(*,'(1x,a,f8.4)') 'intensity neutron thres. (sigma above mean): ',unwp%r_neuthres(2,1)           
        write(*,'(1x,a,f8.4)') 'maximum correlation for intensity neutrons:  ',unwp%r_neuthres(2,2)        

        i_bcnt = 0              !init. number of bootstrap points for first patch
        do i=0, infp%i_rsamps-1 !set azimuth shift for patch to 0.0 samples across swath
          r_xofr(i) = 0.0
        end do

      endif

      i_azovlp = i_azcomlin + 2*NFFT            !overlap between patches
      i_azskip = i_azbuf - i_azovlp             !number of lines to skip for next patch
      unwp%i_ovloff = i_azovlp/2                !offset to the bootstrap phase line
     
      if(mod(i_numlines, i_azskip) .le. i_azovlp) then
        i_numpatch=i_numlines/i_azskip
      else 
        i_numpatch = i_numlines/i_azskip+1
      end if

      write(*,'(/1x,a,i5)') 'azimuth buffer size:                            ',i_azbuf
      write(*,'(1x,a,i5)') 'overlap between azimuth patches:                ',i_azcomlin
      write(*,'(1x,a,i5)') 'total overlap between azimuth patches:          ',i_azovlp
      write(*,'(1x,a,i5)') 'offset in overlap region for phase  bootstrap:  ',unwp%i_ovloff
      write(*,'(1x,a,i5)') 'lines to increment for the next patch:          ',i_azskip
      write(*,'(1x,a,i5)') 'number of patches:                              ',i_numpatch
      b1=1
      b2=2

      do i_patch = 1, i_numpatch        !main processing loop
         j = i_strtline - 1 + (i_patch-1)*i_azskip      !starting line for the patch

         do l = 0, min(i_azbuf - 1, i_numlines+i_strtline-j-2)  !read interferogram
!c            read(INTUNIT,rec=j+l+1, iostat=ierr) (c_intb(k,l), k = 0, infp%i_rsamps-1)
!c            if(ierr .ne. 0) goto 999
              call getLine(intAcc,c_intb(0,l),j+l+1)
!c            read(AMPUNIT,rec=j+l+1, iostat=ierr) (c_ampb(k,l), k = 0, infp%i_rsamps-1)
!c           if(ierr .ne. 0) goto 999
              if (infp%i_useamp .eq. 1) then
                call getLine(ampAcc,c_ampb(0,l),j+l+1)
              else
                  do ia=0, infp%i_rsamps-1
                    c_ampb(ia,l) = cmplx( cabs(c_intb(ia,l)),cabs(c_intb(ia,l))) 
                  end do
              endif
         end do

 999     infp%i_azbufsize = l                           !actual number of lines read
         write(*,'(/1x,a,i4,a,i6,a,i4)') 'PATCH:',i_patch,'   starting line:',j,'    lines read: ',infp%i_azbufsize          
!c     
!c     set the azimuth seed location in the middle of the overlap
!c     note: the overlap is set arbitrarily to allow this fixed skew
!c     case to work, however if the skew difference between patches
!c     exceeds NFFT, then this will fail.
     
         infp%i_sline = 0
         infp%i_eline = infp%i_azbufsize - 1

         if(i_patch .eq. 1)then
           i_sl = 0
           i_el = infp%i_azbufsize - i_azovlp/2 - 1
         endif
         if((i_patch .gt. 1) .and. (i_patch .lt.i_numpatch))then
           i_sl = i_azovlp/2
           i_el = infp%i_azbufsize - i_azovlp/2 - 1
         endif
         if(i_patch .eq. i_numpatch)then
           i_sl = i_azovlp/2
           i_el = infp%i_azbufsize - 1
         endif

         if (i_numpatch .eq. 1) then                    !test if only 1 patch
          i_sl = 0
          i_el = infp%i_azbufsize - 1
         endif

         write(*,'(1x,a,i4,a,i4)') 'starting output line: ',i_sl+1,'   ending output line: ',i_el+1

!c
!c        Filter Interferogram 
!c
         call intf_filt(c_intb, c_ampb,  c_intb_filt)

         if(infp%i_filtopt .eq. 1)then          !write out filtered interferogram
             if(filtAcc .gt. 0) then
                do l=i_sl, i_el
!c             write(FILTUNIT,rec=j+l+1) (c_intb_filt(k,l), k=0, infp%i_rsamps-1)
                   call setLine(filtAcc,c_intb_filt(0,l),j+l+1)
                end do
            endif
         endif
!c
!c       Estimate phase gradient
!c
         if(infp%i_slope .eq. 1)then
           call ph_slope(c_intb_filt,  c_pslope)
         end if
!c
!c      Estimate correlation coefficients
!c
        call intf_cc(c_intb, c_intb_filt, c_ampb, c_pslope,  r_cc(0,0,1), r_cc(0,0,2), 
     $    r_cc(0,0,3), r_sigma)
      
         if(infp%i_cc_std .eq. 1)then           !write out standard correlation
             if(corrAcc.gt.0)then
                do l=i_sl, i_el
!c             write(CCUNIT,rec=j+l+1) (r_cc(k,l,1), k=0, infp%i_rsamps-1)
                    call setLine_r4(corrAcc,r_cc(0,l,1),j+l+1)
                end do 
            end if
         end if

         if(infp%i_cc_norm .eq. 1)then          !write out slope normalized correlation
             if(gccAcc .gt. 0)then
                do l=i_sl, i_el
!c             write(GCCUNIT,rec=j+l+1) (r_cc(k,l,2), k=0, infp%i_rsamps-1)
                    call setLine_r4(gccAcc,r_cc(0,l,2),j+l+1)
                end do 
            end if
         end if
 
         if(infp%i_cc_sigma .eq. 1)then         !write out phase standard deviation and correlation
             if(phsigcorrAcc .gt. 0) then
                do l=i_sl, i_el
!c             write(SIGMAUNIT,rec=j+l+1)(r_sigma(k,l), k=0, infp%i_rsamps-1)
!c            setLine_r4(sigmaAcc,r_sigma(0,l),j+l+1)

!c             write(SIGMACCUNIT,rec=j+l+1)(r_cc(k,l,3), k=0, infp%i_rsamps-1)
                    call setLine_r4(phsigcorrAcc,r_cc(0,l,3),j+l+1)
                end do 
            end if
         end if

! debugging     write(*,*)' finished writing phase correlation for patch'

         if(i_unwrap_flag .eq. 1)then           !test for unwrapping 
           if(i_patch .eq. 1) then
              write(*,*)'marking patch spixel, epixel',unwp%i_spixel,unwp%i_epixel
              do l = 0, NBL-1
                 do i =  unwp%i_spixel, unwp%i_epixel
                    r_bphase(i,l) = 0.0
                    r_bamp(i,l) = 0.0 !mark as not unwrapped 
                 end do
              end do  
           end if

           call unw_rt( c_intb_filt, c_ampb, c_pslope,
     $          r_cc(0,0,infp%i_cc_layer), r_xofr
     $          ,r_unw, b_PatchTrees, b_all_unwrap,
     $          r_bphase,r_bamp, i_complist, i_unw_tot)
           
           call abs_phase(r_unw, c_ampb, r_amp, b_all_unwrap,r_bphase,
     $          r_bamp, i_complist, i_patch)
          
          
           !!If amplitude file not provide, use int amplitude
           !!if (infp%i_useamp .eq. 0) then
           !!   do ia=i_sl,i_el
           !!     r_amp(0:infp%i_rsamps-1,ia) = cabs(c_intb(0:infp%i_rsamps-1,ia))
           !!   enddo
           !!endif



!           write(*,*)'DEBUG: done unwrapping, writing patch'
           do ia = i_sl, i_el
!c              write(UNWUNIT,rec=(j+ia)+1)(r_unw(k,ia), k=0, infp
!c     $             %i_rsamps-1)
                call setLineBand_r4(unwAcc, r_amp(0,ia),j+ia+1, b1)
                call setLineBand_r4(unwAcc, r_unw(0,ia),j+ia+1, b2)
           end do           

           if (conncompAcc.gt.0) then
               do ia = i_sl, i_el
                  call setLineBand(conncompAcc, b_all_unwrap(0,ia),j+ia+1, b1)
               enddo
           endif

        end if                  !end of test for unwrapping selected
      end do                    !loop over all the lines in the file

      DEALLOCATE( c_intb, c_ampb, c_intb_filt, c_pslope )
      DEALLOCATE( r_cc, r_sigma, r_unw, r_amp)
      DEALLOCATE( r_bamp, r_xofr, b_patchTrees, b_all_unwrap )
      
      write(*,'(/a/)') '*** Normal Completion  ***'
      end
