!c This program has been upgraded for matching ScanSAR full-aperture images.
!c 1. setting the following two parameters to use:
!c      winsize_filt = 8
!c      i_covs_filt = 64
!c 2. use same original parameter values as you use in the original program for
!c    ScanSAR full-aperture image matching.
!c 3. Original matching method is still available by setting winsize_filt to 1.
!c    In this case, i_covs_filt has no effects.

!c Cunren Liang, 25-DEC-2019, Caltech

!c****************************************************************

      subroutine ampcor(imgAccessor1, imgAccessor2, band1, band2)

      use ampcorState
      implicit none

!c     INPUT VARIABLES

!c     PARAMETER STATEMENTS:

      integer i_maxsamp
      integer i_ovs,i_srchpp
      parameter(i_ovs=2)
      parameter(i_srchpp=4)

      integer i_dump_images,i_sinc_fourier,i_sinc,i_fourier
      parameter(i_dump_images=0,i_sinc=1,i_fourier=2) !i_dump_images=1 means dump debug feature is on
      parameter(i_sinc_fourier=i_sinc)

      integer i_new,i_old,i_rdf,i_real,i_complex
      parameter(i_new=1,i_old=2,i_rdf=3,i_real=1,i_complex=2)

      integer i_sinc_window
      parameter(i_sinc_window=2)

      integer MAXDECFACTOR      ! maximum lags in interpolation kernels
      parameter(MAXDECFACTOR=4096)

      integer MAXINTKERLGH      ! maximum interpolation kernel length
      parameter (MAXINTKERLGH=256)

      integer MAXINTLGH         ! maximum interpolation kernel array size
      parameter (MAXINTLGH=MAXINTKERLGH*MAXDECFACTOR)

      integer i_log             ! LFN for log file/screen as appropriate
      parameter (i_log=6)

!c     INPUT VARIABLES:

      integer i_wsyj, i_wsxj
      integer i_n2wsyi,i_n2wsxi,i_n2wsyj,i_n2wsxj,i_index,i_indexi
      integer i_ovss
      integer i_srchp
      integer band1, band2, i_lineno

!c     OUTPUT VARIABLES:

      integer i_shiftx,i_shifty
      real*4 r_shfty,r_shftx,r_peak,r_meani,r_meanj
      real*4 r_stdvi,r_stdvj,r_noise,r_eval1
      real*4 r_eval2,r_evec1(2),r_evec2(2)
      integer i_flag,i_edge(2)

!c     LOCAL VARIABLES:

      character*120 a_debugfile

      integer i_x,i_xx
      integer i_y,i_yy

      real r_snr,r_outside
      integer i_xlu, i_ylu
      integer i_mag1, i_mag2

      integer*8 :: imgAccessor1, imgAccessor2

      complex, dimension(:,:), allocatable :: c_refimg
      complex, dimension(:,:), allocatable :: c_srchimg
      real, dimension(:,:), allocatable ::  r_refimg
      real, dimension(:,:), allocatable ::  r_srchimg
      real, dimension(:,:), allocatable :: r_corr
      complex, dimension(:), allocatable :: c_corr, c_corrt
      complex, dimension(:), allocatable :: c_dataout, c_dataout2
      real, dimension(:,:), allocatable :: r_imgi, r_imgj, r_imgc
      real, dimension(:,:), allocatable :: r_imgios, r_imgjos, r_imgcos

      complex, dimension(:), allocatable :: c_chipref, c_chipsch
      complex, dimension(:), allocatable :: c_ossch, c_osref

      integer i_wxd,i_wyd,i_q,i_qq,i_centerxj,i_centeryj
      integer i,j,k,l,i_centerxi,i_centeryi,i_cnta,i_xp,i_yp
      integer i_nn(2),i_dir,i_shiftxos,i_shiftyos
      integer i_inarg,i_nnphy(2),i_unit
      real r_peakos,r_shftxos,r_shftyos,r_covos(3),r_snros
      real r_shftxosc,r_shftyosc,r_mean_cor, r_cov(3)
      integer i_wsxios,i_wsyios,i_wsxjos,i_wsyjos,i_wsox,i_wsoy,i_status
      real r_maxi,r_maxj
      integer ncr,i_wsxjp,i_wsyjp
      integer i_input_style,i_datatype(2)
      character*3 a_style

      integer i_iout,i_jout,i_frac,i_index2,i_index3
      real r_iout,r_jout,r_sincwgt,r_frac


      integer i_cpeak(2),iargc,i_px,i_py,i_p1,i_p2
      real r_max,r_oscoroff(2)
      real r_csrchx,r_csrchy

      integer i_select,i_weight
      integer i_numset
      integer i_err

      integer i_decfactor       ! Range migration decimation Factor
      integer i_intplength      ! Range migration interpolation kernel length
      real*4  r_fdelay          ! Range migration filter delay
      real*4 r_fintp(0:MAXINTLGH) ! interpolation kernel values
      real*8 r_relfiltlen,r_pedestal,r_beta

      integer*4 ii, jj
!c      logical ll

!c     SAVE STATEMENTS:

!      save r_imgi,r_imgj,r_imgc
!      save c_refimg,c_srchimg,r_refimg,r_srchimg

!      save c_chipref,c_chipsch,c_osref,c_ossch,r_corr,c_corr,c_corrt,c_dataout,c_dataout2

!c     FUNCTION STATEMENTS:

      integer nextpower
      real*4 t0, t1, t2, t3, t4, t5
      real*4 seconds
      external seconds
      integer count

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!c    for ScanSAR full-aperture mode matching
      real*4 r_shfty_filt,r_shftx_filt
      integer i_shiftx_filt,i_shifty_filt
      !integer winsize_filt
      integer i_wsyi_filt
      integer i_wsyj_filt
      integer iii, num_pixel
      !integer i_covs_filt, i_cw_filt
      integer i_cw_filt
      real, dimension(:,:), allocatable :: r_imgi_filt, r_imgj_filt, r_imgc_filt
      complex, dimension(:), allocatable :: c_corr_filt, c_corrt_filt
      real tmp_corr
      integer continue_fine_matching

!c    added to avoid accessing indexes out of bounds (segmentation fault) 02-FEB-2020
      integer x_index
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      write(6,*) 'Input Bands: ', band1, band2

      write(6,*) ' XXX start timer'
      t0 = seconds(0.0)           ! start timer

      i_datatype(1) = i_complex   !will be changed according to user input
      i_datatype(2) = i_complex
      i_input_style = i_old
!c Begin
!c Modified by Giangi. These  values are set before function call ampcor from python module, but then were changed 
!c to i_covs=32, i_cw = 16  and a_datatype(1,2) = complex which were the default values (before were read 
!c from in file at this point). I commented out those lines


!c     sinc interploation kernel

      i_decfactor = 4096
      i_weight = 1
      r_pedestal = 0.0
      r_beta = .75
      r_relfiltlen = 6.0

      call fill_sinc(r_beta,r_relfiltlen,i_decfactor,i_weight, r_pedestal,i_intplength,r_fdelay,r_fintp) 

      do i=3,14
         k=2**i
         call cfft1d_jpl(k,c_osref,0)
      end do

      !c Change input types based on user input
      if(index(a_datatype(1),'complex') .ne. 0)then
         i_datatype(1) = i_complex
         i_mag1 = 0
      elseif(index(a_datatype(1),'real') .ne. 0)then
         i_datatype(1) = i_real
      elseif(index(a_datatype(1),'mag') .ne. 0)then
          i_datatype(1) = i_complex
          i_mag1 = 1
      else
         write(i_log,'(a)') 'WARNING - did not understand reference image data type'
         write(i_log,'(a)') 'Expecting complex or real'
         write(i_log,'(a)') 'Your input was '//a_datatype(1)
         write(i_log,'(a)') ' '
      endif

      if(index(a_datatype(2),'complex') .ne. 0)then
         i_datatype(2) = i_complex
         i_mag2 = 0
      elseif(index(a_datatype(2),'real') .ne. 0)then
         i_datatype(2) = i_real
      elseif(index(a_datatype(1),'mag') .ne. 0)then
          i_datatype(2) = i_complex
          i_mag2 = 1
      else
         write(i_log,'(a)') 'WARNING - did not understand search image data type'
         write(i_log,'(a)') 'Expecting complex or real'
         write(i_log,'(a)') 'Your input was '//a_datatype(2)
         write(i_log,'(a)') ' '
      endif

      i_srchx = max(i_srchx,1)
      i_srchy = max(i_srchy,1)

      i_wsxj = i_wsxi+2*i_srchx
      i_wsyj = i_wsyi+2*i_srchy

      i_srchp = min(i_srchy,i_srchx,i_srchpp)

      i_wsxjp = i_wsxi + 2*i_srchp
      i_wsyjp = i_wsyi + 2*i_srchp

      i_n2wsxi = 2**(nextpower(i_wsxi))
      i_n2wsyi = 2**(nextpower(i_wsyi))

      i_n2wsxj = 2**(nextpower(i_wsxjp))
      i_n2wsyj = 2**(nextpower(i_wsyjp))

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!c    for ScanSAR full-aperture mode matching
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !input parameter reference values:
      !winsize_filt should be power of 2 to faciliate subsequent processing
      !winsize_filt = 8
      !i_covs_filt = i_covs
      !i_covs_filt = 64
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      
      !better be power of 2 for fft
      i_cw_filt = (i_wsyj - i_wsyi) / winsize_filt
      i_wsyi_filt = i_wsyi / winsize_filt
      i_wsyj_filt = i_wsyj / winsize_filt

      if(winsize_filt .ne. 1)then
        write(6,*) ''
        write(6,*) '   *** using ScanSAR full-aperture mode for gross offset estimation ***'
        write(6,*) 'azimuth multi-looking window size: ', winsize_filt
        write(6,*) 'number of azimuth samples before and after multi-looking (reference): ', i_wsyi, i_wsyi_filt
        write(6,*) 'number of azimuth samples before and after multi-looking (secondary): ', i_wsyj, i_wsyj_filt
        write(6,*) 'azimuth covariance surface oversampling factor: ', i_covs_filt
        write(6,*) 'total number of azimuth samples to be oversampled', i_cw_filt
        write(6,*) ''
      endif

      if(i_cw_filt .lt. 4)then
        write(6,*) 'ERROR - number of samples availabe for estating gross offset is too small:', i_cw_filt
        write(6,*) '        the value is computed as 2*i_srchy/winsize_filt'
        write(6,*) '        current i_srchy (azimuth search window size):', i_srchy
        write(6,*) '        current winsize_filt (azimuth filtering window size):', winsize_filt
        return
      endif

      if(i_cw_filt .ne. 2**nextpower(i_cw_filt))then
        write(6,*) 'WARNING - number of samples availabe for estating gross offset is NOT power of 2:', i_cw_filt
        write(6,*) '          the value is computed as 2*i_srchy/winsize_filt'
        write(6,*) '          better to make it power of 2 for FFT'
        write(6,*) '          current i_srchy (azimuth search window size):', i_srchy
        write(6,*) '          current winsize_filt (azimuth filtering window size):', winsize_filt
      endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      !c Set max width to largest of the width
      i_maxsamp = max(i_samples(1), i_samples(2))

      allocate( c_refimg(i_maxsamp,i_wsyi) )
      allocate( c_srchimg(i_maxsamp,i_wsyj) )
      allocate( r_refimg(i_maxsamp,i_wsyi) )
      allocate( r_srchimg(i_maxsamp,i_wsyj) )
      allocate( r_corr(i_covs*i_cw,i_covs*i_cw) )
      allocate( c_corr(i_covs*i_cw*i_covs*i_cw) )
      allocate( c_dataout2(i_covs*i_cw*i_covs*i_cw) )
      allocate( c_dataout(i_covs*i_cw*i_cw) )
      allocate( c_corrt(i_cw*i_cw) )

      allocate( r_imgi(i_wsxi,i_wsyi))
      allocate( r_imgj(i_wsxj,i_wsyj))
      allocate( r_imgc(i_wsxj,i_wsyj))
      allocate( r_imgios(i_ovs*i_wsxi,i_ovs*i_wsyi))
      allocate( r_imgjos(i_ovs*i_wsxjp,i_ovs*i_wsyjp))
      allocate( r_imgcos(i_ovs*i_wsxjp,i_ovs*i_wsyjp))

       allocate( c_chipref(i_n2wsxi*i_n2wsyi) )
       allocate( c_chipsch(i_n2wsxj*i_n2wsyj) )
       allocate( c_ossch(i_ovs*i_n2wsxj*i_ovs*i_n2wsyj) )
       allocate( c_osref(i_ovs*i_n2wsxi*i_ovs*i_n2wsyi) )

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!c    for ScanSAR full-aperture mode matching
      allocate( r_imgi_filt(i_wsxi,i_wsyi_filt))
      allocate( r_imgj_filt(i_wsxj,i_wsyj_filt))
      allocate( r_imgc_filt(i_wsxj,i_wsyj_filt))
      allocate( c_corr_filt(i_covs_filt*i_cw_filt*i_covs_filt*i_cw_filt) )
      allocate( c_corrt_filt(i_cw_filt*i_cw_filt) )
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!c-------------------------------
!c        begin ruggedize ... a bunch of input checking

         if(i_datatype(1).ne.i_complex .and. i_datatype(1).ne.i_real)then
           write(i_log,'(a)') 'WARNING - Do not understand data type for reference image'
           write(i_log,'(a,i1,a,i1,a)') 'Expecting flag to be real (',i_real,') or complex (',i_complex,')'
           write(i_log,'(a,i10)') 'Data type flag set to ',i_datatype(1)
           i_datatype(1) = i_complex
          write(i_log,'(a,i1,a)') 'Resetting type flag to be complex (',i_complex,')' 
          write(i_log,'(a)') ' '
         endif
         if(i_datatype(2).ne.i_complex .and. i_datatype(2).ne.i_real)then
           write(i_log,'(a)') 'WARNING - Do not understand data type for search image'
           write(i_log,'(a,i1,a,i1,a)') 'Expecting flag to be real (',i_real,') or complex (',i_complex,')'
           write(i_log,'(a,i10)') 'Data type flag set to ',i_datatype(2)
           i_datatype(2) = i_complex
          write(i_log,'(a,i1,a)') 'Resetting type flag to be complex (',i_complex,')' 
          write(i_log,'(a)') ' '
         endif



         if(i_samples(1).gt.i_maxsamp)then
           write(i_log,'(a)') 'ERROR - Requesting processing of too wide a file'
           write(i_log,'(a,1x,i10,a)') '             Image 1 width is ',i_samples(1),' pixels'
           write(i_log,'(a,1x,i10,a)') 'Maximum allowed file width is ',i_maxsamp   ,' pixels'
            deallocate( c_refimg )
            deallocate( c_srchimg )
            deallocate( r_refimg )
            deallocate( r_srchimg )
           return  ! EMG
         endif
         if(i_samples(2).gt.i_maxsamp)then
           write(i_log,'(a)') 'ERROR - Requesting processing of too wide a file'
           write(i_log,'(a,1x,i10,a)') 'Image 2 width is ',i_samples(2),' pixels'
           write(i_log,'(a,1x,i10,a)') 'Maximum allowed file width is ',i_maxsamp   ,' pixels'
           deallocate( c_refimg )
           deallocate( c_srchimg )
           deallocate( r_refimg )
           deallocate( r_srchimg )
           return  ! EMG
         endif



!c        read in i_wsyi lines of data into the refimg buffer for each chip
!c        read in i_wsyj=i_wsyi+2*i_srchy lines of data into the srchimg buffer for each chip
!c        read in i_wsxi samples of data into the refimg buffer for each chip
!c        read in i_wsxj=i_wsxi+2*i_srchx samples of data into the srchimg buffer for each chip


         if(i_srchx.lt.5)then
           write(i_log,'(a)') 'CAUTION - Requesting very small search window pull in'
           write(i_log,'(a,1x,i10,a)') 'Reference Window Size is             ',i_wsxi ,' sample pixels'
           write(i_log,'(a,1x,i10,a)') 'Number of Search Pixels is           ',i_srchx ,' sample pixels'
           write(i_log,'(a)') 'The rule of thumb is that the search window pull in is at least 5'
           write(i_log,'(a)') 'pixels and is less than the reference window size divided by 5. '
           jj = max(5,nint(float(i_wsxi)/6.0))
           write(i_log,'(a,1x,i10,a)') 'Suggested Number of Search Pixels is ',jj,' sample pixels'
           write(i_log,'(a)') ' '
         endif

         ii = nint(float(i_wsxi)/float(i_srchx))
         if(ii.lt.5)then
           write(i_log,'(a)') 'CAUTION - Requesting very large search window pull in'
           write(i_log,'(a,1x,i10,a)') 'Reference Window Size is             ',i_wsxi ,' sample pixels'
           write(i_log,'(a,1x,i10,a)') 'Number of Search Pixels is           ',i_srchx ,' sample pixels'
           write(i_log,'(a)') 'The rule of thumb is that the search window pull in is at least 5'
           write(i_log,'(a)') 'pixels and is less than the reference window size divided by 5. '
           jj = max(5,nint(float(i_wsxi)/6.0))
           write(i_log,'(a,1x,i10,a)') 'Suggested Number of Search Pixels is ',jj,' sample pixels'
           write(i_log,'(a)') ' '
           write(i_log,'(a)') ' '
         endif

         if(i_srchy.lt.5)then
           write(i_log,'(a)') 'CAUTION - Requesting very small search window pull in'
           write(i_log,'(a,1x,i10,a)') 'Reference Window Size is             ',i_wsyi ,' line pixels'
           write(i_log,'(a,1x,i10,a)') 'Number of Search Pixels is           ',i_srchy ,' line pixels'
           write(i_log,'(a)') 'The rule of thumb is that the search window pull in is at least 5'
           write(i_log,'(a)') 'pixels and is less than the reference window size divided by 5. '
           jj = max(5,nint(float(i_wsyi)/6.0))
           write(i_log,'(a,1x,i10,a)') 'Suggested Number of Search Pixels is ',jj,' line pixels'
           write(i_log,'(a)') ' '
         endif

         ii = nint(float(i_wsyi)/float(i_srchy))
         if(ii.lt.5)then
           write(i_log,'(a)') 'CAUTION - Requesting very large search window pull in'
           write(i_log,'(a,1x,i10,a)') 'Reference Window Size is             ',i_wsyi ,' line pixels'
           write(i_log,'(a,1x,i10,a)') 'Number of Search Pixels is           ',i_srchy ,' line pixels'
           write(i_log,'(a)') 'The rule of thumb is that the search window pull in is at least 5'
           write(i_log,'(a)') 'pixels and is less than the reference window size divided by 5. '
           jj = max(5,nint(float(i_wsyi)/6.0))
           write(i_log,'(a,1x,i10,a)') 'Suggested Number of Search Pixels is ',jj,' line pixels'
           write(i_log,'(a)') ' '
           write(i_log,'(a)') ' '
         endif

         if(i_cw.lt.8)then
           write(i_log,'(a)') 'WARNING - Covariance Surface Window Size Very Small'
           write(i_log,'(a)') 'It is the number of pixels in the Correlation Surface to oversample.'
           write(i_log,'(a)') 'Minimum Recommended Value for the Covariance Surface Window Size is 8.'
           write(i_log,'(a,1x,i3,a)') 'Requested covariance surface window size of ',i_cw,' pixels'
           write(i_log,'(a)') ' '
         endif

           write(i_log,'(a,1x,i4,a)') 'Requested resolving shifts to 1/',i_covs*2,' of a pixel'
           write(i_log,'(a)') ' '

         i_strtsamp = max(i_strtsamp,1)
         i_endsamp  = min(i_endsamp,i_samples(1))

         if(i_skipline.lt.i_wsyi .or. i_skipsamp.lt.i_wsxi)then
           write(i_log,'(a)') 'INFORMATION - you choose skips which are small for your window sizes'
           write(i_log,'(a)') 'Normally the skip size is bigger than the box size'
           write(i_log,'(a,i10,a,i10)') 'Across your skip is ',i_skipsamp,' but your window is ',i_wsxi
           write(i_log,'(a,i10,a,i10)') 'Down   your skip is ',i_skipline,' but your window is ',i_wsyi
           write(i_log,'(a)') 'This means that the image chips are larger than the separation between chips'
           write(i_log,'(a)') ' '
         endif

         r_covth = min(r_covth,999.999998)


         i_avgx = max(1,i_avgx)
         i_avgy = max(1,i_avgy)
         if(i_avgx.gt.1 .or. i_avgy.gt.1)then
           write(i_log,'(a)')    'INFORMATION - You are looking down the data before cross correlation.'
           write(i_log,'(a,i4)') 'Averaging the samples across the file by a factor of ',i_avgx
           write(i_log,'(a,i4)') 'Averaging the lines   down   the file by a factor of ',i_avgy
           write(i_log,'(a)')    ' '
         endif

!c        end ruggedize


      count = 0
!c loop over data begins. initialize number of rows in output table
      numRowTable = 0
      do i_y=i_strtline+i_srchy,i_endline+i_srchy,i_skipline

!c ----------------------------------------------------------------
!c NOTE:
!c   i_wsyi is the Reference image Window Size in line pixels
!c   i_samples(1) is pixel width of image 1
!c   i_samples(2) is pixel width of image 2
!c   c_refimg(1,i_yy): image lines are read into each c_refimg(r,c) fortran "column"
!c                     fortran 2-dimensional arrays are column-wise contiguous
!c   void getImageLine_f(void *pyTiledSampleAccessor, int *lineNumber, char* line, int *lineSizeBytes)
!c        Since C-based function, lineNumber is zero-based indexing
!c ----------------------------------------------------------------

         count = count + 1
         write(6,*) 'At line = ',i_y-i_srchy

         i_centeryi = i_y + (i_wsyi-1)/2.
         i_ylu = nint(r_scaley*i_y)


         if(i_datatype(1) .eq. i_complex)then
            t2 = seconds(t0)           ! start timer
!c           Search lines from current image line i_y down to the i_wsyi lines below
            do i_yy = 1,i_wsyi
               i_xx = band1
               i_lineno = i_y+i_yy-2
!               call getLine(imgAccessor1, c_refimg(:,i_yy), i_y+i_yy-2)
               if ((i_lineno .lt. 1).or.(i_lineno .gt. i_lines(1))) then
                    c_refimg(:,i_yy) = cmplx(0.,0.)
               else
                    call getLineBand(imgAccessor1, c_refimg(:,i_yy), i_xx, i_lineno)
               endif

               if(i_mag1 .ne. 0) then
                   do i_xx=1,i_samples(1)
                    c_refimg(i_xx,i_yy) = cmplx(cabs(c_refimg(i_xx,i_yy)),0.0)
                   enddo
               endif
            end do
            t3 = seconds(t0)           ! start timer

         elseif(i_datatype(1) .eq. i_real)then

            do i_yy = 1,i_wsyi
               i_xx = band1
               i_lineno = i_y+i_yy-2
!               call getLine(imgAccessor1, r_refimg(:,i_yy), i_y+i_yy-2)
               if ((i_lineno .lt. 1).or.(i_lineno .gt. i_lines(1))) then
                    r_refimg(:,i_yy) = 0.0
               else
                   call getLineBand(imgAccessor1, r_refimg(:,i_yy), i_xx, i_lineno)
               endif

               do i_xx=1,i_samples(1)
                  c_refimg(i_xx,i_yy) = cmplx(r_refimg(i_xx,i_yy),0.0)
               enddo
            end do

         endif


         if(i_datatype(2) .eq. i_complex)then

            t2 = seconds(t0)           ! start timer
            do i_yy = 1,i_wsyj
               i_xx = band2
               i_lineno = i_ylu+i_yy-2-i_srchy+i_grossy
!               call getLine(imgAccessor2, c_srchimg(:,i_yy), i_ylu+i_yy-2-i_srchy+i_grossy)

               if ((i_lineno .lt. 1).or.(i_lineno .gt. i_lines(2))) then
                    c_srchimg(:,i_yy) = cmplx(0.,0.)
               else
                    call getLineBand(imgAccessor2, c_srchimg(:,i_yy), i_xx, i_lineno)
               endif

                if(i_mag2 .ne. 0) then
                   do i_xx=1,i_samples(2)
                    c_srchimg(i_xx,i_yy) = cmplx(cabs(c_srchimg(i_xx,i_yy)),0.0)
                   enddo
               endif

            end do
            t3 = seconds(t0)           ! start timer

         elseif(i_datatype(2) .eq. i_real)then

            do i_yy = 1,i_wsyj
               i_xx = band2
               i_lineno = i_ylu+i_yy-2-i_srchy+i_grossy
!               call getLine(imgAccessor2, c_srchimg(:,i_yy), i_ylu+i_yy-2-i_srchy+i_grossy)

               if ((i_lineno .lt. 1).or.(i_lineno .gt. i_lines(2))) then
                    r_srchimg(:,i_yy) = 0.0
               else
                   call getLineBand(imgAccessor2, r_srchimg(:,i_yy), i_xx, i_lineno)
               endif

               do i_xx=1,i_samples(2)
                  c_srchimg(i_xx,i_yy) = cmplx(r_srchimg(i_xx,i_yy),0.0)
               enddo
            end do

         endif

         t4 = seconds(t0)           ! start timer

         do i_x=i_strtsamp+i_srchx,i_endsamp+i_srchx,i_skipsamp

            i_centerxi = i_x+(i_wsxi-1)/2.
            i_xlu = nint(r_scalex*i_x)

!c     get the reference image and search images

            do i_yy = 1,i_wsyi
               do i_xx = 1,i_wsxi

                  x_index = i_x+i_xx-1
                  if ((x_index .lt. 1).or.(x_index .gt. i_maxsamp)) then
                     r_imgi(i_xx,i_yy) = 0.0
                  else
                     r_imgi(i_xx,i_yy) = cabs(c_refimg(x_index,i_yy))
                  endif

               end do
            end do

            do i_yy = 1, i_wsyj
               do i_xx = 1 , i_wsxj

                  x_index = i_xlu+i_xx-1-i_srchx+i_grossx
                  if ((x_index .lt. 1).or.(x_index .gt. i_maxsamp)) then
                     r_imgj(i_xx,i_yy) = 0.0
                  else
                     r_imgj(i_xx,i_yy) = cabs(c_srchimg(x_index,i_yy))
                  endif

               end do
            end do

!c     dump the reference and search images

            if(i_dump_images .eq. 1)then
               a_debugfile = 'refimg_input.dat'
               call dump_chip_r4(a_debugfile,r_imgi,1,i_wsxi,1,i_wsyi,i_wsxi,i_wsyi)
               a_debugfile = 'srchimg_input.dat'
               call dump_chip_r4(a_debugfile,r_imgj,1,i_wsxj,1,i_wsyj,i_wsxj,i_wsyj)
            endif

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!c    for ScanSAR full-aperture mode matching

!c          correlate azimuth-multiple-looked images
            if (winsize_filt .ne. 1) then
!c          get azimuth-multiple-looked images
              do i_yy = 1,i_wsyi_filt
                 do i_xx = 1,i_wsxi
                    r_imgi_filt(i_xx,i_yy) = 0.
                    num_pixel = 0
                    do iii = 1,winsize_filt
                      if(r_imgi(i_xx,(i_yy-1)*winsize_filt+iii) .ne. 0) then
                        num_pixel = num_pixel + 1
                        r_imgi_filt(i_xx,i_yy) = r_imgi_filt(i_xx,i_yy) + r_imgi(i_xx,(i_yy-1)*winsize_filt+iii)**2
                      endif
                    end do
                    if(num_pixel .ne. 0) then
                      r_imgi_filt(i_xx,i_yy) = sqrt(r_imgi_filt(i_xx,i_yy)/num_pixel)
                    endif
                 end do
              end do


              do i_yy = 1, i_wsyj_filt
                 do i_xx = 1 , i_wsxj
                    r_imgj_filt(i_xx,i_yy) = 0.
                    num_pixel = 0
                    do iii = 1,winsize_filt
                      if(r_imgj(i_xx,(i_yy-1)*winsize_filt+iii) .ne. 0) then
                        num_pixel = num_pixel + 1
                        r_imgj_filt(i_xx,i_yy) = r_imgj_filt(i_xx,i_yy) + r_imgj(i_xx,(i_yy-1)*winsize_filt+iii)**2
                      endif
                    end do
                    if(num_pixel .ne. 0) then
                      r_imgj_filt(i_xx,i_yy) = sqrt(r_imgj_filt(i_xx,i_yy)/num_pixel)
                    endif
                 end do
              end do

!c            correlate images
!c            i_avgx = i_avgy = 1 
              call correlate(r_imgi_filt,r_imgj_filt,i_wsxi,i_wsyi_filt,i_wsxj,
     &           i_wsyj_filt,1,1,1,r_meani,r_stdvi,r_meanj,
     &           r_stdvj,r_peak,r_noise,r_cov,r_eval1,
     &           r_eval2,r_evec1,r_evec2,r_imgc_filt,i_shiftx_filt,i_shifty_filt,i_edge,
     &           i_flag,l_debug)

              r_shftx_filt=float(i_shiftx_filt) - i_srchx + i_grossx
              r_shfty_filt=float(i_shifty_filt) - float(i_srchy)/winsize_filt + float(i_grossy)/winsize_filt
              r_shfty_filt=r_shfty_filt*winsize_filt

              if(i_flag .eq. 0 .and. i_edge(1) .eq. 0 .and.
     &           i_edge(2) .eq. 0)then !found a potentially good data point

                r_outside = 0.0
                i_cnta = 0
                do l=max(i_shifty_filt-9,1),min(i_shifty_filt+11,i_wsyj_filt-i_wsyi_filt)
                  do k=max(i_shiftx_filt-9,1),min(i_shiftx_filt+11,i_wsxj-i_wsxi)
                    i_cnta = i_cnta + 1
                    r_outside = r_outside + r_imgc_filt(k,l)**2
                  enddo
                enddo
                r_outside = r_outside - r_peak**2
                r_outside = r_outside/(i_cnta-1)

                r_snr = r_peak**2/max(r_outside,1.e-10)

                if(r_snr .gt. r_snrth .and. r_cov(1) .lt. r_covth .and. r_cov(2) .lt. r_covth)then
                  continue_fine_matching = 1
                else
                  continue_fine_matching = 0
                endif

              else
                continue_fine_matching = 0
              endif

!c            compute finer offset by oversampling covariance surface
              if(continue_fine_matching .ne. 0)then
                r_max = 0.0
                r_mean_cor = 0.0
                i_cnta = 0
                i_px = i_shiftx_filt+1
                i_py = i_shifty_filt+1

                do i_yy=-i_cw_filt/2,i_cw_filt/2-1

                    do i_xx=-i_cw_filt/2,i_cw_filt/2-1

                      i_index = (i_yy+i_cw_filt/2)*i_cw_filt + i_xx + i_cw_filt/2 + 1

                      if (i_xx+i_px .ge. 1 .and. i_xx+i_px .le. (2*i_srchx+1)*1 .and.
     &                         i_yy+i_py .ge. 1 .and. i_yy+i_py .le. (2*i_srchy+1)*1 )then
                        c_corrt_filt(i_index) = cmplx(abs(r_imgc_filt(i_xx+i_px,i_yy+i_py)/r_peak),0.)
                        r_mean_cor = r_mean_cor + cabs(c_corrt_filt(i_index))
                        i_cnta = i_cnta + 1
                      else
                        c_corrt_filt(i_index) = cmplx(0.0, 0.0)
                      endif

                      if(cabs(c_corrt_filt(i_index)) .gt. r_max)then
                        r_max = cabs(c_corrt_filt(i_index))
                        i_p1 = i_xx
                        i_p2 = i_yy
                      endif

                  enddo

                enddo

!c              substract off the mean
!c              looks like it does not do anything, so can comment out these
!c                r_mean_cor = r_mean_cor/max(i_cnta,1)
!c                r_mean_cor = 0.0
!c                do i_yy=-i_cw_filt/2,i_cw_filt/2-1
!c                  do i_xx=-i_cw_filt/2,i_cw_filt/2-1
!c                    i_index = (i_yy+i_cw_filt/2)*i_cw_filt + i_xx + i_cw_filt/2 + 1
!c                    c_corrt_filt(i_index) = c_corrt_filt(i_index) - cmplx(r_mean_cor,0.0)
!c                  enddo
!c                enddo

!c              oversample via Fourier transforms
!c              forward fft the data
                i_nn(1) = i_cw_filt
                i_nn(2) = i_cw_filt
                i_dir = 1
                call fourn2d(c_corrt_filt,i_nn,i_dir)

!c              spread the spectral data out for inverse transforms
                i_nn(1) = i_cw_filt*i_covs_filt
                i_nn(2) = i_cw_filt*i_covs_filt
                i_dir = -1

                do k=1,i_nn(2)
                   do l=1,i_nn(1)
                      i_index = (k-1)*i_nn(1) + l
                      c_corr_filt(i_index) = 0.0
                   enddo
                enddo

                do l=1,i_cw_filt/2
                   do k=1,i_cw_filt/2
                      i_index = (k-1)*i_nn(1) + l
                      i_indexi = (k-1)*i_cw_filt + l
                      c_corr_filt(i_index) = c_corrt_filt(i_indexi)
                      i_index = l + (i_nn(2)-i_cw_filt/2+k-1)*i_nn(1)
                      i_indexi = l + (k+i_cw_filt/2-1)*i_cw_filt
                      c_corr_filt(i_index) = c_corrt_filt(i_indexi)
                      i_index = i_nn(1)-i_cw_filt/2+l + (k-1)*i_nn(2)
                      i_indexi = l+i_cw_filt/2 + (k-1)*i_cw_filt
                      c_corr_filt(i_index) = c_corrt_filt(i_indexi)
                      i_index = i_nn(1)-i_cw_filt/2+l + (i_nn(2)-i_cw_filt/2+k-1)*i_nn(1)
                      i_indexi = l+i_cw_filt/2 + (k+i_cw_filt/2-1)*i_cw_filt
                      c_corr_filt(i_index) = c_corrt_filt(i_indexi)
                   enddo
                enddo

!c              inverse transform
                call fourn2d(c_corr_filt,i_nn,i_dir)


!c              detect the peak
                 r_max=0.
                 do i_yy=1,i_cw_filt*i_covs_filt
                    do i_xx=1,i_cw_filt*i_covs_filt
                       i_index = (i_yy-1)*i_cw_filt*i_covs_filt + i_xx
                       tmp_corr = cabs(c_corr_filt(i_index))/((i_cw_filt**2)*(i_cw_filt*i_covs_filt)**2)
                       if (abs(i_xx-i_cw_filt*i_covs_filt/2) .le. i_covs_filt .and.
     &                          abs(i_yy-i_cw_filt*i_covs_filt/2) .le. i_covs_filt) then
                          if (tmp_corr .ge. r_max) then
                             r_max = tmp_corr
                             i_cpeak(1) = i_xx - i_cw_filt/2*i_covs_filt
                             i_cpeak(2) = i_yy - i_cw_filt/2*i_covs_filt
                          endif
                       endif
                    enddo
                 enddo

                 r_oscoroff(1) = float(i_cpeak(1)-1)/float(i_covs_filt)
                 r_oscoroff(2) = float(i_cpeak(2)-1)/float(i_covs_filt)
                 r_oscoroff(2) = r_oscoroff(2) * winsize_filt

                 r_shftx = r_oscoroff(1)/1 + r_shftx_filt + i_xlu - i_x
                 r_shfty = r_oscoroff(2)/1 + r_shfty_filt + i_ylu - i_y

                 !get integer values for subsequent use. note that all four variables
                 !are used, so they need to be consistent
                 r_shftx = nint(r_shftx)
                 r_shfty = nint(r_shfty)
                 i_shiftx = nint(r_shftx) + i_srchx - i_grossx
                 i_shifty = nint(r_shfty) + i_srchy - i_grossy
              endif
            else
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!c          correlate original images
            call correlate(r_imgi,r_imgj,i_wsxi,i_wsyi,i_wsxj,
     &           i_wsyj,i_avgx,i_avgy,1,r_meani,r_stdvi,r_meanj,
     &           r_stdvj,r_peak,r_noise,r_cov,r_eval1,
     &           r_eval2,r_evec1,r_evec2,r_imgc,i_shiftx,i_shifty,i_edge,
     &           i_flag,l_debug)

            r_shftx=float(i_shiftx*i_avgx) - i_srchx + i_grossx
            r_shfty=float(i_shifty*i_avgy) - i_srchy + i_grossy

!c     decide with points are good matches and print out the match values

            if(i_flag .eq. 0 .and. i_edge(1) .eq. 0 .and.
     &           i_edge(2) .eq. 0)then !found a potentially good data point

!c     compute the "snr"

               if(l_display)then
                  write(6,*) ' '
                  write(6,*) 'Correlation Surface at ',i_centerxi,
     &                 i_centeryi
                  do l=max(i_shifty-3,1),min(i_shifty+5,i_wsyj-i_wsyi)
                     write(6,178) (r_imgc(k,l)**2./r_peak**2.,
     &                    k=max(i_shiftx-3,1),min(i_shiftx+5,i_wsxj-i_wsxi))
 178                 format(1x,9(f6.3,1x))
                  enddo
               endif

               r_outside = 0.0
               i_cnta = 0
               do l=max(i_shifty-9,1),min(i_shifty+11,i_wsyj-i_wsyi)
                  do k=max(i_shiftx-9,1),min(i_shiftx+11,i_wsxj-i_wsxi)
                     i_cnta = i_cnta + 1
                     r_outside = r_outside + r_imgc(k,l)**2
                  enddo
               enddo
               r_outside = r_outside - r_peak**2
               r_outside = r_outside/(i_cnta-1)

               r_snr = r_peak**2/max(r_outside,1.e-10)
!ccccc               write(6,'(a,1x,2(f20.10,1x))') 'Peak/SNR = ',r_peak,r_snr
               if(r_snr .gt. r_snrth .and. r_cov(1) .lt. r_covth .and. r_cov(2) .lt. r_covth)then
                continue_fine_matching = 1
               else
                continue_fine_matching = 0
               endif
            else
              continue_fine_matching = 0
            endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!c          this is just used to keep the original indentation level
            if(1 .eq. 1)then
               if(continue_fine_matching .ne. 0)then
!c     oversample the region around the peak 2 to 1 to estimate the fractional offset

!c     write the reference image and search image around the peak into arrays

                  do i_yy=1,i_wsyi
                     do i_xx=1,i_wsxi
                        i_index = (i_yy-1)*i_n2wsxi + i_xx
                        if(i_x+i_xx-1 .ge. 1 .and. i_x+i_xx-1 .le. i_samples(1))then
                           c_chipref(i_index) = c_refimg(i_x+i_xx-1,i_yy)
                        else
                           c_chipref(i_index) = cmplx(0.0,0.0)
                        endif
                     enddo
                  enddo

                  do i_yy=1,i_wsyi
                     do i_xx=i_wsxi+1,i_n2wsxi
                        i_index = (i_yy-1)*i_n2wsxi + i_xx
                        c_chipref(i_index) = cmplx(0.0,0.0)
                     enddo
                  enddo

                  do i_yy=i_wsyi+1,i_n2wsyi
                     do i_xx=1,i_n2wsxi
                        i_index = (i_yy-1)*i_n2wsxi + i_xx
                        c_chipref(i_index) = cmplx(0.0,0.0)
                     enddo
                  enddo

!c     now the search image

                  do i_yy=1,i_wsyjp
                     do i_xx=1,i_wsxjp
                        i_index = (i_yy-1)*i_n2wsxj + i_xx
                        if(i_xlu+i_xx+i_shiftx*i_avgx-i_srchp+i_grossx - i_srchx .gt. 1 .and.
     &                       i_xlu+i_xx+i_shiftx*i_avgx-i_srchp+i_grossx - i_srchx .le. i_samples(2) .and.
     &                       i_yy+i_avgy*i_shifty-i_srchy+(i_srchy-i_srchp) .ge. 1 .and.
     &                       i_yy+i_avgy*i_shifty-i_srchy+(i_srchy-i_srchp) .le. i_wsyj)then
                           c_chipsch(i_index) = c_srchimg(i_xlu+i_xx+i_shiftx*i_avgx-i_srchp+i_grossx-i_srchx-1,
     &                          i_yy+i_shifty*i_avgy-i_srchy+(i_srchy-i_srchp))
                        else
                           c_chipsch(i_index) = cmplx(0.0,0.0)
                        endif
                     enddo
                  enddo

                  do i_yy=1,i_wsyjp
                     do i_xx=i_wsxjp+1,i_n2wsxj
                        i_index = (i_yy-1)*i_n2wsxj + i_xx
                        c_chipsch(i_index) = cmplx(0.0,0.0)
                     enddo
                  enddo

                  do i_yy=i_wsyjp+1,i_n2wsyj
                     do i_xx=1,i_n2wsxj
                        i_index = (i_yy-1)*i_n2wsxj + i_xx
                        c_chipsch(i_index) = cmplx(0.0,0.0)
                     enddo
                  enddo

!c     Dump the reference and search chip images to disk

                  if(i_dump_images .eq. 1)then
                     a_debugfile = 'chip_ref.dat'
                     call dump_chip_c8(a_debugfile,c_chipref,1,i_n2wsxi,1,i_n2wsyi,i_n2wsxi,i_n2wsyi)
                     a_debugfile = 'chip_srch.dat'
                     call dump_chip_c8(a_debugfile,c_chipsch,1,i_n2wsxj,1,i_n2wsyj,i_n2wsxj,i_n2wsyj)
                  endif

!c     Deramp data prior to FFT

                  call derampc(c_chipref,i_n2wsxi,i_n2wsyi)
                  call derampc(c_chipsch,i_n2wsxj,i_n2wsyj)

!c     forward fft the data

                  i_nn(1) = i_n2wsxj
                  i_nn(2) = i_n2wsyj

                  i_dir = 1

                  call fourn2d(c_chipsch,i_nn,i_dir)

                  i_nn(1) = i_n2wsxi
                  i_nn(2) = i_n2wsyi

                  call fourn2d(c_chipref,i_nn,i_dir)

!c     dump forward FFT of data

                  if(i_dump_images .eq. 1)then
                     a_debugfile = 'forwardfft_ref.dat'
                     call dump_chip_c8(a_debugfile,c_chipref,1,i_n2wsxi,1,i_n2wsyi,i_n2wsxi,i_n2wsyi)
                     a_debugfile = 'forwardfft_srch.dat'
                     call dump_chip_c8(a_debugfile,c_chipsch,1,i_n2wsxj,1,i_n2wsyj,i_n2wsxj,i_n2wsyj)
                  endif

!c     spread the spectral data out for inverse transforms

                  i_nn(1) = i_n2wsxi*i_ovs
                  i_nn(2) = i_n2wsyi*i_ovs

                  i_dir = -1

                  do k=1,i_nn(2)
                     do l=1,i_nn(1)
                        i_index = l + (k-1)*i_nn(1)
                        c_osref(i_index) = cmplx(0.0,0.0)
                     enddo
                  enddo

                  do k=1,i_n2wsyi/2
                     do l=1,i_n2wsxi/2
                        i_index = (k-1)*i_nn(1) + l
                        i_indexi = (k-1)*i_n2wsxi + l
                        c_osref(i_index) = c_chipref(i_indexi)
                        i_index = (i_nn(2) - i_n2wsyi/2 + k - 1)*i_nn(1) + l
                        i_indexi = (k + i_n2wsyi/2 - 1)*i_n2wsxi + l
                        c_osref(i_index) = c_chipref(i_indexi)
                        i_index = (k-1)*i_nn(1) + i_nn(1) - i_n2wsxi/2 + l
                        i_indexi = (k-1)*i_n2wsxi + i_n2wsxi/2 + l
                        c_osref(i_index) = c_chipref(i_indexi)
                        i_index = (i_nn(2) - i_n2wsyi/2 + k - 1)*i_nn(1) + i_nn(1) - i_n2wsxi/2 + l
                        i_indexi = (k + i_n2wsyi/2 - 1)*i_n2wsxi + l + i_n2wsxi/2
                        c_osref(i_index) = c_chipref(i_indexi)
                     enddo
                  enddo

!c     dump zero-padded frequency domain data

                  if(i_dump_images .eq. 1)then
                     a_debugfile = 'osfreqdomain_ref.dat'
                     call dump_chip_c8(a_debugfile,c_osref,1,i_n2wsxi*i_ovs,1,i_n2wsyi*i_ovs,i_n2wsxi*i_ovs,i_n2wsyi*i_ovs)
                  endif

                  call fourn2d(c_osref,i_nn,i_dir)

                  i_nn(1) = i_n2wsxj*i_ovs
                  i_nn(2) = i_n2wsyj*i_ovs
                  i_dir = -1

                  do l=1,i_nn(1)
                     do k=1,i_nn(2)
                        i_index = l + (k-1)*i_nn(1)
                        c_ossch(i_index) = cmplx(0.0,0.0)
                     enddo
                  enddo

                  do k=1,i_n2wsyj/2
                     do l=1,i_n2wsxj/2
                        i_index = (k-1)*i_nn(1) + l
                        i_indexi = (k-1)*i_n2wsxj + l
                        c_ossch(i_index) = c_chipsch(i_indexi)
                        i_index = (i_nn(2) - i_n2wsyj/2 + k - 1)*i_nn(1) + l
                        i_indexi = (k + i_n2wsyj/2 - 1)*i_n2wsxj + l
                        c_ossch(i_index) = c_chipsch(i_indexi)
                        i_index = (k-1)*i_nn(1) + i_nn(1) - i_n2wsxj/2 + l
                        i_indexi = (k-1)*i_n2wsxj + i_n2wsxj/2 + l
                        c_ossch(i_index) = c_chipsch(i_indexi)
                        i_index = (i_nn(2) - i_n2wsyj/2 + k - 1)*i_nn(1) + i_nn(1) - i_n2wsxj/2 + l
                        i_indexi = (k + i_n2wsyj/2 - 1)*i_n2wsxj + l + i_n2wsxj/2
                        c_ossch(i_index) = c_chipsch(i_indexi)
                     enddo
                  enddo

!c     dump zero-padded frequency domain data

                  if(i_dump_images .eq. 1)then
                     a_debugfile = 'osfreqdomain_srch.dat'
                     call dump_chip_c8(a_debugfile,c_ossch,1,i_n2wsxj*i_ovs,1,i_n2wsyj*i_ovs,i_n2wsxj*i_ovs,i_n2wsyj*i_ovs)
                  endif

!c     inverse transform

                  call fourn2d(c_ossch,i_nn,i_dir)

!c     dump the oversampled complex image data

                  if(i_dump_images .eq. 1)then
                     a_debugfile = 'cmplx_os_ref.dat'
                     call dump_chip_c8(a_debugfile,c_osref,1,i_n2wsxi*i_ovs,1,i_n2wsyi*i_ovs,i_n2wsxi*i_ovs,i_n2wsyi*i_ovs)
                     a_debugfile = 'cmplx_os_srch.dat'
                     call dump_chip_c8(a_debugfile,c_ossch,1,i_n2wsxj*i_ovs,1,i_n2wsyj*i_ovs,i_n2wsxj*i_ovs,i_n2wsyj*i_ovs)
                  endif

!c     detect images and put into correlation arrays

                  do i_yy=1,i_wsyi*i_ovs
                     do i_xx=1,i_wsxi*i_ovs
                        i_index = i_xx + (i_yy-1)*i_n2wsxi*i_ovs
                        r_imgios(i_xx,i_yy) = cabs(c_osref(i_index)/(i_n2wsxi*i_n2wsyi))
                     enddo
                  enddo

                  do i_yy=1,i_wsyjp*i_ovs
                     do i_xx=1,i_wsxjp*i_ovs
                        i_index = i_xx + (i_yy-1)*i_n2wsxj*i_ovs
                        r_imgjos(i_xx,i_yy) = cabs(c_ossch(i_index))/(i_n2wsxj*i_n2wsyj)
                     enddo
                  enddo

!c     dump the detected image chips used for cross correlation

                  if(i_dump_images .eq. 1)then
                     a_debugfile = 'detected_os_ref.dat'
                     call dump_chip_r4(a_debugfile,r_imgios,1,i_n2wsxi*i_ovs,1,i_n2wsyi*i_ovs,i_n2wsxi*i_ovs,i_n2wsyi*i_ovs)
                     a_debugfile = 'detected_os_srch.dat'
                     call dump_chip_r4(a_debugfile,r_imgjos,1,i_n2wsxj*i_ovs,1,i_n2wsyj*i_ovs,i_n2wsxj*i_ovs,i_n2wsyj*i_ovs)
                  endif

!c     correlate the oversampled chips

                  i_wsxios = i_wsxi*i_ovs
                  i_wsyios = i_wsyi*i_ovs
                  i_wsxjos = i_wsxjp*i_ovs
                  i_wsyjos = i_wsyjp*i_ovs
                  i_wsox = i_wsxjos - (i_wsxios-1)
                  i_wsoy = i_wsyjos - (i_wsyios-1)

                  i_ovss = 1

                  call correlate(r_imgios,r_imgjos,i_wsxios,i_wsyios,
     &                 i_wsxjos,i_wsyjos,1,1,i_ovss,r_meani,r_stdvi,
     &                 r_meanj,r_stdvj,r_peakos,
     &                 r_noise,r_covos,r_eval1,r_eval2,r_evec1,r_evec2,
     &                 r_imgcos,i_shiftxos,i_shiftyos,i_edge,i_flag,l_debug)

                  r_shftxos = float(i_shiftxos)/i_ovs - float((i_wsox-1)/2)/i_ovs + r_shftx
                  r_shftyos = float(i_shiftyos)/i_ovs - float((i_wsoy-1)/2)/i_ovs + r_shfty

!c     display the correlation surface

                  if(l_display)then
                     write(6,*) ' '
                     write(6,*) 'Correlation Surface of oversamples image at ',i_centerxi,i_centeryi
                     do l= max(i_shiftyos-3,1),min(i_shiftyos+5,i_wsoy)
                        write(6,178) (r_imgcos(k,l)**2/r_peakos**2,k=max(i_shiftxos-3,1),min(i_shiftxos+5,i_wsox))
                     enddo
                  endif

!c     dump the correlation surface

                  if(i_dump_images .eq. 1)then
                     a_debugfile = 'correlation_surface.dat'
                     call dump_chip_r4(a_debugfile,r_imgcos,1,i_wsox,1,i_wsoy,i_wsox,i_wsoy)
                  endif

                  r_outside = 0.0
                  i_cnta = 0
                  do l=max(i_shiftyos-9,1),min(i_shiftyos+11,i_wsoy)
                     do k=max(i_shiftxos-9,1),min(i_shiftxos+11,i_wsox)
                        i_cnta = i_cnta + 1
                        r_outside = r_outside + r_imgcos(k,l)**2
                     enddo
                  enddo
                  r_outside = r_outside - r_peakos**2
                  r_outside = r_outside/(i_cnta-1)
                  r_snros = r_peakos**2/min(r_outside,1.e10)

                  r_snros = 10.
                  r_covos(1) = 0.
                  r_covos(2) = 0.

                  if(r_snros .gt. r_snrth .and. r_covos(1) .lt. r_covth .and. r_covos(2) .lt. r_covth)then

!c     oversample the oversampled correlation surface

                     r_max = 0.0
                     r_mean_cor = 0.0
                     i_cnta = 0
                     i_px = i_shiftxos+1
                     i_py = i_shiftyos+1

                     do i_yy=-i_cw/2,i_cw/2-1

                        do i_xx=-i_cw/2,i_cw/2-1

                           i_index = (i_yy+i_cw/2)*i_cw + i_xx + i_cw/2 + 1

                           if (i_xx+i_px .ge. 1 .and. i_xx+i_px .le. (2*i_srchp+1)*i_ovs .and.
     &                          i_yy+i_py .ge. 1 .and. i_yy+i_py .le. (2*i_srchp+1)*i_ovs )then
                              c_corrt(i_index) = cmplx(abs(r_imgcos(i_xx+i_px,i_yy+i_py)/r_peakos),0.)
                              r_mean_cor = r_mean_cor + cabs(c_corrt(i_index))
                              i_cnta = i_cnta + 1
                           else
                              c_corrt(i_index) = cmplx(0.0, 0.0)
                           endif

                           if(cabs(c_corrt(i_index)) .gt. r_max)then
                              r_max = cabs(c_corrt(i_index))
                              i_p1 = i_xx
                              i_p2 = i_yy
                           endif

                        enddo

                     enddo

!c     substract off the mean

                     r_mean_cor = r_mean_cor/max(i_cnta,1)
                     r_mean_cor = 0.0
                     do i_yy=-i_cw/2,i_cw/2-1
                        do i_xx=-i_cw/2,i_cw/2-1
                           i_index = (i_yy+i_cw/2)*i_cw + i_xx + i_cw/2 + 1
                           c_corrt(i_index) = c_corrt(i_index) - cmplx(r_mean_cor,0.0)
                        enddo
                     enddo

!c     dump the correlation around peak used for oversampling

                     if(i_dump_images .eq. 1)then
                        a_debugfile = 'corrsurf_peak.dat'
                        call dump_chip_c8(a_debugfile,c_corrt,1,i_cw,1,i_cw,i_cw,i_cw)
                     endif

!c     oversample the correlation surface

                     if(i_sinc_fourier .eq. i_sinc)then

!c     Use SINC interpolation to oversample the correlation surface. Note will cheat and
!c     and do a series of 1-d interpolations. Assume correlation function is periodic and
!c     do a circular convolution.

                        do i_yy=-i_cw/2,i_cw/2-1

                           do i_xx=-i_sinc_window*i_covs,i_sinc_window*i_covs

                              i_index2 = (i_yy + i_cw/2)*i_covs*i_cw + i_xx + i_cw*i_covs/2 + 1

                              c_dataout(i_index2) = 0.

                              r_jout = float(i_xx + i_cw*i_covs/2 + i_covs)/i_covs + r_fdelay
                              i_jout = int(r_jout)

                              r_frac = r_jout - i_jout
                              i_frac = int(r_frac*i_decfactor)
                              r_sincwgt = 0.0

                              do k=0,i_intplength-1
                                 if(i_jout-k .lt. 1)then
                                    i_index = (i_yy+i_cw/2)*i_cw + (i_jout-k+i_cw)
                                 elseif(i_jout-k .gt. i_cw)then
                                    i_index = (i_yy+i_cw/2)*i_cw + (i_jout-k-i_cw)
                                 else
                                    i_index = (i_yy+i_cw/2)*i_cw + (i_jout-k)
                                 endif
                                 c_dataout(i_index2) = c_dataout(i_index2) +
     +                                c_corrt(i_index)*r_fintp(k + i_frac*i_intplength)
                                 r_sincwgt = r_sincwgt + r_fintp(k + i_frac*i_intplength)
                              enddo
                              c_dataout(i_index2) = c_dataout(i_index2)/r_sincwgt
                           enddo

                        enddo

                        if(i_dump_images .eq. 1)then
                           a_debugfile = 'sinc_stagext.dat'
                           call dump_chip_c8(a_debugfile,c_dataout,1,i_cw*i_covs,1,i_cw,i_cw*i_covs,i_cw)
                        endif

!c     along track resample

!c                        do i_yy=(-i_cw/2)*i_covs,i_cw/2*i_covs-1

!c                           do i_xx=(-i_cw/2)*i_covs,i_cw/2*i_covs-1

                        do i_yy=-i_sinc_window*i_covs,i_sinc_window*i_covs

                           do i_xx=-i_sinc_window*i_covs,i_sinc_window*i_covs

                              i_index2 = (i_yy + i_cw*i_covs/2)*i_cw*i_covs + i_xx + i_cw*i_covs/2 + 1

                              c_dataout2(i_index2) = 0.

                              r_iout = float(i_yy +i_cw*i_covs/2 + i_covs)/i_covs + r_fdelay
                              i_iout = int(r_iout)

                              r_frac = r_iout - i_iout
                              i_frac = int(r_frac*i_decfactor)
                              r_sincwgt = 0.0

                              do k=0,i_intplength-1
                                 if(i_iout-k .lt. 1)then
                                    i_index = i_iout - k + i_cw
                                 elseif(i_iout-k .gt. i_cw)then
                                    i_index = i_iout - k - i_cw
                                 else
                                    i_index = i_iout - k
                                 endif
                                 i_index3 = (i_index-1)*i_cw*i_covs + i_xx + i_cw*i_covs/2 + 1
                                 c_dataout2(i_index2) = c_dataout2(i_index2) +
     +                                c_dataout(i_index3)*r_fintp(k + i_frac*i_intplength)
                                 r_sincwgt = r_sincwgt + r_fintp(k + i_frac*i_intplength)
                              enddo
                              c_dataout2(i_index2) = c_dataout2(i_index2)/r_sincwgt
                              c_corr(i_index2) = c_dataout2(i_index2)

                           enddo

                        enddo

                        if(i_dump_images .eq. 1)then
                           a_debugfile = 'sinc_stageat.dat'
                           call dump_chip_c8(a_debugfile,c_dataout2,1,i_cw*i_covs,1,i_cw*i_covs,i_cw*i_covs,i_cw*i_covs)
                        endif

                     elseif(i_sinc_fourier .eq. i_fourier)then

!c     oversample via Fourier transforms

!c     forward fft the data

                        i_nn(1) = i_cw
                        i_nn(2) = i_cw
                        i_dir = 1

                        call fourn2d(c_corrt,i_nn,i_dir)

!c     dump the correlation around peak used for oversampling

                        if(i_dump_images .eq. 1)then
                           a_debugfile = 'fowfft_corrsurf_peak.dat'
                           call dump_chip_c8(a_debugfile,c_corrt,1,i_cw,1,i_cw,i_cw,i_cw)
                        endif

!c     spread the spectral data out for inverse transforms

                        i_nn(1) = i_cw*i_covs
                        i_nn(2) = i_cw*i_covs
                        i_dir = -1

                        do k=1,i_nn(2)
                           do l=1,i_nn(1)
                              i_index = (k-1)*i_nn(1) + l
                              c_corr(i_index) = 0.0
                           enddo
                        enddo

                        do l=1,i_cw/2
                           do k=1,i_cw/2
                              i_index = (k-1)*i_nn(1) + l
                              i_indexi = (k-1)*i_cw + l
                              c_corr(i_index) = c_corrt(i_indexi)
                              i_index = l + (i_nn(2)-i_cw/2+k-1)*i_nn(1)
                              i_indexi = l + (k+i_cw/2-1)*i_cw
                              c_corr(i_index) = c_corrt(i_indexi)
                              i_index = i_nn(1)-i_cw/2+l + (k-1)*i_nn(2)
                              i_indexi = l+i_cw/2 + (k-1)*i_cw
                              c_corr(i_index) = c_corrt(i_indexi)
                              i_index = i_nn(1)-i_cw/2+l + (i_nn(2)-i_cw/2+k-1)*i_nn(1)
                              i_indexi = l+i_cw/2 + (k+i_cw/2-1)*i_cw
                              c_corr(i_index) = c_corrt(i_indexi)
                           enddo
                        enddo

!c     dump the zero-padded correlation surface

                        if(i_dump_images .eq. 1)then
                           a_debugfile = 'zpadded_corrsurf_peak.dat'
                           call dump_chip_c8(a_debugfile,c_corr,1,i_cw*i_covs,1,i_cw*i_covs,i_cw*i_covs,i_cw*i_covs)
                        endif

!c     inverse transform

                        call fourn2d(c_corr,i_nn,i_dir)

!c     dump the detected oversampled correlation surface

                        if(i_dump_images .eq. 1)then
                           a_debugfile = 'corrsurf_os.dat'
                           call dump_chip_c8(a_debugfile,c_corr,1,i_cw*i_covs,1,i_cw*i_covs,i_cw*i_covs,i_cw*i_covs)
                        endif

                     endif      !sinc vs fourier oversample

!c     detect the peak

                     r_max=0.
                     do i_yy=1,i_cw*i_covs
                        do i_xx=1,i_cw*i_covs
                           i_index = (i_yy-1)*i_cw*i_covs + i_xx
                           if(i_sinc_fourier .eq. i_fourier)then
                              r_corr(i_xx,i_yy) = cabs(c_corr(i_index))/((i_cw**2)*(i_cw*i_covs)**2)
                           else
                              r_corr(i_xx,i_yy) = cabs(c_corr(i_index))
                           endif
                           if (abs(i_xx-i_cw*i_covs/2) .le. i_covs .and.
     &                          abs(i_yy-i_cw*i_covs/2) .le. i_covs) then
                              if (r_corr(i_xx,i_yy) .ge. r_max) then
                                 r_max = r_corr(i_xx,i_yy)
                                 i_cpeak(1) = i_xx - i_cw/2*i_covs
                                 i_cpeak(2) = i_yy - i_cw/2*i_covs
                              endif
                           endif
                        enddo
                     enddo

!c     dump the detected oversampled correlation surface

                     if(i_dump_images .eq. 1)then
                        a_debugfile = 'detected_corrsurf.dat'
                        call dump_chip_r4(a_debugfile,r_corr,1,i_cw*i_covs,1,i_cw*i_covs,i_cw*i_covs,i_cw*i_covs)
                     endif

                     r_oscoroff(1) = float(i_cpeak(1)-1)/float(i_covs)
                     r_oscoroff(2) = float(i_cpeak(2)-1)/float(i_covs)

                     r_shftxosc = r_oscoroff(1)/i_ovs + r_shftxos + i_xlu - i_x
                     r_shftyosc = r_oscoroff(2)/i_ovs + r_shftyos + i_ylu - i_y
                     r_snr = min(r_snr,9999.99999)

!cc -- write to outfile ch 15.
!cc                     t2 = seconds(t0)           ! start timer

                     numRowTable = numRowTable + 1
                     i_centerxiArr(numRowTable) = i_centerxi
                     i_centeryiArr(numRowTable) = i_centeryi
                     r_shftxoscArr(numRowTable) = r_shftxosc
                     r_shftyoscArr(numRowTable) = r_shftyosc
                     r_snrArr(numRowTable) = r_snr
                     r_cov1Arr(numRowTable) = r_cov(1)
                     r_cov2Arr(numRowTable) = r_cov(2)
                     r_cov3Arr(numRowTable) = r_cov(3)
!c                     write(15,151) i_centerxi,r_shftxosc,i_centeryi,r_shftyosc,
!c     &                    r_snr,r_cov(1),r_cov(2),r_cov(3)
                     t3 = seconds(t0)           ! start timer
!cc                     write(6,*) 'XXX time for writing ch 15 ', t3-t2
! 151                 format(1x,i7,1x,f9.3,1x,i7,1x,f9.3,1x,f10.5,1x,f10.6,1x,f10.6,1x,f10.6)
! 150                 format(1x,i7,1x,f9.3,1x,f9.3,1x,f9.3,1x,i7,1x,f9.3,1x,f9.3,1x,f9.3,1x,
!     &                    f10.5,1x,f10.3,1x,f10.3,1x,f10.3)

                  else

                     write(6,*) 'Bad match at level 2'

                  endif         !thresholds second pass

               else

                  write(6,*) 'Bad match at level 1'

               endif            !thresholds

            endif               !not edge point or no data point

            if(i_dump_images .eq. 1)then
!c               stop  ! EMG
                go to 999   ! to close open files and return.
            endif

         enddo                  !samples loop (j)

         t5 = seconds(t0)           ! start timer
         write(6,*) 'XXX time for inner loop ', t5-t4

      enddo                     !line loop (i)

 489  continue

      t1=seconds(t0)
      write(6,*) 'Elapsed time. ', t1


!c Close files before exiting
!c If we reach this point from the go to 999 line above, then something
!c different probably needs to be done to handle closing unit 115.  What I
!c am doing here treats unit 115 the same as what was done before; this only
!c fixes the units that are not specifically in the MPI branch.
 999  continue
         deallocate( c_refimg )
         deallocate( c_srchimg )
         deallocate( r_refimg )
         deallocate( r_srchimg )
         deallocate( r_corr)
         deallocate( c_corr)
         deallocate( c_dataout2)
         deallocate( c_dataout)
         deallocate( c_corrt)
         deallocate( r_imgi)
         deallocate( r_imgj)
         deallocate( r_imgc)
         deallocate( r_imgios)
         deallocate( r_imgjos)
         deallocate( r_imgcos)
         deallocate( c_chipref)
         deallocate( c_chipsch)
         deallocate( c_ossch)
         deallocate( c_osref)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!c    for ScanSAR full-aperture mode matching
         deallocate( r_imgi_filt)
         deallocate( r_imgj_filt)
         deallocate( r_imgc_filt)
         deallocate( c_corr_filt)
         deallocate( c_corrt_filt)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      close(13)
      close(14)
!c     close(15)

      end

c****************************************************************

      subroutine correlate(r_imgi,r_imgj,i_wsxi,i_wsyi,i_wsxj,
     &     i_wsyj,i_avgx,i_avgy,i_ovs,r_meani,r_stdvi,r_meanj,r_stdvj,
     &     r_peak,r_noise,r_cov,r_eval1,r_eval2,
     &     r_evec1,r_evec2,r_imgc,i_shftx,i_shfty,i_edge,i_flag,
     &     l_debug)

c****************************************************************
c**
c**   FILE NAME: correlate.f
c**
c**   DATE WRITTEN: /10/10/92
c**
c**   PROGRAMMER:Scott Hensley / Scott Shaffer
c**
c**   FUNCTIONAL DESCRIPTION: This routine will do amplitude correlation
c
c**   on two specified input files.
c**
c**   ROUTINES CALLED:none
c**
c**   NOTES: none
c**
c**   UPDATE LOG:
c**
c**   Date      Description                              Person
c**   ----      -----------                              ------
c**   /12/12/94   Modified to work with real data.         SH
c**   /02/22/95   Modified to work oversampled data.      SS/SH
c**
c*****************************************************************

      implicit none

c     INPUT VARIABLES:

      integer i_wsyi,i_wsxi,i_wsyj,i_wsxj,i_ovs
      integer i_avgy,i_avgx,i_wsayi,i_wsaxi
      integer i_wsayj,i_wsaxj,i_wsaxyi,i_wsaxyj

      real r_imi(i_wsxj,i_wsyj)
      real r_imgc(i_wsxj,i_wsyj)
      real r_imj(i_wsxj,i_wsyj)
      real r_imgi(i_wsxi,i_wsxi)
      real r_imgj(i_wsxj,i_wsxj)

c     OUTPUT VARIABLES:
      real*4 r_shfty,r_shftx,r_peak,r_shrp,r_meani,r_meanj
      real*4 r_stdvi,r_stdvj,r_noise,r_cov(3),r_eval1,r_sum
      real*4 r_eval2,r_evec1(2),r_evec2(2)

c     LOCAL VARIABLES:
      integer i,j,m,n,ix,iy,ixx,iyy,i_shfty,i_shftx,io
      integer i_cnti,i_cntj,i_cntai,i_cntaj,i_edge(2),i_flag

      real r_sumc,r_sumi,r_smqi
      real r_denom

      real, dimension(:,:), allocatable :: r_sumj, r_smqj
      real, dimension(:,:), allocatable :: r_crpd, r_corr, r_corn

      real*4 r_dxx,r_dyy,r_dxy,r_n2,r_n4,r_u,r_u2

      logical l_init,l_debug

c     DATA STATEMENTS:
      data l_init /.false./

C     FUNCTION STATEMENTS:

c     PROCESSING STEPS:

      if(l_debug)then
         write(6,*) ' '
         write(6,*) ' Debug Statements ** Inputs ** '
         write(6,*) 'r_imgi(1,1),r_imgj(1,1) = ',
     &        r_imgi(1,1),r_imgj(1,1)
         write(6,*)
     &        ' r_imgi(i_wsxi,i_wsyi),r_imgj(i_wsxj,i_wsyj) = ',
     &        r_imgi(i_wsxi,i_wsyi),r_imgj(i_wsxj,i_wsyj)
         write(6,*) 'i_wsxi and i_wsyi = ',i_wsxi,i_wsyi
         write(6,*) 'i_wsxj and i_wsyj = ',i_wsxj,i_wsyj
         write(6,*) 'i_avgx and i_avgy = ',i_avgx,i_avgy
         write(6,*) 'r_meani and r_stdvi = ',r_meani,r_stdvi
         write(6,*) 'r_meanj and r_stdvj = ',r_meanj,r_stdvj
         write(6,*) 'r_peak and r_noise = ',r_peak,r_noise
         write(6,*) 'r_shftx and r_shfty = ',r_shftx,r_shfty
         write(6,*) 'i_edge and i_flag = ',i_edge(1),i_edge(2),i_flag
      endif


      allocate(r_sumj(0:i_wsxj,0:i_wsyj))
      allocate(r_smqj(0:i_wsxj,0:i_wsyj))
      allocate(r_crpd(0:i_wsxj,0:i_wsyj))
      allocate(r_corr(0:i_wsxj,0:i_wsyj))
      allocate(r_corn(0:i_wsxj,0:i_wsyj))

      i_edge(1)=0
      i_edge(2)=0
      if ( i_avgy .le. 0 ) i_avgy=1
      if ( i_avgx .le. 0 ) i_avgx=1
      i_wsayi=i_wsyi/i_avgy
      i_wsaxi=i_wsxi/i_avgx
      i_wsayj=i_wsyj/i_avgy
      i_wsaxj=i_wsxj/i_avgx
      i_wsaxyi=i_wsayi*i_wsaxi
      i_wsaxyj=i_wsayj*i_wsaxj/i_ovs

      r_cov(1)=0.
      r_cov(2)=0.
      r_cov(3)=0.

c     compute mean and standard deviations on blocks

      i_cntai = 0
      i_cntaj = 0
      r_sumi = 0.
      r_smqi = 0.
      do iy=1,i_wsayj
         do ix=1,i_wsaxj
            r_imgc(ix,iy) = 0.
            r_imi(ix,iy) = 0.
            r_imj(ix,iy) = 0.
            i_cnti=0
            i_cntj=0
            if(i_avgy .ne. 1 .or. i_avgx .ne. 1)then
               do iyy=(iy-1)*i_avgy+1,iy*i_avgy
                  do ixx=(ix-1)*i_avgx+1,ix*i_avgx
                     if ( iyy .le. i_wsyi .and. ixx .le. i_wsxi ) then
                        if ( r_imgi(ixx,iyy) .ne. 0. ) then
                           i_cnti = i_cnti+1
                           r_imi(ix,iy) = r_imi(ix,iy) + r_imgi(ixx,iyy)
                        endif
                     endif
                     if ( r_imgj(ixx,iyy) .ne. 0. ) then
                        i_cntj = i_cntj+1
                        r_imj(ix,iy) = r_imj(ix,iy) + r_imgj(ixx,iyy)
                     endif
                  enddo
               enddo
               if ( i_cnti .ne. 0 ) then
                  i_cntai = i_cntai+1
                  r_imi(ix,iy) = r_imi(ix,iy)/i_cnti
                  r_sumi = r_sumi + r_imi(ix,iy)
                  r_smqi = r_smqi + r_imi(ix,iy)**2
               endif
               if ( i_cntj .ne. 0 ) then
                  r_imj(ix,iy) = r_imj(ix,iy)/i_cntj
                  i_cntaj = i_cntaj+1
               endif
            else
               r_imj(ix,iy) = r_imgj(ix,iy)
               if(ix .le. i_wsxi .and. iy .le. i_wsyi)then
                  r_imi(ix,iy) = r_imgi(ix,iy)
                  if(r_imi(ix,iy) .ne. 0)then
                     i_cntai = i_cntai+1
                     r_sumi = r_sumi + r_imi(ix,iy)
                     r_smqi = r_smqi + r_imi(ix,iy)**2
                  endif
               endif
               if(r_imj(ix,iy) .ne. 0)then
                  i_cntaj = i_cntaj+1
               endif
            endif               !no averaging
         enddo
      enddo


      if ( i_cntai .ne. 0 ) then
         r_meani = r_sumi/i_cntai
         r_stdvi = sqrt((r_smqi/i_cntai)-r_meani**2)
      else
         r_meani = 0.
      endif

      if (i_cntai .ge. 0.9*i_wsaxyi .and.
     &     i_cntaj .ge. 0.9*i_wsaxyj ) then !have enough real estate

         do iy=0,i_wsayj-1
            r_sumj(0,iy) = 0.
            r_smqj(0,iy) = 0.
            do io = 1,i_ovs
               r_sumj(io,iy) = 0.
               r_smqj(io,iy) = 0.
               do ix=0,(i_wsaxi-1)*i_ovs,i_ovs
                  r_sumj(io,iy) = r_sumj(io,iy) + r_imj(ix+io,iy+1)
                  r_smqj(io,iy) = r_smqj(io,iy) + r_imj(ix+io,iy+1)**2
               enddo
            enddo

            do ix=i_ovs+1,i_wsaxj - (i_wsaxi-1)*i_ovs
               r_sumj(ix,iy) = r_sumj(ix-i_ovs,iy) - r_imj(ix-i_ovs,iy+1
     &              ) +r_imj(ix+(i_wsaxi-1)*i_ovs,iy+1)
               r_smqj(ix,iy) = r_smqj(ix-i_ovs,iy) - r_imj(ix-i_ovs,iy+1
     &              )**2 +r_imj(ix+(i_wsaxi-1)*i_ovs,iy+1)**2
            enddo
         enddo

         do ix=0,i_wsaxj - (i_wsaxi-1)*i_ovs-1
            do io=1,i_ovs
               r_sumj(ix,io-1)=0.
               r_smqj(ix,io-1)=0.
               do iy=0,(i_wsayi-1)*i_ovs,i_ovs
                  r_sumj(ix,io-1) = r_sumj(ix,io-1)+r_sumj(ix+1,iy+io-1)
                  r_smqj(ix,io-1) = r_smqj(ix,io-1)+r_smqj(ix+1,iy+io-1)
               enddo
            enddo

            do iy=i_ovs,i_wsayj - (i_wsayi-1)*i_ovs-1
               r_sumj(ix,iy) = r_sumj(ix,iy-i_ovs) - r_sumj(ix+1,iy
     &              -i_ovs)+r_sumj(ix+1,iy+(i_wsayi-1)*i_ovs)
               r_smqj(ix,iy) = r_smqj(ix,iy-i_ovs) - r_smqj(ix+1,iy
     &              -i_ovs)+r_smqj(ix+1,iy+(i_wsayi-1)*i_ovs)
            enddo
         enddo

c         type *,' '
c         do ix=0,i_wsaxj - (i_wsaxi-1)*i_ovs-1
c            do iy=0,i_wsayj - (i_wsayi-1)*i_ovs-1
c               r_sum=0.
c               do ixx=ix+1,ix+i_wsaxi*i_ovs,i_ovs
c                  do iyy=iy+1,iy+i_wsayi*i_ovs,i_ovs
c                     r_sum=r_sum+r_imj(ixx,iyy)
c                  enddo
c               enddo
c               type *,ix,iy,r_sumj(ix,iy),r_sum,r_sumj(ix,iy)-r_sum
c            enddo
c         enddo

         i_shftx = 0
         i_shfty = 0
         r_peak = -9.e27
         do m=0,i_wsaxj - (i_wsaxi-1)*i_ovs-1
            do n=0,i_wsayj - (i_wsayi-1)*i_ovs-1
               r_sumc = 0.
               do j=1,i_wsayi
                  do i=1,i_wsaxi
                     r_sumc = r_sumc + r_imi(i,j)*r_imj((i-1)*i_ovs+m+1,(j-1)*i_ovs+n+1)
                  enddo
               enddo
               r_crpd(m,n) = r_sumc
               r_corr(m,n) = r_sumc - r_meani*r_sumj(m,n)
               r_denom = (r_stdvi*sqrt((r_smqj(m,n)*i_wsaxyi)-
     &              (r_sumj(m,n))**2))
               if ( r_denom .gt. 0. ) then
                  r_corn(m,n) = r_corr(m,n)/r_denom
               else
                  r_corn(m,n) = 0.
               endif
               r_imgc(m+1,n+1) = r_corn(m,n)
c               if(i_wsxi .eq. 112)then
c                  type*, 'r_c = ',m,n,r_corn(m,n),r_crpd(m,n),r_meani*r_sumj(m,n),
c     +                 r_crpd(m,n)-r_meani*r_sumj(m,n),r_sumj(m,n),r_denom
c               endif
               if ( r_peak .lt. r_corn(m,n)) then
                  r_peak = r_corn(m,n)
                  i_shftx = m
                  i_shfty = n
               endif
            enddo
         enddo

c     commpute the curvature of the corrrelation surface to estimate the
c     goodness of the match

         if ( r_peak .gt. 0. ) then

            ix = i_shftx
            iy = i_shfty
            if ( iy .eq. 0 .or. iy .eq. i_wsayj - (i_wsayi-1)*i_ovs-1 )
     &           i_edge(1)=1
            if ( ix .eq. 0 .or. ix .eq. i_wsaxj - (i_wsaxi-1)*i_ovs-1 )
     &           i_edge(2)=1
            r_shftx = float(ix*i_avgx)/i_ovs
            r_shfty = float(iy*i_avgy)/i_ovs
            r_meanj = r_sumj(ix,iy)/i_wsaxyi
            r_stdvj = sqrt((r_smqj(ix,iy)/i_wsaxyi)-r_meanj**2)
            r_shrp = (r_peak-(r_corn(max(ix-1,1),iy)+
     &           r_corn(min(ix+1,i_wsaxj - (i_wsaxi-1)*i_ovs-1),iy))/2.)
            i_flag = 0

            if ( ix .eq. 0 ) then
               if ( iy .eq. 0 ) then
                  r_dxx = -(r_corn(ix+1,iy)+r_corn(ix+1,iy)-
     &                 2*r_corn(ix,iy))/(i_avgx**2)
                  r_dyy = -(r_corn(ix,iy+1)+r_corn(ix,iy+1)-
     &                 2*r_corn(ix,iy))/(i_avgy**2)
                  r_dxy = 0.
                  r_dxx = r_dxx/4 ! added emperically
                  r_dyy = r_dyy/4
                  r_dxy = r_dxy/4
                  r_peak = r_peak/4
               else if ( iy .eq. i_wsayj - (i_wsayi-1)*i_ovs-1 ) then
                  r_dxx = -(r_corn(ix+1,iy)+r_corn(ix+1,iy)-
     &                 2*r_corn(ix,iy))/(i_avgx**2)
                  r_dyy = -(r_corn(ix,iy-1)+r_corn(ix,iy-1)-
     &                 2*r_corn(ix,iy))/(i_avgy**2)
                  r_dxy = 0
                  r_dxx = r_dxx/4 ! added emperically
                  r_dyy = r_dyy/4
                  r_dxy = r_dxy/4
                  r_peak = r_peak/4
               else
                  r_dxx = -(r_corn(ix+1,iy)+r_corn(ix+1,iy)-
     &                 2*r_corn(ix,iy))/(i_avgx**2)
                  r_dyy = -(r_corn(ix,iy+1)+r_corn(ix,iy-1)-
     &                 2*r_corn(ix,iy))/(i_avgy**2)
                  r_dxy = 2*(r_corn(ix+1,iy+1)-
     &                 r_corn(ix+1,iy-1))/(4*i_avgx*i_avgy)
                  r_dxx = r_dxx/2 ! added emperically
                  r_dyy = r_dyy/2
                  r_dxy = r_dxy/2
                  r_peak = r_peak/2
               endif
            else if ( ix .eq. i_wsaxj - (i_wsaxi-1)*i_ovs-1 ) then
               if ( iy .eq. 0 ) then
                  r_dxx = -(r_corn(ix-1,iy)+r_corn(ix-1,iy)-
     &                 2*r_corn(ix,iy))/(i_avgx**2)
                  r_dyy = -(r_corn(ix,iy+1)+r_corn(ix,iy+1)-
     &                 2*r_corn(ix,iy))/(i_avgy**2)
                  r_dxy = 0
                  r_dxx = r_dxx/4 ! added emperically
                  r_dyy = r_dyy/4
                  r_dxy = r_dxy/4
                  r_peak = r_peak/4
               else if ( iy .eq. i_wsayj - (i_wsayi-1)*i_ovs-1 ) then
                  r_dxx = -(r_corn(ix-1,iy)+r_corn(ix-1,iy)-
     &                 2*r_corn(ix,iy))/(i_avgx**2)
                  r_dyy = -(r_corn(ix,iy-1)+r_corn(ix,iy-1)-
     &                 2*r_corn(ix,iy))/(i_avgy**2)
                  r_dxy = 0
                  r_dxx = r_dxx/4 ! added emperically
                  r_dyy = r_dyy/4
                  r_dxy = r_dxy/4
                  r_peak = r_peak/4
               else
                  r_dxx = -(r_corn(ix-1,iy)+r_corn(ix-1,iy)-
     &                 2*r_corn(ix,iy))/(i_avgx**2)
                  r_dyy = -(r_corn(ix,iy+1)+r_corn(ix,iy-1)-
     &                 2*r_corn(ix,iy))/(i_avgy**2)
                  r_dxy = 2*(r_corn(ix-1,iy-1)-
     &                 r_corn(ix-1,iy+1))/(4*i_avgx*i_avgy)
                  r_dxx = r_dxx/2 ! added emperically
                  r_dyy = r_dyy/2
                  r_dxy = r_dxy/2
                  r_peak = r_peak/2
               endif
            else if ( iy .eq. 0 ) then
               r_dxx = -(r_corn(ix+1,iy)+r_corn(ix-1,iy)-
     &              2*r_corn(ix,iy))/(i_avgx**2)
               r_dyy = -(r_corn(ix,iy+1)+r_corn(ix,iy+1)-
     &              2*r_corn(ix,iy))/(i_avgy**2)
               r_dxy = 2*(r_corn(ix+1,iy+1)-
     &              r_corn(ix-1,iy+1))/(4*i_avgx*i_avgy)
               r_dxx = r_dxx/2  ! added emperically
               r_dyy = r_dyy/2
               r_dxy = r_dxy/2
               r_peak = r_peak/2
            else if ( iy .eq. i_wsayj - (i_wsayi-1)*i_ovs-1 ) then
               r_dxx = -(r_corn(ix+1,iy)+r_corn(ix-1,iy)-
     &              2*r_corn(ix,iy))/(i_avgx**2)
               r_dyy = -(r_corn(ix,iy-1)+r_corn(ix,iy-1)-
     &              2*r_corn(ix,iy))/(i_avgy**2)
               r_dxy = 2*(r_corn(ix-1,iy-1)-
     &              r_corn(ix+1,iy-1))/(4*i_avgx*i_avgy)
               r_dxx = r_dxx/2  ! added emperically
               r_dyy = r_dyy/2
               r_dxy = r_dxy/2
               r_peak = r_peak/2
            else
               r_dxx = -(r_corn(ix+1,iy)+r_corn(ix-1,iy)-
     &              2*r_corn(ix,iy))/(i_avgx**2)
               r_dyy = -(r_corn(ix,iy+1)+r_corn(ix,iy-1)-
     &              2*r_corn(ix,iy))/(i_avgy**2)
               r_dxy = (r_corn(ix+1,iy+1)+
     &              r_corn(ix-1,iy-1)-r_corn(ix+1,iy-1)-
     &              r_corn(ix-1,iy+1))/(4*i_avgx*i_avgy)
            endif

            r_n2 = max(1.-r_peak,0.e0)
            r_noise = sqrt(r_n2)
            r_dxx = r_dxx*i_wsaxyi
            r_dyy = r_dyy*i_wsaxyi
            r_dxy = r_dxy*i_wsaxyi

            r_n4 = r_n2**2
            r_n2 = r_n2*2
            r_n4 = r_n4*.5*i_wsaxyi

            r_u = r_dxy**2-r_dxx*r_dyy
            r_u2 = r_u**2       !                    *i_avgx*i_avgy/i_wsaxyi
            if ( r_u .eq. 0 ) then
               r_cov(1)=99.
               r_cov(2)=99.
               r_cov(3)=0.
               i_flag=1
            else
               r_cov(1)=(-r_n2*r_u*r_dyy+r_n4*(r_dyy**2+r_dxy**2))
     &              /r_u2
               r_cov(2)=(-r_n2*r_u*r_dxx+r_n4*(r_dxx**2+r_dxy**2))
     &              /r_u2
               r_cov(3)=((r_n2*r_u      -r_n4*(r_dxx+r_dyy))*r_dxy)
     &              /r_u2
            endif
            r_u=sqrt((r_cov(1)+r_cov(2))**2.-4.*(r_cov(1)*r_cov(2)-
     &           r_cov(3)**2))
            r_eval1=(r_cov(1)+r_cov(2)+r_u)/2.
            r_eval2=(r_cov(1)+r_cov(2)-r_u)/2.
            if ( r_eval1 .le. 0 .or. r_eval2 .le. 0 ) then
            endif

            if ( r_cov(3) .eq. 0 ) then
               if ( r_cov(1) .ge. r_cov(2) ) then
                  r_evec1(1)=1.
                  r_evec1(2)=0.
                  r_evec2(1)=0.
                  r_evec2(2)=1.
               else
                  r_evec1(1)=0.
                  r_evec1(2)=1.
                  r_evec2(1)=1.
                  r_evec2(2)=0.
               endif
            else
               if ( r_cov(1)-r_eval1 .ne. 0. ) then
                  r_evec1(1)=-r_cov(3)/(r_cov(1)-r_eval1)
               else
                  write(6,*) 'e vector 1 error'
                  r_evec1(1)=999.
               endif
               r_evec1(2)=1.
               r_u=sqrt(r_evec1(1)**2+r_evec1(2)**2)
               r_evec1(1)=r_evec1(1)/r_u
               r_evec1(2)=r_evec1(2)/r_u

               if ( r_cov(1)-r_eval2 .ne. 0. ) then
                  r_evec2(1)=-r_cov(3)/(r_cov(1)-r_eval2)
               else
                  write(6,*) 'e vector 2 error'
                  r_evec2(1)=999.
               endif
               r_evec2(2)=1.
               r_u=sqrt(r_evec2(1)**2+r_evec2(2)**2)
               r_evec2(1)=r_evec2(1)/r_u
               r_evec2(2)=r_evec2(2)/r_u
            endif

            r_evec1(1)=r_evec1(1)*sqrt(abs(r_eval1))
            r_evec1(2)=r_evec1(2)*sqrt(abs(r_eval1))
            r_evec2(1)=r_evec2(1)*sqrt(abs(r_eval2))
            r_evec2(2)=r_evec2(2)*sqrt(abs(r_eval2))

         else

            r_shfty=0
            r_shftx=0
            r_shrp=0.
            i_flag=1
            write(6,*) 'correlation error'

         endif

      else

         r_shfty=0
         r_shftx=0
         r_shrp=0.
         i_flag=1

      endif


      deallocate(r_sumj)
      deallocate(r_smqj)
      deallocate(r_crpd)
      deallocate(r_corr)
      deallocate(r_corn)

      if(l_debug)then
         write(6,*) ' '
         write(6,*) 'Exit values'
         write(6,*) 'i_wsxi and i_wsyi = ',i_wsxi,i_wsyi
         write(6,*) 'i_wsxj and i_wsyj = ',i_wsxj,i_wsyj
         write(6,*) 'i_avgx and i_avgy = ',i_avgx,i_avgy
         write(6,*) 'r_meani and r_stdvi = ',r_meani,r_stdvi
         write(6,*) 'r_meanj and r_stdvj = ',r_meanj,r_stdvj
         write(6,*) 'r_peak and r_noise = ',r_peak,r_noise
         write(6,*) 'r_cov = ',r_cov(1),r_cov(2),r_cov(3)
         write(6,*) 'r_eval1 and r_eval2 = ',r_eval1,r_eval2
         write(6,*) 'r_evec1 and r_evec2 = ',r_evec1(1),r_evec1(2),
     &        r_evec2(1), r_evec2(2)
         write(6,*) 'r_shftx and r_shfty = ',r_shftx,r_shfty
         write(6,*) 'i_edge and i_flag = ',i_edge(1),i_edge(2),i_flag
      endif

      return

      end

cc--------------------------------------------------

      subroutine derampc(c_img,i_dimx,i_dimy)

      implicit none
      integer i_dimx,i_dimy,i,j
      complex c_img(i_dimx,i_dimy),c_phdn,c_phac
      real r_phac,r_phdn

      c_phdn = cmplx(0.,0.)
      c_phac = cmplx(0.,0.)

      do i=1,i_dimx-1
         do j=1,i_dimy
            c_phac = c_phac + c_img(i,j)*conjg(c_img(i+1,j))
         enddo
      enddo

      do i=1,i_dimx
         do j=1,i_dimy-1
            c_phdn = c_phdn + c_img(i,j)*conjg(c_img(i,j+1))
         enddo
      enddo

      if(cabs(c_phdn) .eq. 0)then
         r_phdn = 0.0
      else
         r_phdn = atan2(aimag(c_phdn),real(c_phdn))
      endif

      if(cabs(c_phac) .eq. 0)then
         r_phac = 0.0
      else
         r_phac = atan2(aimag(c_phac),real(c_phac))
      endif

c       write(6,*) 'Phase across, down = ',r_phac,r_phdn

      do i=1,i_dimx
         do j=1,i_dimy
            c_img(i,j) = c_img(i,j)*cmplx(cos(r_phac*i+r_phdn*j),
     &           sin(r_phac*i+r_phdn*j))
         enddo
      enddo

      end

cc--------------------------------------------------

      subroutine fourn2d(data,nn,isign)

      complex data(*), d(16384)
      integer nn(2),n,is

      is = -isign
      n = nn(1)
      do i = 1,nn(2)
         call cfft1d_jpl(nn(1),data(1+nn(1)*(i-1)),is)
      end do

      do i = 1,nn(1)

         do j = 1,nn(2)
            d(j) = data(i+nn(1)*(j-1))
         end do

         call cfft1d_jpl(nn(2),d,is)

         do j = 1 , nn(2)
            if(is .eq. 1)then
               d(j) = d(j)*nn(1)*nn(2)
           endif
           data(i+nn(1)*(j-1)) = d(j)
         end do

      end do

      return
      end

c****************************************************************

            integer function nextpower(i_num)

c****************************************************************
c**
c**   FILE NAME: nextpower.f
c**
c**   DATE WRITTEN: 6/1/97
c**
c**   PROGRAMMER: Scott Hensley
c**
c**   FUNCTIONAL DESCRIPTION: Computes the closest number which is a
c**   power of two and returns the exponent of two for the number that
c**   is the first power of two exceeding the input number.
c**
c**   ROUTINES CALLED:
c**
c**   NOTES:
c**
c**   UPDATE LOG:
c**
c**   Date Changed        Reason Changed                  CR # and Version #
c**   ------------       ----------------                 -----------------
c**
c*****************************************************************

      implicit none

c     INCLUDE FILES:

c     PARAMETER STATEMENTS:

c     INPUT VARIABLES:

      integer i_num

c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      real*8 r_num,r_log2,r_log2numm1
      integer i_temp

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

      data r_log2 /.301029995664d0/

c     FUNCTION STATEMENTS:

c     SAVE STATEMENTS:

      save r_log2

c     PROCESSING STEPS:

      r_num = i_num

      r_log2numm1 = dlog10(r_num - .5d0)/r_log2

      nextpower = int(r_log2numm1)+1

      end

c****************************************************************

      subroutine dump_chip_c8(a_filename,c_data,i_startsamp,
     +     i_endsamp,i_startline,i_endline,i_physical_samps,
     +     i_physical_lines)

c****************************************************************
c**
c**   FILE NAME: dump_chip_c8.f
c**
c**   DATE WRITTEN: 7/3/2002
c**
c**   PROGRAMMER: Scott Hensley
c**
c**   FUNCTIONAL DESCRIPTION: This routine will take data
c**   in a 2-D array and output into a direct access file.
c**
c**   ROUTINES CALLED:
c**
c**   NOTES:
c**
c**   UPDATE LOG:
c**
c**   Date Changed        Reason Changed                  CR # and Version #
c**   ------------       ----------------                 -----------------
c**
c*****************************************************************

      implicit none

c     INCLUDE FILES:

c     PARAMETER STATEMENTS:

      integer i_unit
      parameter(i_unit=99)

c     INPUT VARIABLES:

      character*(*) a_filename
      integer i_physical_samps,i_physical_lines
      complex*8 c_data(i_physical_samps,i_physical_lines)
      integer i_startline,i_endline
      integer i_startsamp,i_endsamp

c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i_samples,i,j,i_sl

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION STATEMENTS:

c     SAVE STATEMENTS:

c     PROCESSING STEPS:

c     open file

      i_samples = i_endsamp - i_startsamp + 1
      i_sl = index(a_filename,' ') - 1

      write(6,*) ' '
      write(6,'(a)') 'Opening direct access complex file: '//a_filename(1:i_sl)
      write(6,'(a,1x,i10)') 'Record length: ',i_samples

      open(i_unit,file=a_filename,form='unformatted',access='direct',recl=8*i_samples)

      do i=i_startline,i_endline
         write(i_unit,rec=i-i_startline+1) (c_data(j,i),j=i_startsamp,i_endsamp)
      enddo

      close(i_unit)

      end

c****************************************************************

      subroutine dump_chip_r4(a_filename,r_data,i_startsamp,
     +     i_endsamp,i_startline,i_endline,i_physical_samps,
     +     i_physical_lines)

c****************************************************************
c**
c**   FILE NAME: dump_chip_r4.f
c**
c**   DATE WRITTEN: 7/3/2002
c**
c**   PROGRAMMER: Scott Hensley
c**
c**   FUNCTIONAL DESCRIPTION: This routine will take data
c**   in a 2-D array and output into a direct access file.
c**
c**   ROUTINES CALLED:
c**
c**   NOTES:
c**
c**   UPDATE LOG:
c**
c**   Date Changed        Reason Changed                  CR # and Version #
c**   ------------       ----------------                 -----------------
c**
c*****************************************************************

      implicit none

c     INCLUDE FILES:

c     PARAMETER STATEMENTS:

      integer i_unit
      parameter(i_unit=99)

c     INPUT VARIABLES:

      character*(*) a_filename
      integer i_physical_samps,i_physical_lines
      real*4 r_data(i_physical_samps,i_physical_lines)
      integer i_startline,i_endline
      integer i_startsamp,i_endsamp

c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i_samples,i,j,i_sl

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION STATEMENTS:

c     SAVE STATEMENTS:

c     PROCESSING STEPS:

c     open file

      i_samples = i_endsamp - i_startsamp + 1
      i_sl = index(a_filename,' ') - 1

      write(6,*) ' '
      write(6,'(a)') 'Opening direct access real*4 file: '//a_filename(1:i_sl)
      write(6,'(a,1x,i10)') 'Record length: ',i_samples

      open(i_unit,file=a_filename,form='unformatted',access='direct',recl=4*i_samples)

      do i=i_startline,i_endline
         write(i_unit,rec=i-i_startline+1) (r_data(j,i),j=i_startsamp,i_endsamp)
      enddo

      close(i_unit)

      end

c****************************************************************

      subroutine fill_sinc(r_beta,r_relfiltlen,i_decfactor,i_weight,
     +     r_pedestal,i_intplength,r_fdelay,r_fintp)

c****************************************************************
c**
c**   FILE NAME: fill_sinc.f
c**
c**   DATE WRITTEN: 2/2/98
c**
c**   PROGRAMMER: Scott Hensley
c**
c**   FUNCTIONAL DESCRIPTION: This routine computes the sinc interpolation
c**   coefficients needed by the processor for various range and azimuth
c**   interpolations.
c**
c**   ROUTINES CALLED:
c**
c**   NOTES:
c**
c**   UPDATE LOG:
c**
c**   Date Changed        Reason Changed                  CR # and Version #
c**   ------------       ----------------                 -----------------
c**
c*****************************************************************

      implicit none

c     INCLUDE FILES:

c     PARAMETER STATEMENTS:

      integer MAXDECFACTOR      ! maximum lags in interpolation kernels
      parameter(MAXDECFACTOR=4096)

      integer MAXINTKERLGH      ! maximum interpolation kernel length
      parameter (MAXINTKERLGH=256)

      integer MAXINTLGH         ! maximum interpolation kernel array size
      parameter (MAXINTLGH=MAXINTKERLGH*MAXDECFACTOR)

c     INPUT VARIABLES:

      integer i_decfactor,i_weight
      real*8 r_beta,r_relfiltlen,r_pedestal

c     OUTPUT VARIABLES:

      integer i_intplength      ! Range migration interpolation kernel length
      real*4  r_fdelay          ! Range migration filter delay
      real*4 r_fintp(0:MAXINTLGH) ! interpolation kernel values

c     LOCAL VARIABLES:

      real*8 r_filter(0:MAXINTLGH)
      integer i,j,i_filtercoef

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION STATEMENTS:

c     SAVE STATEMENTS:

c     PROCESSING STEPS:

c     get sinc

      call sinc_coef(r_beta,r_relfiltlen,i_decfactor,r_pedestal,
     +     i_weight,i_intplength,i_filtercoef,r_filter(0))

      r_fdelay = i_intplength/2.d0

      do i = 0 , i_intplength - 1
         do j = 0 , i_decfactor - 1
            r_fintp(i+j*i_intplength) = r_filter(j+i*i_decfactor)
         enddo
      enddo

      end

c****************************************************************

      subroutine sinc_coef(r_beta,r_relfiltlen,i_decfactor,r_pedestal,
     +     i_weight,i_intplength,i_filtercoef,r_filter)

c****************************************************************
c**
c**   FILE NAME: sinc_coef.f
c**
c**   DATE WRITTEN: 10/15/97
c**
c**   PROGRAMMER: Scott Hensley
c**
c**   FUNCTIONAL DESCRIPTION: The number of data values in the array
c**   will always be the interpolation length * the decimation factor,
c**   so this is not returned separately by the function.
c**
c**   ROUTINES CALLED:
c**
c**   NOTES:
c**
c**   UPDATE LOG:
c**
c**   Date Changed        Reason Changed                  CR # and Version #
c**   ------------       ----------------                 -----------------
c**
c*****************************************************************

      implicit none

c     INPUT VARIABLES:

      real*8 r_beta             !the "beta" for the filter
      real*8 r_relfiltlen       !relative filter length
      integer i_decfactor       !the decimation factor
      real*8 r_pedestal         !pedestal height
      integer i_weight          !0 = no weight , 1=weight

c     OUTPUT VARIABLES:

      integer i_intplength      !the interpolation length
      integer i_filtercoef      !number of coefficients
      real*8 r_filter(*)        !an array of data values

c     LOCAL VARIABLES:

      real*8 r_alpha,pi,r_wgt,r_s,r_fct,r_wgthgt,r_soff,r_wa
      integer i_psfl,i,ii

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

C     FUNCTION STATEMENTS:

c     PROCESSING STEPS:

      pi = 4.d0*atan(1.d0)

c     number of coefficients

      i_intplength = nint(r_relfiltlen/r_beta)
      i_filtercoef = i_intplength*i_decfactor
      r_wgthgt = (1.d0 - r_pedestal)/2.d0
      r_soff = (i_filtercoef - 1.d0)/2.d0

      do i=0,i_filtercoef-1
         r_wa = i - r_soff
         r_wgt = (1.d0 - r_wgthgt) + r_wgthgt*cos((pi*r_wa)/r_soff)
         r_s = r_wa*r_beta/dble(i_decfactor)
         if(r_s .ne. 0.0)then
            r_fct = sin(pi*r_s)/(pi*r_s)
         else
            r_fct = 1.0
         endif
         if(i_weight .eq. 1)then
            r_filter(i+1) = r_fct*r_wgt
         else
            r_filter(i+1) = r_fct
         endif
      enddo

      end

c****************************************************************

      subroutine write_template(i_unit)

c****************************************************************
c**
c**   FILE NAME: ampcor.f
c**
c**   DATE WRITTEN: 8/15/02
c**
c**   PROGRAMMER: Scott Hensley
c**
c**   FUNCTIONAL DESCRIPTION: Write a template file for user.
c**
c**   ROUTINES CALLED:
c**
c**   NOTES:
c**
c**   UPDATE LOG:
c**
c**   Date Changed        Reason Changed                  CR # and Version #
c**   ------------       ----------------                 -----------------
c**
c*****************************************************************

      implicit none

c     INCLUDE FILES:

c     PARAMETER STATEMENTS:

c     INPUT VARIABLES:

      integer i_unit

c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION STATEMENTS:

c     SAVE STATEMENTS:

c     PROCESSING STEPS:

      write(i_unit,'(a)') '                             AMPCOR RDF INPUT FILE'
      write(i_unit,*) ' '
      write(i_unit,'(a)') 'Data Type for Reference Image Real or Complex                   (-)    =  Complex   ![Complex , '//
     +     'Real , RMG1 , RMG2]'
      write(i_unit,'(a)') 'Data Type for Search Image Real or Complex                      (-)    =  Complex   ![Complex , '//
     +     'Real , RMG1 , RMG2]'
      write(i_unit,*) ' '
      write(i_unit,'(a)') '                                                                          !If file is a line '//
     +     'interleaved (i.e. RMG)'
      write(i_unit,'(a)') '                                                                          !file then RMG1 one '//
     +     'uses the first data'
      write(i_unit,'(a)') '                                                                          !layer and RMG2 uses '//
     +     'the secoond data layer'
      write(i_unit,*) ' '
      write(i_unit,'(a)') 'INPUT/OUTPUT FILES'
      write(i_unit,*) ' '
      write(i_unit,'(a)') 'Reference Image Input File                                      (-)    =  file1'
      write(i_unit,'(a)') 'Search Image Input File                                         (-)    =  file2'
      write(i_unit,'(a)') 'Match Output File                                               (-)    =  outfile'
      write(i_unit,*) ' '
      write(i_unit,'(a)') 'MATCH REGION'
      write(i_unit,*) ' '
      write(i_unit,'(a)') 'Number of Samples in Reference/Search Images                    (-)    =  width_ref width_srch'//
     +     '   !Must be less than 18000'
      write(i_unit,'(a)') 'Start, End and Skip Lines in Reference Image                    (-)    =  firstline lastline skip_y'
      write(i_unit,'(a)') 'Start, End and Skip Samples in Reference Image                  (-)    =  firstpix width skip_x'
      write(i_unit,*) ' '
      write(i_unit,'(a)') '                                                                          !Provides location of '//
     +     'match windows in'
      write(i_unit,'(a)') '                                                                          !imagery. Note it is '//
     +     'possible to match with'
      write(i_unit,'(a)') '                                                                          !skip setting less '//
     +     'than the window size, of'
      write(i_unit,'(a)') '                                                                          !course the matches '//
     +     'will NOT be independent.'
      write(i_unit,*) ' '
      write(i_unit,'(a)') 'MATCH PARAMETERS'
      write(i_unit,*) ' '
      write(i_unit,'(a)') 'Reference Window Size Samples/Lines                             (-)    =  window_size_x window_size_y'
      write(i_unit,'(a)') 'Search Pixels Samples/Lines                                     (-)    =  search_x search_y'
      write(i_unit,*) ' '
      write(i_unit,'(a)') '                                                                          !window size plus '//
     +     '2*(search window size)'
      write(i_unit,'(a)') '                                                                          !must be less than '//
     +     '512. Note to get best'
      write(i_unit,'(a)') '                                                                          !oversampling of the '//
     +     'correlation surface should'
      write(i_unit,'(a)') '                                                                          !set the search '//
     +     'window to 5 or greater, otherwise'
      write(i_unit,'(a)') '                                                                          !sinc interpolator '//
     +     'does not have enough support.'
      write(i_unit,*) ' '
      write(i_unit,'(a)') 'Pixel Averaging Samples/Lines                                   (-)    =  pix_ave_x pix_ave_y'
      write(i_unit,*) ' '
      write(i_unit,'(a)') '                                                                          !If you expect '//
     +     'subpixel matching accuracy'
      write(i_unit,'(a)') '                                                                          !then this '//
     +     'SHOULD BE SET TO ONE!'
      write(i_unit,*) ' '
      write(i_unit,'(a)') 'Covariance Surface Oversample Factor and Window Size            (-)    =  oversample_fact window_size'
      write(i_unit,*) ' '
      write(i_unit,'(a)') '                                                                          !oversample factor '//
     +     'determine how much'
      write(i_unit,'(a)') '                                                                          !oversampling via '//
     +     'sinc interpolation is done'
      write(i_unit,'(a)') '                                                                          !for the covarinance '//
     +     'surface. Two times this'
      write(i_unit,'(a)') '                                                                          !number is the '//
     +     'quantization level of the matches,'
      write(i_unit,'(a)') '                                                                          !e.g. if '//
     +     'oversample = 64 the 128 of a pixel'
      write(i_unit,'(a)') '                                                                          !quantization '//
     +     'error. Window size is how many pixels'
      write(i_unit,'(a)') '                                                                          !in the '//
     +     'CORRELATION SURFACE to oversample. Best'
      write(i_unit,'(a)') '                                                                          !results '//
     +     'should have number > 8.'
      write(i_unit,*) ' '
      write(i_unit,'(a)') 'Mean Offset Between Reference and Search Images Samples/Lines   (-)    =  iX0 iY0'
      write(i_unit,*) ' '
      write(i_unit,'(a)') '                                                                          !Convention used '//
     +     'that position in ref image plus'
      write(i_unit,'(a)') '                                                                          !offset is equal '//
     +     'to position in image 2.'
      write(i_unit,*) ' '
      write(i_unit,'(a)') 'MATCH THRESHOLDS AND DEBUG DATA'
      write(i_unit,*) ' '
      write(i_unit,'(a)') 'SNR and Covariance Thresholds                                   (-)    =  snr_thresh cov_thresh'
      write(i_unit,*) ' '
      write(i_unit,'(a)') '                                                                          !Eliminates matches '//
     +     'based on SNR threshold (SNR must be'
      write(i_unit,'(a)') '                                                                          !greater than '//
     +     'this threshold) and Covariance threshold'
      write(i_unit,'(a)') '                                                                          !(cross track '//
     +     'and along track SQRT(COV) must be LESS THAN'
      write(i_unit,'(a)') '                                                                          !than this '//
     +     'threshold in PIXELS. Typical values depend'
      write(i_unit,'(a)') '                                                                          !on type of '//
     +     'imagery being matched.'
      write(i_unit,*) ' '
      write(i_unit,'(a)') 'Debug and Display Flags T/F                                     (-)    =  f t'

      close(i_unit)

      end

cc-------------------------------------------

      real*4 function seconds(t0)
      real*4 t0
      real*8 secondo

      seconds = secondo(-1) - t0

      return
      end
