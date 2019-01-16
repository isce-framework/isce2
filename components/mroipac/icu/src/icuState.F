!c    type DEFINTIONS:

      module icuState


      !c size.inc
      integer NUMSUBS           ! number of subroutines to debug
      parameter (NUMSUBS=100)

!c     PARAMETERS FOR UNWRAPPING

      integer NFFT,STEP,PSD_WIN,NR_RLGT,NI_AZ,WIN_MAX
      parameter (NFFT=32, STEP=NFFT/4, PSD_WIN=3)

      integer MEM_TREES_FACTOR, MEM_RES_FACTOR
      parameter (MEM_TREES_FACTOR=6, MEM_RES_FACTOR=3)

      parameter (WIN_MAX=13)

      integer*4 MAXNEUTYPES                   !subroutine gen_neutrons max number of types of neutrons
      integer*4 MAXTHRES                      !max number of thresholds for the neutrons
      integer*4 MAXCCOMP
      parameter (MAXNEUTYPES=2, MAXTHRES=2, MAXCCOMP=32)

      integer*4 MAX_GRASS
      parameter(MAX_GRASS=65536)                !max length of ping-pong lists for growing perimeter

      !c structures.inc
!c     INTERFEROGRAM FORMATION PARAMETERS

      type  inf 
         sequence
         integer*4   i_filtopt         ! filtering flag (0=no filter, 1=filtering)
         integer*4   i_filttype        ! interferogram filter flag (0=LP, 1=PS)
         integer*4   i_useamp          ! flag to determine if amplitude used (0=no, 1=yes)
         integer*4   i_cc_std          ! standard correlation flag (0=no, 1=yes)
         integer*4   i_cc_norm         ! slope normalized correlation flag (0=no, 1=yes)
         integer*4   i_cc_sigma        ! phase standard deviation derived corr. flag (0=no, 1=yes)
         integer*4   i_cc_layer        ! correlation used to generate mask: 1=std, 2=norm, 3=sigma
         integer*4   i_slope           ! phase gradient flag (0=no, 1=yes)
         integer*4   i_cc_winsz        ! size of correlation estimation window
         integer*4   i_phs_winsz       ! size of phase standard deviation estimation window
         real*4      r_ps_alpha        ! power spectrum filtering exponent alpha
         real*4      r_lpwinrng        ! range window size for low pass filter
         real*4      r_lpwinaz         ! azimuth window size for low pass filter
         integer*4   i_azbufsize       ! number of lines in the interferogram patch
         integer*4   i_sline           ! first valid interferogram line in the patch (0 based index)
         integer*4   i_eline           ! last valid interferogram line in the patch (0 based index)
         integer*4   i_rsamps          ! number of samplesline of the interferogram patch
         integer*4   i_ssamp           ! first valid sample of interferogram (0 based index)
         integer*4   i_esamp           ! last valid sample of interferogram (0 based index)
      end type inf

      type  unw  
         sequence
         integer*4   i_neutypes(MAXNEUTYPES)           ! number of types of neutrons
         real*4      r_neuthres(MAXNEUTYPES, MAXTHRES) !array for neutron thresholds
         integer*4   i_tree_sets       ! number of realizations of the trees 
         real*4      r_ccthr_min       ! minimum correlation threshold for unwrapping
         real*4      r_ccthr_max       ! maximum correlation threshold for unwrapping
         real*4      r_ccthr_incr      ! increment of correlation threshold
         real*4      r_phvarmax        ! maximum phase variance to accept bootstrap (radians)
         integer*4   i_ovloff          ! offset of phase bootstap data line
         integer*4   i_minbootpts      ! minimum number of points to attempt bootstrap
         integer*4   i_minbootlns      ! minimum number of lines to attempt bootstrap
         integer*4   i_ambadjust       ! ambiguity adjustment to unwrapped phase
         integer*4   i_tree_type       ! tree type (0=GZW, 1=CC)
         integer*4   i_spixel          ! range pixel number of first good unwrapped point (0 based index)
         integer*4   i_epixel          ! range pixel number of last good unwrapped point  (0 based index)
         integer*4   i_sunw            ! first good line of unwrap buffer (0 based index)
         integer*4   i_eunw            ! last good line of unwrap buffer  (0 based index)
      end type unw

      !constants.inc

      character*255  VERSIONID
      parameter (VERSIONID='V2.0   Jun 25, 2013')

      integer FORWARD_FFT,INVERSE_FFT     !FFT Conventions
      parameter (FORWARD_FFT = -1)
      parameter (INVERSE_FFT = 1)

      integer i_rmcoef,i_mccoef,i_pscoef    !flags for filter selection
      parameter(i_rmcoef=1,i_mccoef=2,i_pscoef=3)


!c       PARAMETERS FOR UNWRAPPING

      integer*1 PLUS, MINUS, CHARGE, CUT, VISIT, LAWN,
     &          TREE, TWIG, NEUTRON, LCORR                                       !phase unwrapping flag values
      parameter(PLUS=1, MINUS=2, CHARGE=3, CUT=4,  VISIT=8,  LAWN=16,
     &          TREE=16,  TWIG=32, NEUTRON=64, LCORR=-128) 
      
      integer*4  NBL    !number of lines in the bootstrap phase and amplitude arrays
      parameter(NBL=3)

      integer*4 TREE_GZW, TREE_CC               !tree types
      parameter(TREE_GZW=0, TREE_CC=1)

      integer*4 MBL
      parameter(MBL=64)                 !maximum branch length
      
      integer*4 BOOT, NOBOOT
      parameter (BOOT=1, NOBOOT=0)

!c       BUFFER CONSTANTS

!c     for ampbuf buffer

      integer i_mag,i_phs,i_cor,i_lks
      parameter(i_mag=1,i_phs=2,i_cor=3,i_lks=4)        
      
!c       for MAPBUF layer values
      
      integer i_amp,i_hgt,i_herr,i_sslp,i_cslp,i_acur,i_scur,i_lutr,
     &        i_luts,i_ptn
      parameter(i_amp=1,i_hgt=2,i_herr=4,i_cslp=5,i_sslp=6,i_acur=7,
     &          i_scur=8)
      parameter(i_lutr=9,i_luts=10,i_ptn=0)
      
      
!c       MATHEMATICAL CONSTANTS    
      
      real*8 pi         !if you need an explanation, give up
      parameter(pi=3.1415926539d0)
      
      real*4 PI_SP, TWO_PI_SP
      parameter (PI_SP=3.1415926536, TWO_PI_SP=6.2831853072)

      real*8 TWO_PI             !if you need an explanation, give up
      parameter(TWO_PI=2.d0*pi)
      
      real*8 r_dtor,dtor        !multiplicative constant to convert from degrees to radians
      parameter(r_dtor=pi/180.d0,dtor=pi/180.d0)
      
      real*8 r_rtod,rtod        !multiplicative constant to convert from radians tp degrees
      parameter(r_rtod=180.d0/pi,rtod=180.d0/pi)
      
      real*4 RTD, DTR
      parameter(RTD = 57.2957795131, DTR = .0174532925199)

      !All the structure instances
      type (unw) unwp      !unwrapping parameters 
      type (inf) infp        !interferogram filtering and correlation

      integer i_strtline,i_numlines
      integer i_azbuf,i_azcomlin
      integer i_unwrap_flag


      end module icuState
