        subroutine mdxsub(a_cmd,i_maxbuff,readfunc)

          implicit none

          integer I_WKSPACE
          integer I_MAXCOLS
          integer I_MAXROWS
          integer I_MAXSAMP
          parameter(I_WKSPACE =  100000)
          parameter(I_MAXCOLS =  100000)
          parameter(I_MAXROWS =  100000)
          parameter(I_MAXSAMP =  100000)

          integer I_BMAX
          parameter(I_BMAX=200)     ! Maximum number of buffered commands

          integer I_EMAX
          parameter(I_EMAX=200)     ! Maximum number of buffered expose commands

          integer I_FMAX            ! Maximum number of data files
          parameter(I_FMAX= 6)

          integer I_CMAX            ! Maximum number of data channels
          parameter(I_CMAX=10)

          integer I_DMAX            ! Maximum number of displays
          parameter(I_DMAX=10)

          integer I_KMAX            ! Maximum number of color tables in pulldown menu
          parameter(I_KMAX=20)


c     INPUT VARIABLES:

          integer i_inarg
          character*255 a_inarg(255)


c     Some Useful Local Variables

          character*255 a_value
          character*255 a_title
          character*255 a_sss(I_CMAX)
          character*200 a_nullstr
          character*255 a_label
          character*255 a_command
          character*255 a_fmt
          character*(*) a_cmd

          character*255 a_workdir
          character*255 a_colordir

          character*20  a_colorname(I_KMAX)
          character*255 a_colorfile(I_KMAX)
          integer i_colormax
          integer i_colorset

          integer i
          integer j
          integer ix
          integer iy
          integer ib
          integer ie
          integer i_r
          integer i_c
          integer i_d
          integer ir
          integer ic

          integer i_arg
          integer i_row
          integer i_col
          integer i_typ

          integer i_tmp
          integer i_dat  ! Data file counter
          integer i_set  ! Set Counter
          integer i_chn
          integer i_sss
          integer i_pid
          integer i_opr
          integer i_pfmt
          integer i_pset
          integer i_sset
          integer i_tset

          integer i_loop
          integer i_flip
          integer i_stat
          integer i_dflag
          integer i_value
          integer i_field

          integer i_default

          byte b_buff(4)
          integer i_buff
          equivalence(b_buff,i_buff)

          integer i_endian

          integer i_cnt
          integer i_err
          integer i_flg
          integer i_pos
          integer i_max
          integer i_roff
          integer i_log
          integer i_dec

          real*4 r_data(0:I_MAXCOLS)
          real*4 r_data2(0:I_MAXCOLS)
          integer*4 i_data(0:I_MAXCOLS)
          integer*4 i_data2(0:I_MAXCOLS)
          real*8 r_sqr
          real*8 r_sum
          real*8 r_avg
          real*8 r_std
          real*4 r_zmstrt
          real*4 r_expn
          real*4 r_setmin
          real*4 r_setmax

          real*4 r_dnx(3)
          real*4 r_eux(3)
          real*4 r_loc(3)

          real*4 r_pi
          real*4 r_rtod

          real*4  r_a
          real*4  r_e2

          integer i_smode
          integer i_samps
          integer i_rsamps(I_MAXSAMP)
          integer i_csamps(I_MAXSAMP)
          integer i_tsamps(I_MAXSAMP)
          real*4  r_wsamps(I_MAXSAMP)
          real*4  r_ssamps(I_MAXSAMP)
          real*4  r_vsamps(I_MAXSAMP,I_CMAX)
          real*4  r_row(I_MAXSAMP)
          real*4  r_col(I_MAXSAMP)
          real*4  r_rowlow
          real*4  r_rowhigh
          real*4  r_collow
          real*4  r_colhigh
          real*4  r_path

          real*4 r_wdth
          real*4 r_spce
          real*4 r_dist

          integer ii
          integer jj
          integer iii
          integer i_cc
          integer i_rr
          integer i_clast
          integer i_rlast


c     structure / set_structure / s_set(-I_FMAX:I_CMAX)
          character*200 a_setname(-I_FMAX:I_CMAX) ! Parameter name
          character*200 a_setfile(-I_FMAX:I_CMAX) ! Data filename
          character*200 a_setinfo(-I_FMAX:I_CMAX) ! Header filename
          character*200 a_setproj(-I_FMAX:I_CMAX) ! Projection name
          character*16  a_setvnul(-I_FMAX:I_CMAX) ! Hex string of null value
          integer i_setunit(-I_FMAX:I_CMAX) ! Unit number to read set
          integer i_setrows(-I_FMAX:I_CMAX) ! Number of rows in set
          integer i_setcols(-I_FMAX:I_CMAX) ! Number of columns in set
          integer i_setshdr(-I_FMAX:I_CMAX) ! Number of bytes in set header
          integer i_setstlr(-I_FMAX:I_CMAX) ! Number of bytes in set trailer
          integer i_setrhdr(-I_FMAX:I_CMAX) ! Number of bytes in row header
          integer i_setrtlr(-I_FMAX:I_CMAX) ! Number of bytes in row trailer
          integer i_setchdr(-I_FMAX:I_CMAX) ! Number of bytes in column header
          integer i_setctlr(-I_FMAX:I_CMAX) ! Number of bytes in column trailer
          integer i_setvend(-I_FMAX:I_CMAX) ! Endian flag
          integer i_setvfmt(-I_FMAX:I_CMAX) ! Method to decode columns
          real*4  r_setrmlt(-I_FMAX:I_CMAX) ! Row Scale  for set
          real*4  r_setradr(-I_FMAX:I_CMAX) ! Row Offset for set
          real*4  r_setcmlt(-I_FMAX:I_CMAX) ! Column Scale  for set
          real*4  r_setcadr(-I_FMAX:I_CMAX) ! Column Offset for set
          real*4  r_setvmlt(-I_FMAX:I_CMAX) ! Value Scale  for set
          real*4  r_setvadr(-I_FMAX:I_CMAX) ! Value Offset for set
          real*4  r_setvmin(-I_FMAX:I_CMAX) ! Minimum valid value
          real*4  r_setvmax(-I_FMAX:I_CMAX) ! Maximum valid value
          real*4  r_setvavg(-I_FMAX:I_CMAX) ! Average value in set
          real*4  r_setvstd(-I_FMAX:I_CMAX) ! Standard deviation of values in set
          real*4  r_setpegv(3,-I_FMAX:I_CMAX) ! Set Peg
          byte    b_setvnul(0:16,-I_FMAX:I_CMAX) ! Invalid value
c     end structure

          integer i_dsp
          integer i_dspselect
c     structure / dspinfo / s_dsp
          character*200 a_dspctbl(-I_FMAX:I_CMAX) ! Color table file
          integer i_dspcnt
          integer i_dspchnl         ! Number of sets to display
          integer i_dspaddr(-I_FMAX:I_CMAX) ! Add  auto Scale flag
          integer i_dspmult(-I_FMAX:I_CMAX) ! Mult auto Scale flag
          integer i_dspmixv(-I_FMAX:I_CMAX) ! Method to mix set (add, multiply, max, avg)
          integer i_dspnumt(-I_FMAX:I_CMAX) ! Number of entries in color table
          integer i_dspmode(-I_FMAX:I_CMAX)
          integer i_dspdvdc(-I_FMAX:I_CMAX)
          integer i_dspactv(0:I_DMAX,-I_CMAX:I_CMAX)
          real*4  r_dspredt(0:255,-I_FMAX:I_CMAX) ! Values of red color table
          real*4  r_dspgrnt(0:255,-I_FMAX:I_CMAX) ! Values of green color table
          real*4  r_dspblut(0:255,-I_FMAX:I_CMAX) ! Values of blue color table
          real*4  r_dspcplw(-I_FMAX:I_CMAX) ! Discard if below value
          real*4  r_dspcphi(-I_FMAX:I_CMAX) ! Discard if above value
          real*4  r_dspexpn(-I_FMAX:I_CMAX) ! Exponent to raise data
          real*4  r_dspaddr(-I_FMAX:I_CMAX) ! Shift data by value
          real*4  r_dspwrap(-I_FMAX:I_CMAX) ! Wrap data by value
          real*4  r_dspmult(-I_FMAX:I_CMAX) ! Multiply data by value
          real*4  r_dspvmin(-I_FMAX:I_CMAX) ! Min value to display
          real*4  r_dspvmax(-I_FMAX:I_CMAX) ! Max value to display
          real*4  r_dspval1(-I_FMAX:I_CMAX)
          real*4  r_dspval2(-I_FMAX:I_CMAX)
          real*4  r_dspval3(-I_FMAX:I_CMAX)
c     end structure

c     structure / dspinfo / s_win
          character*200 a_dsptitle(0:I_DMAX) ! Window title
          integer i_winactv(0:I_DMAX)
          integer i_winrows(0:I_DMAX) ! rows offset
          integer i_wincols(0:I_DMAX) ! sample offset
          integer i_wincadr(0:I_DMAX) ! column offset to start of window
          integer i_winradr(0:I_DMAX) ! row offset to start of window
          integer i_winselc(0:I_DMAX) ! Set active flag
          real*4  r_winzoom(0:I_DMAX) ! Zoom factor
c     end structure

          integer i_winx                ! initial window size on screen
          integer i_winy                ! initial window size on screen

          integer i_wxs(6,-10:10)   ! window x size
          integer i_wys(6,-10:10)   ! window y size
          integer i_vxs(6,-10:10)   ! viewport x size
          integer i_vys(6,-10:10)   ! viewport y size
          integer i_vxo(6,-10:10)   ! viewport x offset
          integer i_vyo(6,-10:10)   ! viewport y offset

          integer i_int
          integer i_bpl
          integer i_ncx
          integer i_nrx
          integer i_enrx2
          integer i_ponly
          integer i_indx(0:I_WKSPACE)
          real*4  r_rdat(0:I_WKSPACE)
          real*4  r_gdat(0:I_WKSPACE)
          real*4  r_bdat(0:I_WKSPACE)
c          save r_rdat, r_gdat,r_bdat

          character*255 a_file
          character*120 a_filename
          character*255 a_ptsfile
          character*120 a_label1
          character*120 a_label2
          character*160  a_labels(0:20)
          character*160  a_data(0:20)
          character*160  a_elabl(0:20)
          character*160  a_edata(0:20)
          character*120 a_nullclr
          character*120 a_lcolor
          integer*4     i_nullclr(3)
          integer*4     stat,i_stat32(13),i_err32

          integer*4 i_msgid
          character*160 a_message

          character*200 a_out
          byte b_out(3*I_WKSPACE)
          equivalence(a_out,b_out)

          real*4 r_value
          real*4 r_val(I_WKSPACE)

          integer i_w
          integer i_win
          integer i_evn

          integer i_val
          integer i_key
          integer i_asc

          integer i_act

          integer i_debug

          integer i_done
          integer i_wait
          integer i_cntl
          integer i_shft
          integer i_abort
          integer i_pinit
          integer i_scroll
          integer i_eventmod       ! number of lines read between X window event calls
          integer i_rcenter
          integer i_ccenter
          integer i_cpos
          integer i_rpos
          integer i_cdsp
          integer i_cset

          integer i_qubeset

          integer i_show
          integer i_region
          integer i_start
          integer i_newpoint

          integer i_event(0:10)
          integer i_button

          integer i_ecnt
          integer i_edat(0:10,I_EMAX) ! Expose Buffer data
          integer i_ecmd(0:10)

          integer i_bcnt
          integer i_bdat(0:10,I_BMAX) ! Action Buffer data
          integer i_brow           ! Number of lines in action
          integer i_blks            ! Number of blocks needed to complete action
          integer i_strt
          integer i_stop
          integer i_incr
          integer i_coff

          integer i_close

          integer i_redraw(I_DMAX)
          integer i_cw
          integer i_ch
          integer i_widget
          integer i_menu

          integer i_edsp
          integer i_ewin
          integer i_eevn
          integer i_ecol
          integer i_erow
          integer i_encx
          integer i_enrx

          integer i_lat
          integer i_lon
          integer i_str

          integer i_pcpad
          integer i_prpad

          character*10 a_rowfrmt
          character*10 a_colfrmt

          character*120 a_hdrfile
          integer i_lsize
          integer i_ssize
          real*8 r_peg(3)
          real*8 r_lat
          real*8 r_lon
          character*120 a_type
          real*8 r_str(2)
          real*8 r_spc(2)
          integer i_mbytes
          integer i_dbytes
          real r_mmul
          real r_madd
          real r_dmul
          real r_dadd

          real r_median
          real r_space

          integer i_ewupdate
          integer i_rstat
          byte b_data(0:3)

          character*20  a_tname(5)
          character*1   a_twait(5)
          character*120 a_tcmnd(5)

          character*120 a_clickcmd(6)


c     FUNCTIONS

          integer rdflen
          external rdflen

          character*40 rdflower
          external rdflower

          integer rdfnum
          external rdfnum

          integer initdk
          external initdk

          real*8 rdir
          external rdir

          real*4  wrap            ! Height wrap variables
          external wrap

          integer i_CnvrtFmt
          external i_CnvrtFmt

          integer i_setvbyt
          external i_setvbyt

          character*18 version_mdx
          external version_mdx

          integer version_gx
          external version_gx

#ifdef IO64
          integer*8 i_fbytes
          integer*8 i_maxbuff

          integer*8 readfunc
          external readfunc

          integer*8 i_getfsize
          external i_getfsize

          integer*8 i_eight
          external i_eight

#else
          integer*4 i_fbytes
          integer*4 i_maxbuff

          integer*4 readfunc
          external readfunc

          integer*4 i_getfsize
          external i_getfsize

          integer*4 i_eight
          external i_eight

#endif


c     PROCESSING STEPS:

c
c     Initialize pi and conversions
c
          r_pi = 4.d0*atan(1.0d0)
          r_rtod = 180.0d0/r_pi


c
c     Initialize datum stuff
c
          r_a =  6378137.0
          r_e2 = 0.00669438

c
c     Determine endian ness of machine
c
         b_buff(1) = 0
         b_buff(2) = 0
         b_buff(3) = 0
         b_buff(4) = 1
         if (i_buff .eq. 1) then   ! Big Endian
            i_endian = 1
         else                      ! Little Endian
            i_endian = -1
         end if


c     Note - The order that parameters get set is as follows:
c     1) Internal parameter initialization set at the top of the program
c     2) Parameters read in from the mdx.ini file in the local or home
c     directory
c     3) Parameters on command line prior to any file name specified
c     5) Parameters on command line prior to any specified set
c     6) Parameters read in from the implicit file header for a given
c     set
c     7) Parameters on command line following set specification
c
c     headers can also be explicitly identified on the command line and
c     will
c     be given priority indicated by their location in the command line.


c
c     Initialize set stuff
c

          a_setname(0) = ' '
          a_setfile(0) = ' '
          a_setinfo(0) = ' '
          a_setproj(0) = ' '
          i_setunit(0) = 0
          i_setrows(0) = 1000000
          i_setcols(0) = 0
          i_setshdr(0) = 0
          i_setstlr(0) = 0
          i_setrhdr(0) = 0
          i_setrtlr(0) = 0
          i_setchdr(0) = 0
          i_setctlr(0) = 0
          i_setvend(0) = i_endian
          i_setvfmt(0) = 4          ! REAL*4

          r_setrmlt(0) =  1.0
          r_setradr(0) =  0.0

          r_setcmlt(0) =  1.0
          r_setcadr(0) =  0.0

          r_setvmlt(0) =  1.0d0
          r_setvadr(0) =  0.0d0
          r_setvmin(0) = -1.0d27
          r_setvmax(0) =  1.0d27
          a_setvnul(0) = ' '

          a_filename = ' '

          a_labels(0) = 'Display Parameters'
          a_labels(1) = 'Scale Mode:'
          a_labels(2) = 'SDEV Factor:'
          a_labels(3) = 'Offset:'
          a_labels(4) = 'Exponent:'
          a_labels(5) = 'Min Valid:'
          a_labels(6) = 'Max Valid:'
          a_labels(7) = 'Min Clip:'
          a_labels(8) = 'Max Clip:'
          a_labels(9) = 'Color Table:'
          a_labels(10) = ' '

          a_data(0) = '0'
          a_data(1) = '1'
          a_data(2) = '2'
          a_data(3) = '3'
          a_data(4) = '4'
          a_data(5) = '5'
          a_data(6) = '6'
          a_data(7) = '7'
          a_data(8) = '8'
          a_data(9) = '9'
          a_data(10) = ' '

C
C     rjm: Initialize win column start and rol start
C          This is needed for "-P" print option
C          And, what the heck, initial i_data too.
C
         do i = 0,I_MAXCOLS
           i_data(i) = 0
           i_data2(i) = 0
         end do
         do i = 0,I_DMAX
            i_wincadr(i) = 0
            i_winradr(i) = 0
         end do

c
c     Initialize display stuff
c
          i_done = 0
          i_bcnt = 0
          i_ecnt = 0
          i_wait = 0                ! 0 = wait for event
          i_r = -1

          a_title = ' '
          i_dspchnl = 0             ! Number of channels
          i_winrows(0) = 0          ! Number of rows
          i_wincols(0) = 0          ! Number of columns
          r_winzoom(0) = 1.0        ! Print Zoom factor
          i_winx = 0
          i_winy = 0

          r_dspcplw(0) = -1.e27  ! Clip if below value
          r_dspcphi(0) =  1.e27  ! Clip if above value
          r_dspvmin(0) = -1.e27  ! Discard if below value
          r_dspvmax(0) =  1.e27  ! Discard if above value
          r_dspexpn(0) =  1.     ! raise data to pwr
          r_dspaddr(0) =  0.     ! Shift data by value
          r_dspwrap(0) =  0.     ! Wrap data by value
          r_dspmult(0) =  0.     ! Multiply data by value
          r_dspval1(0) =  2.0    ! Value used in computing auto scale
          r_dspval2(0) = 90.0    ! Value used in computing auto scale
          r_dspval3(0) =  1.0    ! Value used in computing auto scale
          i_dspaddr(0) =  1.     ! Flag to enable auto scale
          i_dspmult(0) =  1.     ! Flag to enable auto scale
          i_dspmixv(0) =  2      ! Method to mix sets (add, multiply, max, avg)
          i_dspmode(0) =  3      ! Autoscale to 90%
          i_dspdvdc(0) =  0
          a_dspctbl(0) = ' '     ! Default color table

          r_winzoom(1) = 1.0        ! Screen Zoom default

          i_menu = 1
          i_close = 1

          do i=1, I_DMAX
            i_dspactv(i,0) = -1
            do j=1,I_CMAX
              i_dspactv(i,-j) = -1
              i_dspactv(i, j) = -1
            end do
            i_redraw(i) = 0
          end do

          do i=0,20
            a_elabl(i) = ' '
            a_edata(i) = ' '
          end do

          a_nullclr='0,0,255'

          a_lcolor='white'

          i_abort=0
          i_debug = 2
          i_eventmod = 10
          i_scroll = 0
          i_pinit = 0
          i_ponly = 0
          i_pfmt = 1
          i_pset = 0
          i_sset = 0
          i_tset = 0
          i_cntl = 0
          i_shft = 0
          i_key = 0

          i_region = 0
          i_act = 0

          i_smode = 1
          r_wdth = 0.
          r_spce = 0.1
          i_samps = 0
          i_show = 0

          i_pcpad = 31
          i_prpad = 50

          i_cdsp = -1
          i_cset = 0

          i_qubeset = 0

          i_ccenter = 0
          i_rcenter = 0
          r_lat = -3*r_pi
          r_lon = -3*r_pi

          i_r = -2
          i_dspselect = 0

          i_ewupdate = 0

          do i=1,5
            a_tname(i)=' '
            a_twait(i)=' '
            a_tcmnd(i)=' '
          end do
          a_tname(1) = 'Plot Location'
          a_tname(2) = 'Plot Profile'

          a_workdir = './'
          a_colordir = './'

          a_ptsfile = ' '

          a_colorname(1) = 'Other'
          a_colorfile(1) = '?'
          a_colorname(2) = 'White'
          a_colorfile(2) = 'white'
          a_colorname(3) = 'Black'
          a_colorfile(3) = 'black'
          a_colorname(4) = 'Bitmap'
          a_colorfile(4) = 'bitmap'
          a_colorname(5) = 'Grey'
          a_colorfile(5) = 'grey'
          a_colorname(6) = 'Red'
          a_colorfile(6) = 'red'
          a_colorname(7) = 'Green'
          a_colorfile(7) = 'green'
          a_colorname(8) = 'Blue'
          a_colorfile(8) = 'blue'
          a_colorname(9) = 'CMY'
          a_colorfile(9) = 'cmy'
          a_colorname(10) = 'BGW'
          a_colorfile(10) = 'bgw'
          i_colormax=10

          do i=1,6
            a_clickcmd(i) = ' '
          end do

c
c     Read in MDX default file
c
          call get_mdxdefaults(a_tname,a_tcmnd,a_twait,a_nullclr,i_pcpad,i_prpad,r_winzoom,
     &          a_workdir,a_colordir,a_colorname,a_colorfile,i_colormax,i_close,a_clickcmd)

c
c     Read in command line
c
          call rdf_getfields(a_cmd,i_inarg,a_inarg)
          if (i_inarg .eq. 0) then
            return
          else
            i_arg = 0
            i_dat = 0
            i_set = 0
            i_chn = 0
            i_tmp = 0
            do while(i_arg .lt. i_inarg)
              i_arg=i_arg + 1
              a_value = a_inarg(i_arg)
              i_int=1
              do i=1,rdflen(a_value)
                if (index("1234567890",a_value(i:i)) .eq. 0) i_int=0
              end do
c              write(6,*) 'i_arg,a_value=',i_arg,' ',a_value(1:60)
              if (a_value .eq. ' ') then
                                ! error
              else if (a_value .eq. '-V') then
                write(6,*) '  '
                write(6,'(1x,a,a18,a)'     ) '   <<  mdx    Version  ',version_mdx(),        '  >>   '
                write(6,'(1x,a,f5.1,13x,a)') '   <<  graphx Version  ',float(version_gx()),'  >>   '
                write(6,*) '  '

              else if (a_value .eq. '-s' .or. a_value .eq. '-samples' .or.
     &              a_value .eq. '-cols' .or. a_value .eq. '-columns') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) i_setcols(i_tmp)
              else if (i_int .eq. 1) then  ! also number of columns
                read(a_value,*) i_setcols(i_tmp)
              else if (a_value .eq. '-l' .or. a_value .eq. '-lines' .or.
     &              a_value .eq. '-rows') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) i_setrows(i_tmp)
              else if (a_value .eq. '-col' .or. a_value .eq. '-c' .or. a_value .eq. '-cpos') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) i_ccenter
              else if (a_value .eq. '-row' .or. a_value .eq. '-r' .or. a_value .eq. '-rpos') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) i_rcenter
              else if (a_value .eq. '-lat' .or. a_value .eq. '-latitude') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) r_lat
                r_lat = r_lat/r_rtod
              else if (a_value .eq. '-lon' .or. a_value .eq. '-longitude') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) r_lon
                r_lon = r_lon/r_rtod
              else if (a_value .eq. '-shdr' .or. a_value .eq. '-set_hddr') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) i_setshdr(i_tmp)
              else if (a_value .eq. '-rhdr' .or. a_value .eq. '-row_hddr') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) i_setrhdr(i_tmp)
              else if (a_value .eq. '-chdr' .or. a_value .eq. '-col_hddr') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) i_setchdr(i_tmp)
              else if (a_value .eq. '-stlr' .or. a_value .eq. '-set_tail') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) i_setstlr(i_tmp)
              else if (a_value .eq. '-rtlr' .or. a_value .eq. '-row_tail') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) i_setrtlr(i_tmp)
              else if (a_value .eq. '-ctlr' .or. a_value .eq. '-col_tail') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) i_setctlr(i_tmp)
              else if (a_value .eq. '-vfmt' .or. a_value .eq. '-val_frmt') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                if (i_CnvrtFmt(a_value) .gt. 0) then
                  i_setvfmt(i_tmp) = i_CnvrtFmt(a_value)
                else
                  write(6,*) '*** Warning ***  Could not parse value format for set: ',
     &                a_setname(i_tmp)(1:max(1,rdflen(a_setname(i_tmp)))),
     &                '   ',a_value
                end if
              else if (a_value .eq. '-rmlt' .or. a_value .eq. '-row_mult') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) r_setrmlt(i_tmp)
              else if (a_value .eq. '-radr' .or. a_value .eq. '-row_addr') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) r_setradr(i_tmp)
              else if (a_value .eq. '-cmlt' .or. a_value .eq. '-col_mult') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) r_setcmlt(i_tmp)
              else if (a_value .eq. '-cadr' .or. a_value .eq. '-col_addr') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) r_setcadr(i_tmp)
              else if (a_value .eq. '-vmlt' .or. a_value .eq. '-val_mult') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) r_setvmlt(i_tmp)
              else if (a_value .eq. '-vadr' .or. a_value .eq. '-val_addr') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) r_setvadr(i_tmp)
              else if (a_value .eq. '-plat' .or. a_value .eq. '-set_plat') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) r_setpegv(1,i_tmp)
              else if (a_value .eq. '-plon' .or. a_value .eq. '-set_plon') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) r_setpegv(2,i_tmp)
              else if (a_value .eq. '-phdg' .or. a_value .eq. '-set_phdg') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) r_setpegv(3,i_tmp)
              else if (a_value .eq. '-proj' .or. a_value .eq. '-set_proj') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) a_setproj(i_tmp)
              else if (a_value .eq. '-min' .or. a_value .eq. '-vmin' .or. a_value .eq. '-minval' .or. a_value .eq. '-val_minv')
     &              then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) r_setvmin(i_tmp)
              else if (a_value .eq. '-max' .or. a_value .eq. '-vmax' .or. a_value .eq. '-maxval' .or. a_value .eq. '-val_maxv')
     &              then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) r_setvmax(i_tmp)
              else if (a_value .eq. '-e' .or. a_value .eq. '-exp')
     &              then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) r_dspexpn(i_tmp)
              else if (a_value .eq. '-clpmin' .or. a_value .eq. '-minclp')
     &              then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) r_dspcplw(i_tmp)
              else if (a_value .eq. '-clpmax' .or. a_value .eq. '-maxclp')
     &              then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) r_dspcphi(i_tmp)
              else if (a_value .eq. '-val_endi') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                if (a_value .eq. 'little_endian' .or. a_value .eq. 'LITTLE_ENDIAN') then
                  i_setvend(i_tmp) = -1
                else if (a_value .eq. 'big_endian' .or. a_value .eq. 'BIG_ENDIAN') then
                  i_setvend(i_tmp) = 1
                else
                  write(6,*) '*** Warning *** Could not parse endian-ness',a_value
                end if
              else if (a_value .eq. '-bs' .or. a_value .eq. '-B' .or. a_value .eq. '-bswap') then
                i_setvend(i_tmp) = -i_setvend(i_tmp)
              else if (a_value .eq. '-LE' .or. a_value .eq. '-le' .or. a_value .eq. '-little') then
                i_setvend(i_tmp) = -1
              else if (a_value .eq. '-BE' .or. a_value .eq. '-be' .or. a_value .eq. '-big') then
                i_setvend(i_tmp) = 1
              else if (a_value .eq. '-D' .or. a_value .eq. '-dc' .or. a_value .eq. '-dvdc' .or.
     &                 a_value .eq. '-dx' .or. a_value .eq. '-dvdx' .or. a_value .eq. '-slope') then
                i_dspdvdc(i_tmp)=1
              else if (a_value .eq. '-d' .or. a_value .eq. '-wrap') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) r_dspwrap(i_tmp)
                if (r_dspwrap(i_tmp) .ne. 0.0) then
                  r_dspmult(i_tmp) = r_dspwrap(i_tmp)
                  i_dspmult(i_tmp) = 0
                  i_dspaddr(i_tmp) = 0
                  i_dspmode(i_tmp) = 6
                end if
              else if (a_value .eq. '-a' .or. a_value .eq. '-addr' .or.
     &              a_value .eq. '-add' .or. a_value .eq. '-daddr') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) r_dspaddr(i_tmp)
                i_dspaddr(i_tmp) = 0
              else if (a_value .eq. '-m' .or. a_value .eq. '-mult' .or. a_value .eq. '-dmult') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) r_dspmult(i_tmp)
                i_dspmult(i_tmp) = 0
                i_dspmode(i_tmp) = 1
              else if (a_value .eq. '-f' .or. a_value .eq. '-fact' .or. a_value .eq. '-sdev') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) r_dspval1(i_tmp)
                i_dspmult(i_tmp) = 1
                i_dspmode(i_tmp) = 2
              else if (a_value .eq. '-p' .or. a_value .eq. '-percent' .or. a_value .eq. '-%') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) r_dspval2(i_tmp)
                i_dspmult(i_tmp) = 1
                i_dspmode(i_tmp) = 3
              else if (a_value .eq. '-cw' .or. a_value .eq. '-cws' .or. a_value .eq. '-charlie') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) r_dspval3(i_tmp)
                i_dspmult(i_tmp) = 1
                i_dspmode(i_tmp) = 5
              else if (a_value .eq. '-SDEV' ) then
                r_dspval1(i_tmp) = 2
                i_dspmult(i_tmp) = 1
                i_dspaddr(i_tmp) = 1
                i_dspmode(i_tmp) = 2
              else if (a_value .eq. '-PER' ) then
                r_dspval2(i_tmp) = 90
                i_dspmult(i_tmp) = 1
                i_dspmode(i_tmp) = 3
              else if (a_value .eq. '-CW' ) then
                r_dspval3(i_tmp) = 1
                i_dspmult(i_tmp) = 1
                i_dspaddr(i_tmp) = 0
                i_dspmode(i_tmp) = 5
                r_dspaddr(i_tmp) = 0.
              else if (a_value .eq. '-WRAP' ) then
                i_dspmode(i_tmp) = 6
                r_dspwrap(i_tmp) = r_pi
                i_dspaddr(i_tmp) = 0
                r_dspaddr(i_tmp) = 0.
              else if (a_value .eq. '-z' .or. a_value .eq. '-zoom') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) r_winzoom(1)
                if (r_winzoom(1) .lt. 0.) r_winzoom(1) = abs(1./r_winzoom(1))
              else if (a_value .eq. '-pz' .or. a_value .eq. '-pzoom') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) r_winzoom(0)
                if (r_winzoom(0) .lt. 0.) r_winzoom(0) = abs(1./r_winzoom(0))
              else if (a_value .eq. '-vx' .or. a_value .eq. '-vxsize') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) i_winx
              else if (a_value .eq. '-vy' .or. a_value .eq. '-vysize') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) i_winy
              else if (a_value .eq. '-pcpad' .or. a_value .eq. '-pc') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) i_pcpad
              else if (a_value .eq. '-prpad' .or. a_value .eq. '-pr') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) i_prpad
              else if (a_value .eq. '-mix') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                if (a_value .eq. '+') then
                  i_dspmixv(i_tmp) = 1
                else if (a_value .eq. 'x' .or. a_value .eq. 'X') then
                  i_dspmixv(i_tmp) = 2
                else
                  read(a_value,*) i_dspmixv(i_tmp)
                end if
              else if (a_value .eq. '-cmap' .or. a_value .eq. '-ctable') then
                i_arg=i_arg+1
                a_dspctbl(i_tmp)=a_inarg(i_arg)
                do i=1,i_colormax
                  if (a_dspctbl(i_tmp) .eq. a_colorname(i)) a_dspctbl(i_tmp)=a_colorfile(i)
                end do
              else if (a_value .eq. '-null') then
                i_arg=i_arg+1
                a_setvnul(i_tmp)=a_inarg(i_arg)
              else if (a_value .eq. '-nc' .or. a_value .eq. '-null_color' .or
     &              . a_value .eq. '-cnull') then
                i_arg=i_arg+1
                a_nullclr = a_inarg(i_arg)
              else if (a_value .eq. '-lc' .or. a_value .eq. '-line_color' .or
     &              . a_value .eq. '-cline') then
                i_arg=i_arg+1
                a_lcolor = a_inarg(i_arg)
              else if (a_value .eq. '-workdir' .or. a_value .eq. '-work_dir') then
                i_arg=i_arg+1
                a_workdir = a_inarg(i_arg)
                i_cnt=rdflen(a_workdir)
                if (a_workdir(i_cnt:i_cnt) .ne. '/') a_workdir=a_workdir(1:i_cnt)//'/'
              else if (a_value .eq. '-colordir' .or. a_value .eq. '-color_dir') then
                i_arg=i_arg+1
                a_colordir = a_inarg(i_arg)
                i_cnt=rdflen(a_colordir)
                if (a_colordir(i_cnt:i_cnt) .ne. '/') a_colordir=a_colordir(1:i_cnt)//'/'
              else if (a_value .eq. '-emod') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) i_eventmod
                i_eventmod = max(1,min(1000,i_eventmod))
              else if (a_value .eq. '-debug') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                read(a_value,*) i_debug
              else if (a_value .eq. '-points' .or. a_value .eq. '-pts') then
                i_arg=i_arg+1
                a_ptsfile = a_inarg(i_arg)
                i_smode = 0
                i_show = 1
                i_event(0) = 1          ! Display
                i_event(1) = 0          ! Window
                i_event(2) = 12         ! Event
                i_event(3) = 0
                i_event(4) = 0
                i_event(5) = 43
                i_event(6) = -1
c                write(6,*) 'i_bcnt2 =',i_bcnt
                call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
              else if (a_value .eq. '-click' .or. a_value .eq. '-clk') then
                i_arg=i_arg+1
                a_clickcmd(1) = a_inarg(i_arg)
              else if (a_value .eq. '-ON' .or. a_value .eq. '-on') then
                i_dspactv(1,i_tmp) = 1
              else if (a_value .eq. '-OFF' .or. a_value .eq. '-off') then
                i_dspactv(1,i_tmp) = 0
              else if (a_value .eq. '-active') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                do i=1,I_CMAX
                  if (a_value(i:i) .eq. '1') then
                    i_dspactv(1,i) = 1
                  else if (a_value(i:i) .eq. '0') then
                    i_dspactv(1,i) = 0
                  end if
                end do
              else if (a_value .eq. '-P' .or. a_value .eq. '-ponly') then
                i_ponly = 1
              else if (a_value .eq. '-NM' .or. a_value .eq. '-nomenu') then
                i_menu = 0
              else if (a_value .eq. '-M' .or. a_value .eq. '-menu') then
                i_menu = 1
              else if (a_value .eq. '-NC' .or. a_value .eq. '-NOCLOSE') then
                i_close = 0
              else if (a_value .eq. '-C' .or. a_value .eq. '-CLOSE') then
                i_close = 1
              else if (a_value .eq. '-SRTM' .or. a_value .eq. '-srtm' .or. a_value .eq. '-SRTM30') then
                a_setname(-i_dat) = 'SRTM-dte'
                i_setchdr(-i_dat) = 0
                i_setctlr(-i_dat) = 0
                i_setvfmt(-i_dat) = 2
                i_setcols(-i_dat) = 3601
                i_setrows(-i_dat) = 3601
              else if (a_value .eq. '-c8' .or. a_value .eq. '-complex*8') then
                a_setname(-i_dat) = 'C8-Mag C8-Pha'
                i_setchdr(-i_dat) = 0
                i_setctlr(-i_dat) = 0
                i_setvfmt(-i_dat) = 6
              else if (a_value .eq. '-c2' .or. a_value .eq. '-complex*2') then
                a_setname(-i_dat) = 'C2-Mag C2-Pha'
                i_setchdr(-i_dat) = 0
                i_setctlr(-i_dat) = 0
                i_setvfmt(-i_dat) = 10
              else if (a_value .eq. '-c4' .or. a_value .eq. '-complex*4') then
                a_setname(-i_dat) = 'C4-Mag C4-Pha'
                i_setchdr(-i_dat) = 0
                i_setctlr(-i_dat) = 0
                i_setvfmt(-i_dat) = 12
              else if (a_value .eq. '-c8iq' .or. a_value .eq. '-complex*8iq') then
                a_setname(-i_dat) = 'C8-I C8-Q'
                i_setchdr(-i_dat) = 0
                i_setctlr(-i_dat) = 0
                i_setvfmt(-i_dat) = 6
              else if (a_value .eq. '-rmg' ) then
                a_setname(-i_dat) = 'RMG-Mag RMG-Hgt'
                i_setrhdr(-i_dat) = 0
                i_setrtlr(-i_dat) = 0
                i_setvfmt(-i_dat) = 4
              else if (a_value .eq. '-rmgi' ) then
                a_setname(-i_dat) = 'RMG-Mag RMG-Pha'
                i_setrhdr(-i_dat) = 0
                i_setrtlr(-i_dat) = 0
                i_setvfmt(-i_dat) = 4
              else if (a_value .eq. '-b1' .or. a_value .eq. '-byte' .or. a_value .eq. '-b') then
                i_setvfmt(i_tmp) = 0
              else if (a_value .eq. '-i1' .or. a_value .eq. '-integer*1') then
                i_setvfmt(i_tmp) = 1
              else if (a_value .eq. '-i2' .or. a_value .eq. '-integer*2' .or. a_value .eq. '-si2') then
                i_setvfmt(i_tmp) = 2
              else if (a_value .eq. '-i4' .or. a_value .eq. '-integer*4') then
                i_setvfmt(i_tmp) = 3
              else if (a_value .eq. '-r4' .or. a_value .eq. '-real*4') then
                i_setvfmt(i_tmp) = 4
              else if (a_value .eq. '-r8' .or. a_value .eq. '-real*8') then
                i_setvfmt(i_tmp) = 5
              else if (a_value .eq. '-c8mag' .or. a_value .eq. 'cmag') then
                i_setvfmt(i_tmp) = 6
              else if (a_value .eq. '-c8pha' .or. a_value .eq. 'cpha') then
                i_setvfmt(i_tmp) = 7
              else if (a_value .eq. '-b2' .or. a_value .eq. '-byte*2' .or. a_value .eq. '-byte2') then
                i_setvfmt(i_tmp) = 8
              else if (a_value .eq. '-stokes11' .or. a_value .eq. '-compressed_stokes') then
                i_setvfmt(i_tmp) = 9
              else if (a_value .eq. '-c2mag') then
                i_setvfmt(i_tmp) = 10
              else if (a_value .eq. '-c2pha') then
                i_setvfmt(i_tmp) = 11
              else if (a_value .eq. '-c4mag') then
                i_setvfmt(i_tmp) = 12
              else if (a_value .eq. '-c4pha') then
                i_setvfmt(i_tmp) = 13
              else if (a_value .eq. '-r4mag' .or. a_value .eq. '-real*4_mag') then
                i_setvfmt(i_tmp) = 14
              else if (a_value .eq. '-h' .or. a_value .eq. '-hdr') then
                i_arg=i_arg+1
                a_setinfo(i_tmp)=a_inarg(i_arg)
                a_nullstr=' '
                call get_setinfo(            a_nullstr,
     &              a_setinfo(i_tmp),
     &              a_setproj(i_tmp),
     &              i_setunit(i_tmp),
     &              i_setrows(i_tmp),
     &              i_setcols(i_tmp),
     &              i_setshdr(i_tmp),
     &              i_setstlr(i_tmp),
     &              i_setrhdr(i_tmp),
     &              i_setrtlr(i_tmp),
     &              i_setchdr(i_tmp),
     &              i_setctlr(i_tmp),
     &              i_setvend(i_tmp),
     &              i_setvfmt(i_tmp),
     &              r_setvmlt(i_tmp),
     &              r_setvadr(i_tmp),
     &              r_setvmin(i_tmp),
     &              r_setvmax(i_tmp),
     &              a_setvnul(i_tmp),
     &              r_setrmlt(i_tmp),
     &              r_setradr(i_tmp),
     &              r_setcmlt(i_tmp),
     &              r_setcadr(i_tmp),
     &              r_setpegv(1,i_tmp),
     &              r_dspaddr(i_tmp),
     &              r_dspmult(i_tmp),
     &              r_dspwrap(i_tmp),
     &              r_dspexpn(i_tmp),
     &              r_dspcplw(i_tmp),
     &              r_dspcphi(i_tmp),
     &              r_dspval1(i_tmp),
     &              r_dspval2(i_tmp),
     &              r_dspval3(i_tmp),
     &              i_dspmode(i_tmp),
     &              i_dspaddr(i_tmp),
     &              i_dspmult(i_tmp),
     &              i_dspmixv(i_tmp),
     &              i_dspdvdc(i_tmp),
     &              a_dspctbl(i_tmp)   )
                call get_setinfo(            a_setname(i_tmp),
     &              a_setinfo(i_tmp),
     &              a_setproj(i_tmp),
     &              i_setunit(i_tmp),
     &              i_setrows(i_tmp),
     &              i_setcols(i_tmp),
     &              i_setshdr(i_tmp),
     &              i_setstlr(i_tmp),
     &              i_setrhdr(i_tmp),
     &              i_setrtlr(i_tmp),
     &              i_setchdr(i_tmp),
     &              i_setctlr(i_tmp),
     &              i_setvend(i_tmp),
     &              i_setvfmt(i_tmp),
     &              r_setvmlt(i_tmp),
     &              r_setvadr(i_tmp),
     &              r_setvmin(i_tmp),
     &              r_setvmax(i_tmp),
     &              a_setvnul(i_tmp),
     &              r_setrmlt(i_tmp),
     &              r_setradr(i_tmp),
     &              r_setcmlt(i_tmp),
     &              r_setcadr(i_tmp),
     &              r_setpegv(1,i_tmp),
     &              r_dspaddr(i_tmp),
     &              r_dspmult(i_tmp),
     &              r_dspwrap(i_tmp),
     &              r_dspexpn(i_tmp),
     &              r_dspcplw(i_tmp),
     &              r_dspcphi(i_tmp),
     &              r_dspval1(i_tmp),
     &              r_dspval2(i_tmp),
     &              r_dspval3(i_tmp),
     &              i_dspmode(i_tmp),
     &              i_dspaddr(i_tmp),
     &              i_dspmult(i_tmp),
     &              i_dspmixv(i_tmp),
     &              i_dspdvdc(i_tmp),
     &              a_dspctbl(i_tmp)   )
              else if (a_value .eq. '-top' .or. a_value .eq. '-air' .or. a_value .eq. '-maghdr' .or. a_value .eq. '-dtehdr') then
                i_arg=i_arg+1
                a_hdrfile=a_inarg(i_arg)

                i_mbytes=4
                a_type='sch'
                r_mmul=1.0
                r_madd=0.0
                r_dmul=1.0
                r_dadd=0.0
                call read_hdr(a_hdrfile,i_lsize,i_ssize,r_peg,a_type,
     &              r_str,r_spc,i_mbytes,i_dbytes,r_mmul,r_madd,
     &              r_dmul,r_dadd,i_err)

                    a_setinfo(i_tmp) = ' '
                    a_setproj(i_tmp) = a_type
                    i_setunit(i_tmp) = 0
                    i_setrows(i_tmp) = i_lsize
                    i_setcols(i_tmp) = i_ssize
                    i_setshdr(i_tmp) = 0
                    i_setstlr(i_tmp) = 0
                    i_setrhdr(i_tmp) = 0
                    i_setrtlr(i_tmp) = 0
                    i_setchdr(i_tmp) = 0
                    i_setctlr(i_tmp) = 0
                    if (a_value .eq. '-maghdr') then
                      if (i_mbytes .eq. 1) then
                        i_setvfmt(i_tmp) = i_CnvrtFmt('BYTE')
                      else if (i_mbytes .eq. 2) then
                        i_setvfmt(i_tmp) = i_CnvrtFmt('BYTE*2')
                      else if (i_mbytes .eq. 4) then
                        i_setvfmt(i_tmp) = i_CnvrtFmt('REAL*4')
                      else
                        i_setvfmt(i_tmp) = i_CnvrtFmt('REAL*4')
                      end if
                      r_setvmlt(i_tmp) = r_mmul
                      r_setvadr(i_tmp) = r_madd
                    else
                      if (i_dbytes .eq. 1) then
                        i_setvfmt(i_tmp) = i_CnvrtFmt('BYTE')
                      else if (i_dbytes .eq. 2) then
                        i_setvfmt(i_tmp) = i_CnvrtFmt('BYTE*2')
                      else if (i_dbytes .eq. 4) then
                        i_setvfmt(i_tmp) = i_CnvrtFmt('REAL*4')
                      else
                        i_setvfmt(i_tmp) = i_CnvrtFmt('REAL*4')
                      end if
                      r_setvmlt(i_tmp) = r_dmul
                      r_setvadr(i_tmp) = r_dadd
                    end if
c                    r_setvmin(i_tmp) =
c                    r_setvmax(i_tmp) =
c                    a_setvnul(i_tmp) =
                    r_setrmlt(i_tmp) = r_spc(1)
                    r_setradr(i_tmp) = r_str(1)+r_spc(1)
                    r_setcmlt(i_tmp) = r_spc(2)
                    r_setcadr(i_tmp) = r_str(2)+r_spc(2)
                    r_setpegv(1,i_tmp) = r_peg(1)
                    r_setpegv(2,i_tmp) = r_peg(2)
                    r_setpegv(3,i_tmp) = r_peg(3)

                    if (a_setfile(i_tmp) .ne. ' ') then
                      if(a_setname(i_tmp) .eq. ' ') write(a_setname(i_tmp),'(a,i1)') 'Set_',abs(i_tmp)
                      a_setinfo(i_tmp) = a_setfile(i_tmp)(1:max(1,rdflen(a_setfile(i_tmp))))//'.mdx'
                      call put_setinfo(a_setname(i_tmp),
     &                a_setinfo(i_tmp),
     &                a_setproj(i_tmp),
     &                i_setunit(i_tmp),
     &                i_setrows(i_tmp),
     &                i_setcols(i_tmp),
     &                i_setshdr(i_tmp),
     &                i_setstlr(i_tmp),
     &                i_setrhdr(i_tmp),
     &                i_setrtlr(i_tmp),
     &                i_setchdr(i_tmp),
     &                i_setctlr(i_tmp),
     &                i_setvend(i_tmp),
     &                i_setvfmt(i_tmp),
     &                r_setvmlt(i_tmp),
     &                r_setvadr(i_tmp),
     &                r_setvmin(i_tmp),
     &                r_setvmax(i_tmp),
     &                a_setvnul(i_tmp),
     &                r_setrmlt(i_tmp),
     &                r_setradr(i_tmp),
     &                r_setcmlt(i_tmp),
     &                r_setcadr(i_tmp),
     &                r_setpegv(1,i_tmp)   )
                    end if
              else if (a_value .eq. '-set') then
                i_arg=i_arg+1
                a_value = a_inarg(i_arg)
                i_chn = i_chn + 1
                if (i_dat .gt. 0) then
                  i_set = min(i_set + 1,I_CMAX)
                  i_tmp = i_set
                  call copy_setdata(-i_dat,i_tmp,
     &                          i_setunit,i_setrows,i_setcols,
     &                          a_setname,a_setfile,a_setinfo,a_setproj,
     &                          i_setshdr,i_setstlr,i_setrhdr,i_setrtlr,i_setchdr,i_setctlr,
     &                          r_setrmlt,r_setradr,r_setcmlt,r_setcadr,r_setvmlt,r_setvadr,
     &                          i_setvend,i_setvfmt,r_setvmin,r_setvmax,a_setvnul,r_setpegv)

                  call copy_dspdata(-i_dat,i_tmp,
     &                          r_dspaddr,r_dspmult,r_dspwrap,r_dspexpn,r_dspcplw,r_dspcphi,
     &                          r_dspval1,r_dspval2,r_dspval3,i_dspmode,i_dspaddr,i_dspmult,i_dspmixv,
     &                          a_dspctbl,i_dspdvdc)

                  i_dspactv(1,i_tmp) = i_dspactv(1,-i_dat)

                  a_setname(i_tmp) = a_value
                  if (a_setinfo(i_tmp) .eq. ' ' ) then
                    a_setinfo(i_tmp)=a_setfile(i_tmp
     &                    )(1:rdflen(a_setfile(i_tmp)))//'.mdx'
                  end if
c     type *,'looking1 at ',a_setinfo(i_tmp),' ',i_tmp
                  call get_setinfo(          a_setname(i_tmp),
     &                 a_setinfo(i_tmp),
     &                 a_setproj(i_tmp),
     &                 i_setunit(i_tmp),
     &                 i_setrows(i_tmp),
     &                 i_setcols(i_tmp),
     &                 i_setshdr(i_tmp),
     &                 i_setstlr(i_tmp),
     &                 i_setrhdr(i_tmp),
     &                 i_setrtlr(i_tmp),
     &                 i_setchdr(i_tmp),
     &                 i_setctlr(i_tmp),
     &                 i_setvend(i_tmp),
     &                 i_setvfmt(i_tmp),
     &                 r_setvmlt(i_tmp),
     &                 r_setvadr(i_tmp),
     &                 r_setvmin(i_tmp),
     &                 r_setvmax(i_tmp),
     &                 a_setvnul(i_tmp),
     &                 r_setrmlt(i_tmp),
     &                 r_setradr(i_tmp),
     &                 r_setcmlt(i_tmp),
     &                 r_setcadr(i_tmp),
     &                 r_setpegv(1,i_tmp),
     &                 r_dspaddr(i_tmp),
     &                 r_dspmult(i_tmp),
     &                 r_dspwrap(i_tmp),
     &                 r_dspexpn(i_tmp),
     &                 r_dspcplw(i_tmp),
     &                 r_dspcphi(i_tmp),
     &                 r_dspval1(i_tmp),
     &                 r_dspval2(i_tmp),
     &                 r_dspval3(i_tmp),
     &                 i_dspmode(i_tmp),
     &                 i_dspaddr(i_tmp),
     &                 i_dspmult(i_tmp),
     &                 i_dspmixv(i_tmp),
     &                 i_dspdvdc(i_tmp),
     &                 a_dspctbl(i_tmp)   )

                else

                  if (i_set .eq. 1) then
                    a_setname(0) = a_value
                  else
                    a_setname(0) =
     &                    a_setname(0)(1:max(1,rdflen(a_setname(0))))//'
     &                    '//a_value
                  end if
                end if
              else if (a_value(1:1) .eq. '-' .and. a_value .ne. '-file') then   ! implicit set name
                a_value = a_value(2:)
                i_chn = i_chn + 1
                if (i_dat .gt. 0) then
                  i_set = min(i_set + 1,I_CMAX)
                  i_tmp = i_set
                  call copy_setdata(-i_dat,i_tmp,
     &                          i_setunit,i_setrows,i_setcols,
     &                          a_setname,a_setfile,a_setinfo,a_setproj,
     &                          i_setshdr,i_setstlr,i_setrhdr,i_setrtlr,i_setchdr,i_setctlr,
     &                          r_setrmlt,r_setradr,r_setcmlt,r_setcadr,r_setvmlt,r_setvadr,
     &                          i_setvend,i_setvfmt,r_setvmin,r_setvmax,a_setvnul,r_setpegv)

                  call copy_dspdata(-i_dat,i_tmp,
     &                          r_dspaddr,r_dspmult,r_dspwrap,r_dspexpn,r_dspcplw,r_dspcphi,
     &                          r_dspval1,r_dspval2,r_dspval3,i_dspmode,i_dspaddr,i_dspmult,i_dspmixv,
     &                          a_dspctbl,i_dspdvdc)

                  i_dspactv(1,i_tmp) = i_dspactv(1,-i_dat)

                  a_setname(i_tmp) = a_value
                  if (a_setinfo(i_tmp) .eq. ' ' ) then
                    a_setinfo(i_tmp)=a_setfile(i_tmp
     &                    )(1:rdflen(a_setfile(i_tmp)))//'.mdx'
                  end if
                  call get_setinfo(          a_setname(i_tmp),
     &                 a_setinfo(i_tmp),
     &                 a_setproj(i_tmp),
     &                 i_setunit(i_tmp),
     &                 i_setrows(i_tmp),
     &                 i_setcols(i_tmp),
     &                 i_setshdr(i_tmp),
     &                 i_setstlr(i_tmp),
     &                 i_setrhdr(i_tmp),
     &                 i_setrtlr(i_tmp),
     &                 i_setchdr(i_tmp),
     &                 i_setctlr(i_tmp),
     &                 i_setvend(i_tmp),
     &                 i_setvfmt(i_tmp),
     &                 r_setvmlt(i_tmp),
     &                 r_setvadr(i_tmp),
     &                 r_setvmin(i_tmp),
     &                 r_setvmax(i_tmp),
     &                 a_setvnul(i_tmp),
     &                 r_setrmlt(i_tmp),
     &                 r_setradr(i_tmp),
     &                 r_setcmlt(i_tmp),
     &                 r_setcadr(i_tmp),
     &                 r_setpegv(1,i_tmp),
     &                 r_dspaddr(i_tmp),
     &                 r_dspmult(i_tmp),
     &                 r_dspwrap(i_tmp),
     &                 r_dspexpn(i_tmp),
     &                 r_dspcplw(i_tmp),
     &                 r_dspcphi(i_tmp),
     &                 r_dspval1(i_tmp),
     &                 r_dspval2(i_tmp),
     &                 r_dspval3(i_tmp),
     &                 i_dspmode(i_tmp),
     &                 i_dspaddr(i_tmp),
     &                 i_dspmult(i_tmp),
     &                 i_dspmixv(i_tmp),
     &                 i_dspdvdc(i_tmp),
     &                 a_dspctbl(i_tmp)   )

                else

                  if (i_set .eq. 1) then
                    a_setname(0) = a_value
                  else
                    a_setname(0) =
     &                    a_setname(0)(1:max(1,rdflen(a_setname(0))))//'
     &                    '//a_value
                  end if
                end if
              else
                if (a_value .eq. '-file') then
                  i_arg=i_arg+1
                  a_value = a_inarg(i_arg)
                end if
c                write(6,*) 'a_value=',a_value,i_dat,i_chn  !@#$%
                if (i_dat .gt. 0 .and. i_chn .eq. 0) then ! last file had no sets specified
                  if (a_setname(-i_dat) .eq. ' ') then
                    a_label = a_setfile(-i_dat)
                    if (a_label(1:1) .ne. '=') then
                      do  while(max(index(a_label,'/'),index(a_label,'.')) .gt. 0 .and.
     &                         max(index(a_label,'/'),index(a_label,'.')) .lt. rdflen(a_label)-1)
                        a_label = a_label(max(index(a_label,'/'),index(a_label,'.'))+1:)
                      end do
                    end if
                    a_setname(-i_dat) = a_label
                  end if
c                  write(6,*) 'i_dat=',i_dat
c                  write(6,*) 'a_setname=',a_setname(-i_dat)   !@#$%

                  call rdf_getfields(a_setname(-i_dat),i_sss,a_sss)
                  do i_chn=1,i_sss
                    i_set = min(i_set + 1,I_CMAX)
                    i_tmp = i_set
                    call copy_setdata(-i_dat,i_tmp,
     &                          i_setunit,i_setrows,i_setcols,
     &                          a_setname,a_setfile,a_setinfo,a_setproj,
     &                          i_setshdr,i_setstlr,i_setrhdr,i_setrtlr,i_setchdr,i_setctlr,
     &                          r_setrmlt,r_setradr,r_setcmlt,r_setcadr,r_setvmlt,r_setvadr,
     &                          i_setvend,i_setvfmt,r_setvmin,r_setvmax,a_setvnul,r_setpegv)

                    call copy_dspdata(-i_dat,i_tmp,
     &                          r_dspaddr,r_dspmult,r_dspwrap,r_dspexpn,r_dspcplw,r_dspcphi,
     &                          r_dspval1,r_dspval2,r_dspval3,i_dspmode,i_dspaddr,i_dspmult,i_dspmixv,
     &                          a_dspctbl,i_dspdvdc)
                    a_setname(i_tmp) = a_sss(i_chn)

                    i_dspactv(1,i_tmp) = i_dspactv(1,-i_dat)

                    if (a_setinfo(i_tmp) .eq. ' ' ) then
                      a_setinfo(i_tmp)=a_setfile(i_tmp
     &                       )(1:rdflen(a_setfile(i_tmp)))//'.mdx'
                    end if
c                    write(6,*) 'i_chn=',i_chn,i_tmp   !@#$%
                    call get_setinfo(a_setname(i_tmp),
     &                    a_setinfo(i_tmp),
     &                    a_setproj(i_tmp),
     &                    i_setunit(i_tmp),
     &                    i_setrows(i_tmp),
     &                    i_setcols(i_tmp),
     &                    i_setshdr(i_tmp),
     &                    i_setstlr(i_tmp),
     &                    i_setrhdr(i_tmp),
     &                    i_setrtlr(i_tmp),
     &                    i_setchdr(i_tmp),
     &                    i_setctlr(i_tmp),
     &                    i_setvend(i_tmp),
     &                    i_setvfmt(i_tmp),
     &                    r_setvmlt(i_tmp),
     &                    r_setvadr(i_tmp),
     &                    r_setvmin(i_tmp),
     &                    r_setvmax(i_tmp),
     &                    a_setvnul(i_tmp),
     &                    r_setrmlt(i_tmp),
     &                    r_setradr(i_tmp),
     &                    r_setcmlt(i_tmp),
     &                    r_setcadr(i_tmp),
     &                    r_setpegv(1,i_tmp),
     &                    r_dspaddr(i_tmp),
     &                    r_dspmult(i_tmp),
     &                    r_dspwrap(i_tmp),
     &                    r_dspexpn(i_tmp),
     &                    r_dspcplw(i_tmp),
     &                    r_dspcphi(i_tmp),
     &                    r_dspval1(i_tmp),
     &                    r_dspval2(i_tmp),
     &                    r_dspval3(i_tmp),
     &                    i_dspmode(i_tmp),
     &                    i_dspaddr(i_tmp),
     &                    i_dspmult(i_tmp),
     &                    i_dspmixv(i_tmp),
     &                    i_dspdvdc(i_tmp),
     &                    a_dspctbl(i_tmp)   )

                  end do
                end if
                i_dat = i_dat+1
                i_tmp = -i_dat
                i_chn = 0
c                write(6,*) 'hello=',i_dat,i_tmp,i_chn  !@#$%
                call copy_setdata(0,i_tmp,
     &                           i_setunit,i_setrows,i_setcols,
     &                           a_setname,a_setfile,a_setinfo,a_setproj,
     &                           i_setshdr,i_setstlr,i_setrhdr,i_setrtlr,i_setchdr,i_setctlr,
     &                           r_setrmlt,r_setradr,r_setcmlt,r_setcadr,r_setvmlt,r_setvadr,
     &                           i_setvend,i_setvfmt,r_setvmin,r_setvmax,a_setvnul,r_setpegv)

                call copy_dspdata(0,i_tmp,
     &                           r_dspaddr,r_dspmult,r_dspwrap,r_dspexpn,r_dspcplw,r_dspcphi,
     &                           r_dspval1,r_dspval2,r_dspval3,i_dspmode,i_dspaddr,i_dspmult,i_dspmixv,
     &                           a_dspctbl,i_dspdvdc)

                i_dspactv(1,i_tmp) = i_dspactv(1,0)

                a_setfile(i_tmp) = a_value

c                write(6,*) 'hello2=',i_dat,i_tmp,i_chn  !@#$%
c                write(6,*) 'hello3=',a_setinfo(i_tmp) !@#$%
                if (a_setinfo(i_tmp) .eq. ' ' .and. a_setfile(i_tmp)(1:1) .ne. '=' .and. a_setfile(i_tmp) .ne. 'internal') then
                  a_setinfo(i_tmp)=a_setfile(i_tmp
     &                 )(1:rdflen(a_setfile(i_tmp)))//'.mdx'
c                  write(6,*) 'openning=',a_setname(i_tmp) !@#$%
                  call get_airsarinfo(         a_setname(i_tmp),  ! Only executes if no header is specified
     &              a_setfile(i_tmp),
     &              a_setproj(i_tmp),
     &              i_setunit(i_tmp),
     &              i_setrows(i_tmp),
     &              i_setcols(i_tmp),
     &              i_setshdr(i_tmp),
     &              i_setstlr(i_tmp),
     &              i_setrhdr(i_tmp),
     &              i_setrtlr(i_tmp),
     &              i_setchdr(i_tmp),
     &              i_setctlr(i_tmp),
     &              i_setvend(i_tmp),
     &              i_setvfmt(i_tmp),
     &              r_setvmlt(i_tmp),
     &              r_setvadr(i_tmp),
     &              r_setvmin(i_tmp),
     &              r_setvmax(i_tmp),
     &              a_setvnul(i_tmp),
     &              r_setrmlt(i_tmp),
     &              r_setradr(i_tmp),
     &              r_setcmlt(i_tmp),
     &              r_setcadr(i_tmp),
     &              r_setpegv(1,i_tmp),
     &              a_dspctbl(i_tmp)   )
c                    write(6,*) 'openning1=',a_setname(i_tmp) !@#$%
                  call get_pdsinfo(            a_setname(i_tmp),  ! Only executes if no header is specified
     &              a_setfile(i_tmp),
     &              a_setproj(i_tmp),
     &              i_setunit(i_tmp),
     &              i_setrows(i_tmp),
     &              i_setcols(i_tmp),
     &              i_setshdr(i_tmp),
     &              i_setstlr(i_tmp),
     &              i_setrhdr(i_tmp),
     &              i_setrtlr(i_tmp),
     &              i_setchdr(i_tmp),
     &              i_setctlr(i_tmp),
     &              i_setvend(i_tmp),
     &              i_setvfmt(i_tmp),
     &              r_setvmlt(i_tmp),
     &              r_setvadr(i_tmp),
     &              r_setvmin(i_tmp),
     &              r_setvmax(i_tmp),
     &              a_setvnul(i_tmp),
     &              r_setrmlt(i_tmp),
     &              r_setradr(i_tmp),
     &              r_setcmlt(i_tmp),
     &              r_setcadr(i_tmp),
     &              r_setpegv(1,i_tmp),
     &              a_dspctbl(i_tmp) , i_debug   )
c                    write(6,*) 'openning2=',a_setname(i_tmp) !@#$%
                  call get_cubinfo(            a_setname(i_tmp),  ! Only executes if no header is specified
     &              a_setfile(i_tmp),
     &              a_setproj(i_tmp),
     &              i_setunit(i_tmp),
     &              i_setrows(i_tmp),
     &              i_setcols(i_tmp),
     &              i_setshdr(i_tmp),
     &              i_setstlr(i_tmp),
     &              i_setrhdr(i_tmp),
     &              i_setrtlr(i_tmp),
     &              i_setchdr(i_tmp),
     &              i_setctlr(i_tmp),
     &              i_setvend(i_tmp),
     &              i_setvfmt(i_tmp),
     &              r_setvmlt(i_tmp),
     &              r_setvadr(i_tmp),
     &              r_setvmin(i_tmp),
     &              r_setvmax(i_tmp),
     &              a_setvnul(i_tmp),
     &              r_setrmlt(i_tmp),
     &              r_setradr(i_tmp),
     &              r_setcmlt(i_tmp),
     &              r_setcadr(i_tmp),
     &              r_setpegv(1,i_tmp),
     &              a_dspctbl(i_tmp) , i_debug   )
                end if
c                write(6,*) 'openning3=',a_setinfo(i_tmp) !@#$%
                call get_setinfo(              a_setname(i_tmp),
     &              a_setinfo(i_tmp),
     &              a_setproj(i_tmp),
     &              i_setunit(i_tmp),
     &              i_setrows(i_tmp),
     &              i_setcols(i_tmp),
     &              i_setshdr(i_tmp),
     &              i_setstlr(i_tmp),
     &              i_setrhdr(i_tmp),
     &              i_setrtlr(i_tmp),
     &              i_setchdr(i_tmp),
     &              i_setctlr(i_tmp),
     &              i_setvend(i_tmp),
     &              i_setvfmt(i_tmp),
     &              r_setvmlt(i_tmp),
     &              r_setvadr(i_tmp),
     &              r_setvmin(i_tmp),
     &              r_setvmax(i_tmp),
     &              a_setvnul(i_tmp),
     &              r_setrmlt(i_tmp),
     &              r_setradr(i_tmp),
     &              r_setcmlt(i_tmp),
     &              r_setcadr(i_tmp),
     &              r_setpegv(1,i_tmp),
     &              r_dspaddr(i_tmp),
     &              r_dspmult(i_tmp),
     &              r_dspwrap(i_tmp),
     &              r_dspexpn(i_tmp),
     &              r_dspcplw(i_tmp),
     &              r_dspcphi(i_tmp),
     &              r_dspval1(i_tmp),
     &              r_dspval2(i_tmp),
     &              r_dspval3(i_tmp),
     &              i_dspmode(i_tmp),
     &              i_dspaddr(i_tmp),
     &              i_dspmult(i_tmp),
     &              i_dspmixv(i_tmp),
     &              i_dspdvdc(i_tmp),
     &              a_dspctbl(i_tmp)   )
c                   write(6,*) 'set name = ',a_setname(i_tmp)
              end if !@#$%

            end do


          end if

          if (i_dat .gt. 0 .and. i_chn .eq. 0) then ! last file had no sets specified
            if (a_setname(-i_dat) .eq. ' ') then
              a_label = a_setfile(-i_dat)
              if (a_label(1:1) .ne. '=') then
                do  while(max(index(a_label,'/'),index(a_label,'.')) .gt. 0 .and.
     &                max(index(a_label,'/'),index(a_label,'.')) .lt. rdflen(a_label)-1)
                  a_label = a_label(max(index(a_label,'/'),index(a_label,'.'))+1:)
                end do
              end if
              a_setname(-i_dat) = a_label
            end if
            if (a_setname(-i_dat) .eq. ' ') then
              write(a_setname(-i_dat),'(a,i2)') 'Set ',i_dat
            end if
            call rdf_getfields(a_setname(-i_dat),i_sss,a_sss)
            do i_chn=1,i_sss
              i_set = min(i_set + 1,I_CMAX)
              i_tmp = i_set
              call copy_setdata(-i_dat,i_tmp,
     &                        i_setunit,i_setrows,i_setcols,
     &                        a_setname,a_setfile,a_setinfo,a_setproj,
     &                        i_setshdr,i_setstlr,i_setrhdr,i_setrtlr,i_setchdr,i_setctlr,
     &                        r_setrmlt,r_setradr,r_setcmlt,r_setcadr,r_setvmlt,r_setvadr,
     &                        i_setvend,i_setvfmt,r_setvmin,r_setvmax,a_setvnul,r_setpegv)

              call copy_dspdata(-i_dat,i_tmp,
     &                        r_dspaddr,r_dspmult,r_dspwrap,r_dspexpn,r_dspcplw,r_dspcphi,
     &                        r_dspval1,r_dspval2,r_dspval3,i_dspmode,i_dspaddr,i_dspmult,i_dspmixv,
     &                        a_dspctbl,i_dspdvdc)

              i_dspactv(1,i_tmp) = i_dspactv(1,-i_dat)

              a_setname(i_tmp) = a_sss(i_chn)

              if (a_setinfo(i_tmp) .eq. ' ' ) then
                a_setinfo(i_tmp)=a_setfile(i_tmp
     &              )(1:rdflen(a_setfile(i_tmp)))//'.mdx'
              end if
              call get_setinfo(                a_setname(i_tmp),
     &           a_setinfo(i_tmp),
     &           a_setproj(i_tmp),
     &           i_setunit(i_tmp),
     &           i_setrows(i_tmp),
     &           i_setcols(i_tmp),
     &           i_setshdr(i_tmp),
     &           i_setstlr(i_tmp),
     &           i_setrhdr(i_tmp),
     &           i_setrtlr(i_tmp),
     &           i_setchdr(i_tmp),
     &           i_setctlr(i_tmp),
     &           i_setvend(i_tmp),
     &           i_setvfmt(i_tmp),
     &           r_setvmlt(i_tmp),
     &           r_setvadr(i_tmp),
     &           r_setvmin(i_tmp),
     &           r_setvmax(i_tmp),
     &           a_setvnul(i_tmp),
     &           r_setrmlt(i_tmp),
     &           r_setradr(i_tmp),
     &           r_setcmlt(i_tmp),
     &           r_setcadr(i_tmp),
     &           r_setpegv(1,i_tmp),
     &           r_dspaddr(i_tmp),
     &           r_dspmult(i_tmp),
     &           r_dspwrap(i_tmp),
     &           r_dspexpn(i_tmp),
     &           r_dspcplw(i_tmp),
     &           r_dspcphi(i_tmp),
     &           r_dspval1(i_tmp),
     &           r_dspval2(i_tmp),
     &           r_dspval3(i_tmp),
     &           i_dspmode(i_tmp),
     &           i_dspaddr(i_tmp),
     &           i_dspmult(i_tmp),
     &           i_dspmixv(i_tmp),
     *           i_dspdvdc(i_tmp),
     &           a_dspctbl(i_tmp)   )

            end do
          end if

          do i_chn = 1,i_set        ! In order to make the -c8 option work correctly
            if (a_setname(i_chn) .eq. 'C8-Mag') then
              i_setvfmt(i_chn) = 6
            else if (a_setname(i_chn) .eq. 'C8-Pha') then
              i_setvfmt(i_chn) = 7
              r_dspwrap(i_chn) = 2.0d0*r_pi
              i_dspmode(i_chn) = 6
              a_dspctbl(i_chn) = 'cmy'
            else if (a_setname(i_chn) .eq. 'C8-I') then
              i_setvfmt(i_chn) = 4
              i_setchdr(i_chn) = 0
              i_setctlr(i_chn) = 4
            else if (a_setname(i_chn) .eq. 'C8-Q') then
              i_setvfmt(i_chn) = 4
              i_setchdr(i_chn) = 4
              i_setctlr(i_chn) = 0
            else if (a_setname(i_chn) .eq. 'C2-Mag') then
              i_setvfmt(i_chn) = 10
            else if (a_setname(i_chn) .eq. 'C2-Pha') then
              i_setvfmt(i_chn) = 11
              r_dspwrap(i_chn) = 2.0d0*r_pi
              i_dspmode(i_chn) = 6
              a_dspctbl(i_chn) = 'cmy'
            else if (a_setname(i_chn) .eq. 'C4-Mag') then
              i_setvfmt(i_chn) = 12
            else if (a_setname(i_chn) .eq. 'C4-Pha') then
              i_setvfmt(i_chn) = 13
              r_dspwrap(i_chn) = 2.0d0*r_pi
              i_dspmode(i_chn) = 6
              a_dspctbl(i_chn) = 'cmy'
            else if (a_setname(i_chn) .eq. 'SRTM-dte') then
              i_setvfmt(i_chn) = 2
              r_dspwrap(i_chn) = 200.
              r_dspmult(i_chn) = r_dspwrap(i_chn)
              i_dspmult(i_chn) = 0
              i_dspaddr(i_chn) = 0
              i_dspmode(i_chn) = 6
              a_dspctbl(i_chn) = 'cmy'
              i_str=0
              do while (index(a_setfile(i_chn)(i_str+1:),'/') .gt. 0)
                i_str=i_str+index(a_setfile(i_chn)(i_str+1:),'/')
              end do
              read(a_setfile(i_chn)(i_str+1:),'(x,i2.2)') i_lat
              i_lat=i_lat+1.0
              if (a_setfile(i_chn)(i_str+1:i_str+1) .eq. 'S') i_lat=-i_lat
              read(a_setfile(i_chn)(i_str+1:),'(4x,i3.3)') i_lon
              if (a_setfile(i_chn)(i_str+4:i_str+4) .eq. 'W') i_lon=-i_lon
              i_setcols(i_chn) = 3601
              i_setrows(i_chn) = 3601
              i_setvend(i_chn) = 1
              a_setproj(i_chn) = 'eqa'
              r_setcadr(i_chn) = i_lon
              r_setcmlt(i_chn) = 1.0d0/3600
              r_setradr(i_chn) = i_lat
              r_setrmlt(i_chn) = -1.0d0/3600
              r_setvmin(i_chn) = -10000.
            end if
          end do

          do i_chn = 1,i_set        ! In order to make the -rmg option work correctly
            if (a_setname(i_chn) .eq. 'RMG-Mag') then
              i_setrtlr(i_chn) = 4*i_setcols(i_chn)
              i_setvfmt(i_chn) = 4
            else if (a_setname(i_chn) .eq. 'RMG-Hgt') then
              i_setrhdr(i_chn) = 4*i_setcols(i_chn)
              i_setvfmt(i_chn) = 4
              a_dspctbl(i_chn) = 'cmy'
            else if (a_setname(i_chn) .eq. 'RMG-Pha') then
              i_setrhdr(i_chn) = 4*i_setcols(i_chn)
              i_setvfmt(i_chn) = 4
              a_dspctbl(i_chn) = 'cmy'
            end if
          end do

          do i_chn = 1,i_set        ! In order to make QUBE Data work better
            if (a_setname(i_chn) .eq. 'QUBE' .and. i_qubeset .eq. 0) then
              a_dspctbl(i_chn) = 'cmy'
              i_qubeset=1
            else if (a_setname(i_chn) .eq. 'QUBE' .and. i_qubeset .eq. 1) then
              a_dspctbl(i_chn) = 'grey'
              i_dspdvdc(i_chn) = 1
              i_qubeset=0
            end if
          end do

          do i_chn = 1,i_set        !
            if (i_dspdvdc(i_chn) .eq. 1) then
              a_setname(i_chn) = 'd('//a_setname(i_chn)(1:max(1,rdflen(a_setname(i_chn))))//')/dc'
              i_setvfmt(i_chn) = -i_setvfmt(i_chn)
            end if
          end do

          do i_chn = 1,i_set        ! set b_setvnul
            b_setvnul(0,i_chn) = rdflen(a_setvnul(i_chn))/2
            if (i_debug .eq. -5 .or. i_debug .ge. 5) write(6,*) 'i_chn,len,a_setvnul(i_chn) ',i_chn,b_setvnul(0
     &        ,i_chn),'#',a_setvnul(i_chn),'#'
            do i=1,16
              if (i .le. b_setvnul(0,i_chn)) then
                read(a_setvnul(i_chn)(2*i-1:2*i),fmt='(z2.2)')
     &              b_setvnul(i,i_chn)
              else
                b_setvnul(i,i_chn) = 0
              end if
              if (b_setvnul(i,i_chn) .ge. 128) b_setvnul(i,i_chn) = b_setvnul(i,i_chn)-256
              if (i_debug .eq. -10 .or. i_debug .ge. 10) write(6,*) 'b_setvnul = ',i_chn,i,b_setvnul(i,i_chn)
            end do
          end do


          do i_chn = 1,i_set  ! correct sign of utm northing spacing
            if (rdflower(a_setproj(i_chn)) .eq. 'utm') r_setrmlt(i_chn)=-r_setrmlt(i_chn)
          end do

          if (index(a_nullclr,',') .ne. 0) then
            i_val = index(a_nullclr,'(')
            if (i_val .gt. 0) a_nullclr = a_nullclr(i_val+1:)
            i_val = index(a_nullclr,')')
            if (i_val .gt. 2) a_nullclr = a_nullclr(:i_val-1)
            read(a_nullclr,*) i_nullclr
          else
            if (i_debug .ge. 6) write(6,*) 'looking up color: ',a_nullclr
            call init_dsp(a_lcolor,i_debug)
            call get_colorrgb(a_nullclr,i_nullclr)
          end if
          if (i_debug .ge. 5) write(6,*) 'Setting null color: ',i_nullclr

          do i_chn = 1,i_set
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) ' '
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'setname = ',a_setname(i_chn)(1:50)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'setfile = ',a_setfile(i_chn)(1:50)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'setinfo = ',a_setinfo(i_chn)(1:50)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'setproj = ',a_setproj(i_chn)(1:50)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'setrows = ',i_setrows(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'setcols = ',i_setcols(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'setshdr = ',i_setshdr(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'setstlr = ',i_setstlr(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'setrhdr = ',i_setrhdr(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'setrtlr = ',i_setrtlr(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'setchdr = ',i_setchdr(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'setctlr = ',i_setctlr(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'setvend = ',i_setvend(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'setfrmt = ',i_setvfmt(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'setrmlt = ',r_setrmlt(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'setradr = ',r_setradr(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'setcmlt = ',r_setcmlt(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'setcadr = ',r_setcadr(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'setvmlt = ',r_setvmlt(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'setvadr = ',r_setvadr(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'setvmin = ',r_setvmin(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'setvmax = ',r_setvmax(i_chn)

            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'dspmode = ',i_dspmode(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'dspaddr = ',r_dspaddr(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'dspmult = ',r_dspmult(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'dspwrap = ',r_dspwrap(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'dspexpn = ',r_dspexpn(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'dspcplw = ',r_dspcplw(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'dspcphi = ',r_dspcphi(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'dspval1 = ',r_dspval1(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'dspval2 = ',r_dspval2(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'dspval3 = ',r_dspval3(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'dspmode = ',i_dspmode(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'dspdvdx = ',i_dspdvdc(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'dspaddr = ',i_dspaddr(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'dspmult = ',i_dspmult(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'dspmixv = ',i_dspmixv(i_chn)
            if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'dspctbl = ',a_dspctbl(i_chn)(1:50)

          end do

          if (i_setcols(1) .eq. 0) then
            write(6,*) 'Number of columns not specified'
            stop ' '
          end if

c     open image files

          do i_chn = 1,i_set

            if (a_setfile(i_chn) .ne. ' ' .and. a_setfile(i_chn)(1:1) .ne. '=') then
              if (a_setfile(i_chn)(1:max(1,rdflen(a_setfile(i_chn)))) .eq. 'internal' .and. i_maxbuff .gt. 0) then
                i_setunit(i_chn) = -i_chn
                i_fbytes = readfunc(1,i_chn,i_eight(0),0,b_data)
                if (i_debug .ge. 3) write(6,*) 'internal buffer size=',i_fbytes
                i_fbytes = min(i_fbytes,i_maxbuff)
              else
                i_setunit(i_chn) = initdk(20+i_chn,a_setfile(i_chn))
                i_fbytes = i_getfsize(i_setunit(i_chn))
                if (i_setunit(i_chn) .le. 0) stop 'set number less than or equal to zero'
              end if
              if (i_fbytes .gt. 0) then
                i_setrows(i_chn) = min(i_eight(i_setrows(i_chn)),(i_fbytes
     &              -i_setshdr(i_chn)-i_setstlr(i_chn))/((i_setvbyt(i_setvfmt(i_chn))
     &              +i_setchdr(i_chn)+i_setctlr(i_chn))*i_setcols(i_chn)+i_setrhdr(i_chn)+i_setrtlr(i_chn)
     &              ))

c                i_setunit(i_chn) = 20+i_chn
c                open(unit=i_setunit(i_chn),file=a_setfile(i_chn),status='old',
c     &            form='unformatted',access='direct',recl=i_setcols(i_chn)*i_setvbyt(i_chn))
                 if (i_debug .eq. 2) write(6,*) 'Opening file: ',a_setfile(i_chn)(1:60)
                 if (i_debug .gt. 2) write(6,*) 'Opening file: ',a_setfile(i_chn)(1:60),' ',i_setrows(i_chn),i_setcols(i_chn)
              else
                write(6,*) 'Error opening: ',a_setfile(i_chn)(1:60),
     &              i_chn,i_fbytes
                stop ' '
              end if

              i_pos = 0
              do while (index(a_setfile(i_chn)(i_pos+1:),'/') .ne. 0)
                i_pos = i_pos + index(a_setfile(i_chn)(i_pos+1:),'/')
              end do
              if (a_filename .eq. ' ') then
                a_filename = a_setfile(i_chn)(i_pos+1:)
              else
                if (a_filename .ne. a_setfile(i_chn)(i_pos+1:)) then ! Only show filename once if same for all channels
                  a_filename=a_filename(1:max(rdflen(a_filename),1))/
     &                 /', '//a_setfile(i_chn)(i_pos+1:)
                end if
              end if


c
c     Compute data stats
c

              if (.false.) then ! disable mean and std calc on raw data file

                if (i_debug .ge. 3) write(6,'(1x,a,i3)') 'Computing set stats for set: ',i_chn
                if (i_debug .ge. 4) write(6,*) 'Number of rows/cols: ',i_setrows(i_chn)
     &           ,i_setcols(i_chn)

                i_err = 0
                i_cnt = 0
                r_sum = 0.
                r_sqr = 0.
                r_setvavg(i_chn) = 0.0
                r_setvstd(i_chn) = 0.0
                do i_row = 0,i_setrows(i_chn)-1,min(max(i_setrows(i_chn)/100,1),20000)
                  do i_col = 0, i_setcols(i_chn)-1, min(max(i_setcols(i_chn)/100,1),20000)
                    !@#$%
                      if (a_setfile(i_chn)(1:1) .ne. '=') then
                        call readdat(i_setunit(i_chn),
     &                          i_setrows(i_chn),
     &                          i_setcols(i_chn),
     &                          i_setshdr(i_chn),
     &                          i_setstlr(i_chn),
     &                          i_setrhdr(i_chn),
     &                          i_setrtlr(i_chn),
     &                          i_setchdr(i_chn),
     &                          i_setctlr(i_chn),
     &                          i_setvend(i_chn)*i_endian,
     &                          i_setvfmt(i_chn),
     &                          r_setvmlt(i_chn),
     &                          r_setvadr(i_chn),
     &                          r_setvmin(i_chn),
     &                          r_setvmax(i_chn),
     &                          b_setvnul(0,i_chn),
     &                          i_row,i_col-1,3,r_data,i_data,readfunc,i_err)
                      else
                        do j=0,2
                          r_data(j)=0
                          i_data(j)=0
                        end do
                        i_opr=1
                        i_flg=0
                        a_value = ' '
                        do i = 2,rdflen(a_setfile(i_chn))+1
                          if (index('+-*/x',a_setfile(i_chn)(i:i)) .gt. 0 .or. a_setfile(i_chn)(i:i) .eq. ' ') then
                            if (a_value .ne. ' ') then
                              read(a_value,*) r_value
                              if (i_opr .eq. 1) then
                                r_data(1)=r_data(1)+r_value
                              else if (i_opr .eq. 2) then
                                r_data(1)=r_data(1)-r_value
                              else if (i_opr .eq. 3) then
                                r_data(1)=r_data(1)*r_value
                              else if (i_opr .eq. 4) then
                                r_data(1)=r_data(1)/r_value
                              end if
                              a_value = ' '
                            end if
                          end if
                          if(a_setfile(i_chn)(i:i) .eq. '+') then
                            i_opr=1
                          else if (a_setfile(i_chn)(i:i) .eq. '-') then
                            i_opr=2
                          else if (a_setfile(i_chn)(i:i) .eq. 'x') then
                            i_opr=3
                          else if (a_setfile(i_chn)(i:i) .eq. '/') then
                            i_opr=4
                          else if (a_setfile(i_chn)(i:i) .eq. 's' .or. a_setfile(i_chn)(i:i) .eq. 'S') then
                            i_flg=1
                          else if (i_flg .eq. 1) then
                            i_flg = 0
                            i_tmp = max(min((ichar(a_setfile(i_chn)(i:i))-ichar('0')),i_set+1),1)
                            if (i_tmp .gt. i_set) i_tmp=0
                            if (i_tmp .ne. 0) then
                              call readdat(i_setunit(i_tmp),
     &                          i_setrows(i_tmp),
     &                          i_setcols(i_tmp),
     &                          i_setshdr(i_tmp),
     &                          i_setstlr(i_tmp),
     &                          i_setrhdr(i_tmp),
     &                          i_setrtlr(i_tmp),
     &                          i_setchdr(i_tmp),
     &                          i_setctlr(i_tmp),
     &                          i_setvend(i_tmp)*i_endian,
     &                          i_setvfmt(i_tmp),
     &                          r_setvmlt(i_tmp),
     &                          r_setvadr(i_tmp),
     &                          r_setvmin(i_tmp),
     &                          r_setvmax(i_tmp),
     &                          b_setvnul(0,i_tmp),
     &                          i_row,i_col-1,3,r_data2,i_data2,readfunc,i_err)
                              i_data(1)=i_data(1)+i_data2(1)
                              if (i_opr .eq. 1) then
                                r_data(1)=r_data(1)+r_data2(1)
                              else if (i_opr .eq. 2) then
                                r_data(1)=r_data(1)-r_data2(1)
                              else if (i_opr .eq. 3) then
                                r_data(1)=r_data(1)*r_data2(1)
                              else if (i_opr .eq. 4) then
                                r_data(1)=r_data(1)/r_data2(1)
                              else
                                r_data(1)=r_data(1)+r_data2(1)
                              end if
                            end if
                          else
                            a_value = a_value(1:max(1,rdflen(a_value)))//a_setfile(i_chn)(i:i)
                          end if
                        end do
                      end if
                   !@#$%
                    if (i_data(1) .eq. 0) then
                      i_cnt = i_cnt + 1
                      r_val(min(i_cnt,I_WKSPACE)) = r_data(1)
                      r_sum = r_sum + dble(r_data(1))
                      r_sqr = r_sqr + dble(r_data(1))**2
                    end if
                  end do
                end do

                if (i_cnt .gt. 0) then
                  r_setvavg(i_chn) = r_sum/max(i_cnt,1)
                  r_setvstd(i_chn) = sqrt(max(1.d-99,(r_sqr/max(i_cnt,1))
     &              -r_setvavg(i_chn)**2))
                  if (i_debug .ge. 3) write(6,*) 'avg/std = ',r_setvavg(i_chn),r_setvstd(i_chn),i_cnt
c                  call median(0.5,min(i_cnt,I_WKSPACE),r_val,r_median)
                  if (i_debug .ge. 4) write(6,*) 'average = ',r_setvavg(i_chn),i_cnt
                  if (i_debug .ge. 4) write(6,*) 'median  = ',r_median,i_cnt
                end if

              end if

            end if
          enddo


          do i_chn = 1,i_set
              if (a_setfile(i_chn)(1:1) .eq. '=') then
                if (i_setrows(i_chn) .eq. 0) i_setrows(i_chn) = i_setrows(1)
                if (i_setcols(i_chn) .eq. 0) i_setcols(i_chn) = i_setcols(1)
              end if
          end do

c
c     Set up row/column formats
c
          write(a_rowfrmt,'(a,i2.2,a)') '(i',min(10,max(1,int(1+alog10(float(i_setrows(1)))))),')'
          write(a_colfrmt,'(a,i2.2,a)') '(i',min(10,max(1,int(1+alog10(float(i_setcols(1)))))),')'
          if (i_debug .ge. 6) write(6,*) 'row/col fmt = ',a_rowfrmt,' ',a_colfrmt

c
c     Set some Color Table defaults
c
          do i=1,i_set
            a_value = rdflower(a_setname(i))
            if (a_dspctbl(i) .eq. ' ') then
              if (a_value(1:3) .eq. 'set') then
                if (i .eq. 1) a_dspctbl(i) = 'grey'
                if (i .eq. 2) a_dspctbl(i) = 'cmy'
                if (i .eq. 3) a_dspctbl(i) = 'bitmap'
              else if (a_value .eq. 'mag' .or.
     &               a_value .eq. 'rcs' .or.
     &               a_value .eq. 'amp' .or.
     &               a_value .eq. 'amplitude' .or.
     &               a_value .eq. 'magnitude' ) then
                a_dspctbl(i) = 'grey'
              else if (a_value .eq. 'dte' .or.
     &               a_value .eq. 'hgt' .or.
     &               a_value .eq. 'pha' .or.
     &               a_value .eq. 'height' .or.
     &               a_value .eq. 'phase' ) then
                a_dspctbl(i) = 'cmy'
              else if (a_value .eq. 'vv' ) then
                a_dspctbl(i) = 'blue'
              else if (a_value .eq. 'red' ) then
                a_dspctbl(i) = 'red'
              else if (a_value .eq. 'green' ) then
                a_dspctbl(i) = 'green'
              else if (a_value .eq. 'blue' ) then
                a_dspctbl(i) = 'blue'
              else if (a_value .eq. 'hh' ) then
                a_dspctbl(i) = 'green'
              else if (a_value .eq. 'airsar-dem') then
                a_dspctbl(i) = 'cmy'
              else if (a_value .eq. 'airsar-mag') then
                a_dspctbl(i) = 'grey'
              else if (a_value .eq. 'airsar-cor') then
                a_dspctbl(i) = 'grey'
              else if (a_value .eq. 'airsar-m11') then
                a_dspctbl(i) = 'grey'
              else
                a_dspctbl(i) = 'grey'
              end if
            end if
          end do

c
c     Initialize graphics
c
          do i_d=1,I_DMAX
            i_winactv(i_d)=0
          end do
          i_dsp=1
          i_winrows(i_dsp) = min(nint(i_setrows(1)*r_winzoom(i_dsp)),32000)
          i_wincols(i_dsp) = min(nint(i_setcols(1)*r_winzoom(i_dsp)),32000)
          i_winradr(i_dsp) = 0
          i_wincadr(i_dsp) = 0
          if (i_ponly .eq. 0) then
            call create_dsp(a_filename,i_winrows(i_dsp),i_wincols(i_dsp),i_winy,i_winx,
     &         a_setname(1),i_set,i_d,i_menu,a_tname,i_close,a_lcolor,i_debug)
            if (i_debug .ge. 6) write(6,*) 'i_dsp=',i_dsp
            if (i_d .lt. 1 .or. i_d .gt. I_DMAX) stop 'Error creating Display'
            call get_wininfo(i_d,1,i_vxo(i_d,1),i_vyo(i_d,1),i_vxs(i_d,1),
     &        i_vys(i_d,1),i_wxs(i_d,1),i_wys(i_d,1),i_widget)
c     &        i_vys(i_d,1),i_cw,i_ch,i_widget)
            if (i_debug .ge. 6) write(6,*) 'from get_win',i_vxo(i_d,1),i_vyo(i_d,1),i_vxs(i_d,1)
     &       ,i_vys(i_d,1)
            i_winactv(i_d) = 1
            do i=1, I_CMAX
              if (i .le. i_set) then
                if (i_dspactv(i_d,i) .lt. 0) then
                  i_dspactv(i_d,i) = 1
                end if
                call set_button_shadow(i_d,i+1,i_dspactv(i_d,i),i_debug)
c                call get_colortable(a_colordir,a_dspctbl(i),i_dspnumt(i),r_dspredt(0,i),r_dspgrnt(0,i),r_dspblut(0,i),i_debug)
              else
                i_dspactv(i_d,i) = 0
              end if
            end do
          else
            i_vxo(1,1)=0
            i_vyo(1,1)=0
            i_vxs(1,1)=i_wincols(1)
            i_vys(1,1)=i_winrows(1)
          end if
          do i=1,i_set
            call get_colortable(a_colordir,a_dspctbl(i),i_dspnumt(i),r_dspredt(0,i),r_dspgrnt(0,i),r_dspblut(0,i),i_debug)
          end do

c
c     Start Managing Window
c
c
c     Set up to Compute display stats
c

          do i_chn = 1,i_set
            i_event(0) = i_chn
            i_event(1) = 1
            i_event(2) = 11
            i_event(3) = 0
            i_event(4) = 0
            i_event(5) = 0
            i_event(6) = 0
            if (i_debug .ge. 6) write(6,*) 'i_bcnt2 =',i_bcnt
            call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
          end do

          if (i_ponly .ne. 0) then
            i_event(0) = 0    ! tells data to go to print file instead of screen
            i_event(1) = 1
            i_event(2) = 1
            i_event(3) = (i_vxo(1,1)*r_winzoom(0))/r_winzoom(1)
            i_event(4) = (i_vyo(1,1)*r_winzoom(0))/r_winzoom(1)
            i_event(5) = ((min(i_vxs(1,1)+i_vxo(1,1),i_wincols(1))-i_vxo(1,1))*
     &          r_winzoom(0))/r_winzoom(1)
            i_event(6) = ((min(i_vys(1,1)+i_vyo(1,1),i_winrows(1))-i_vyo(1,1))*
     &          r_winzoom(0))/r_winzoom(1)
            i_event(7) = 0
            i_event(8) = i_event(4)
            i_event(9) = i_event(6)
            do i_chn=1,i_set
              i_dspactv(0,i_chn) = 1
            end do
            call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
            i_dspselect=i_dsp
            if (i_debug .eq. -22 .or. i_debug .ge. 22) write(6,*) 'Printing:',i_event(3),i_event(4),i_event(5),i_event(6)
          else
            if (r_lat .ge. -2.0d0*r_pi .and. r_lon .ge. -2.0d0*r_pi) then
              r_eux(1)=r_lat
              r_eux(2)=r_lon
              r_eux(3)=0.0
c              write(6,*) 'r_eux=',r_eux
              call get_coordinates(a_setproj(1),r_setpegv(1,1),r_dnx,r_eux,2,i_debug,i_err)
c              i_rcenter=((r_dnx(1)-r_setradr(1))/r_setrmlt(1)-i_winradr(1) )*r_winzoom(1)
c              write(6,*) 'r_dnx=',r_dnx
c              write(6,*) 'real center=',((r_dnx(1)-r_setradr(1))/r_setrmlt(1)-i_winradr(1) ),((r_dnx(2)-r_setcadr(1))/r_setcmlt(1)-i_wincadr(1) )
              i_rcenter=((r_dnx(1)-r_setradr(1))/r_setrmlt(1)-i_winradr(1) )
              i_ccenter=((r_dnx(2)-r_setcadr(1))/r_setcmlt(1)-i_wincadr(1) )
            end if
            if (i_debug .eq. -9 .or. i_debug .ge. 9) write(6,*) 'moving scroll to',i_ccenter,i_rcenter
            i_d=1
            if (i_winradr(i_d).ne.max(min(i_rcenter-int((32000/2)/r_winzoom(i_d)),i_setrows(1)-int(32000/r_winzoom(i_d))),0))then
               i_winradr(i_d)=max(min(i_rcenter-int((32000/2)/r_winzoom(i_d)),i_setrows(1)-int(32000/r_winzoom(i_d))),0)
               i_redraw(i_d)=1
            end if
            if (i_wincadr(i_d).ne.max(min(i_ccenter-int((32000/2)/r_winzoom(i_d)),i_setcols(1)-int(32000/r_winzoom(i_d))),0))then
               i_wincadr(i_d)=max(min(i_ccenter-int((32000/2)/r_winzoom(i_d)),i_setcols(1)-int(32000/r_winzoom(i_d))),0)
               i_redraw(i_d)=1
            end if
            if (i_redraw(i_d) .eq. 1) then
               i_redraw(i_d) = 0
               i_event(0) = i_d   ! Redraw window
               i_event(1) = 1
               i_event(2) = 1
               i_event(3) = i_vxo(i_d,1)-5
               i_event(4) = i_vyo(i_d,1)-5
               i_event(5) = i_vxs(i_d,1)
               i_event(6) = i_vys(i_d,1)
               if (i_debug .ge. 6) write(6,*) 'i_bcnt2 =',i_bcnt
               call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
            end if
            i_cpos = nint((i_ccenter-i_wincadr(i_d))*r_winzoom(i_d)-0.5*i_vxs(i_d,1))
            i_rpos = nint((i_rcenter-i_winradr(i_d))*r_winzoom(i_d)-0.5*i_vys(i_d,1))
            call move_scroll(i_d,1,i_cpos,i_rpos)
          end if
          do while(i_done .eq. 0 .or. i_pinit .ne. 0)
            if (i_ponly .eq. 0) then
              i_event(0) = -1
            else
              i_event(0) = 0
            end if
            do while(i_event(0) .ne. 0 .and. i_done .eq. 0 .and. i_bcnt .lt. I_BMAX)
              call getevent(i_wait,i_event)
              if (i_debug .ge. 4) then
                if (i_event(0) .ne. 0 .and. i_event(2) .ne. 9 .and. i_debug .ge. 5) then
                  write(6,'(1x,a,7i10)')
     &                 'i_event=',i_event(0),i_event(1),i_event(2)
     &                 ,i_event(3),i_event(4),i_event(5),i_event(6)
                end if
              end if

              call buffer_cmd(i_event,i_bdat,i_bcnt,1,I_BMAX,i_abort,i_debug)
              if (i_debug .ge. 6) write(6,*) 'i_bcnt =',i_bcnt

              if (i_wait .eq. 0 .and. (i_bcnt .gt. 0 .or. i_ecnt .gt. 0)) then ! Update wait flag
                i_wait = 1
                iy = 0
                if (i_debug .ge. 7) write(6,*) 'turning wait off',i_wait
     &              ,i_bcnt
              end if
              if (i_wait .eq. 1 .and. (i_bcnt .eq. 0 .and. i_ecnt .eq. 0)) then
                i_wait = 0
                if (i_debug .ge. 7) write(6,*) 'turning wait on',i_wait
              end if
            end do

            if (i_bcnt .gt. 0) then ! Execute oldest action in buffer
              i_dsp = i_bdat(0,1)
              i_win = i_bdat(1,1)
              i_evn = i_bdat(2,1)

              if (i_dsp .lt. 0 .or. i_dsp .gt. I_DMAX) then
                                        ! do nothing
              else if (i_evn .eq. 1) then ! Expose Command
                if (i_win .eq. 1) then ! Window 1
                  i_col = i_bdat(3,1)
                  i_row = i_bdat(4,1)
                  i_ncx = i_bdat(5,1)
                  i_nrx = i_bdat(6,1) ! number of lines in expose event
                  if (i_ecnt .eq. I_EMAX) write(6,*) ' *** Warning *** - Too many expose commands to buffer, Skipping: ',i_col,i_row,i_ncx,i_nrx
                  i_ecnt=min(i_ecnt+1,I_EMAX)
                  do i=0,10
                    i_edat(i,i_ecnt)=i_bdat(i,1)
                  end do
                  i_scroll=0
                  if (i_ecnt .gt. 2) then ! Check if commands can be combined
                    if (i_edat(0,i_ecnt-1) .eq. i_bdat(0,1)) then ! Both events from the same display
                      if (i_edat(4,i_ecnt-1) .eq. i_bdat(4,1) .and.
     &                    i_edat(6,i_ecnt-1) .eq. i_bdat(6,1) ) then   ! Top and bottom edges line up
                        if (i_edat(3,i_ecnt-1)+i_edat(5,i_ecnt-1) .eq. i_bdat(3,1)) then   ! Scoll right
                          i_ecnt=i_ecnt-1
                          i_edat(5,i_ecnt) = i_edat(5,i_ecnt)+i_bdat(5,1)
                          i_scroll=0
                          if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'Merge Scroll Right  ',i_edat(3,i_ecnt),i_edat(5,i_ecnt)
                        else if (i_bdat(3,1)+i_bdat(5,1) .eq. i_edat(3,i_ecnt-1)) then ! Scoll left
                          i_ecnt=i_ecnt-1
                          i_edat(3,i_ecnt) = i_bdat(3,1)
                          i_edat(5,i_ecnt) = i_edat(5,i_ecnt)+i_bdat(5,1)
                          i_scroll=0
                          if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'Merge Scroll Left  ',i_edat(3,i_ecnt),i_edat(5,i_ecnt)
                        else
                          if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'Horizontal Scroll not contiguous',
     &                        i_edat(3,i_ecnt-1)+i_edat(5,i_ecnt-1),i_bdat(3,1),i_bdat(3,1)+i_bdat(5,1),i_edat(3,i_ecnt-1)
                        end if
                      else if (i_edat(3,i_ecnt-1) .eq. i_bdat(3,1) .and.
     &                    i_edat(5,i_ecnt-1) .eq. i_bdat(5,1) ) then   ! Left and right edges line upe
                        if (i_edat(4,i_ecnt-1)+i_edat(6,i_ecnt-1) .eq. i_bdat(4,1)) then   ! Scoll Down
                          i_ecnt=i_ecnt-1
                          i_edat(6,i_ecnt) = i_edat(6,i_ecnt)+i_bdat(6,1)
                          i_scroll=0
                          if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'Merge Scroll Down  ',i_edat(4,i_ecnt),i_edat(6,i_ecnt)
                        else if (i_bdat(4,1)+i_bdat(6,1) .eq. i_edat(4,i_ecnt-1)) then ! Scoll up
                          i_ecnt=i_ecnt-1
                          i_edat(4,i_ecnt) = i_bdat(4,1)
                          i_edat(6,i_ecnt) = i_edat(6,i_ecnt)+i_bdat(6,1)
                          i_scroll=1
                          if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'Merge Scroll Up  ',i_edat(4,i_ecnt),i_edat(6,i_ecnt)
                        else
                          if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'Vertical Scroll not contiguous',
     &                        i_edat(4,i_ecnt-1)+i_edat(6,i_ecnt-1),i_bdat(4,1),i_bdat(4,1)+i_bdat(6,1),i_edat(4,i_ecnt-1)
                        end if
                      end if
                    end if
                  end if
                  i_edat(7,i_ecnt)=i_scroll
                  i_edat(8,i_ecnt)=i_edat(4,i_ecnt)
                  i_edat(9,i_ecnt)=i_edat(6,i_ecnt)
                  do i=2,i_ecnt
                    i_d=i_edat(0,i)
                    i_w=i_edat(1,i)
                    if (i_d .gt. 0 .and. i_w .eq. 1) then
                      if (i_edat(3,i) .gt. i_vxo(i_d,i_w)+i_vxs(i_d,i_w)) then
                        i_edat(6,i) = -2  ! delete command
                        if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) '** Deleting expose: ',1
                      else if (i_edat(3,i)+i_edat(5,i) .lt. i_vxo(i_d,i_w)) then
                        i_edat(6,i) = -2 ! delete command
                        if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) '** Deleting expose: ',2
                      else if (i_edat(4,i) .gt. i_vyo(i_d,i_w)+i_vys(i_d,i_w)) then
                        i_edat(6,i) = -2  ! delete command
                        if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) '** Deleting expose: ',3
                      else if (i_edat(4,i)+i_edat(6,i) .lt. i_vyo(i_d,i_w)) then
                        i_edat(6,i) = -2 ! delete command
                        if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) '** Deleting expose: ',4
                      else
                        i_max=min(i_edat(3,i)+i_edat(5,i),i_vxo(i_d,i_w)+i_vxs(i_d,i_w)+1)
                        i_edat(3,i) = max(i_edat(3,i),i_vxo(i_d,i_w)-1)
                        i_edat(5,i) = i_max-i_edat(3,i)

c                       write(6,*) '***** ',i,i_edat(4,i),i_edat(6,i),i_vyo(i_d,i_w),i_vys(i_d,i_w)
                        i_max=min(i_edat(4,i)+i_edat(6,i),i_vyo(i_d,i_w)+i_vys(i_d,i_w)+1)
                        i_edat(4,i) = max(i_edat(4,i),i_vyo(i_d,i_w)-1)
                        i_edat(6,i) = i_max-i_edat(4,i)
                        i_edat(8,i)=i_edat(4,i)
                        i_edat(9,i)=i_edat(6,i)
c                       write(6,*) '***** ',i,i_edat(4,i),i_edat(6,i),i_vyo(i_d,i_w),i_vys(i_d,i_w)
                      end if
                    end if
                  end do
                if (i_debug .ge. 7) write(6,*) 'i_ecnt0= ',i_ecnt,
     &              i_bdat(1,i_ecnt),i_bdat(2,i_ecnt),i_bdat(3,i_ecnt),
     &              i_bdat(4,i_ecnt),i_bdat(5,i_ecnt),i_bdat(6,i_ecnt)
                end if
              else if (i_evn .eq. 2) then ! Configure window event
c               if (i_win .eq. 1 .and.. i_bdat(3,1) .lt. i_vxo(i_dsp,i_win)) then ! remember if scrolling up or down
c                 i_scroll = 1
c               else
c                 i_scroll = 0
c               end if
                i_vxo(i_dsp,i_win) = i_bdat(3,1) ! offset of viewport
                i_vyo(i_dsp,i_win) = i_bdat(4,1) ! offset of viewport
                i_wxs(i_dsp,i_win) = i_bdat(5,1) ! size of window
                i_wys(i_dsp,i_win) = i_bdat(6,1) ! size of window
                if (i_win .eq. 1) then
                  if (i_cset .le. 0 .and. i_cdsp .ge. 0) then
                    i_cdsp = -1
                    if (i_debug .ge. 8) write(6,*) '--Setting cdsp = -1'
                  end if
                  if (i_debug .ge. 8) write(6,*) 'i_cdsp,i_cset = ',i_cdsp,i_cset
                  if (i_debug .ge. 6) write(6,*) '** config1 =',i_dsp,i_win,i_bdat(3,1)
     &                 ,i_bdat(4,1),i_bdat(5,1),i_bdat(6,1)
                  if (i_debug .eq. -21 .and. i_win .eq. 1) write(6,*) 'vxo,vyo =',i_vxo(i_dsp,i_win),i_vyo(i_dsp,i_win)
                end if
              else if (i_evn .eq. 3) then ! Configure window event
                i_vxs(i_dsp,i_win) = i_bdat(5,1) ! size of viewport
                i_vys(i_dsp,i_win) = i_bdat(6,1) ! size of viewport
                if (i_win .eq. 1) then
                  if (i_cset .le. 0 .and. i_cdsp .ge. 0) then
                    i_cdsp = -1
                    if (i_debug .ge. 8) write(6,*) '--Setting cdsp = -1'
                  end if
                  if (i_debug .ge. 8) write(6,*) 'i_cdsp,i_cset = ',i_cdsp,i_cset
                  if (i_debug .ge. 6 .or. i_debug .eq. -6) write(6,*) '** config2 =',i_dsp,i_win,i_bdat(3,1),
     &                  i_bdat(4,1),i_bdat(5,1),i_bdat(6,1)
                  if (i_debug .eq. -21 .and. i_win .eq. 1) write(6,*) 'vxs,vys =',i_vxs(i_dsp,i_win),
     &                  i_vys(i_dsp,i_win),i_wxs(i_dsp,i_win),i_wys(i_dsp,i_win)
                end if

              else if (i_evn .eq. 4) then ! Click in window
                i_button = i_bdat(3,1)
                i_col    = i_bdat(4,1)
                i_row    = i_bdat(5,1)
                if (i_debug .eq. -12 .or. i_debug .ge. 12) write(6,*) 'press win/button=',i_win,i_button
c                write(6,*) 'click event: ',i_dsp,i_win,i_button
                if (i_win .eq. 0) then
                  ! do nothing
                else if (i_button .eq. 4 .and. i_win .eq. 1) then
                  if (i_key .eq. 0) then ! Scroll Bar up
                    if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*)
     &                    'Scroll up',i_vxo(i_dsp,1),min(i_wys(i_dsp,1),max(0,i_vyo(i_dsp,1)-int(0.2*i_vys(i_dsp,1))))
!                   call move_scroll(i_dsp,1,i_vxo(i_dsp,1),min(i_wys(i_dsp,1),max(0,i_vyo(i_dsp,1)-int(0.2*i_vys(i_dsp,1)))))
                    call move_scroll(i_dsp,1,i_vxo(i_dsp,1),max(0,i_vyo(i_dsp,1)-int(0.2*i_vys(i_dsp,1))))
                  else ! Scroll Bar left
!                   call move_scroll(i_dsp,1,min(i_wxs(i_dsp,1),max(0,i_vxo(i_dsp,1)-int(0.2*i_vxs(i_dsp,1)))),i_vyo(i_dsp,1))
                    call move_scroll(i_dsp,1,max(0,i_vxo(i_dsp,1)-int(0.2*i_vxs(i_dsp,1))),i_vyo(i_dsp,1))
                  end if
                else if (i_button .eq. 5 .and. i_win .eq. 1) then
                  if (i_key .eq. 0) then  ! Scroll Bar down
!                   write(6,*) 'xxx ',i_wys(i_dsp,1),i_vyo(i_dsp,1),i_vys(i_dsp,1)
                    if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*)
     &                    'Scroll down',i_vxo(i_dsp,1),min(i_wys(i_dsp,1),max(0,i_vyo(i_dsp,1)+int(0.2*i_vys(i_dsp,1))))
!                   call move_scroll(i_dsp,1,i_vxo(i_dsp,1),min(i_wys(i_dsp,1),max(0,i_vyo(i_dsp,1)+int(0.2*i_vys(i_dsp,1)))))
                    call move_scroll(i_dsp,1,i_vxo(i_dsp,1),max(0,i_vyo(i_dsp,1)+int(0.2*i_vys(i_dsp,1))))
                  else ! Scroll Bar right
!                   call move_scroll(i_dsp,1,min(i_wxs(i_dsp,1),max(0,i_vxo(i_dsp,1)+int(0.2*i_vxs(i_dsp,1)))),i_vyo(i_dsp,1))
                    call move_scroll(i_dsp,1,max(0,i_vxo(i_dsp,1)+int(0.2*i_vxs(i_dsp,1))),i_vyo(i_dsp,1))
                  end if
                else if (i_button .eq. 6 .and. i_win .eq. 1) then
                  if (i_key .ne. 0) then ! Scroll Bar up
                    if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*)
     &                    'Scroll up',i_vxo(i_dsp,1),min(i_wys(i_dsp,1),max(0,i_vyo(i_dsp,1)-int(0.2*i_vys(i_dsp,1))))
!                   call move_scroll(i_dsp,1,i_vxo(i_dsp,1),min(i_wys(i_dsp,1),max(0,i_vyo(i_dsp,1)-int(0.2*i_vys(i_dsp,1)))))
                    call move_scroll(i_dsp,1,i_vxo(i_dsp,1),max(0,i_vyo(i_dsp,1)-int(0.2*i_vys(i_dsp,1))))
                  else ! Scroll Bar left
!                   call move_scroll(i_dsp,1,min(i_wxs(i_dsp,1),max(0,i_vxo(i_dsp,1)-int(0.2*i_vxs(i_dsp,1)))),i_vyo(i_dsp,1))
                    call move_scroll(i_dsp,1,max(0,i_vxo(i_dsp,1)-int(0.2*i_vxs(i_dsp,1))),i_vyo(i_dsp,1))
                  end if
                else if (i_button .eq. 7 .and. i_win .eq. 1) then
                  if (i_key .ne. 0) then  ! Scroll Bar down
!                   write(6,*) 'xxx ',i_wys(i_dsp,1),i_vyo(i_dsp,1),i_vys(i_dsp,1)
                    if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*)
     &                    'Scroll down',i_vxo(i_dsp,1),min(i_wys(i_dsp,1),max(0,i_vyo(i_dsp,1)+int(0.2*i_vys(i_dsp,1))))
!                   call move_scroll(i_dsp,1,i_vxo(i_dsp,1),min(i_wys(i_dsp,1),max(0,i_vyo(i_dsp,1)+int(0.2*i_vys(i_dsp,1)))))
                    call move_scroll(i_dsp,1,i_vxo(i_dsp,1),max(0,i_vyo(i_dsp,1)+int(0.2*i_vys(i_dsp,1))))
                  else ! Scroll Bar right
!                   call move_scroll(i_dsp,1,min(i_wxs(i_dsp,1),max(0,i_vxo(i_dsp,1)+int(0.2*i_vxs(i_dsp,1)))),i_vyo(i_dsp,1))
                    call move_scroll(i_dsp,1,max(0,i_vxo(i_dsp,1)+int(0.2*i_vxs(i_dsp,1))),i_vyo(i_dsp,1))
                  end if
                else if (i_button .ge. 8) then  !
                  ! do nothing
                else if (i_win .eq. 1) then ! Click in window 1
c                  write(6,*) 'in window 1'
                  i_event(0) = i_dsp
                  i_event(1) = i_win
                  i_event(2) = 13
                  i_event(3) = i_button
                  i_event(4) = i_col
                  i_event(5) = i_row
                  i_event(6) = 1
c                  write(6,*) 'adding event to buffer =',i_bcnt,i_dsp,i_win,4,-i_button
                  call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                else if (i_win .eq. -1) then ! Click in label for window 1
                  if (i_key .eq. 0 .or. a_setproj(1) .eq. ' ' .or.
     &                           r_setrmlt(1) .eq. 0. .or. r_setcmlt(1) .eq. 0.) then
                    if (i_debug .ge. 5) write(6,*) 'i_bdat(3,1)=',i_bdat(3,1)
                    do i=0,20
                      a_elabl(i)=' '
                      a_edata(i)=' '
                    end do
                    a_elabl(1)='Enter Col, Row: '
                    a_edata(1)=' '
c                    do i=1,10
c                      write(86,*) a_elabl(i)
c                      do j=1,160
c                        write(86,*) ichar(a_elabl(i)(j:j)),' ',a_elabl(i)(j:j)
c                      end do
c                    end do
                    call entry_window(0,a_elabl,a_edata)
                    i_dspselect=i_dsp
                    a_elabl(1)=' '
                    a_edata(1)=' '
                  else
                    do i=0,20
                      a_elabl(i)=' '
                      a_edata(i)=' '
                    end do
                    a_elabl(2)='Enter Lat,Lon: '
                    a_edata(2)=' '
                    call entry_window(0,a_elabl,a_edata)
                    i_dspselect=i_dsp
                    a_elabl(2)=' '
                    a_edata(2)=' '
                  end if
                else if (i_win .ge. 2 .and. i_win .le. i_set+1) then  ! Click on set button  ! max(4,min(i_set+1+2*i_close,I_CMAX+2)) ) then ! Click on Buttons
                  if (i_shft .eq. 0) then
                    if (i_button .le. 0) then
                      if (i_debug .ge. 1) write(6,*) 'Button press error',i_button
                    else if (i_button .eq. 1) then
                      if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'i_dsp ',i_dsp
     &                      ,'   Button',i_win-1,' - left click',i_dsp
                      do i_chn=1,i_set
                        if (i_chn .eq. i_win-1) then
                          i_dspactv(i_dsp,i_win-1) = 1
                          call set_button_shadow(i_dsp,i_chn+1,1,i_debug)
                        else
                          i_dspactv(i_dsp,i_chn) = 0
                          call set_button_shadow(i_dsp,i_chn+1,0,i_debug)
                        end if
                      end do
                    else if (i_button .eq. 2) then
                      if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'i_dsp ',i_dsp
     &                      ,'   Button',i_win-1,' - middle click'
     &                      ,i_dsp
                      i_dspactv(i_dsp,i_win-1) = 1-i_dspactv(i_dsp,i_win-1)
                      if (i_dspactv(i_dsp,i_win-1) .eq. 1) then
                        call set_button_shadow(i_dsp,i_win,1,i_debug)
                      else
                        call set_button_shadow(i_dsp,i_win,0,i_debug)
                      end if
                    else if (i_button .eq. 3) then
                      if (i_debug .eq. -4 .or. i_debug .ge. 4) write(6,*) 'i_dsp ',i_dsp
     &                      ,'   Button',i_win-1,' - right click',i_dsp
                      i_chn=i_win-1
                      a_labels(0)=a_setname(i_chn)(1:max(1,rdflen(a_setname(i_chn))))//' Parameters'
                      if (i_dspmode(i_chn) .eq. 1) then
                        a_data(1)='1|Range|SDEV|PER|NORM|CW|Wrap'
                      else if (i_dspmode(i_chn) .eq. 2) then
                        a_data(1)='2|Range|SDEV|PER|NORM|CW|Wrap'
                      else if (i_dspmode(i_chn) .eq. 3) then
                        a_data(1)='3|Range|SDEV|PER|NORM|CW|Wrap'
                      else if (i_dspmode(i_chn) .eq. 4) then
                        a_data(1)='4|Range|SDEV|PER|NORM|CW|Wrap'
                      else if (i_dspmode(i_chn) .eq. 5) then
                        a_data(1)='5|Range|SDEV|PER|NORM|CW|Wrap'
                      else
                        a_data(1)='6|Range|SDEV|PER|NORM|CW|Wrap'
                      end if
c                      write(a_data(1),'(I10)') i_dspmode(i_chn)
                      if (i_dspmode(i_chn) .eq. 1) then
                        a_labels(2)='Range:'
                        write(a_data(2),'(f15.4)') r_dspmult(i_chn)
                        a_labels(3)='Offset:'
                        write(a_data(3),'(f15.4)') r_dspaddr(i_chn)
                      else if (i_dspmode(i_chn) .eq. 2) then
                        a_labels(2)='SDEV Factor:'
                        write(a_data(2),'(f15.2)') r_dspval1(i_chn)
                        a_labels(3)='Offset:'
                        write(a_data(3),'(f15.4)') r_dspaddr(i_chn)
                      else if (i_dspmode(i_chn) .eq. 3) then
                        a_labels(2)='Percent:'
                        write(a_data(2),'(f15.2)') r_dspval2(i_chn)
                        a_labels(3) = '|'
                        a_data(3)   = ' '
                      else if (i_dspmode(i_chn) .eq. 4) then
                        a_labels(2)=' '
                        write(a_data(2),'(f15.2)') r_dspmult(i_chn)
                        a_labels(3)='|'
                        a_data(3)  =' '
                        write(a_data(3),'(f15.4)') r_dspaddr(i_chn)
                      else if (i_dspmode(i_chn) .eq. 5) then
                        a_labels(2)='CW Scale:'
                        write(a_data(2),'(f15.2)') r_dspval3(i_chn)
                        a_labels(3) = '|'
                        a_data(3)   = ' '
                      else
                        a_labels(2)='Wrap:'
                        write(a_data(2),'(f15.4)') r_dspwrap(i_chn)
                        a_labels(3)='Offset:'
                        write(a_data(3),'(f15.4)') r_dspaddr(i_chn)
                      end if
                      call encodeval(r_dspexpn(i_chn),a_data(4))
                      call encodeval(r_setvmin(i_chn),a_data(5))
                      call encodeval(r_setvmax(i_chn),a_data(6))
                      call encodeval(r_dspcplw(i_chn),a_data(7))
                      call encodeval(r_dspcphi(i_chn),a_data(8))
c                      write(a_data(4),'(f15.4)') r_dspexpn(i_chn)
c                      write(a_data(5),'(f15.4)') r_setvmin(i_chn)
c                      write(a_data(6),'(f15.4)') r_setvmax(i_chn)
c                      write(a_data(7),'(f15.4)') r_dspcplw(i_chn)
c                      write(a_data(8),'(f15.4)') r_dspcphi(i_chn)
                      i_colorset=0
                      do i=1,i_colormax
                        if (a_dspctbl(i_chn) .eq. a_colorfile(i)) i_colorset=i
                      end do
                      if (i_colorset .gt. 0) then
                        write(a_data(9),'(i2)') i_colorset
                        do i=1,i_colormax
                          a_data(9)=a_data(9)(1:rdflen(a_data(9)))//'|'//a_colorname(i)
                        end do
                        if (a_data(9)(1:1) .eq. ' ') a_data(9)=a_data(9)(2:)
                      else
                        a_data(9)=a_dspctbl(i_chn)
                      end if
                      call entry_window(i_chn,a_labels,a_data)
c                      call entry_window(i_chn,a_labels,a_data)  ! Hack to get around some memory bug
                      if (i_win .eq. 5) then
c                          call mv_getfile(a_filename)
                      end if
                    end if
                    if (i_dsp .gt. 0) then ! Redraw window 1 if event from a display click
                      i_event(0) = i_dsp
                      i_event(1) = 1
                      i_event(2) = 1
                      i_event(3) = i_vxo(i_dsp,1)-5
                      i_event(4) = i_vyo(i_dsp,1)-5
                      i_event(5) = i_vxs(i_dsp,1)
                      i_event(6) = i_vys(i_dsp,1)
                      if (i_debug .ge. 6) write(6,*) 'i_bcnt2 =',i_bcnt
                      call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                    end if
                  else ! Create Colorbar
                    i_shft = 0
                    i_chn=i_win-1
                    a_file=a_workdir(1:rdflen(a_workdir))//'colorbar.agr'
                    open(unit=97,file=a_file,status='unknown',form='formatted')
                    do i=0,min(i_dspnumt(i_chn)-1,252)
                      write(97,'(5(a,i3),a)') '@map color ',i+2,' to (',
     &                 int(255*r_dspredt(i,i_chn)),', ',int(255*r_dspgrnt(i,i_chn)),', ',int(255*r_dspblut(i,i_chn)),'), "',i,' "'
                    end do
                    write(97,'(4(a,i3),a)') '@map color ',255,' to (',150,', ',150,', ',150,'), "Grey"'
c                    write(6,*) 'dspmult = ',r_dspmult(i_chn),r_dspmult(i_chn)/5,alog10(r_dspmult(i_chn)/5.)
c                    write(6,*) 'nintlog = ',nint(alog10(r_dspmult(i_chn)/5.))
c                    write(6,*) 'spacing = ',10.**nint(alog10(r_dspmult(i_chn)/5))
                    r_space = 10.**nint(alog10(r_dspmult(i_chn)/5))
                    if (r_space .gt. r_dspmult(i_chn)/4) r_space = r_space/2
                    if (r_space .gt. r_dspmult(i_chn)/4) r_space = r_space/2
                    if (r_space .gt. r_dspmult(i_chn)/4) r_space = r_space/2
                    if (r_space .lt. r_dspmult(i_chn)/8) r_space = r_space*2
                    if (r_space .lt. r_dspmult(i_chn)/8) r_space = r_space*2
                    if (r_space .lt. r_dspmult(i_chn)/8) r_space = r_space*2
c                    write(6,'(a,e15.4)') '@xaxis  tick major ',r_space

                    write(97,'(a)') '@version 50114'
                    write(97,'(a)') '@g0 on'
                    write(97,'(a)') '@with g0'
                    write(97,'(a,e15.4)') '@    world xmin ',r_dspaddr(i_chn)
                    write(97,'(a,e15.4)') '@    world xmax ',r_dspaddr(i_chn)+r_dspmult(i_chn)
                    write(97,'(a,e15.4)') '@xaxis  tick major ',r_space
                    write(97,'(a)') '@view xmin 0.10'
                    write(97,'(a)') '@view xmax 0.55'
                    write(97,'(a)') '@view ymin 0.85'
                    write(97,'(a)') '@view ymax 0.90'
                    write(97,'(a)') '@xaxis  on'
                    write(97,'(a)') '@yaxis  off'
                    write(97,'(a)') '@s0 symbol 2'
                    write(97,'(a)') '@s0 symbol size 0.2'
                    write(97,'(a)') '@s0 symbol fill color 1'
                    write(97,'(a)') '@s0 symbol fill pattern 1'
                    write(97,'(a)') '@s0 symbol linewidth 1.0'
                    write(97,'(a)') '@s0 symbol linestyle 0'
                    write(97,'(a)') '@s0 linestyle 0'
                    write(97,'(a)') '@s0 fill pattern 1'
                    write(97,'(a)') '@s0 line type 0'
                    write(97,'(a)') '@subtitle "Colorbar for '//a_setname(i_chn)(1:rdflen(a_setname(i_chn)))//'"'
                    write(97,'(a)') '@type xycolor'
                    do i=0,499
                      do j=0,100
                          r_value = max(r_dspcplw(i_chn),min(r_dspcphi(i_chn),i*r_dspmult(i_chn)/500+r_dspaddr(i_chn)))     ! Clip data
                          r_value = (r_value-r_dspaddr(i_chn))                              ! Shift data
                          if (i_dspmode(i_chn) .eq. 6) then                                 ! Wrap data
                            r_value = wrap(r_value,r_dspwrap(i_chn))
                          end if
                          r_value = r_value/r_dspmult(i_chn)                                ! Scale data
                          if (r_dspexpn(i_chn) .ne. 1.0) then                               ! Compress data
                            r_value = min(1.0,max(0.0,r_value))**r_dspexpn(i_chn)
                          end if
                          i_value = max(0,min(i_dspnumt(i_chn)-1,int(i_dspnumt(i_chn)*r_value)))
                        write(97,*) i*r_dspmult(i_chn)/500+r_dspaddr(i_chn),j/100.,min(i_value+2,254)
                      end do
                    end do
                    close(97)
                    a_command = 'xmgrace -noask -barebones -geometry 500x200 '//a_file(1:rdflen(a_file))//' &'
                    write(6,*) 'Displaying Colorbar for ',a_setname(i_chn)(1:rdflen(a_setname(i_chn)))
                    call system(a_command)

                  end if
                else if (i_win .eq.  max(4,min(i_set+1+2*i_close,I_CMAX+2)) ) then ! Click on close button
                  if (i_close .eq. 1) call destroy_display(i_dsp) ! Closes Display  change to: i_done = 1 if to quit whole program
                end if


              else if (i_evn .eq. 5) then ! button Release
                if (i_win .eq. 1 ) then ! button Release in window 1
                  i_button = i_bdat(3,1)
                  i_col    = i_bdat(4,1)
                  i_row    = i_bdat(5,1)
                  if (i_debug .eq. -12 .or. i_debug .ge. 12) write(6,*) 'release win/button=',i_win,i_button
                  if (i_win .eq. 1 .and. (i_button .eq. 1 .or. i_button .eq. 3)) then
                    i_event(0) = i_dsp
                    i_event(1) = i_win
                    i_event(2) = 13
                    i_event(3) = i_button
                    i_event(4) = i_col
                    i_event(5) = i_row
                    i_event(6) = 3
c                    write(6,*) 'adding event to buffer =',i_bcnt,i_dsp,i_win,4,-i_button
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                  end if
                end if
                i_button = 0
              else if (i_evn .eq. 6) then ! Key Press
                i_key = i_bdat(3,1)
                i_asc = i_bdat(6,1)
                if (i_debug .eq. -7 .or. i_debug .ge. 7) write(6,*) 'Key Press:   ',i_key,i_asc
                if (i_key .eq. 62 .or. i_asc .eq. 65507) then
                  i_cntl = 1
                else if (i_key .eq. 64 .or. i_asc .eq. 65505) then
                  i_shft = 1
                else if (i_pset .eq. 1) then
                  if (i_asc .eq. ichar('p') .or. i_asc .eq. ichar('P')) then
                    i_event(0)=i_dsp
                    i_event(1)=0
                    i_event(2)=0
                    i_event(3)=0
                    i_event(4)=6
                    i_event(5)=1
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                  else if (i_asc .eq. ichar('f') .or. i_asc .eq. ichar('F')) then
                    i_event(0)=i_dsp
                    i_event(1)=0
                    i_event(2)=0
                    i_event(3)=0
                    i_event(4)=6
                    i_event(5)=2
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                  else if (i_asc .eq. ichar('s') .or. i_asc .eq. ichar('S')) then
                    i_event(0)=i_dsp
                    i_event(1)=0
                    i_event(2)=0
                    i_event(3)=0
                    i_event(4)=6
                    i_event(5)=3
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                  end if
                  i_pset = 0
                  i_cntl = 0
                else if (i_sset .eq. 1) then
                  if (i_asc .eq. ichar('p') .or. i_asc .eq. ichar('P')) then
                    i_event(0)=i_dsp
                    i_event(1)=0
                    i_event(2)=0
                    i_event(3)=0
                    i_event(4)=5
                    i_event(5)=1
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                  else if (i_asc .eq. ichar('l') .or. i_asc .eq. ichar('L')) then
                    i_event(0)=i_dsp
                    i_event(1)=0
                    i_event(2)=0
                    i_event(3)=0
                    i_event(4)=5
                    i_event(5)=2
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                  else if (i_asc .eq. ichar('r') .or. i_asc .eq. ichar('R')) then
                    i_event(0)=i_dsp
                    i_event(1)=0
                    i_event(2)=0
                    i_event(3)=0
                    i_event(4)=5
                    i_event(5)=3
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                  else if (i_asc .eq. ichar('c') .or. i_asc .eq. ichar('C')) then
                    i_event(0)=i_dsp
                    i_event(1)=0
                    i_event(2)=0
                    i_event(3)=0
                    i_event(4)=5
                    i_event(5)=4
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                  end if
                  i_sset = 0
                  i_cntl = 0
                else if (i_cntl .eq. 1) then
                  if (i_asc .eq. ichar('a') .or. i_asc .eq. ichar('A')) then
                    i_event(0)=i_dsp
                    i_event(1)=0
                    i_event(2)=0
                    i_event(3)=0
                    i_event(4)=1
                    i_event(5)=1
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                    i_cntl = 0
                  else if (i_asc .eq. ichar('q') .or. i_asc .eq. ichar('Q')) then
                    i_event(0)=i_dsp
                    i_event(1)=0
                    i_event(2)=0
                    i_event(3)=0
                    i_event(4)=1
                    i_event(5)=2
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                  else if (i_asc .eq. ichar('d') .or. i_asc .eq. ichar('D')) then
                    i_event(0)=i_dsp
                    i_event(1)=0
                    i_event(2)=0
                    i_event(3)=0
                    i_event(4)=2
                    i_event(5)=1
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                  else if (i_asc .eq. ichar('k') .or. i_asc .eq. ichar('K')) then
                    i_event(0)=i_dsp
                    i_event(1)=0
                    i_event(2)=0
                    i_event(3)=0
                    i_event(4)=2
                    i_event(5)=2
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                    i_cntl = 0
                  else if (i_asc .eq. ichar('r') .or. i_asc .eq. ichar('R')) then   ! Resize Display
                    i_event(0)=i_dsp
                    i_event(1)=0
                    i_event(2)=0
                    i_event(3)=0
                    i_event(4)=2
                    i_event(5)=3
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                  else if (i_asc .eq. ichar('n') .or. i_asc .eq. ichar('N')) then
                    i_event(0)=i_dsp
                    i_event(1)=0
                    i_event(2)=0
                    i_event(3)=0
                    i_event(4)=4
                    i_event(5)=1
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                  else if (i_asc .eq. ichar('+') .or. i_asc .eq. ichar('=')) then
                    i_event(0)=i_dsp
                    i_event(1)=0
                    i_event(2)=0
                    i_event(3)=0
                    i_event(4)=4
                    i_event(5)=2
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                  else if (i_asc .eq. ichar('-') .or. i_asc .eq. ichar('_')) then
                    i_event(0)=i_dsp
                    i_event(1)=0
                    i_event(2)=0
                    i_event(3)=0
                    i_event(4)=4
                    i_event(5)=3
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                  else if ((i_asc .eq. ichar('1') .or. i_asc .eq. ichar('!')) .and. i_set .ge. 1) then ! shortcut to open display parameter window 1
                    i_event(0)=i_dsp
                    i_event(1)=2
                    i_event(2)=4
                    i_event(3)=3
                    i_event(4)=0
                    i_event(5)=0
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                  else if ((i_asc .eq. ichar('2') .or. i_asc .eq. ichar('@')) .and. i_set .ge. 2) then ! shortcut to open display parameter window 2
                    i_event(0)=i_dsp
                    i_event(1)=3
                    i_event(2)=4
                    i_event(3)=3
                    i_event(4)=0
                    i_event(5)=0
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                  else if ((i_asc .eq. ichar('3') .or. i_asc .eq. ichar('#')) .and. i_set .ge. 3) then ! shortcut to open display parameter window 3
                    i_event(0)=i_dsp
                    i_event(1)=4
                    i_event(2)=4
                    i_event(3)=3
                    i_event(4)=0
                    i_event(5)=0
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                  else if ((i_asc .eq. ichar('4') .or. i_asc .eq. ichar('$')) .and. i_set .ge. 4) then ! shortcut to open display parameter window 4
                    i_event(0)=i_dsp
                    i_event(1)=5
                    i_event(2)=4
                    i_event(3)=3
                    i_event(4)=0
                    i_event(5)=0
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                  else if ((i_asc .eq. ichar('5') .or. i_asc .eq. ichar('%')) .and. i_set .ge. 5) then ! shortcut to open display parameter window 5
                    i_event(0)=i_dsp
                    i_event(1)=6
                    i_event(2)=4
                    i_event(3)=3
                    i_event(4)=0
                    i_event(5)=0
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                  else if ((i_asc .eq. ichar('6') .or. i_asc .eq. ichar('^')) .and. i_set .ge. 6) then ! shortcut to open display parameter window 6
                    i_event(0)=i_dsp
                    i_event(1)=6
                    i_event(2)=4
                    i_event(3)=3
                    i_event(4)=0
                    i_event(5)=0
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                  else if (i_asc .eq. ichar('z') .or. i_asc .eq. ichar('Z')) then
                    i_event(0)=i_dsp
                    i_event(1)=0
                    i_event(2)=0
                    i_event(3)=0
                    i_event(4)=4
                    i_event(5)=4
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                    i_cntl = 0
                  else if (i_asc .eq. ichar('p') .or. i_asc .eq. ichar('P')) then
                    i_pset=1
                    if (i_debug .gt. 9) write(6,*) 'i_pset = ',i_pset,i_key,i_asc
                  else if (i_asc .eq. ichar('s') .or. i_asc .eq. ichar('S')) then  ! select menu set
                    i_sset=1
                    if (i_debug .gt. 9) write(6,*) 'i_sset = ',i_sset,i_key,i_asc
                  else if (i_asc .eq. ichar('t') .or. i_asc .eq. ichar('T')) then  ! tool menu set
                    i_tset=1
                    if (i_debug .gt. 9) write(6,*) 'i_tset = ',i_tset,i_key,i_asc
                  end if
                else if (i_asc .eq. ichar('c') .or. i_asc .eq. ichar('C')) then
                  i_samps=0
                  i_redraw(i_dsp) = 1
                  if (i_redraw(i_dsp) .eq. 1) then
                    i_redraw(i_dsp) = 0
                    i_event(0) = i_dsp   ! Redraw window
                    i_event(1) = 1
                    i_event(2) = 1
                    i_event(3) = i_vxo(i_dsp,1)-5
                    i_event(4) = i_vyo(i_dsp,1)-5
                    i_event(5) = i_vxs(i_dsp,1)
                    i_event(6) = i_vys(i_dsp,1)
                    if (i_debug .ge. 6) write(6,*) 'i_bcnt2 =',i_bcnt
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                  end if
                else if (i_asc .eq. ichar('d') .or. i_asc .eq. ichar('D')) then
                  i_show=1-i_show
                  do i=0,20
                    a_elabl(i)=' '
                    a_edata(i)=' '
                  end do
                  a_elabl(6)='Display: '
                  write(a_edata(6),'(i1,a)') i_show+1,'|Off|On'  ! Sample Display ON/Off
                  a_elabl(7)='Mode: '
                  write(a_edata(7),'(i1,a)') i_smode+1,'|None|Point|Line|Region'
                  call entry_window(0,a_elabl,a_edata)
                  i_dspselect=i_dsp
                  a_elabl(6)=' '
                  a_edata(6)=' '
                  a_elabl(7)=' '
                  a_edata(7)=' '
                  i_event(0) = i_dsp   ! Redraw window
                  i_event(1) = 1
                  i_event(2) = 1
                  i_event(3) = i_vxo(i_dsp,1)-5
                  i_event(4) = i_vyo(i_dsp,1)-5
                  i_event(5) = i_vxs(i_dsp,1)
                  i_event(6) = i_vys(i_dsp,1)
                  if (i_debug .ge. 6) write(6,*) 'i_bcnt2 =',i_bcnt
                  call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                else if (i_asc .eq. ichar('n') .or. i_asc .eq. ichar('N')) then
                  i_smode=0
                  do i=0,20
                    a_elabl(i)=' '
                    a_edata(i)=' '
                  end do
                  a_elabl(6)='Display: '
                  write(a_edata(6),'(i1,a)') i_show+1,'|Off|On'  ! Sample Display ON/Off
                  a_elabl(7)='Mode: '
                  write(a_edata(7),'(i1,a)') i_smode+1,'|None|Point|Line|Region'
                  call entry_window(0,a_elabl,a_edata)
                  i_dspselect=i_dsp
                  a_elabl(6)=' '
                  a_edata(6)=' '
                  a_elabl(7)=' '
                  a_edata(7)=' '
                else if (i_asc .eq. ichar('p') .or. i_asc .eq. ichar('P')) then
                  i_smode=1
                  i_show=1
                  do i=0,20
                    a_elabl(i)=' '
                    a_edata(i)=' '
                  end do
                  a_elabl(6)='Display: '
                  write(a_edata(6),'(i1,a)') i_show+1,'|Off|On'  ! Sample Display ON/Off
                  a_elabl(7)='Mode: '
                  write(a_edata(7),'(i1,a)') i_smode+1,'|None|Point|Line|Region'
                  call entry_window(0,a_elabl,a_edata)
                  i_dspselect=i_dsp
                  a_elabl(6)=' '
                  a_edata(6)=' '
                  a_elabl(7)=' '
                  a_edata(7)=' '
                else if (i_asc .eq. ichar('l') .or. i_asc .eq. ichar('L')) then
                  i_smode=2
                  i_show=1
                  do i=0,20
                    a_elabl(i)=' '
                    a_edata(i)=' '
                  end do
                  a_elabl(6)='Display: '
                  write(a_edata(6),'(i1,a)') i_show+1,'|Off|On'  ! Sample Display ON/Off
                  a_elabl(7)='Mode: '
                  write(a_edata(7),'(i1,a)') i_smode+1,'|None|Point|Line|Region'
                  call entry_window(0,a_elabl,a_edata)
                  i_dspselect=i_dsp
                  a_elabl(6)=' '
                  a_edata(6)=' '
                  a_elabl(7)=' '
                  a_edata(7)=' '
                else if (i_asc .eq. ichar('r') .or. i_asc .eq. ichar('R')) then
                  i_smode=3
                  i_show=1
                  do i=0,20
                    a_elabl(i)=' '
                    a_edata(i)=' '
                  end do
                  a_elabl(6)='Display: '
                  write(a_edata(6),'(i1,a)') i_show+1,'|Off|On'  ! Sample Display ON/Off
                  a_elabl(7)='Mode: '
                  write(a_edata(7),'(i1,a)') i_smode+1,'|None|Point|Line|Region'
                  call entry_window(0,a_elabl,a_edata)
                  i_dspselect=i_dsp
                  a_elabl(6)=' '
                  a_edata(6)=' '
                  a_elabl(7)=' '
                  a_edata(7)=' '
                end if
              else if (i_evn .eq. 7) then ! Key Release
                if (i_bdat(3,1) .eq. 62 .or. i_bdat(6,1) .eq. 65507) i_cntl = 0
                if (i_bdat(3,1) .eq. 64 .or. i_bdat(6,1) .eq. 65505) i_shft = 0
                if (i_debug .eq. -7 .or. i_debug .ge. 7) write(6,*) 'Key Release: ',i_bdat(3,1),i_bdat(6,1)
c                i_cntl = 0
                i_key = 0
                i_asc = 0
              else if (i_evn .eq. 8) then ! Destroy Window event
                if (i_debug .ge. 6) write(6,*) '*** Window Destroyed: ',i_dsp,i_win
                i_winactv(i_dsp) = 0
                do ib = 1,i_bcnt ! Clear out any remaining event in buffer for destroyed window
                  if (i_bdat(0,ib) .eq. i_dsp .and. i_bdat(1,ib) .eq. i_win) then
                    do ie = 0,10
                      i_bdat(ie,ib) = 0
                    end do
                  end if
                end do
                i_done = 1
                do i_d=1,I_DMAX
                  if (i_winactv(i_d) .eq. 1) i_done = 0
                end do
              else if (i_evn .eq. 9) then ! Mouse motion
                i_button = nint(i_bdat(3,1)/256.)
                if (i_debug .eq. -12 .or. i_debug .ge. 12) write(6,*) 'button in motion = ',i_bdat(3,1),nint(i_bdat(3,1)/256.)
                if (i_button .eq. 4) then
                  i_button = 3
                end if
                i_col    = i_bdat(4,1)
                i_row    = i_bdat(5,1)
                if (i_debug .eq. -12 .or. i_debug .ge. 12) write(6,*) 'motion win/button=',i_win,i_button
                if (i_win .eq. 1 .and. (i_button .eq. 1 .or. i_button .eq. 3)) then
                  i_event(0) = i_dsp
                  i_event(1) = i_win
                  i_event(2) = 13
                  i_event(3) = i_button
                  i_event(4) = i_col
                  i_event(5) = i_row
                  i_event(6) = 2
c                  write(6,*) 'i_bcnt2 =',i_bcnt
                  call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                end if
              else if (i_evn .eq. 10) then ! Entry Window change
                i_chn    = i_bdat(0,1)
                if (i_chn .lt. 0) write(6,*) '*** i_chn Error ***   ',i_chn
                i_field  = i_bdat(4,1)
                i_value  = i_bdat(5,1)
                i_msgid  = i_bdat(6,1)
                call get_message(i_msgid,a_message)
                if (i_chn .gt. 0) then
                  if (i_field .ne. 9) then
                    if (index(a_message,'.') .eq. 0 .and. index(a_message,'*') .eq. 0 .and.
     &                 rdflen(a_message) .gt. 0) a_message=a_message(1:rdflen(a_message))//'.'
                  end if
                  if (i_debug .ge. 6) write(6,*) 'channel=',i_chn,'   field=',i_field,
     &                                           '   ival=',i_value,'  msg=',a_message(1:30)
                  if (i_field .eq. 1) then
                    i_ewupdate=1
                    if (i_value .eq. 1) then                       ! Range Mode
                      i_dspmult(i_chn) = 0
                      i_dspaddr(i_chn) = 0
                      i_dspmode(i_chn)=i_value
                    else if (i_value .eq. 2) then                  ! SDEV Mode
                      if (r_dspval1(i_chn) .eq. 0) r_dspval1(i_chn)=2.0
                      if (i_dspmode(i_chn) .eq. 2) then
                        i_dspaddr(i_chn) = 1
                      end if
                      i_dspmult(i_chn)=1
                      i_dspmode(i_chn)=i_value
                    else if (i_value .eq. 3) then                  ! Percent Mode
                      if (r_dspval2(i_chn) .eq. 0) r_dspval2(i_chn)=90.0
                      i_dspaddr(i_chn)=1
                      i_dspmult(i_chn)=1
                      i_dspmode(i_chn)=i_value
                    else if (i_value .eq. 5) then                  ! CW Mode
                      if (r_dspval3(i_chn) .eq. 0) r_dspval3(i_chn)=1.0
                      r_dspaddr(i_chn)=0
                      i_dspaddr(i_chn)=0
                      i_dspmult(i_chn)=1
                      i_dspmode(i_chn)=i_value
                    else if (i_value .eq. 6) then
                      if (i_dspmode(i_chn) .ne. 6) then
                        r_dspwrap(i_chn) = r_dspmult(i_chn)
                        r_dspaddr(i_chn) = 0
                        i_dspmult(i_chn) = 0
                        i_dspaddr(i_chn) = 0
                      end if
                      i_dspmode(i_chn)=i_value
                    end if
                  else if (i_field .eq. 2) then
                    call decodeval(a_message,r_value,i_err)
c                    read(a_message,*,iostat=i_err) r_value
                    if (i_err .ne. 0) then
                      i_ewupdate = 1
                    else
                      if (i_dspmode(i_chn) .eq. 1) then
                        if (r_value .ne. 0.) then
                          r_dspmult(i_chn) = r_value
                          i_dspmult(i_chn) = 0
                        else
                          i_dspmult(i_chn) = 1
                          r_dspval1(i_chn) = 2
                        end if
                      else if (i_dspmode(i_chn) .eq. 2) then
                        if (r_value .ne. 0. ) then
                          r_dspval1(i_chn) = r_value
                          i_dspmult(i_chn) = 1
                        else
                          if (r_dspval1(i_chn) .eq. 0) r_dspval1(i_chn)=2.0
                        end if
                      else if (i_dspmode(i_chn) .eq. 3) then
                        if (r_value .ne. 0.) then
                          r_dspval2(i_chn) = r_value
                          i_dspmult(i_chn) = 1
                        else
                          if (r_dspval2(i_chn) .eq. 0) r_dspval2(i_chn)=90.0
                        end if
                      else if (i_dspmode(i_chn) .eq. 4) then
                         ! undefined
                      else if (i_dspmode(i_chn) .eq. 5) then
                        if (r_value .ne. 0.) then
                          r_dspval3(i_chn) = r_value
                          i_dspmult(i_chn) = 1
                        else
                          if (r_dspval3(i_chn) .eq. 0) r_dspval1(i_chn)=1.0
                        end if
                      else if (i_dspmode(i_chn) .eq. 6) then
                        if (r_value .ne. 0.) r_dspwrap(i_chn) = r_value
                        r_dspmult(i_chn)=r_dspwrap(i_chn)
                      end if
                    end if
                  else if (i_field .eq. 3) then
                    call decodeval(a_message,r_value,i_err)
                    if (i_err .ne. 0) then
                      i_ewupdate = 1
                    else
c                     if (r_value .ne. r_dspaddr(i_chn)) i_ewupdate = 1
                      r_dspaddr(i_chn) = r_value
                      i_dspaddr(i_chn)=0
                    end if
                  else if (i_field .eq. 4) then
                    call decodeval(a_message,r_value,i_err)
                    if (i_err .ne. 0) then
                      i_ewupdate = 1
                    else
c                     if (r_value .ne. r_dspexpn(i_chn)) i_ewupdate = 1
                      r_dspexpn(i_chn) = r_value
                    end if
                  else if (i_field .eq. 5) then
                    call decodeval(a_message,r_value,i_err)
                    if (i_err .ne. 0) then
                      i_ewupdate = 1
                    else
c                     if (r_value .ne. r_setvmin(i_chn)) i_ewupdate = 1
                      r_setvmin(i_chn) = r_value
                    end if
                  else if (i_field .eq. 6) then
                    call decodeval(a_message,r_value,i_err)
                    if (i_err .ne. 0) then
                      i_ewupdate = 1
                    else
c                     if (r_value .ne. r_setvmax(i_chn)) i_ewupdate = 1
                      r_setvmax(i_chn) =  r_value
                    end if
                  else if (i_field .eq. 7) then
                    call decodeval(a_message,r_value,i_err)
                    if (i_err .ne. 0) then
                      i_ewupdate = 1
                    else
c                     if (r_value .ne. r_dspcplw(i_chn)) i_ewupdate = 1
                      r_dspcplw(i_chn) = r_value
                    end if
                  else if (i_field .eq. 8) then
                    call decodeval(a_message,r_value,i_err)
                    if (i_err .ne. 0) then
                      i_ewupdate = 1
                    else
c                     if (r_value .ne. r_dspcphi(i_chn)) i_ewupdate = 1
                      r_dspcphi(i_chn) = r_value
                    end if
                  else if (i_field .eq. 9) then
c                    if (a_message .eq. ' ') then
                    if (i_value .ne. 0) then
                      i_colorset = i_value
                      if (i_colorset .gt. 0 .and. i_colorset .le. i_colormax) then
                        if (a_colorfile(i_colorset) .ne. a_dspctbl(i_chn)) i_ewupdate = 1
                        a_dspctbl(i_chn)=a_colorfile(i_colorset)
                        if (i_colorset .eq. 1) a_dspctbl(i_chn)=' '
                        i_colorset = 0
                      end if
                    else
                      if (a_message .eq. ' ') a_message='?'
                      if (a_message .ne. a_dspctbl(i_chn)) i_ewupdate = 1
                      a_dspctbl(i_chn)=a_message
                      if (index(a_dspctbl(i_chn),' - not found.  Using grey') .gt. 1) then
                        a_dspctbl(i_chn)=a_dspctbl(i_chn)(1:index(a_dspctbl(i_chn),' - not found.  Using grey')-1)
                      end if
                    end if
                    call get_colortable(a_colordir,a_dspctbl(i_chn),i_dspnumt(i_chn),
     &                    r_dspredt(0,i_chn),r_dspgrnt(0,i_chn),r_dspblut(0,i_chn),i_debug)
                  end if
c                 if (i_dspmult(i_chn) .eq. 1 .or. i_dspaddr(i_chn) .eq. 1) then
                  if (1 .eq. 1) then
                    i_event(0) = i_chn
                    i_event(1) = 1
                    i_event(2) = 11
                    i_event(3) = 0
                    i_event(4) = 0
                    i_event(5) = 0
                    i_event(6) = 0
                    if (i_debug .ge. 6) write(6,*) 'Going to recompute mean/Std   i_bcnt2 =',i_bcnt
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                  end if
                  if (1 .eq. i_ewupdate) then        ! Add command to buffer to re-draw entry window
                    i_event(0) = 0
                    i_event(1) = i_chn+1
                    i_event(2) = 4
                    i_event(3) = 3
                    i_event(4) = 0
                    i_event(5) = 0
                    i_event(6) = 0
                    if (i_debug .ge. 6) write(6,*) 'Going to redraw entry window   i_bcnt2 =',i_bcnt
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                    i_ewupdate = 0
                  end if
                  do i_d=1,I_DMAX
                    if (i_debug .ge. 6) write(6,*) 'i_winactv=',i_winactv(i_d),i_d
                    if (i_winactv(i_d) .eq. 1) then
                      i_event(0) = i_d
                      i_event(1) = 1
                      i_event(2) = 1
                      i_event(3) = i_vxo(i_d,1)-5
                      i_event(4) = i_vyo(i_d,1)-5
                      i_event(5) = i_vxs(i_d,1)
                      i_event(6) = i_vys(i_d,1)
                      if (i_debug .ge. 6) write(6,*) 'i_bcnt2 =',i_bcnt
                      call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                    end if
                  end do
c              write(a_data(1),'(i10)') i_dspmode(i_chn)
c                 if (i_dspmode(i_chn) .eq. 1) then
c                   a_labels(2)='Range:'
c                   write(a_data(2),'(f15.4)') r_dspmult(i_chn)
c              else if (i_dspmode(i_chn) .eq. 2) then
c                   a_labels(2)='SDEV Factor:'
c                    write(a_data(2),'(f15.2)') r_dspval1(i_chn)
c                 else
c                   a_labels(2)='Wrap:'
c                   write(a_data(2),'(f15.4)') r_dspwrap(i_chn)
c              end if
c              write(a_data(3),'(f15.4)') r_dspaddr(i_chn)
c              write(a_data(4),'(f15.4)') r_setvmin(i_chn)
c              write(a_data(5),'(f15.4)') r_setvmax(i_chn)
c              write(a_data(6),'(f15.4)') r_dspcplw(i_chn)
c              write(a_data(7),'(f15.4)') r_dspcphi(i_chn)
c              a_data(8)=a_dspctbl(i_chn)
c                 call entry_window(i_chn,a_labels,a_data)
                 else
                   if (i_field .eq. 1) then
                     read(a_message,*,iostat=i_err) i_col,i_row
                     if (i_err .eq. 0) then
                       i_rcenter=i_row
                       i_ccenter=i_col
                       i_cdsp=i_dspselect
c                      write(6,*) 'i_key=',i_key
                       if (i_debug .ge. 9) write(6,*) 'moving scroll to',i_ccenter,i_rcenter
                       if (i_winactv(i_dspselect) .ne. 0) then
                         if (i_winradr(i_dspselect) .ne.
     &                        max(min(i_rcenter-int((32000/2)/r_winzoom(i_dspselect)),
     &                           i_setrows(1)-int(32000/r_winzoom(i_dspselect))),0) ) then
                           i_winradr(i_dspselect) =
     &                        max(min(i_rcenter-int((32000/2)/r_winzoom(i_dspselect)),
     &                           i_setrows(1)-int(32000/r_winzoom(i_dspselect))),0)
                           i_redraw(i_dspselect)=1
                         end if
                         if (i_wincadr(i_dspselect) .ne.
     &                        max(min(i_ccenter-int((32000/2)/r_winzoom(i_dspselect)),
     &                           i_setcols(1)-int(32000/r_winzoom(i_dspselect))),0) ) then
                           i_wincadr(i_dspselect) =
     &                        max(min(i_ccenter-int((32000/2)/r_winzoom(i_dspselect)),
     &                           i_setcols(1)-int(32000/r_winzoom(i_dspselect))),0)
                           i_redraw(i_dspselect)=1
                         end if
                         if (i_redraw(i_dspselect) .eq. 1) then
                           i_redraw(i_dspselect) = 0
                           i_event(0) = i_dspselect   ! Redraw window
                           i_event(1) = 1
                           i_event(2) = 1
                           i_event(3) = i_vxo(i_dspselect,1)-5
                           i_event(4) = i_vyo(i_dspselect,1)-5
                           i_event(5) = i_vxs(i_dspselect,1)
                           i_event(6) = i_vys(i_dspselect,1)
                           if (i_debug .ge. 6) write(6,*) 'i_bcnt2 =',i_bcnt
                           call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                         end if
                         i_cpos = nint((i_ccenter-i_wincadr(i_dspselect))*r_winzoom(i_dspselect)-0.5*i_vxs(i_dspselect,1))
                         i_rpos = nint((i_rcenter-i_winradr(i_dspselect))*r_winzoom(i_dspselect)-0.5*i_vys(i_dspselect,1))
                         call move_scroll(i_dspselect,1,i_cpos,i_rpos)
                       end if
                     end if
                   else if (i_field .eq. 2) then
                     read(a_message,*,iostat=i_err) r_lat,r_lon
                     if (i_err .eq. 0) then
                       r_eux(1)=r_lat/r_rtod
                       r_eux(2)=r_lon/r_rtod
                       r_eux(3)=0.0
                       call get_coordinates(a_setproj(1),r_setpegv(1,1),r_dnx,r_eux,2,i_debug,i_err)
                       i_rcenter=((r_dnx(1)-r_setradr(1))/r_setrmlt(1)-i_winradr(1) )
                       i_ccenter=((r_dnx(2)-r_setcadr(1))/r_setcmlt(1)-i_wincadr(1) )
                       i_cdsp=i_dspselect
c                      write(6,*) 'i_key=',i_key
                       if (i_debug .eq. -9 .or. i_debug .ge. 9) write(6,*) 'moving scroll to',i_ccenter,i_rcenter
                       if (i_winactv(i_dspselect) .ne. 0) then
                         if (i_winradr(i_dspselect) .ne.
     &                        max(min(i_rcenter-int((32000/2)/r_winzoom(i_dspselect)),
     &                           i_setrows(1)-int(32000/r_winzoom(i_dspselect))),0) ) then
                           i_winradr(i_dspselect) =
     &                        max(min(i_rcenter-int((32000/2)/r_winzoom(i_dspselect)),
     &                           i_setrows(1)-int(32000/r_winzoom(i_dspselect))),0)
                           i_redraw(i_dspselect)=1
                         end if
                         if (i_wincadr(i_dspselect) .ne.
     &                        max(min(i_ccenter-int((32000/2)/r_winzoom(i_dspselect)),
     &                           i_setcols(1)-int(32000/r_winzoom(i_dspselect))),0) ) then
                           i_wincadr(i_dspselect) =
     &                        max(min(i_ccenter-int((32000/2)/r_winzoom(i_dspselect)),
     &                           i_setcols(1)-int(32000/r_winzoom(i_dspselect))),0)
                           i_redraw(i_dspselect)=1
                         end if
                         if (i_redraw(i_dspselect) .eq. 1) then
                           i_redraw(i_dspselect) = 0
                           i_event(0) = i_dspselect   ! Redraw window
                           i_event(1) = 1
                           i_event(2) = 1
                           i_event(3) = i_vxo(i_dspselect,1)-5
                           i_event(4) = i_vyo(i_dspselect,1)-5
                           i_event(5) = i_vxs(i_dspselect,1)
                           i_event(6) = i_vys(i_dspselect,1)
                           if (i_debug .eq. -6 .or. i_debug .ge. 6) write(6,*) 'i_bcnt2 =',i_bcnt
                           call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                         end if
                         i_cpos = nint((i_ccenter-i_wincadr(i_dspselect))*r_winzoom(i_dspselect)-0.5*i_vxs(i_dspselect,1))
                         i_rpos = nint((i_rcenter-i_winradr(i_dspselect))*r_winzoom(i_dspselect)-0.5*i_vys(i_dspselect,1))
                         call move_scroll(i_dspselect,1,i_cpos,i_rpos)
                       end if
                     end if
                   else if (i_field .eq. 3) then
                     read(a_message,*,iostat=i_err) r_winzoom(0)
                     if (r_winzoom(0) .lt. 0) r_winzoom(0) = abs(1./r_winzoom(0))
                   else if (i_field .eq. 4) then
                     if (i_value .eq. 1) then
                       write(6,*) 'Print Format PPM'
                     else if (i_value .eq. 2) then
                       write(6,*) 'Raw RGB Not Supported - Using PPM'
                     else if (i_value .eq. 3) then
                       write(6,*) 'PostScript Not Supported - Using PPM'
                     else
                       write(6,*) 'Print Format Not Supported - Using PPM'
                     end if
                   else if (i_field .eq. 5) then
                     if (i_debug .ge. 8) write(6,*) '--i_cdsp,i_cset = ',i_cdsp,i_cset
                     if (i_cdsp .ne. 0 .and. i_cdsp .ne. i_dspselect) then
                       i_rcenter=nint((i_vyo(i_dspselect,1)+
     &                           0.5*i_vys(i_dspselect,1))/r_winzoom(i_dspselect))+i_winradr(i_dspselect)
                       i_ccenter=nint((i_vxo(i_dspselect,1)+
     &                           0.5*i_vxs(i_dspselect,1))/r_winzoom(i_dspselect))+i_wincadr(i_dspselect)
                     end if
                     if (i_debug .ge. 6) write(6,*) 'Center was at (col,row): ',i_ccenter,i_rcenter
                     read(a_message,*,iostat=i_err) r_winzoom(i_dspselect)
                     if (r_winzoom(i_dspselect) .lt. 0.) r_winzoom(i_dspselect) = abs(1./r_winzoom(i_dspselect))
                     i_winrows(i_dspselect) = min(nint(i_setrows(1)*r_winzoom(i_dspselect)),32000)
                     i_wincols(i_dspselect) = min(nint(i_setcols(1)*r_winzoom(i_dspselect)),32000)
                     call resize_win(i_dspselect,1,i_wincols(i_dspselect),i_winrows(i_dspselect))
                     if (i_winradr(i_dspselect) .ne.
     &                   max(min(i_rcenter-int((32000/2)/r_winzoom(i_dspselect)),
     &                   i_setrows(1)-int(32000/r_winzoom(i_dspselect))),0)) then
                       i_winradr(i_dspselect) =
     &                   max(min(i_rcenter-int((32000/2)/r_winzoom(i_dspselect)),
     &                   i_setrows(1)-int(32000/r_winzoom(i_dspselect))),0)
                       i_redraw(i_dspselect)=1
                     end if
                     if (i_wincadr(i_dspselect) .ne.
     &                   max(min(i_ccenter-int((32000/2)/r_winzoom(i_dspselect)),
     &                   i_setcols(1)-int(32000/r_winzoom(i_dspselect))),0)) then
                       i_wincadr(i_dspselect) =
     &                   max(min(i_ccenter-int((32000/2)/r_winzoom(i_dspselect)),
     &                   i_setcols(1)-int(32000/r_winzoom(i_dspselect))),0)
                       i_redraw(i_dspselect)=1
                     end if
                     if (i_redraw(i_dspselect) .eq. 1) then
                       i_redraw(i_dspselect) = 0
                       i_event(0) = i_dspselect   ! Redraw window
                       i_event(1) = 1
                       i_event(2) = 1
                       i_event(3) = i_vxo(i_dspselect,1)-5
                       i_event(4) = i_vyo(i_dspselect,1)-5
                       i_event(5) = i_vxs(i_dspselect,1)
                       i_event(6) = i_vys(i_dspselect,1)
                       if (i_debug .ge. 6) write(6,*) 'i_bcnt2 =',i_bcnt
                       call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                     end if
                     i_cpos = nint((i_ccenter-i_wincadr(i_dspselect))*r_winzoom(i_dspselect)-0.5*i_vxs(i_dspselect,1))
                     i_rpos = nint((i_rcenter-i_winradr(i_dspselect))*r_winzoom(i_dspselect)-0.5*i_vys(i_dspselect,1))
                     if (i_debug .ge. 9) write(6,*) 'moving scroll to',i_cpos,i_rpos
                     call move_scroll(i_dspselect,1,i_cpos,i_rpos)
                     i_cset = 1
                     i_event(0) = i_dspselect
                     i_event(1) = 1
                     i_event(2) = 1
                     i_event(3) = i_vxo(i_dspselect,1)-5
                     i_event(4) = i_vyo(i_dspselect,1)-5
                     i_event(5) = i_vxs(i_dspselect,1)
                     i_event(6) = i_vys(i_dspselect,1)
                     if (i_debug .ge. 6) write(6,*) 'i_bcnt2 =',i_bcnt
                     call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                     if (r_winzoom(i_dspselect) .ge. 1.0) then
                       write(a_label,'(a,i3,a)') 'ZOOM:',nint(r_winzoom(i_dspselect)),'x'
                     else
                       write(a_label,'(a,i3,a)') 'ZOOM:',-nint(1.0/r_winzoom(i_dspselect)),'x'
                     end if
                     call display_label(i_dspselect,1,a_label,1)
                   else if (i_field .eq. 6) then  ! Sample Display ON/Off
                     if (i_value .eq. 1) then
                       i_show = 0
                     else
                       i_show = 1
                     end if
                     i_event(0) = i_dspselect   ! Redraw window
                     i_event(1) = 1
                     i_event(2) = 1
                     i_event(3) = i_vxo(i_dspselect,1)-5
                     i_event(4) = i_vyo(i_dspselect,1)-5
                     i_event(5) = i_vxs(i_dspselect,1)
                     i_event(6) = i_vys(i_dspselect,1)
                     if (i_debug .ge. 6) write(6,*) 'i_bcnt2 =',i_bcnt
                     call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                   else if (i_field .eq. 7) then  ! Select Mode
                     i_smode = i_value-1
                   else if (i_field .eq. 8) then  ! Sample Display ON/Off
                     if (i_smode .lt. 3) then
                       read(a_message,*,iostat=i_err) r_wdth
                     else
                       read(a_message,*,iostat=i_err) r_spce
                     end if
                   end if
                 end if
              else if (i_evn .eq. 11) then ! Re-scale set
                i_chn    = abs(i_bdat(0,1))
                if (i_debug .ge. 3) write(6,'(1x,a,i3)') 'Computing display stats for set: ',i_chn
                i_err = 0
                i_cnt = 0
                r_sum = 0.
                r_sqr = 0.
                do i_row = 0,i_setrows(i_chn)-1,min(max(i_setrows(i_chn)/100,1),20000)
                  do i_col = 0, i_setcols(i_chn)-1, min(max(i_setcols(i_chn)/100,1),20000)
                      if (a_setfile(i_chn)(1:1) .ne. '=') then
                        call readdat(i_setunit(i_chn),
     &                          i_setrows(i_chn),
     &                          i_setcols(i_chn),
     &                          i_setshdr(i_chn),
     &                          i_setstlr(i_chn),
     &                          i_setrhdr(i_chn),
     &                          i_setrtlr(i_chn),
     &                          i_setchdr(i_chn),
     &                          i_setctlr(i_chn),
     &                          i_setvend(i_chn)*i_endian,
     &                          i_setvfmt(i_chn),
     &                          r_setvmlt(i_chn),
     &                          r_setvadr(i_chn),
     &                          r_setvmin(i_chn),
     &                          r_setvmax(i_chn),
     &                          b_setvnul(0,i_chn),
     &                          i_row,i_col-1,3,r_data,i_data,readfunc,i_err)
                      else
c                       write(6,*) 'i_row,col=',i_row,i_col
                        do j=0,2
                          r_data(j)=0
                          i_data(j)=0
                        end do
                        i_opr=1
                        i_flg=0
                        a_value = ' '
                        do i = 2,rdflen(a_setfile(i_chn))+1
                          if (index('+-*/x',a_setfile(i_chn)(i:i)) .gt. 0 .or. a_setfile(i_chn)(i:i) .eq. ' ') then
                            if (a_value .ne. ' ') then
                              read(a_value,*) r_value
                              if (i_opr .eq. 1) then
                                r_data(1)=r_data(1)+r_value
                              else if (i_opr .eq. 2) then
                                r_data(1)=r_data(1)-r_value
                              else if (i_opr .eq. 3) then
                                r_data(1)=r_data(1)*r_value
                              else if (i_opr .eq. 4) then
                                r_data(1)=r_data(1)/r_value
                              end if
                              a_value = ' '
                            end if
                          end if
                          if(a_setfile(i_chn)(i:i) .eq. '+') then
                            i_opr=1
                          else if (a_setfile(i_chn)(i:i) .eq. '-') then
                            i_opr=2
                          else if (a_setfile(i_chn)(i:i) .eq. 'x') then
                            i_opr=3
                          else if (a_setfile(i_chn)(i:i) .eq. 's' .or. a_setfile(i_chn)(i:i) .eq. 'S') then
                            i_flg=1
                          else if (i_flg .eq. 1) then
                            i_flg = 0
                            i_tmp = max(min((ichar(a_setfile(i_chn)(i:i))-ichar('0')),i_set+1),1)
                            i_tmp = max(min((ichar(a_setfile(i_chn)(i:i))-48),i_set+1),0)
c                            write(6,*) 'reading ',i_tmp,i_opr,i_data(1)
                            if (i_tmp .gt. i_set) i_tmp=0
                            if (i_tmp .ne. 0) then
                              call readdat(i_setunit(i_tmp),
     &                          i_setrows(i_tmp),
     &                          i_setcols(i_tmp),
     &                          i_setshdr(i_tmp),
     &                          i_setstlr(i_tmp),
     &                          i_setrhdr(i_tmp),
     &                          i_setrtlr(i_tmp),
     &                          i_setchdr(i_tmp),
     &                          i_setctlr(i_tmp),
     &                          i_setvend(i_tmp)*i_endian,
     &                          i_setvfmt(i_tmp),
     &                          r_setvmlt(i_tmp),
     &                          r_setvadr(i_tmp),
     &                          r_setvmin(i_tmp),
     &                          r_setvmax(i_tmp),
     &                          b_setvnul(0,i_tmp),
     &                          i_row,i_col-1,3,r_data2,i_data2,readfunc,i_err)
                              i_data(1)=i_data(1)+i_data2(1)
                              if (i_opr .eq. 1) then
                                r_data(1)=r_data(1)+r_data2(1)
                              else if (i_opr .eq. 2) then
                                r_data(1)=r_data(1)-r_data2(1)
                              else if (i_opr .eq. 3) then
                                r_data(1)=r_data(1)*r_data2(1)
                              else
                                r_data(1)=r_data(1)+r_data2(1)
                              end if
                            end if
                          else
                            a_value = a_value(1:max(1,rdflen(a_value)))//a_setfile(i_chn)(i:i)
                          end if
                        end do
                      end if
                   !@#$%
                    if (i_data(1) .eq. 0) then
                      i_cnt = i_cnt + 1
                      i_c = 1
                      r_data(i_c) = max(r_dspcplw(i_chn),min(r_dspcphi(i_chn ! Clip data
     &                       ),r_data(i_c)))
c                        if (r_dspwrap(i_chn) .ne. 0.0) r_data(i_c) = ! Wrap data
c     &                       wrap(r_data(i_c),r_dspwrap(i_chn))
                      r_val(min(i_cnt,I_WKSPACE)) = r_data(i_c)
                      r_sum = r_sum + dble(r_data(i_c))
                      r_sqr = r_sqr + dble(r_data(i_c))**2.0d0
                    end if
                  end do
                end do
                if (i_debug .ge. 5) write(6,*) 'i_cnt,r_sum,r_sqr = ',i_cnt,r_sum,r_sqr
                if (i_cnt .gt. 0) then
                  r_avg = r_sum/max(i_cnt,1)
                  r_std = sqrt(max(1.d-99,(r_sqr/max(i_cnt,1))-(r_avg)**2))
                  r_setvavg(i_chn)=r_avg
                  r_setvstd(i_chn)=r_std
                  if (i_debug .ge. 4) write(6,*) 'average = ',r_setvavg(i_chn),i_cnt
                  if (i_dspmode(i_chn) .eq. 3) then
                    call median(   (1.-(r_dspval2(i_chn)/100))/2.,min(i_cnt,I_WKSPACE),r_val,r_median)
                    r_dspaddr(i_chn)=r_median
                    call median(1.-(1.-(r_dspval2(i_chn)/100))/2.,min(i_cnt,I_WKSPACE),r_val,r_median)
                    r_dspmult(i_chn)=r_median-r_dspaddr(i_chn)
                    if (i_debug .ge. 4) write(6,*) 'median = ',r_dspaddr(i_chn),r_median,i_cnt
                  else if (i_dspmode(i_chn) .eq. 5) then
                    r_dspaddr(i_chn)=0.
                    r_dspmult(i_chn)=r_avg/(0.7*r_dspval3(i_chn))
                  else
                    if (i_dspaddr(i_chn) .eq. 1) r_dspaddr(i_chn) = r_avg-(r_dspval1(i_chn)*r_std)
                    if (i_dspmult(i_chn) .eq. 1) r_dspmult(i_chn) = 2.*r_dspval1(i_chn)*r_std
                  end if
                else
                  r_dspaddr(i_chn) = 0.0d0
                  r_dspmult(i_chn) = 1.0d0
                end if
                if (i_debug .ge. 3) write(6,*) 'dsp add/mult = ',r_dspaddr(i_chn),r_dspmult(i_chn)
                if (i_bdat(0,1) .lt. 0) then
                  write(a_data(1),'(i10)') i_dspmode(i_chn)
                  if (i_dspmode(i_chn) .eq. 1) then
                    a_labels(2)='Range:'
                    write(a_data(2),'(f15.4)') r_dspmult(i_chn)
                    a_labels(3)='Offset:'
                    write(a_data(3),'(f15.4)') r_dspaddr(i_chn)
                  else if (i_dspmode(i_chn) .eq. 2) then
                    a_labels(2)='SDEV Factor:'
                    write(a_data(2),'(f15.2)') r_dspval1(i_chn)
                    a_labels(3)='Offset:'
                    write(a_data(3),'(f15.4)') r_dspaddr(i_chn)
                  else if (i_dspmode(i_chn) .eq. 3) then
                    a_labels(2)='Percent:'
                    write(a_data(2),'(f15.2)') r_dspval2(i_chn)
                    a_labels(3) = ' '
                    a_data(3)   = ' '
                  else if (i_dspmode(i_chn) .eq. 4) then
                    a_labels(2)=' '
                    write(a_data(2),'(f15.2)') r_dspmult(i_chn)
                    a_labels(3)=' '
                    write(a_data(3),'(f15.4)') r_dspaddr(i_chn)
                  else if (i_dspmode(i_chn) .eq. 5) then
                    a_labels(2)='CW Scale:'
                    write(a_data(2),'(f15.2)') r_dspval3(i_chn)
                    a_labels(3) = ' '
                    a_data(3)   = ' '
                  else
                    a_labels(2)='Wrap:'
                    write(a_data(2),'(f15.4)') r_dspwrap(i_chn)
                    a_labels(3)='Offset:'
                    write(a_data(3),'(f15.4)') r_dspaddr(i_chn)
                  end if
                  call encodeval(r_dspexpn(i_chn),a_data(4))
                  call encodeval(r_setvmin(i_chn),a_data(5))
                  call encodeval(r_setvmax(i_chn),a_data(6))
                  call encodeval(r_dspcplw(i_chn),a_data(7))
                  call encodeval(r_dspcphi(i_chn),a_data(8))
c                  write(a_data(4),'(f15.4)') r_dspexpn(i_chn)
c                  write(a_data(5),'(f15.4)') r_setvmin(i_chn)
c                  write(a_data(6),'(f15.4)') r_setvmax(i_chn)
c                  write(a_data(7),'(f15.4)') r_dspcplw(i_chn)
c                  write(a_data(8),'(f15.4)') r_dspcphi(i_chn)
                  i_colorset=0
                  do i=1,i_colormax
                    if (a_dspctbl(i_chn) .eq. a_colorfile(i)) i_colorset=i
                  end do
                  if (i_colorset .gt. 0) then
                    write(a_data(9),'(i2)') i_colorset
                    do i=1,i_colormax
                      a_data(9)=a_data(9)(1:rdflen(a_data(9)))//'|'//a_colorname(i)
                    end do
                    if (a_data(9)(1:1) .eq. ' ') a_data(9)=a_data(9)(2:)
                  else
                    a_data(9)=a_dspctbl(i_chn)
                  end if
                  call entry_window(i_chn,a_labels,a_data)
                end if
              else if (i_evn .eq. 12) then ! file name
                i_value  = i_bdat(5,1)
                i_msgid  = i_bdat(6,1)
                if (i_msgid .ge. 0) then
                  call get_message(i_msgid,a_message)
                else
                  a_message=a_ptsfile
                  a_ptsfile=' '
                end if
                if (i_debug .eq. -14 .or. i_debug .ge. 14) write(6,*) 'File message = ',i_msgid,i_value,' ',a_message
                if (i_value .eq. 43) then ! import points file
                  open(91,file=a_message,status='old',form='formatted',iostat=i_err)
                  i_samps=0
                  do while(i_err .eq. 0)
                    read(91,'(3i8,5e15.5)',iostat=i_err) i_csamps(i_samps+1),i_rsamps(i_samps+1),i_tsamps(i_samps+1)
                    if (i_err .eq. 0) then
                      i_samps=i_samps+1

                      do i_chn=1,i_set
                        if (1 .eq. 1) then
                          if (a_setfile(i_chn)(1:1) .ne. '=') then
                            call readdat(i_setunit(i_chn),
     &                              i_setrows(i_chn),
     &                              i_setcols(i_chn),
     &                              i_setshdr(i_chn),
     &                              i_setstlr(i_chn),
     &                              i_setrhdr(i_chn),
     &                              i_setrtlr(i_chn),
     &                              i_setchdr(i_chn),
     &                              i_setctlr(i_chn),
     &                              i_setvend(i_chn)*i_endian,
     &                              i_setvfmt(i_chn),
     &                              r_setvmlt(i_chn),
     &                              r_setvadr(i_chn),
     &                              r_setvmin(i_chn),
     &                              r_setvmax(i_chn),
     &                              b_setvnul(0,i_chn),
     &                              i_rr,i_cc-1,3,
     &                              r_data,i_data,readfunc,i_err)
                          else
                            do j=0,2
                              r_data(j)=0
                              i_data(j)=0
                            end do
                            i_opr=1
                            i_flg=0
                            a_value = ' '
                            do iii = 2,rdflen(a_setfile(i_chn))+1
                              if (index('+-*/x',a_setfile(i_chn)(iii:iii)) .gt. 0 .or. a_setfile(i_chn)(iii:iii) .eq. ' ') then
                                if (a_value .ne. ' ') then
                                  read(a_value,*) r_value
                                  if (i_opr .eq. 1) then
                                    r_data(1)=r_data(1)+r_value
                                  else if (i_opr .eq. 2) then
                                    r_data(1)=r_data(1)-r_value
                                    else if (i_opr .eq. 3) then
                                      r_data(1)=r_data(1)*r_value
                                    else if (i_opr .eq. 4) then
                                      r_data(1)=r_data(1)/r_value
                                    end if
                                    a_value = ' '
                                  end if
                                end if
                                if(a_setfile(i_chn)(iii:iii) .eq. '+') then
                                  i_opr=1
                                else if (a_setfile(i_chn)(iii:iii) .eq. '-') then
                                  i_opr=2
                                else if (a_setfile(i_chn)(iii:iii) .eq. 'x') then
                                  i_opr=3
                                else if (a_setfile(i_chn)(iii:iii) .eq. '/') then
                                  i_opr=4
                                else if (a_setfile(i_chn)(iii:iii) .eq. 's' .or. a_setfile(i_chn)(iii:iii) .eq. 'S') then
                                  i_flg=1
                                else if (i_flg .eq. 1) then
                                  i_flg = 0
                                  i_tmp = max(min((ichar(a_setfile(i_chn)(iii:iii))-ichar('0')),i_set+1),1)
                                  if (i_tmp .gt. i_set) i_tmp=0
                                  if (i_tmp .ne. 0) then
                                    call readdat(i_setunit(i_tmp),
     &                                i_setrows(i_tmp),
     &                                i_setcols(i_tmp),
     &                                i_setshdr(i_tmp),
     &                                i_setstlr(i_tmp),
     &                                i_setrhdr(i_tmp),
     &                                i_setrtlr(i_tmp),
     &                                i_setchdr(i_tmp),
     &                                i_setctlr(i_tmp),
     &                                i_setvend(i_tmp)*i_endian,
     &                                i_setvfmt(i_tmp),
     &                                r_setvmlt(i_tmp),
     &                                r_setvadr(i_tmp),
     &                                r_setvmin(i_tmp),
     &                                r_setvmax(i_tmp),
     &                                b_setvnul(0,i_tmp),
     &                                i_rr,i_cc-1,3,
     &                                r_data2,i_data2,readfunc,i_err)
                                      i_data(1)=i_data(1)+i_data2(1)
                                    if (i_opr .eq. 1) then
                                      r_data(1)=r_data(1)+r_data2(1)
                                    else if (i_opr .eq. 2) then
                                      r_data(1)=r_data(1)-r_data2(1)
                                    else if (i_opr .eq. 3) then
                                      r_data(1)=r_data(1)*r_data2(1)
                                    else if (i_opr .eq. 4) then
                                      r_data(1)=r_data(1)/r_data2(1)
                                    else
                                      r_data(1)=r_data(1)+r_data2(1)
                                    end if
                                  end if
                                else
                                  a_value = a_value(1:max(1,rdflen(a_value)))//a_setfile(i_chn)(iii:iii)
                              end if
                            end do
                          end if
                        end  if
                        if (i_samps .gt. 0) r_vsamps(i_samps,i_chn)=r_data(1)
                      end do
                    end if

                  end do
                  close(91)
                  if (i_dspselect .eq. 0) then
c                    write(6,*) 'dspselect = ',i_dspselect,'  setting to 1'
                    i_dspselect =1
                  end if
                  i_show=1
                  i_event(0) = i_dspselect   ! Redraw window
                  i_event(1) = 1
                  i_event(2) = 1
                  i_event(3) = i_vxo(i_dspselect,1)-5
                  i_event(4) = i_vyo(i_dspselect,1)-5
                  i_event(5) = i_vxs(i_dspselect,1)
                  i_event(6) = i_vys(i_dspselect,1)
                  if (i_debug .eq. -6 .or. i_debug .ge. 6) write(6,*) 'i_bcnt2 =',i_bcnt
                  call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                else if (i_value .eq. 44) then
                  open(91,file=a_message,status='unknown',form='formatted',iostat=i_err)
                  if (i_err .eq. 0) then
                    do i=1,i_samps
                      write(91,'(3i8,5e15.5)') i_csamps(i),i_rsamps(i),i_tsamps(i),(r_vsamps(i,i_chn),i_chn=1,i_set)
                    end do
                  end if
                  close(91)
                end if
              else if (i_evn .eq. 13) then ! buffered mouse click event in window 1
                i_button = i_bdat(3,1)
                i_col    = i_bdat(4,1)
                i_row    = i_bdat(5,1)
                i_typ    = i_bdat(6,1)
                if (i_win .eq. 1) then ! Just making sure
                  if (i_button .eq. 0) then
                                        ! do nothing
                  else if (i_button .ne. 2) then ! Left or right click
c                    write(6,*) 'left or right click'
                    i_rcenter=i_row/r_winzoom(i_dsp)+i_winradr(i_dsp)
                    i_ccenter=i_col/r_winzoom(i_dsp)+i_wincadr(i_dsp)
                    if (a_clickcmd(1) .ne. ' ' .and. i_typ .eq. 1) then
                      write(a_command,'(a,4i8,a)') a_clickcmd(1)(1:max(1,rdflen(a_clickcmd(1)))),i_button,i_ccenter,i_rcenter,
     &                    i_typ,' &'
                      if (i_debug .eq. -17 .or. i_debug .ge. 17) write(6,*) 'cmnd:'//a_command(1:70)
                        call system(a_command)
                      end if
                    if (i_debug .eq. -12 .or. i_debug .ge. 12) write(6,*) '   in event 13  (button/typ): ',i_button,i_typ
                    i_cdsp = i_dsp
                    i_cset = 0
                    a_label1=' '
                    a_label2=' '
                    if (i_smode .ne. 0 .and. i_shft .eq. 0 .and. i_typ .eq. 1 .and. (i_region .eq. 0 .or. i_smode .ne. 3)) then
                      i_samps=0
                      i_redraw(i_dsp)=1
                      if (i_redraw(i_dsp) .eq. 1) then
                        i_redraw(i_dsp) = 0
                        i_event(0) = i_dsp   ! Redraw window
                        i_event(1) = 1
                        i_event(2) = 1
                        i_event(3) = i_vxo(i_dsp,1)-5
                        i_event(4) = i_vyo(i_dsp,1)-5
                        i_event(5) = i_vxs(i_dsp,1)
                        i_event(6) = i_vys(i_dsp,1)
                        if (i_debug .ge. 6) write(6,*) 'i_bcnt2 =',i_bcnt
                        call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                      end if
                    end if

                    if (i_samps .eq. 0) then
                      i_newpoint = 1
                    else
                      if (i_samps .eq. 0 .or. i_typ .ne. 2 .or.
     &                        i_rsamps(i_samps) .ne. i_rcenter .or.
     &                        i_csamps(i_samps) .ne. i_ccenter) then
                        i_newpoint = 1
                      else
                        i_newpoint = 0
                      end if
                    end if
                    if (i_newpoint .eq. 1) then
                      i_start=i_samps+1
                      if (i_smode .eq. 0) then
                        ! do nothing
                      else if (i_smode .eq. 1) then   ! Point
                        if (i_button .eq. 1 .or. i_typ .eq. 3) then
                          i_samps = min(i_samps+1,I_MAXSAMP)
                          i_rsamps(i_samps) = i_rcenter
                          i_csamps(i_samps) = i_ccenter
                          i_tsamps(i_samps) = 1
                          r_wsamps(i_samps) = r_wdth
                          r_ssamps(i_samps) = r_spce
                        end if
                      else if (i_smode .eq. 2) then   ! Line
                        if (i_button .eq. 1 .or. i_typ .ne. 2) then
                          i_samps = min(i_samps+1,I_MAXSAMP)
                          i_rsamps(i_samps) = i_rcenter
                          i_csamps(i_samps) = i_ccenter
                          r_wsamps(i_samps) = r_wdth
                          r_ssamps(i_samps) = r_spce
                          if (i_typ .eq. 1) then
                            i_tsamps(i_samps) = 2
                            if (i_debug .eq. -12 .or. i_debug .ge. 12) write(6,*) '   starting line: ',i_samps,i_tsamps(i_samps)
                          else
                            i_tsamps(i_samps) = -2
                            if (i_debug .eq. -12 .or. i_debug .ge. 12) write(6,*) '   continue line: ',i_samps,i_tsamps(i_samps)
                          end if
                        end if
                      else if (i_smode .eq. 3) then   ! Region
c                        write(6,*) 'i_button = ',i_button
                        if (i_region .eq. 0) then
                          if ((i_button .eq. 1 .and. i_typ .eq. 3) .or.
     &                        (i_button .eq. 3 .and. i_typ .eq. 1)  ) then
                            i_samps=min(i_samps+1,I_MAXSAMP)
                            i_rsamps(i_samps)=i_rcenter
                            i_csamps(i_samps)=i_ccenter
                            i_tsamps(i_samps)=+3
                            r_wsamps(i_samps) = r_wdth
                            r_ssamps(i_samps) = r_spce
                            i_region=i_region+1
                            if (i_debug .eq. -12 .or. i_debug .ge. 12) write(6,*) '   start region: ',i_region
                          end if
                        else if (i_typ .eq. 3) then
                          if (i_button .eq. 1) then
                            i_samps=min(i_samps+1,I_MAXSAMP)
                            i_rsamps(i_samps)=i_rcenter
                            i_csamps(i_samps)=i_ccenter
                            i_tsamps(i_samps)=-3
                            r_wsamps(i_samps) = r_wdth
                            r_ssamps(i_samps) = r_spce
                            i_region=i_region+1
                            if (i_debug .eq. -12 .or. i_debug .ge. 12) write(6,*) '   add region:   ',i_region
                          else if (i_button .eq. 3) then
                            if (i_region .eq. 1) then
                              i_samps=min(i_samps+1,I_MAXSAMP)
                              i_rsamps(i_samps)=i_rsamps(i_samps-i_region)
                              i_csamps(i_samps)=i_ccenter
                              i_tsamps(i_samps)=-3
                              r_wsamps(i_samps) = r_wdth
                              r_ssamps(i_samps) = r_spce
                              i_region=i_region+1
                              if (i_debug .eq. -12 .or. i_debug .ge. 12) write(6,*) '   rect 1: ',i_region

                              i_samps=min(i_samps+1,I_MAXSAMP)
                              i_rsamps(i_samps)=i_rcenter
                              i_csamps(i_samps)=i_ccenter
                              i_tsamps(i_samps)=-3
                              r_wsamps(i_samps) = r_wdth
                              r_ssamps(i_samps) = r_spce
                              i_region=i_region+1
                              if (i_debug .eq. -12 .or. i_debug .ge. 12) write(6,*) '   rect 2: ',i_region

                              i_samps=min(i_samps+1,I_MAXSAMP)
                              i_rsamps(i_samps)=i_rcenter
                              i_csamps(i_samps)=i_csamps(i_samps-i_region)
                              i_tsamps(i_samps)=-3
                              r_wsamps(i_samps) = r_wdth
                              r_ssamps(i_samps) = r_spce
                              i_region=i_region+1
                              if (i_debug .eq. -12 .or. i_debug .ge. 12) write(6,*) '   rect 3: ',i_region

                              i_samps=min(i_samps+1,I_MAXSAMP)
                              i_rsamps(i_samps)=i_rsamps(i_samps-i_region)
                              i_csamps(i_samps)=i_csamps(i_samps-i_region)
                              i_tsamps(i_samps)=-3
                              r_wsamps(i_samps) = r_wdth
                              r_ssamps(i_samps) = r_spce
                              i_region=i_region+1
                              if (i_debug .eq. -12 .or. i_debug .ge. 12) write(6,*) '   rect 4: ',i_region
                            else
                              i_samps=min(i_samps+1,I_MAXSAMP)
                              i_rsamps(i_samps)=i_rsamps(i_samps-i_region)
                              i_csamps(i_samps)=i_csamps(i_samps-i_region)
                              i_tsamps(i_samps)=-3
                              r_wsamps(i_samps) = r_wdth
                              r_ssamps(i_samps) = r_spce
                              i_region=i_region+1
                              if (i_debug .eq. -12 .or. i_debug .ge. 12) write(6,*) '   add region:   ',i_region
                            end if
                            i_region = 0
                            if (i_debug .eq. -12 .or. i_debug .ge. 12) write(6,*) '   end of region: ',i_region
                          end if
                        end if
c                      end if
                    end if
                    if (i_debug .eq. -20 .or. i_debug .ge. 20 .and. i_samps .gt. 0) write(6,*) 'i_samps=',i_samps,i_tsamps(i_samps),i_button

                    if (i_show .eq. 1) then ! show lines turned on
                        do i=i_start,i_samps
                          if (i_tsamps(i) .eq. 1) then
                            r_row(1)=r_winzoom(i_dsp)*(i_winrows(i_dsp)/
     &                            r_winzoom(i_dsp)-i_rsamps(i))/float(i_winrows(i_dsp))-2/float(i_winrows(i_dsp))
                            r_row(2)=r_winzoom(i_dsp)*(i_winrows(i_dsp)/
     &                            r_winzoom(i_dsp)-i_rsamps(i))/float(i_winrows(i_dsp))+2/float(i_winrows(i_dsp))
                            r_col(1)=r_winzoom(i_dsp)*i_csamps(i)/float(i_wincols(i_dsp))-2/float(i_winrows(i_dsp))
                            r_col(2)=r_winzoom(i_dsp)*i_csamps(i)/float(i_wincols(i_dsp))+2/float(i_winrows(i_dsp))
                            call plot_data(i_dsp,i_win,2,r_col,r_row)
                            r_row(1)=r_winzoom(i_dsp)*(i_winrows(i_dsp)/
     &                            r_winzoom(i_dsp)-i_rsamps(i))/float(i_winrows(i_dsp))+2/float(i_winrows(i_dsp))
                            r_row(2)=r_winzoom(i_dsp)*(i_winrows(i_dsp)/
     &                            r_winzoom(i_dsp)-i_rsamps(i))/float(i_winrows(i_dsp))-2/float(i_winrows(i_dsp))
                            r_col(1)=r_winzoom(i_dsp)*i_csamps(i)/float(i_wincols(i_dsp))-2/float(i_winrows(i_dsp))
                            r_col(2)=r_winzoom(i_dsp)*i_csamps(i)/float(i_wincols(i_dsp))+2/float(i_winrows(i_dsp))
                            call plot_data(i_dsp,i_win,2,r_col,r_row)
                            if (i_debug .eq. -12 .or. i_debug .ge. 12) write(6,*) '   drawing line: ',i,
     &                           r_col(1),r_row(1),r_col(2),r_row(2)
                          else if (i_tsamps(i) .lt. 0) then
                            r_row(1)=r_winzoom(i_dsp)*(i_winrows(i_dsp)/
     &                            r_winzoom(i_dsp)-i_rsamps(i-1))/float(i_winrows(i_dsp))
                            r_row(2)=r_winzoom(i_dsp)*(i_winrows(i_dsp)/
     &                            r_winzoom(i_dsp)-i_rsamps(i))/float(i_winrows(i_dsp))
                            r_col(1)=r_winzoom(i_dsp)*i_csamps(i-1)/float(i_wincols(i_dsp))
                            r_col(2)=r_winzoom(i_dsp)*i_csamps(i)/float(i_wincols(i_dsp))
                            call plot_data(i_dsp,i_win,2,r_col,r_row)
                            if (i_debug .eq. -12 .or. i_debug .ge. 12) write(6,*) '   drawing line: ',i,
     &                           i_tsamps(i),r_col(1),r_row(1),r_col(2),r_row(2)
                          end if
                        end do
                      end if  ! end i_show

                    end if

                    do i_chn=1,i_set
                    !@#$%
                      if (a_setfile(i_chn)(1:1) .ne. '=') then
                        call readdat(i_setunit(i_chn),
     &                          i_setrows(i_chn),
     &                          i_setcols(i_chn),
     &                          i_setshdr(i_chn),
     &                          i_setstlr(i_chn),
     &                          i_setrhdr(i_chn),
     &                          i_setrtlr(i_chn),
     &                          i_setchdr(i_chn),
     &                          i_setctlr(i_chn),
     &                          i_setvend(i_chn)*i_endian,
     &                          i_setvfmt(i_chn),
     &                          r_setvmlt(i_chn),
     &                          r_setvadr(i_chn),
     &                          r_setvmin(i_chn),
     &                          r_setvmax(i_chn),
     &                          b_setvnul(0,i_chn),
     &                          int((i_row)/r_winzoom(i_dsp))+i_winradr(i_dsp),
     &                          int(i_col/r_winzoom(i_dsp))+i_wincadr(i_dsp)-1,
     &                          3,r_data,i_data,readfunc,i_err)
                      else
                        do j=0,2
                          r_data(j)=0
                          i_data(j)=0
                        end do
                        i_opr=1
                        i_flg=0
                        a_value = ' '
                        do i = 2,rdflen(a_setfile(i_chn))+1
                          if (index('+-*/x',a_setfile(i_chn)(i:i)) .gt. 0 .or. a_setfile(i_chn)(i:i) .eq. ' ') then
                            if (a_value .ne. ' ') then
                              read(a_value,*) r_value
                              if (i_opr .eq. 1) then
                                r_data(1)=r_data(1)+r_value
                              else if (i_opr .eq. 2) then
                                r_data(1)=r_data(1)-r_value
                              else if (i_opr .eq. 3) then
                                r_data(1)=r_data(1)*r_value
                              else if (i_opr .eq. 4) then
                                r_data(1)=r_data(1)/r_value
                              end if
                              a_value = ' '
                            end if
                          end if
                          if(a_setfile(i_chn)(i:i) .eq. '+') then
                            i_opr=1
                          else if (a_setfile(i_chn)(i:i) .eq. '-') then
                            i_opr=2
                          else if (a_setfile(i_chn)(i:i) .eq. 'x') then
                            i_opr=3
                          else if (a_setfile(i_chn)(i:i) .eq. '/') then
                            i_opr=4
                          else if (a_setfile(i_chn)(i:i) .eq. 's' .or. a_setfile(i_chn)(i:i) .eq. 'S') then
                            i_flg=1
                          else if (i_flg .eq. 1) then
                            i_flg = 0
                            i_tmp = max(min((ichar(a_setfile(i_chn)(i:i))-ichar('0')),i_set+1),1)
                            if (i_tmp .gt. i_set) i_tmp=0
                            if (i_tmp .ne. 0) then
                              call readdat(i_setunit(i_tmp),
     &                          i_setrows(i_tmp),
     &                          i_setcols(i_tmp),
     &                          i_setshdr(i_tmp),
     &                          i_setstlr(i_tmp),
     &                          i_setrhdr(i_tmp),
     &                          i_setrtlr(i_tmp),
     &                          i_setchdr(i_tmp),
     &                          i_setctlr(i_tmp),
     &                          i_setvend(i_tmp)*i_endian,
     &                          i_setvfmt(i_tmp),
     &                          r_setvmlt(i_tmp),
     &                          r_setvadr(i_tmp),
     &                          r_setvmin(i_tmp),
     &                          r_setvmax(i_tmp),
     &                          b_setvnul(0,i_tmp),
     &                          int((i_row)/r_winzoom(i_dsp))+i_winradr(i_dsp),
     &                          int(i_col/r_winzoom(i_dsp))+i_wincadr(i_dsp)-1,
     &                          3,r_data2,i_data2,readfunc,i_err)
                              i_data(1)=i_data(1)+i_data2(1)
                              if (i_opr .eq. 1) then
                                r_data(1)=r_data(1)+r_data2(1)
                              else if (i_opr .eq. 2) then
                                r_data(1)=r_data(1)-r_data2(1)
                              else if (i_opr .eq. 3) then
                                r_data(1)=r_data(1)*r_data2(1)
                              else if (i_opr .eq. 4) then
                                r_data(1)=r_data(1)/r_data2(1)
                              else
                                r_data(1)=r_data(1)+r_data2(1)
                              end if
                            end if
                          else
                            a_value = a_value(1:max(1,rdflen(a_value)))//a_setfile(i_chn)(i:i)
                          end if
                        end do
                      end if
                      if(i_samps .gt. 0) r_vsamps(i_samps,i_chn)=r_data(1)

                      if (a_label1 .eq. ' ') then
                        write(a_label,fmt=a_colfrmt,iostat=i_err) int(i_col/r_winzoom(i_dsp))+i_wincadr(i_dsp)
                        a_label1='COL: '//a_label

                        write(a_label,fmt=a_rowfrmt,iostat=i_err) int(i_row/r_winzoom(i_dsp))+i_winradr(i_dsp)
                        a_label1=a_label1(1:rdflen(a_label1))//'   ROW: '//a_label
                      end if

                      if (i_cntl .eq. 1 .and. a_setproj(i_chn) .ne. ' ' .and.
     &                           r_setrmlt(i_chn) .ne. 0. .and. r_setcmlt(i_chn) .ne. 0.) then
                        r_dnx(1) = ((int(i_row/r_winzoom(i_dsp))+i_winradr(i_dsp))*r_setrmlt(i_chn))+r_setradr(i_chn)
                        r_dnx(2) = ((int(i_col/r_winzoom(i_dsp))+i_wincadr(i_dsp))*r_setcmlt(i_chn))+r_setcadr(i_chn)
                        r_dnx(3) = r_data(1)
                        call get_coordinates(a_setproj(i_chn),r_setpegv(1,i_chn),
     &                                         r_dnx,r_eux,1,i_debug,i_err)
                        if (i_err .eq. 0) then
                          r_data(1) = r_eux(3)
                          if (a_label1(1:4) .ne. 'LAT:' .or. rdflower(a_setproj(i_chn)) .ne. 'scx') then

                            if (i_debug .ge. 6) write(6,*) 'i_key = ',i_key
                            if (i_debug .ge. 6) write(6,*) a_setproj(i_chn),r_eux(1)*180.0/r_pi,r_eux(2)*180.0/r_pi
                            write(a_label,'(f10.5)') r_eux(1)*180.0/r_pi
                            a_label1='LAT: '//a_label

                            write(a_label,'(f11.5)',iostat=i_err) r_eux(2)*180.0/r_pi
                            a_label1=a_label1(1:rdflen(a_label1))//'   LON: '//a_label
                          end if
                        end if
                      else if (i_shft .eq. 1 .and.
     &                           r_setrmlt(i_chn) .ne. 0. .and. r_setcmlt(i_chn) .ne. 0.) then
                        r_eux(2) = ((int(i_row/r_winzoom(i_dsp))+i_winradr(i_dsp))*r_setrmlt(i_chn))+r_setradr(i_chn)
                        r_eux(1) = ((int(i_col/r_winzoom(i_dsp))+i_wincadr(i_dsp))*r_setcmlt(i_chn))+r_setcadr(i_chn)
                        r_eux(3) = r_data(1)
                        if (a_label1(1:4) .ne. 'LAT:' ) then

                          if (i_debug .ge. 6) write(6,*) 'i_key = ',i_key
                          if (i_debug .ge. 6) write(6,*) a_setproj(i_chn),r_eux(1),r_eux(2)
                          write(a_label,'(f10.4)') r_eux(1)
                          a_label1='CPS: '//a_label

                          write(a_label,'(f11.4)',iostat=i_err) r_eux(2)
                          a_label1=a_label1(1:rdflen(a_label1))//'   RPS: '//a_label
                        end if

                      end if
                      if (i_data(1) .eq. 0) then ! data valid
                        i_log=nint(alog10(abs(r_data(1))))
                        i_dec=8-nint(alog10(abs(r_setvavg(i_chn))+10*r_setvstd(i_chn)))
                        if (i_debug .gt. 20) write(6,*) 'i_dec=',i_dec
                        if ((i_dec .ge. -2 .and. i_dec .le. 12) .and. abs(i_log) .lt. 8) then
                          write(a_fmt,'(a,i2.2,a)',iostat=i_err) '(f12.',min(10,max(0,i_dec)),')'
                        else
                          a_fmt='(e12.5)'
                        end if
                        write(a_label2(rdflen(a_label2)+1:),fmt=a_fmt,iostat=i_err) r_data(1)
                      else                        ! data invalid
                        i_log=nint(alog10(abs(r_data(1))))
                        i_dec=7-nint(alog10(abs(r_setvavg(i_chn))+10*r_setvstd(i_chn)))
                        if (i_debug .gt. 20) write(6,*) 'i_dec=',i_dec,i_log
                        if (i_dec .ge. -2 .and. i_dec .le. 12 .and. abs(i_log) .eq. 8) then
                          write(a_fmt,'(a,i1,a)',iostat=i_err) '(f12.',min(9,max(0,i_dec)),',a1)'
                        else
                          a_fmt='(e12.4,a1)'
                        end if
                        write(a_label2(rdflen(a_label2)+1:),fmt=a_fmt,iostat=i_err) r_data(1),'*'
                      end if
                    end do
                    if (r_winzoom(i_dsp) .ge. 1.0) then
                      write(a_label,'(a,i3,a)',iostat=i_err) 'ZOOM:',nint(r_winzoom(i_dsp)),'x'
                    else
                      write(a_label,'(a,i3,a)',iostat=i_err) 'ZOOM:',-nint(1.0/r_winzoom(i_dsp)),'x'
                    end if
                    a_label=a_label(1:9)//'  '//a_label1(1:max(1,rdflen(a_label1)))//' '//a_label2
                    call display_label(i_dsp,i_win,a_label,1)
                    if (i_button .eq. 3 .and. i_typ .eq. 3) write(6,*) a_filename(1:max(rdflen(a_filename),1))//'  '//
     &                 a_label1(1:max(1,rdflen(a_label1)))//' '//a_label2(1:max(1,rdflen(a_label2)))
                  else if (i_button .eq. 2)  then ! middle click
                    i_rcenter=i_row/r_winzoom(i_dsp)+i_winradr(i_dsp)
                    i_ccenter=i_col/r_winzoom(i_dsp)+i_wincadr(i_dsp)
                    i_cdsp = i_dsp
                    i_cset = 1
                    if (i_debug .ge. 8) write(6,*) '--i_cdsp,i_cset = ',i_cdsp,i_cset
c                   if (i_debug .ge. 9) write(6,*) 'i_key=',i_key
                    if (i_debug .ge. 9) write(6,*) 'moving scroll to',i_ccenter,i_rcenter
                    do i_d=1,I_DMAX
                      if (i_winactv(i_d) .ne. 0) then
                        if ((i_key .eq. 0 .and. i_d .eq. i_dsp) .or.
     &                        (i_key .ne. 0 .and. i_d .ne. i_dsp)) then
                          if (i_winradr(i_d) .ne. max(min(i_rcenter-int((32000/2)/r_winzoom(i_d)),
     &                           i_setrows(1)-int(32000/r_winzoom(i_d))),0) ) then
                            i_winradr(i_d) =
     &                            max(min(i_rcenter-int((32000/2)/r_winzoom(i_d)),i_setrows(1)-int(32000/r_winzoom(i_d))),0)
                            i_redraw(i_d)=1
                          end if
                          if (i_wincadr(i_d) .ne. max(min(i_ccenter-int((32000/2)/r_winzoom(i_d)),
     &                           i_setcols(1)-int(32000/r_winzoom(i_d))),0) ) then
                            i_wincadr(i_d) =
     &                            max(min(i_ccenter-int((32000/2)/r_winzoom(i_d)),i_setcols(1)-int(32000/r_winzoom(i_d))),0)
                            i_redraw(i_d)=1
                          end if
                          if (i_redraw(i_d) .eq. 1) then
                            i_redraw(i_d) = 0
                            i_event(0) = i_d   ! Redraw window
                            i_event(1) = 1
                            i_event(2) = 1
                            i_event(3) = i_vxo(i_d,1)-5
                            i_event(4) = i_vyo(i_d,1)-5
                            i_event(5) = i_vxs(i_d,1)
                            i_event(6) = i_vys(i_d,1)
                            if (i_debug .ge. 6) write(6,*) 'i_bcnt2 =',i_bcnt
                            call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                          end if
                          i_cpos = nint((i_ccenter-i_wincadr(i_d))*r_winzoom(i_d)-0.5*i_vxs(i_d,1))
                          i_rpos = nint((i_rcenter-i_winradr(i_d))*r_winzoom(i_d)-0.5*i_vys(i_d,1))
                          call move_scroll(i_d,1,i_cpos,i_rpos)
                        end if
                      end if
                    end do
                  end if
                end if

              else if (i_evn .eq. 0) then ! menu select
                i_col = i_bdat(4,1)
                i_row = i_bdat(5,1)
                if (i_debug .ge. 6) write(6,*) 'Menu Item selected:',i_dsp,i_col,i_row
                if (i_col .eq. 0) then
                  if (i_row .eq. 1) then
                    call write_greeting()
                  else if (i_row .eq. 2) then
                    write(6,*) 'Function not implemented yet'
                  end if
                else if (i_col .eq. 1) then
                  if (i_row .eq. 1) then
                    if (i_debug .ge. 2) write(6,*) 'Creating new application'
                    i_arg = 0
                    a_command = 'mdx'
                    do while(i_arg .lt. i_inarg)
                      i_arg=i_arg + 1
                      a_value = a_inarg(i_arg)
                      a_command = a_command(1:rdflen(a_command))//' '//a_value
                    end do
                    a_command=a_command(1:rdflen(a_command))//' &'
                    if (i_debug .ge. 6) write(6,*) 'a_command=',a_command
                    call system(a_command)
                  else if (i_row .eq. 2) then
                    i_done = 1
                  end if
                else if (i_col .eq. 2) then
                  if (i_row .eq. 1) then
                    if (i_debug .ge. 3) write(6,*) 'Creating new display'
                    call create_dsp(a_filename,i_winrows(i_dsp)
     &                    ,i_wincols(i_dsp),i_winy,i_winx,a_setname(1),i_set,i_d
     &                    ,i_menu,a_tname,i_close,a_lcolor,i_debug)
                    if (i_debug .ge. 6) write(6,*) 'i_d =',i_d
                    if (i_d .gt. 0) then
                      i_winactv(i_d) = 1
                      call get_wininfo(i_d,1,i_vxo(i_d,1),i_vyo(i_d,1)
     &                       ,i_vxs(i_d,1),i_vys(i_d,1),i_wxs(i_d,1),i_wys(i_d,1)
c     &                       ,i_vxs(i_d,1),i_vys(i_d,1),i_cw,i_ch
     &                       ,i_widget)
                      if (i_debug .ge. 6) write(6,*) 'from get_win',i_vxo(i_d,1),i_vyo(i_d
     &                       ,1),i_vxs(i_d,1),i_vys(i_d,1)
                      i_winrows(i_d)=i_winrows(i_dsp)
                      i_wincols(i_d)=i_wincols(i_dsp)
                      i_winselc(i_d)=i_winselc(i_dsp)
                      r_winzoom(i_d)=r_winzoom(i_dsp)
                      do i=1, I_CMAX
                        i_dspactv(i_d,i) = i_dspactv(i_dsp,i)
                        if (i .le. i_set) then
                          if (i_dspactv(i_d,i) .eq. 1) then
                            call set_button_shadow(i_d,i+1,1,i_debug)
                          else
                            call set_button_shadow(i_d,i+1,0,i_debug)
                          end if
                        end if
                      end do
                    end if
                  else if (i_row .eq. 2) then
                    call destroy_display(i_dsp)
                  else if (i_row .eq. 3) then  ! Resize Display

                    do i_chn = 1,i_set

                      if (a_setfile(i_chn) .ne. ' ' .and. a_setfile(i_chn)(1:1) .ne. '=') then
                        if (i_setunit(i_chn) .lt. 0) then
                          i_fbytes = readfunc(1,i_chn,i_eight(0),0,b_data)
                          if (i_debug .ge. 3) write(6,*) 'internal buffer size=',i_fbytes
                          i_fbytes = min(i_fbytes,i_maxbuff)
                        else
                          i_fbytes = i_getfsize(i_setunit(i_chn))
c                          write(6,*) 'calling i_getfsize  ',i_fbytes,i_setunit(i_chn)
                        end if
                        if (i_fbytes .gt. 0) then
                          i_setrows(i_chn) = (i_fbytes
     &                      -i_setshdr(i_chn)-i_setstlr(i_chn))/((i_setvbyt(i_setvfmt(i_chn))
     &                      +i_setchdr(i_chn)+i_setctlr(i_chn))*i_setcols(i_chn)+i_setrhdr(i_chn)+i_setrtlr(i_chn))
                        end if
                      end if
                    end do

                    i_winrows(i_dsp) = min(nint(i_setrows(1)*r_winzoom(i_dsp)),32000)
                    i_wincols(i_dsp) = min(nint(i_setcols(1)*r_winzoom(i_dsp)),32000)
                    call resize_win(i_dsp,1,i_wincols(i_dsp),i_winrows(i_dsp))
c                    write(6,*) 'New number of rows = ',i_winrows(i_dsp)
                  end if
                else if (i_col .eq. 3) then ! Set
                  if (i_row .eq. 1) then
                    write(6,*) 'Function not yet implemented'
                  else if (i_row .eq. 2) then
                    write(6,*) 'Function not yet implemented'
                  else if (i_row .eq. 3) then
                    write(6,*) 'Function not yet implemented'
                  end if
                else if (i_col .eq. 4) then ! Zoom
                  if (i_row .le. 3) then
                    if (i_debug .ge. 8) write(6,*) '--i_cdsp,i_cset = ',i_cdsp,i_cset
                    if (i_cdsp .ne. 0 .and. i_cdsp .ne. i_dsp) then
                      i_rcenter=nint((i_vyo(i_dsp,1)+0.5*i_vys(i_dsp,1))/r_winzoom(i_dsp))+i_winradr(i_dsp)
                      i_ccenter=nint((i_vxo(i_dsp,1)+0.5*i_vxs(i_dsp,1))/r_winzoom(i_dsp))+i_wincadr(i_dsp)
                    end if
                    if (i_debug .ge. 6) write(6,*) 'Center was at (col,row): ',i_ccenter,i_rcenter
                    if (i_row .eq. 1) then
                      r_winzoom(i_dsp)=1.
                      if (i_debug .ge. 6) write(6,*) 'Zoom off: ',r_winzoom(i_dsp)
                    else if (i_row .eq. 2) then
                      r_winzoom(i_dsp)=r_winzoom(i_dsp)*2.
                      if (i_debug .ge. 6) write(6,*) 'Zooming in: ',r_winzoom(i_dsp)
                    else if (i_row .eq. 3) then
                      r_winzoom(i_dsp)=r_winzoom(i_dsp)/2.
                      if (i_debug .ge. 6) write(6,*) 'Zooming out: ',r_winzoom(i_dsp)
                    end if
                    i_winrows(i_dsp) = min(nint(i_setrows(1)*r_winzoom(i_dsp)),32000)
                    i_wincols(i_dsp) = min(nint(i_setcols(1)*r_winzoom(i_dsp)),32000)
                    call resize_win(i_dsp,1,i_wincols(i_dsp),i_winrows(i_dsp))
                    if (i_winradr(i_dsp) .ne.
     &                   max(min(i_rcenter-int((32000/2)/r_winzoom(i_dsp)),i_setrows(1)-int(32000/r_winzoom(i_dsp))),0) ) then
                      i_winradr(i_dsp) =
     &                   max(min(i_rcenter-int((32000/2)/r_winzoom(i_dsp)),i_setrows(1)-int(32000/r_winzoom(i_dsp))),0)
                      i_redraw(i_dsp)=1
                    end if
                    if (i_wincadr(i_dsp) .ne.
     &                   max(min(i_ccenter-int((32000/2)/r_winzoom(i_dsp)),i_setcols(1)-int(32000/r_winzoom(i_dsp))),0) ) then
                      i_wincadr(i_dsp) =
     &                   max(min(i_ccenter-int((32000/2)/r_winzoom(i_dsp)),i_setcols(1)-int(32000/r_winzoom(i_dsp))),0)
                      i_redraw(i_dsp)=1
                    end if
                    if (i_redraw(i_dsp) .eq. 1) then
                      i_redraw(i_dsp) = 0
                      i_event(0) = i_dsp   ! Redraw window
                      i_event(1) = 1
                      i_event(2) = 1
                      i_event(3) = i_vxo(i_dsp,1)-5
                      i_event(4) = i_vyo(i_dsp,1)-5
                      i_event(5) = i_vxs(i_dsp,1)
                      i_event(6) = i_vys(i_dsp,1)
                      if (i_debug .ge. 6) write(6,*) 'i_bcnt2 =',i_bcnt
                      call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                    end if
                    i_cpos = nint((i_ccenter-i_wincadr(i_dsp))*r_winzoom(i_dsp)-0.5*i_vxs(i_dsp,1))
                    i_rpos = nint((i_rcenter-i_winradr(i_dsp))*r_winzoom(i_dsp)-0.5*i_vys(i_dsp,1))
                    if (i_debug .ge. 9) write(6,*) 'moving scroll to',i_cpos,i_rpos
                    call move_scroll(i_dsp,1,i_cpos,i_rpos)
                    i_cset = 1
                    i_event(0) = i_dsp
                    i_event(1) = 1
                    i_event(2) = 1
                    i_event(3) = i_vxo(i_dsp,1)-5
                    i_event(4) = i_vyo(i_dsp,1)-5
                    i_event(5) = i_vxs(i_dsp,1)
                    i_event(6) = i_vys(i_dsp,1)
                    if (i_debug .ge. 6) write(6,*) 'i_bcnt2 =',i_bcnt
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                  else if (i_row .eq. 4) then
                    do i=0,20
                      a_elabl(i)=' '
                      a_edata(i)=' '
                    end do
                    a_elabl(5)='Zoom: '
                    if (r_winzoom(i_dsp) .ge. 1.0) then
                      write(a_edata(5),*) nint(r_winzoom(i_dsp))
                    else
                      write(a_edata(5),*) -nint(1.0/r_winzoom(i_dsp))
                    end if
                    call entry_window(0,a_elabl,a_edata)
                    i_dspselect=i_dsp
                    a_elabl(5)=' '
                    a_edata(5)=' '
                  end if
                  if (r_winzoom(i_dsp) .ge. 1.0) then
                    write(a_label,'(a,i3,a)') 'ZOOM:',nint(r_winzoom(i_dsp)),'x'
                  else
                    write(a_label,'(a,i3,a)') 'ZOOM:',-nint(1.0/r_winzoom(i_dsp)),'x'
                  end if
                  call display_label(i_dsp,1,a_label,1)
                else if (i_col .eq. 5) then   ! Select Menu
                  if (i_row .eq. 1) then
                    do i=0,20
                      a_elabl(i)=' '
                      a_edata(i)=' '
                    end do
                    a_elabl(6)='Display: '
                    write(a_edata(6),'(i1,a)') i_show+1,'|Off|On'  ! Sample Display ON/Off
                    a_elabl(7)='Mode: '
                    write(a_edata(7),'(i1,a)') i_smode+1,'|None|Point|Line|Region'
                    if (i_smode .lt.3) then
                      a_elabl(8)='Width: '
                      write(a_edata(8),*) r_wdth
                    else
                      a_elabl(8)='Density: '
                      write(a_edata(8),*) r_spce
                    end if
                    call entry_window(0,a_elabl,a_edata)
                    i_dspselect=i_dsp
                    a_elabl(6)=' '
                    a_edata(6)=' '
                    a_elabl(7)=' '
                    a_edata(7)=' '
                    a_elabl(8)=' '
                    a_edata(8)=' '
                  else if (i_row .eq. 2) then
                    call gx_getfile(a_value,43)
c                    write(6,*) 'file:',a_value(1:70)
                  else if (i_row .eq. 3) then
                    call gx_getfile(a_value,44)
c                    write(6,*) 'file:',a_value(1:70)
                  else if (i_row .eq. 3) then
                  else if (i_row .eq. 4) then
                    i_samps = 0
                    i_redraw(i_dsp) = 1
                    if (i_redraw(i_dsp) .eq. 1) then
                      i_redraw(i_dsp) = 0
                      i_event(0) = i_dsp   ! Redraw window
                      i_event(1) = 1
                      i_event(2) = 1
                      i_event(3) = i_vxo(i_dsp,1)-5
                      i_event(4) = i_vyo(i_dsp,1)-5
                      i_event(5) = i_vxs(i_dsp,1)
                      i_event(6) = i_vys(i_dsp,1)
                      if (i_debug .ge. 6) write(6,*) 'i_bcnt2 =',i_bcnt
                      call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                    end if
                  end if
                else if (i_col .eq. 6) then   ! Print Menu
                  if (i_row .eq. 2) then
                    call get_wininfo(i_dsp,1,i_vxo(i_dsp,1),i_vyo(i_dsp,1)
     &                       ,i_vxs(i_dsp,1),i_vys(i_dsp,1),i_wxs(i_dsp,1),i_wys(i_dsp,1)
     &                       ,i_widget)
                    i_event(0) = 0    ! tells data to go to print file instead of screen
                    i_event(1) = 1
                    i_event(2) = 1
c                    i_event(3) = (i_vxo(i_dsp,1)*r_winzoom(0))/r_winzoom(i_dsp)
c                    i_event(4) = (i_vyo(i_dsp,1)*r_winzoom(0))/r_winzoom(i_dsp)

                    i_event(3) = (max(i_vxo(i_dsp,1)-1,0)/r_winzoom(i_dsp)+i_wincadr(i_dsp))*r_winzoom(0)
                    i_event(4) = (max(i_vyo(i_dsp,1)-1,0)/r_winzoom(i_dsp)+i_winradr(i_dsp))*r_winzoom(0)

c                    i_event(5) = ((min(i_vxs(i_dsp,1)+i_vxo(i_dsp,1),i_wincols(i_dsp))-i_vxo(i_dsp,1))*
c     &                  r_winzoom(0))/r_winzoom(i_dsp)
c                    i_event(6) = ((min(i_vys(i_dsp,1)+i_vyo(i_dsp,1),i_winrows(i_dsp))-i_vyo(i_dsp,1))*
c     &                  r_winzoom(0))/r_winzoom(i_dsp)

                    i_event(5) = max((min(i_vxs(i_dsp,1)-i_pcpad,i_wincols(i_dsp)-i_vxo(i_dsp,1))/
     &                  r_winzoom(i_dsp))*r_winzoom(0),20.)
                    i_event(6) =     (min(i_vys(i_dsp,1)-i_prpad,i_winrows(i_dsp)-i_vyo(i_dsp,1))/
     &                  r_winzoom(i_dsp))*r_winzoom(0)

                    i_event(7) = 0
                    i_event(8) = i_event(4)
                    i_event(9) = i_event(6)
                    if (i_debug .ge. 21 .or. i_debug .eq. -21) write(6,*) 'Print range:',i_event(3),i_event(4),i_event(5),i_event(6)
                    do i_chn=1,i_set
                      i_dspactv(0,i_chn) = i_dspactv(i_dsp,i_chn)
                    end do
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                    i_dspselect=i_dsp
                  else if (i_row .eq. 3) then ! Print setup
                    do i=0,20
                      a_elabl(i)=' '
                      a_edata(i)=' '
                    end do
                    a_elabl(3)='Print Zoom: '
                    if (r_winzoom(0).ge. 1.0) then
                      write(a_edata(3),*) nint(r_winzoom(0))
                    else
                      write(a_edata(3),*) -nint(1.0/r_winzoom(0))
                    end if
                    a_elabl(4)='Format: '
                    write(a_edata(4),'(i1,a)') i_pfmt,'|PPM|Raw RGB|PostScript'
                    call entry_window(0,a_elabl,a_edata)
                    i_dspselect=i_dsp
                    a_elabl(3)=' '
                    a_edata(3)=' '
                    a_elabl(4)=' '
                    a_edata(4)=' '

                  end if
                else if (i_col .eq. 7) then   ! Tool Menu
                  write(6,*) 'Tool: ',a_tname(i_row)(1:max(1,rdflen(a_tname(i_row))))
                  if (i_row .eq. 1) then      ! Plot location
                    if (abs(i_samps) .ge. 1) then
                      a_file=a_workdir(1:rdflen(a_workdir))//'mdx_points.dat'
                      open(unit=19,file=a_file,status='unknown',form='formatted',iostat=i_err)
                      if (i_err .eq. 0) then
                        do i=1,abs(i_samps)
                          if (i_tsamps(i) .gt. 0) write(19,*) ' '
                          write(19,*) i_csamps(i),i_rsamps(i),(r_vsamps(i,j),j=1,i_set),i_tsamps(i)
                        end do
                        close(19)
                        a_value = 'xmgrace -free -noask -pexec "yaxes invert on" '//a_file(1:rdflen(a_file))//' &'
                        call system(a_value)
c                        i_samps=-abs(i_samps)
                      else
                        write(6,*) 'Cant open file:'//a_file(1:50)
                      end if
                    end if
                  else if (i_row .eq. 2) then      ! Plot Profiles
                    if (abs(i_samps) .ge. 2) then
                      a_file=a_workdir(1:rdflen(a_workdir))//'mdx_points.dat'
                      open(unit=19,file=a_file,status='unknown',form='formatted',iostat=i_err)
                      if (i_err .eq. 0) then
                        do i=1,abs(i_samps)
                          if (i .eq. 1) then
                            r_path = 0.0
                          else
                            r_path = r_path + sqrt(float(i_csamps(i)-i_csamps(i-1))**2.0+float(i_rsamps(i)-i_rsamps(i-1))**2.0)
                          end if

                          a_label=' '
                          do j=1,i_set
                             if (i_dspactv(i_dsp,j) .eq. 1) then
                               write(a_label1,*) r_vsamps(i,j)
                               a_label = a_label(1:max(1,rdflen(a_label)))//' '//a_label1
                             end if
                          end do
                          if (i_tsamps(i) .gt. 0) write(19,*) ' '
                          write(19,*) r_path,' ',a_label(1:max(1,rdflen(a_label)))
                          i_clast=i_csamps(i)
                          i_rlast=i_rsamps(i)
                          if (i .lt. abs(i_samps)) then
                            if (i_tsamps(i+1) .eq. -2) then   ! fill in extra points
                              r_dist=sqrt((i_csamps(i+1)-i_csamps(i))**2.+(i_rsamps(i+1)-i_rsamps(i))**2.)
                              do ii=1,int(r_dist/r_spce)
                                i_cc=i_csamps(i)+nint((i_csamps(i+1)-i_csamps(i))*(ii*r_spce)/r_dist)
                                i_rr=i_rsamps(i)+nint((i_rsamps(i+1)-i_rsamps(i))*(ii*r_spce)/r_dist)
                                if (i_cc .ne. i_clast .or. i_rr .ne. i_rlast) then
                                  i_clast = i_cc
                                  i_rlast = i_rr
                                  a_label=' '

                                  do i_chn=1,i_set
                                    if (i_dspactv(i_dsp,i_chn) .eq. 1) then
                                      if (a_setfile(i_chn)(1:1) .ne. '=') then
                                        call readdat(i_setunit(i_chn),
     &                                          i_setrows(i_chn),
     &                                          i_setcols(i_chn),
     &                                          i_setshdr(i_chn),
     &                                          i_setstlr(i_chn),
     &                                          i_setrhdr(i_chn),
     &                                          i_setrtlr(i_chn),
     &                                          i_setchdr(i_chn),
     &                                          i_setctlr(i_chn),
     &                                          i_setvend(i_chn)*i_endian,
     &                                          i_setvfmt(i_chn),
     &                                          r_setvmlt(i_chn),
     &                                          r_setvadr(i_chn),
     &                                          r_setvmin(i_chn),
     &                                          r_setvmax(i_chn),
     &                                          b_setvnul(0,i_chn),
     &                                          i_rr,i_cc-1,3,
     &                                          r_data,i_data,readfunc,i_err)
                                      else
                                        do j=0,2
                                          r_data(j)=0
                                          i_data(j)=0
                                        end do
                                        i_opr=1
                                        i_flg=0
                                        a_value = ' '
                                        do iii = 2,rdflen(a_setfile(i_chn))+1
                                          if (index('+-*/x',a_setfile(i_chn)(iii:iii)) .gt. 0 .or. a_setfile(i_chn)(iii:iii) .eq. ' ') then
                                            if (a_value .ne. ' ') then
                                              read(a_value,*) r_value
                                              if (i_opr .eq. 1) then
                                                r_data(1)=r_data(1)+r_value
                                              else if (i_opr .eq. 2) then
                                                r_data(1)=r_data(1)-r_value
                                              else if (i_opr .eq. 3) then
                                                r_data(1)=r_data(1)*r_value
                                              else if (i_opr .eq. 4) then
                                                r_data(1)=r_data(1)/r_value
                                              end if
                                              a_value = ' '
                                            end if
                                          end if
                                          if(a_setfile(i_chn)(iii:iii) .eq. '+') then
                                            i_opr=1
                                          else if (a_setfile(i_chn)(iii:iii) .eq. '-') then
                                            i_opr=2
                                          else if (a_setfile(i_chn)(iii:iii) .eq. 'x') then
                                            i_opr=3
                                          else if (a_setfile(i_chn)(iii:iii) .eq. '/') then
                                            i_opr=4
                                          else if (a_setfile(i_chn)(iii:iii) .eq. 's' .or. a_setfile(i_chn)(iii:iii) .eq. 'S') then
                                            i_flg=1
                                          else if (i_flg .eq. 1) then
                                            i_flg = 0
                                            i_tmp = max(min((ichar(a_setfile(i_chn)(iii:iii))-ichar('0')),i_set+1),1)
                                            if (i_tmp .gt. i_set) i_tmp=0
                                            if (i_tmp .ne. 0) then
                                              call readdat(i_setunit(i_tmp),
     &                                          i_setrows(i_tmp),
     &                                          i_setcols(i_tmp),
     &                                          i_setshdr(i_tmp),
     &                                          i_setstlr(i_tmp),
     &                                          i_setrhdr(i_tmp),
     &                                          i_setrtlr(i_tmp),
     &                                          i_setchdr(i_tmp),
     &                                          i_setctlr(i_tmp),
     &                                          i_setvend(i_tmp)*i_endian,
     &                                          i_setvfmt(i_tmp),
     &                                          r_setvmlt(i_tmp),
     &                                          r_setvadr(i_tmp),
     &                                          r_setvmin(i_tmp),
     &                                          r_setvmax(i_tmp),
     &                                          b_setvnul(0,i_tmp),
     &                                          i_rr,i_cc-1,3,
     &                                          r_data2,i_data2,readfunc,i_err)
                                              i_data(1)=i_data(1)+i_data2(1)
                                              if (i_opr .eq. 1) then
                                                r_data(1)=r_data(1)+r_data2(1)
                                              else if (i_opr .eq. 2) then
                                                r_data(1)=r_data(1)-r_data2(1)
                                              else if (i_opr .eq. 3) then
                                                r_data(1)=r_data(1)*r_data2(1)
                                              else if (i_opr .eq. 4) then
                                                r_data(1)=r_data(1)/r_data2(1)
                                              else
                                                r_data(1)=r_data(1)+r_data2(1)
                                              end if
                                            end if
                                          else
                                            a_value = a_value(1:max(1,rdflen(a_value)))//a_setfile(i_chn)(iii:iii)
                                          end if
                                        end do
                                      end if
                                      write(a_label1,*) r_data(1)
                                      a_label = a_label(1:max(1,rdflen(a_label)))//' '//a_label1
                                    end  if
                                  end do
                                  write(19,*) r_path+sqrt((i_cc-i_csamps(i))**2.+(i_rr-i_rsamps(i))**2.),' ',
     &                                           a_label(1:max(1,rdflen(a_label)))
                                end if !  i_cc .ne. i_clast .or. i_rr .ne. i_rlast
                              end do
                            end if
                          end if

                        end do
                        close(19)
                        a_value = 'xmgrace -free -noask -nxy '//a_file(1:rdflen(a_file))//' &'
                        call system(a_value)
c                        i_samps=-abs(i_samps)
                      else
                        write(6,*) 'Cant open file:'//a_file(1:50)
                      end if
                    end if
                  else if (i_row .ge. 3 .and. i_row .le. 5) then      ! Tool 3, 4, or 5
                    i_act=0
                    do i=i_set,1,-1
                      if (i_dspactv(i_dsp,i) .eq. 1) i_act = i
                    end do
                    if (i_act .gt. 0) then
                      a_file=a_workdir(1:rdflen(a_workdir))//'mdx_points.dat'
                      open(unit=19,file=a_file,status='unknown',form='formatted',iostat=i_err)
                      if (i_err .eq. 0) then
c                        write(19,*) a_setfile(i_act)(1:rdflen(a_setfile(i_act))),' ',a_setname(i_act)(1:rdflen(a_setname(i_act))),i_setcols(i_act),i_setrows(i_act)
                        do i=1,abs(i_samps)
                          if (i .eq. 1 .or. i_tsamps(i) .gt. 0) then
                            r_path = 0.0
                          else
                            r_path = r_path + sqrt(float(i_csamps(i)-i_csamps(i-1))**2.0+float(i_rsamps(i)-i_rsamps(i-1))**2.0)
                          end if
                          write(19,'(3i8,2e15.5)') i_csamps(i),i_rsamps(i),i_tsamps(i),r_path,r_vsamps(i,i_act)
                        end do
                        close(19)
                        write(a_value,'(9i10,9e15.5)')
     &                      i_setcols(i_act),i_setrows(i_act),i_setvend(i_act),
     &                      i_setvfmt(i_act),i_setshdr(i_act),
     &                      i_setrhdr(i_act),i_setrtlr(i_act),
     &                      i_setchdr(i_act),i_setctlr(i_act),
     &                      r_setvmin(i_act),r_setvmax(i_act),
     &                      r_setrmlt(i_act),r_setradr(i_act),
     &                      r_setcmlt(i_act),r_setcadr(i_act),
     &                      r_setpegv(1,i_act),r_setpegv(2,i_act),r_setpegv(3,i_act)
                        a_value = a_file(1:rdflen(a_file))//' '//a_setfile(i_act)(1:rdflen(a_setfile(i_act)))//' '//a_value
                        a_value = a_tcmnd(i_row)(1:rdflen(a_tcmnd(i_row)))//' '//a_value
                        a_value = a_value(1:rdflen(a_value))//' '//a_twait(i_row)
                        if (i_debug .eq. -9 .or. i_debug .ge. 9) write(6,*) a_value(1:rdflen(a_value))
                        call system(a_value)
                        i_redraw(i_dsp)=1
                      else
                        write(6,*) 'Cant open file:'//a_file(1:50)
                      end if
                    end if
                  end if
                  if (i_redraw(i_dsp) .eq. 1) then
                    i_redraw(i_dsp) = 0
                    i_event(0) = i_dsp   ! Redraw window
                    i_event(1) = 1
                    i_event(2) = 1
                    i_event(3) = i_vxo(i_dsp,1)-5
                    i_event(4) = i_vyo(i_dsp,1)-5
                    i_event(5) = i_vxs(i_dsp,1)
                    i_event(6) = i_vys(i_dsp,1)
                    if (i_debug .ge. 6) write(6,*) 'i_bcnt2 =',i_bcnt
                    call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)
                  end if
                else if (i_col .eq. 8) then
                  ! Do nothing
                end if
              end if

              do ib=1,i_bcnt-1    ! Move Items up in buffer
                do i=0,10
                  i_bdat(i,ib) = i_bdat(i,ib+1)
                end do
              end do
              i_bcnt = i_bcnt-1
            else
              i_cset = 0
              if (i_debug .ge. 8) write(6,*) '--Setting i_cset = 0'
            end if

            if (i_bcnt .eq. 0 .and. i_ecnt .gt. 0 .or. i_r .ge. 0) then   ! Expose Command to execute
              if (i_r .le. -1) then ! just starting to process command
                call get_ecmd(I_WKSPACE,i_ecnt,i_edat,i_ecmd,i_debug)
                i_edsp = i_ecmd(0)
                i_ewin = i_ecmd(1)
                i_eevn = i_ecmd(2)
                i_ecol = i_ecmd(3)
                i_erow = i_ecmd(4)
                i_encx = i_ecmd(5)   ! columns per line
                i_enrx = i_ecmd(6)   ! number of lines in this segment
                if (i_debug .ge. 5) write(6,*) 'initializing expose'
                if (i_edsp .eq. 0 .and. i_pinit .eq. 0) then ! initialize printer data to file
                  i_pinit = 1
                  i_pid = 30
                  a_file=a_workdir(1:rdflen(a_workdir))//'out.ppm'
                  open(unit=i_pid,file=a_file,status='unknown',form='unformatted',
     &                   access='direct',recl=i_encx*3,iostat=i_err)

                  if (i_err .ne. 0) then
                    write(6,*) ' '
                    write(6,*) 'Error Opening out.ppm   PPM file not created.  ',i_err
                    write(6,*)
                    i_err=0
                  else

                    a_label = 'Creating PPM file'
                    if (i_winactv(i_dspselect) .ne. 0 )call display_label(i_dspselect,1,a_label,1)
                    if (i_debug .ge. 2) then
                      write(6,*) ' '
                      write(6,'(1x,a)') 'Creating PPM File -- '
                    end if

                    a_out='P6'
                    b_out(3)=13
                    do i=4,3*i_encx
                      b_out(i)=32
                    end do
                    write(a_value,'(3i15)') i_encx,i_ecmd(9),255
c                    write(6,*) 'a_value=',a_value(1:50)
c                    write(6,*) 'rdflen(a_value)  =',rdflen(a_value)
                    do i = 1,rdflen(a_value)
                      b_out(3*i_encx-rdflen(a_value)-1+i) = ichar(a_value(i:i))
                    end do
                    b_out(i_encx*3) = 13
                    write(i_pid,rec=1,iostat=i_stat) (b_out(i),i=1,i_encx*3)
                  end if


                end if

                i_r = 0

                if (r_winzoom(i_edsp) .ge. 1) then
                  i_strt=i_encx-1
                  i_stop=0
                  i_incr=-1
                  i_coff=mod(i_ecol,nint(r_winzoom(i_edsp)))
c                 write(6,*) 'strt,stop,i_coff=',i_strt,i_stop,i_coff,i_incr
                else
                  i_strt=0
                  i_stop=i_encx-1
                  i_incr=1
                  i_coff=0
                end if
              else ! in the middle of reading data and filling display buffer
                if (i_debug .ge. 6 .and. i_r .eq. 0) write(6,*) 'gathering data for expose'
                    if (i_debug .ge. 5) write(6,*) 'reading at line: ',i_r+1,
     &                         int(i_erow/r_winzoom(i_edsp)),int(i_ecol/r_winzoom(i_edsp)),
     &                         int(i_enrx/r_winzoom(i_edsp)),int(i_encx/r_winzoom(i_edsp))
                if (i_edsp .gt. 0) then
                  if (i_debug .ge. 4 .and. i_r .lt. i_vyo(i_edsp,i_ewin)-i_erow-2)
     &               write(6,*) '    skipping lines at top: ',i_r,' to ',i_vyo(i_edsp,i_ewin)-i_erow-2
                  i_r = max(i_r,i_vyo(i_edsp,i_ewin)-i_erow-2)

                  if (i_debug .ge. 4 .and. i_vyo(i_edsp,i_ewin)+i_vys(i_edsp,i_ewin)-i_erow .lt. i_enrx)
     &               write(6,*) '    skipping lines at bottom: ',i_enrx,' to ',
     &                         min(i_enrx,i_vyo(i_edsp,i_ewin)+i_vys(i_edsp,i_ewin)-i_erow)
                  i_enrx = min(i_enrx,i_vyo(i_edsp,i_ewin)+i_vys(i_edsp,i_ewin)-i_erow)
                end if
                do while (i_bcnt .le. 0 .and. i_r .lt. i_enrx)

c                  write(6,*) 'i_r=',i_r,i_encx-1
                  i_dflag = 0
                  do i_c=0,i_encx-1
                    i_pos=i_c+i_r*i_encx
                    r_rdat(i_pos)=0.0
                    r_gdat(i_pos)=0.0
                    r_bdat(i_pos)=0.0
                    i_indx(i_pos)=0.0
                  end do
                  do i_chn = 1,i_set
                    if (i_dspactv(i_edsp,i_chn) .eq. 1) then
                      if (a_setfile(i_chn)(1:1) .ne. '=') then
                      call readdat(i_setunit(i_chn),
     &                          i_setrows(i_chn),
     &                          i_setcols(i_chn),
     &                          i_setshdr(i_chn),
     &                          i_setstlr(i_chn),
     &                          i_setrhdr(i_chn),
     &                          i_setrtlr(i_chn),
     &                          i_setchdr(i_chn),
     &                          i_setctlr(i_chn),
     &                          i_setvend(i_chn)*i_endian,
     &                          i_setvfmt(i_chn),
     &                          r_setvmlt(i_chn),
     &                          r_setvadr(i_chn),
     &                          r_setvmin(i_chn),
     &                          r_setvmax(i_chn),
     &                          b_setvnul(0,i_chn),
     &                          int((i_erow+i_r)/r_winzoom(i_edsp))+i_winradr(i_edsp),
     &                          int(i_ecol/r_winzoom(i_edsp))+i_wincadr(i_edsp),
     &                          int(i_encx/r_winzoom(i_edsp))+2,
     &                          r_data,i_data,readfunc,i_err)
                      else
                        do j=0,int(i_encx/r_winzoom(i_edsp))+2
                          r_data(j)=0
                          i_data(j)=0
                        end do
                        i_opr=1
                        i_flg=0
                        a_value=' '
                        do i = 2,rdflen(a_setfile(i_chn))+1
                          if (index('+-*/x',a_setfile(i_chn)(i:i)) .gt. 0 .or. a_setfile(i_chn)(i:i) .eq. ' ') then
                            if (a_value .ne. ' ') then
                              read(a_value,*) r_value
                              if (i_opr .eq. 1) then
                                do j=0,int(i_encx/r_winzoom(i_edsp))+2
                                  r_data(j)=r_data(j)+r_value
                                end do
                              else if (i_opr .eq. 2) then
                                do j=0,int(i_encx/r_winzoom(i_edsp))+2
                                  r_data(j)=r_data(j)-r_value
                                end do
                              else if (i_opr .eq. 3) then
                                do j=0,int(i_encx/r_winzoom(i_edsp))+2
                                  r_data(j)=r_data(j)*r_value
                                end do
                              else if (i_opr .eq. 4) then
                                do j=0,int(i_encx/r_winzoom(i_edsp))+2
                                  r_data(j)=r_data(j)/r_value
                                end do
                              end if
                              a_value = ' '
                            end if
                          end if
                          if(a_setfile(i_chn)(i:i) .eq. '+') then
                            i_opr=1
                          else if (a_setfile(i_chn)(i:i) .eq. '-') then
                            i_opr=2
                          else if (a_setfile(i_chn)(i:i) .eq. '*' .or. a_setfile(i_chn)(i:i) .eq. 'x') then
                            i_opr=3
                          else if (a_setfile(i_chn)(i:i) .eq. '/') then
                            i_opr=4
                          else if (a_setfile(i_chn)(i:i) .eq. 's' .or. a_setfile(i_chn)(i:i) .eq. 'S') then
                            i_flg=1
                          else if (i_flg .eq. 1) then
                            i_flg = 0
                            i_tmp = max(min((ichar(a_setfile(i_chn)(i:i))-ichar('0')),i_set+1),1)
                            if (i_tmp .gt. i_set) i_tmp=0
                            if (i_tmp .ne. 0) then
                              call readdat(i_setunit(i_tmp),
     &                          i_setrows(i_tmp),
     &                          i_setcols(i_tmp),
     &                          i_setshdr(i_tmp),
     &                          i_setstlr(i_tmp),
     &                          i_setrhdr(i_tmp),
     &                          i_setrtlr(i_tmp),
     &                          i_setchdr(i_tmp),
     &                          i_setctlr(i_tmp),
     &                          i_setvend(i_tmp)*i_endian,
     &                          i_setvfmt(i_tmp),
     &                          r_setvmlt(i_tmp),
     &                          r_setvadr(i_tmp),
     &                          r_setvmin(i_tmp),
     &                          r_setvmax(i_tmp),
     &                          b_setvnul(0,i_tmp),
     &                          int((i_erow+i_r)/r_winzoom(i_edsp))+i_winradr(i_edsp),
     &                          int(i_ecol/r_winzoom(i_edsp))+i_wincadr(i_edsp),
     &                          int(i_encx/r_winzoom(i_edsp))+2,
     &                          r_data2,i_data2,readfunc,i_err)
                              do j=0,int(i_encx/r_winzoom(i_edsp))+2
                                i_data(j)=i_data(j)+i_data2(j)
                              end do
                              if (i_opr .eq. 1) then
                                do j=0,int(i_encx/r_winzoom(i_edsp))+2
                                  r_data(j)=r_data(j)+r_data2(j)
                                end do
                              else if (i_opr .eq. 2) then
                                do j=0,int(i_encx/r_winzoom(i_edsp))+2
                                  r_data(j)=r_data(j)-r_data2(j)
                                end do
                              else if (i_opr .eq. 3) then
                                do j=0,int(i_encx/r_winzoom(i_edsp))+2
                                  r_data(j)=r_data(j)*r_data2(j)
                                end do
                              else if (i_opr .eq. 4) then
                                do j=0,int(i_encx/r_winzoom(i_edsp))+2
                                  r_data(j)=r_data(j)/r_data2(j)
                                end do
                              else
                                do j=0,int(i_encx/r_winzoom(i_edsp))+2
                                  r_data(j)=r_data(j)+r_data2(j)
                                end do
                              end if
                            end if
                          else
                            a_value = a_value(1:max(1,rdflen(a_value)))//a_setfile(i_chn)(i:i)
                          end if
                        end do
                      end if
                      do i_c = i_strt,i_stop,i_incr
                        if (r_winzoom(i_edsp) .ne. 1.) then
                          r_data(i_c) = r_data(int((i_c+i_coff)/r_winzoom(i_edsp)))
                          i_data(i_c) = i_data(int((i_c+i_coff)/r_winzoom(i_edsp)))
                        end if
                        i_pos=i_c+i_r*i_encx
                        if (i_data(i_c) .eq. 0 .and. i_indx(i_pos) .eq. 0) then
                          r_data(i_c) = max(r_dspcplw(i_chn),min(r_dspcphi(i_chn),r_data(i_c)))     ! Clip data
                          r_data(i_c) = (r_data(i_c)-r_dspaddr(i_chn))                              ! Shift data
                          if (i_dspmode(i_chn) .eq. 6) then                                         ! Wrap data
                            r_data(i_c) = wrap(r_data(i_c),r_dspwrap(i_chn))
                          end if
                          r_data(i_c) = r_data(i_c)/r_dspmult(i_chn)                                ! Scale data
                          if (r_dspexpn(i_chn) .ne. 1.0) then                                       ! Compress data
                            r_data(i_c) = min(1.0,max(0.0,r_data(i_c)))**r_dspexpn(i_chn)
                          end if
                          i_data(i_c) = max(0,min(i_dspnumt(i_chn)-1,int(i_dspnumt(i_chn)*r_data(i_c))))
                          if (i_dflag .eq. 0) then
                            r_rdat(i_pos) = r_dspredt(i_data(i_c),i_chn)
                            r_gdat(i_pos) = r_dspgrnt(i_data(i_c),i_chn)
                            r_bdat(i_pos) = r_dspblut(i_data(i_c),i_chn)
                          else
                            if (i_dspmixv(i_chn) .eq. 1)  then ! add
                              r_rdat(i_pos) =r_rdat(i_pos)+r_dspredt(i_data(i_c),i_chn)
                              r_gdat(i_pos) =r_gdat(i_pos)+r_dspgrnt(i_data(i_c),i_chn)
                              r_bdat(i_pos) =r_bdat(i_pos)+r_dspblut(i_data(i_c),i_chn)
                            else if (i_dspmixv(i_chn) .eq. 2) then ! mult
                              r_rdat(i_pos) =r_rdat(i_pos)*r_dspredt(i_data(i_c),i_chn)
                              r_gdat(i_pos) =r_gdat(i_pos)*r_dspgrnt(i_data(i_c),i_chn)
                              r_bdat(i_pos) =r_bdat(i_pos)*r_dspblut(i_data(i_c),i_chn)
                            else if (i_dspmixv(i_chn) .eq. 3) then ! max
                              r_rdat(i_pos) = max(r_rdat(i_pos),r_dspredt(i_data(i_c),i_chn))
                              r_gdat(i_pos) = max(r_gdat(i_pos),r_dspgrnt(i_data(i_c),i_chn))
                              r_bdat(i_pos) = max(r_bdat(i_pos),r_dspblut(i_data(i_c),i_chn))
                            end if
     &
                          end if
                        else
                          i_indx(i_pos) = 1 ! mark pixel as bad and set color to default background
                          r_rdat(i_pos) = max(0.0,min(0.9999,(i_nullclr(1)/255.)))
                          r_gdat(i_pos) = max(0.0,min(0.9999,(i_nullclr(2)/255.)))
                          r_bdat(i_pos) = max(0.0,min(0.9999,(i_nullclr(3)/255.)))
                        end if
                      end do
                      i_dflag = 1
                    end if
                  end do  ! Loop over channels
                  if (i_ponly .eq. 0 .and. mod(i_r,i_eventmod) .eq. 1) then
                    call getevent(1,i_event)
                    if (i_debug .ge. 5) then
                       if (i_event(0) .ne. 0) then
                          write(6,'(1x,a,7i10)')
     &                        'i_event=',i_event(0),i_event(1),i_event(2)
     &                        ,i_event(3),i_event(4),i_event(5),i_event(6)
                       end if
                    end if
                    if (i_event(2) .ne. 9 .or. i_button .ne. 0) call buffer_cmd(i_event,i_bdat,i_bcnt,2,I_BMAX,i_abort,i_debug)

                  end if
                  i_r = i_r+1
                end do  ! Loop over rows
                if (i_r .ge. i_enrx) then
                  if (i_r .eq. i_enrx) then
                    r_sum=0.0d0
                    i_bpl = i_encx
                    i_enrx2= i_r
                    if (i_edsp .ne. 0) then
                      if (i_debug .eq. -3 .or. i_debug .ge. 3) write(6,*) 'call disp',i_edsp,i_ewin,i_ecol,i_erow,i_encx,i_enrx2,i_bpl
                      call display_img(i_edsp,i_ewin,i_ecol,i_erow,i_encx,i_enrx2,i_bpl,r_rdat,r_gdat,r_bdat)
                      if (i_show .eq. 1) then ! show lines turned on
                        do i=1,i_samps
                          if (.true.) then
                            r_row(1)=r_winzoom(i_edsp)*(i_winrows(i_edsp)/r_winzoom(i_edsp)-i_rsamps(max(1,i-1)))
     &                         /float(i_winrows(i_edsp))
                            r_row(2)=r_winzoom(i_edsp)*(i_winrows(i_edsp)/r_winzoom(i_edsp)-i_rsamps(i         ))
     &                         /float(i_winrows(i_edsp))
                            r_col(1)=r_winzoom(i_edsp)*i_csamps(max(1,i-1))/float(i_wincols(i_edsp))
                            r_col(2)=r_winzoom(i_edsp)*i_csamps(i         )/float(i_wincols(i_edsp))
                            r_rowlow=min(r_row(1),r_row(2))
                            r_rowhigh=max(r_row(1),r_row(2))
                            r_collow=min(r_col(1),r_col(2))
                            r_colhigh=max(r_col(1),r_col(2))
                            if (r_rowlow*i_winrows(i_edsp) .ge. i_wys(i_edsp,1)-(i_erow) .or.
     &                          r_rowhigh*i_winrows(i_edsp) .le. i_wys(i_edsp,1)-(i_erow+i_enrx2)) then
c                            if (r_rowlow*i_winrows(i_edsp) .ge. (i_erow+i_enrx2) .or. r_rowhigh*i_winrows(i_edsp) .le. i_erow) then
c                              write(6,*) 'row expose: ',int(r_rowlow*i_winrows(i_edsp)),int(r_rowhigh*i_winrows(i_edsp)),i_wys(i_edsp,1)-(i_erow+i_enrx2),i_wys(i_edsp,1)-(i_erow)
c                              write(6,*) 'row expose: ',int(r_rowlow*i_winrows(i_edsp)),int(r_rowhigh*i_winrows(i_edsp)),i_erow,i_erow+i_enrx2
                            else
                              if (r_collow*i_wincols(i_edsp) .ge. i_ecol+i_encx .or. r_colhigh*i_wincols(i_edsp) .le. i_ecol) then
                                ! do nothing
                              else
                                if (i_tsamps(i) .eq. 1) then
                                  r_row(1)=r_winzoom(i_edsp)*(i_winrows(i_edsp)/
     &                                  r_winzoom(i_edsp)-i_rsamps(i))/float(i_winrows(i_edsp))-2/float(i_winrows(i_edsp))
                                  r_row(2)=r_winzoom(i_edsp)*(i_winrows(i_edsp)/
     &                                  r_winzoom(i_edsp)-i_rsamps(i))/float(i_winrows(i_edsp))+2/float(i_winrows(i_edsp))
                                  r_col(1)=r_winzoom(i_edsp)*i_csamps(i)/float(i_wincols(i_edsp))-2/float(i_winrows(i_edsp))
                                  r_col(2)=r_winzoom(i_edsp)*i_csamps(i)/float(i_wincols(i_edsp))+2/float(i_winrows(i_edsp))
                                  call plot_data(i_edsp,1,2,r_col,r_row)
                                  r_row(1)=r_winzoom(i_edsp)*(i_winrows(i_edsp)/
     &                                  r_winzoom(i_edsp)-i_rsamps(i))/float(i_winrows(i_edsp))+2/float(i_winrows(i_edsp))
                                  r_row(2)=r_winzoom(i_edsp)*(i_winrows(i_edsp)/
     &                                  r_winzoom(i_edsp)-i_rsamps(i))/float(i_winrows(i_edsp))-2/float(i_winrows(i_edsp))
                                  r_col(1)=r_winzoom(i_edsp)*i_csamps(i)/float(i_wincols(i_edsp))-2/float(i_winrows(i_edsp))
                                  r_col(2)=r_winzoom(i_edsp)*i_csamps(i)/float(i_wincols(i_edsp))+2/float(i_winrows(i_edsp))
                                  call plot_data(i_edsp,1,2,r_col,r_row)
                                  if (i_debug .eq. -12 .or. i_debug .ge. 12) write(6,*) 're-drawing point: ',i,
     &                                 r_col(1)+2,r_row(1)-2
                                else if (i_tsamps(i) .lt. 0) then
                                  r_row(1)=r_winzoom(i_edsp)*(i_winrows(i_edsp)/
     &                                  r_winzoom(i_edsp)-i_rsamps(max(1,i-1)))/float(i_winrows(i_edsp))
                                  r_row(2)=r_winzoom(i_edsp)*(i_winrows(i_edsp)/
     &                                  r_winzoom(i_edsp)-i_rsamps(i))/float(i_winrows(i_edsp))
                                  r_col(1)=r_winzoom(i_edsp)*i_csamps(max(1,i-1))/float(i_wincols(i_edsp))
                                  r_col(2)=r_winzoom(i_edsp)*i_csamps(i)/float(i_wincols(i_edsp))
                                  call plot_data(i_edsp,1,2,r_col,r_row)
                                  if (i_debug .eq. -12 .or. i_debug .ge. 12) write(6,*) 're-drawing line: ',i,
     &                                 r_col(1),r_row(1),r_col(2),r_row(2)
                                end if
                              end if
                            end if
                          end if
                        end do
                      end if  ! end i_show

                    else
                      if (i_debug .gt. 3) write(6,*) 'Printing block: ',i_ecmd(7),i_erow,i_ecmd(8)
                      if (mod(i_ecmd(7)+1,1000) .eq. 0) write(6,*) '  Blocks remaining: ',i_ecmd(7)+1
                      do ir=0,i_enrx2-1

                        do ic=0,min(i_encx,I_WKSPACE)-1
                          b_out(ic*3+1) = max(0,min(255,int(r_rdat(ic+ir*i_encx)*256)))
                          b_out(ic*3+2) = max(0,min(255,int(r_gdat(ic+ir*i_encx)*256)))
                          b_out(ic*3+3) = max(0,min(255,int(r_bdat(ic+ir*i_encx)*256)))
                        end do

                        write(i_pid,rec=2+ir+i_erow-i_ecmd(8),iostat=i_stat) (b_out(ib),ib=1,3*i_encx)

                      end do

                      if (i_ecmd(7) .eq. 0) then
                        close(i_pid,iostat=i_stat)
                        i_pinit=0
                        if (i_ponly .ne. 0) then
                          i_ponly=0
                          i_done=1
                        end if
                        a_label = 'Print Complete'
                        if (i_winactv(i_dspselect) .ne. 0 )call display_label(i_dspselect,1,a_label,1)
                        if (i_debug .ge. 2) then
                          write(6,*) 'Print file complete'
                          write(6,*) ' '
                        end if
                      end if
                    end if
                  end if
                  i_r = -1
                  if (i_debug .ge. 6) write(6,*) 'i_ecnt2=',i_ecnt
                end if

              end if
            end if

          end do


c     !@#&%
          write(6,*) ' '
          write(6,*) 'mdx Done'
          write(6,*) ' '

        end

        subroutine get_mdxdefaults(a_tname,a_tcmnd,a_twait,a_nullclr,i_pcpad,i_prpad,r_winzoom,
     &                  a_workdir,a_colordir,a_colorname,a_colorfile,i_colormax,i_close,a_clickcmd)

          implicit none

          integer       I_KMAX
          parameter     (I_KMAX=20)

          character*20  a_tname(5)
          character*1   a_twait(5)
          character*120 a_tcmnd(5)
          character*120 a_clickcmd(6)
          character*120 a_nullclr
          character*20  a_colorname(I_KMAX)
          character*255 a_colorfile(I_KMAX)
          integer       i_colormax
          integer       i_close
          real*4        r_winzoom(0:5)

          character*255 a_workdir
          character*255 a_colordir

          integer i
          integer j
          integer i_len
          integer i_found
          integer i_inarg
          integer i_prpad
          integer i_pcpad
          character*255 a_inarg(255)
          character*120 a_home

          character*120 a_keyw
          character*120 a_valu
          character*120 a_unit
          character*120 a_dimn
          character*120 a_elem
          character*120 a_oper
          character*120 a_cmnt

          integer rdflen
          external rdflen

          integer rdfnum
          external rdfnum

          character*40 rdflower
          external rdflower

          call getenv('HOME',a_home)
          if (a_home .ne. ' ') then
            a_home=a_home(1:rdflen(a_home))//'/.MDXinit'
          else
            a_home='.MDXinit'
          end if
          call rdf_init('ERROR_SCREEN=OFF')
          call rdf_clear()
          call rdf_read(a_home)

c          write(6,*) 'rdfnum = ',rdfnum()
          do i=1,rdfnum()
            call rdf_viewcols(i,a_keyw,a_valu,a_unit,a_dimn,a_elem,a_oper,a_cmnt)
c            write(6,*) ' a_keyw:',a_keyw
c            write(6,*) ' a_valu:',a_valu
            a_keyw=rdflower(a_keyw)
            if (a_keyw .eq. ' ') then
              ! do nothing
            else if (a_keyw .eq. 'tool3') then
              i_found=3
              call rdf_getfields(a_valu,i_inarg,a_inarg)
              if (i_inarg .ge. 2) then
                a_tname(i_found)=a_inarg(1)
                a_tcmnd(i_found)=a_inarg(2)
                a_twait(i_found)=' '
                do j=3,i_inarg
                  if (a_inarg(j) .ne. '&') then
                    a_tcmnd(i_found)=a_tcmnd(i_found)(1:rdflen(a_tcmnd(i_found)))//' '//a_inarg(j)
                  else
                    a_twait(i_found)='&'
                  end if
                end do
              end if
            else if (a_keyw .eq. 'tool4') then
              i_found=4
              call rdf_getfields(a_valu,i_inarg,a_inarg)
              if (i_inarg .ge. 2) then
                a_tname(i_found)=a_inarg(1)
                a_tcmnd(i_found)=a_inarg(2)
                a_twait(i_found)=' '
                do j=3,i_inarg
                  if (a_inarg(j) .ne. '&') then
                    a_tcmnd(i_found)=a_tcmnd(i_found)(1:rdflen(a_tcmnd(i_found)))//' '//a_inarg(j)
                  else
                    a_twait(i_found)='&'
                  end if
                end do
              end if
            else if (a_keyw .eq. 'tool5') then
              i_found=5
              call rdf_getfields(a_valu,i_inarg,a_inarg)
              if (i_inarg .ge. 2) then
                a_tname(i_found)=a_inarg(1)
                a_tcmnd(i_found)=a_inarg(2)
                a_twait(i_found)=' '
                do j=3,i_inarg
                  if (a_inarg(j) .ne. '&') then
                    a_tcmnd(i_found)=a_tcmnd(i_found)(1:rdflen(a_tcmnd(i_found)))//' '//a_inarg(j)
                  else
                    a_twait(i_found)='&'
                  end if
                end do
              end if
            else if (a_keyw .eq. 'click') then
              i_found=1
              call rdf_getfields(a_valu,i_inarg,a_inarg)
              if (i_inarg .ge. 2) then
                a_clickcmd(1)=a_inarg(2)
                do j=3,i_inarg
                    a_clickcmd(i_found)=a_clickcmd(i_found)(1:rdflen(a_clickcmd(i_found)))//' '//a_inarg(j)
                end do
              end if
            else if (a_keyw .eq. 'addtool') then
              i_found=0
              do j=5,3,-1
                if (a_tname(j) .eq. ' ') i_found=j
              end do
              if (i_found .ne. 0) then
                call rdf_getfields(a_valu,i_inarg,a_inarg)
                if (i_inarg .ge. 2) then
                  a_tname(i_found)=a_inarg(1)
                  a_tcmnd(i_found)=a_inarg(2)
                  a_twait(i_found)=' '
                  do j=3,i_inarg
                    if (a_inarg(j) .ne. '&') then
                      a_tcmnd(i_found)=a_tcmnd(i_found)(1:rdflen(a_tcmnd(i_found)))//' '//a_inarg(j)
                    else
                      a_twait(i_found)='&'
                    end if
                  end do
                end if
              end if
            else if (a_keyw .eq. 'null_color') then
              a_nullclr=a_valu
            else if (a_keyw .eq. 'addcmap') then
              call rdf_getfields(a_valu,i_inarg,a_inarg)
              if (i_inarg .eq. 2) then
                i_colormax=i_colormax+1
                if (i_colormax .gt. I_KMAX) then
                  i_colormax=I_KMAX
                  write(6,*) 'i_colormax error'
                end if
                a_colorname(i_colormax)=a_inarg(1)
                a_colorfile(i_colormax)=a_inarg(2)
c                write(6,*) 'adding color file: ',a_colorfile(i_colormax)
              end if
            else if (a_keyw .eq. 'delcmap') then
              call rdf_getfields(a_valu,i_inarg,a_inarg)
              if (i_inarg .eq. 1) then
                i_found=0
                do j=1,i_colormax
                  if (a_colorname(j) .eq. a_inarg(1)) i_found=i_found+1
                  if (j+i_found .lt. I_KMAX) then
                    a_colorfile(j)=a_colorfile(j+i_found)
                    a_colorname(j)=a_colorname(j+i_found)
                  else
                    a_colorfile(j)=' '
                    a_colorname(j)=' '
                  end if
                end do
                i_colormax=i_colormax-i_found
              end if
            else if (a_keyw .eq. 'pcpad') then
              read(a_valu,*) i_pcpad
            else if (a_keyw .eq. 'prpad') then
              read(a_valu,*) i_prpad
            else if (a_keyw .eq. 'close') then
              if (a_valu .eq. 'on' .or. a_valu .eq. 'ON') i_close=1
              if (a_valu .eq. 'off' .or. a_valu .eq. 'OFF') i_close=0
            else if (a_keyw .eq. 'zoom') then
              read(a_valu,*) r_winzoom(1)
              if (r_winzoom(1) .lt. 0.) r_winzoom(1) = abs(1./r_winzoom(1))
            else if (a_keyw .eq. 'pzoom') then
              read(a_valu,*) r_winzoom(0)
              if (r_winzoom(0) .lt. 0.) r_winzoom(0) = abs(1./r_winzoom(0))
            else if (a_keyw .eq. 'workdir') then
              a_workdir=a_valu
              if (a_workdir .eq. ' ') a_workdir='./'
              i_len=rdflen(a_workdir)
              if (a_workdir(i_len:i_len) .ne. '/') a_workdir=a_workdir(1:i_len)//'/'
            else if (a_keyw .eq. 'colordir') then
              a_colordir=a_valu
              if (a_colordir .eq. ' ') a_colordir='./'
              i_len=rdflen(a_colordir)
              if (a_colordir(i_len:i_len) .ne. '/') a_colordir=a_colordir(1:i_len)//'/'
            end if
          end do
          call rdf_clear()
          return
        end

        subroutine encodeval(r_data,a_data)

          implicit none

          real*4 r_data
          character*(*) a_data

          character*20 a_fmt
          integer i
          integer i_err

            a_data='*'
            i=13
            do while (index(a_data,'*') .ne. 0 .and. i .gt. 1)
              i=i-1
              if (i .ge. 10) then
                write(a_fmt,fmt='(a,i2,a)',iostat=i_err) '(f15.',i,')'
              else
                write(a_fmt,fmt='(a,i1,a)',iostat=i_err) '(f15.',i,')'
              end if
              write(a_data,fmt=a_fmt,iostat=i_err) r_data
            end do
            if (index(a_data,'*') .ne. 0) then
              write(a_data,fmt='(e15.8)',iostat=i_err) r_data
            end if
          return
        end

        subroutine decodeval(a_data,r_data,i_err)

          implicit none

          real*4 r_data
          character*(*) a_data
          integer i_err

          integer i_loc

            i_loc=max(index(a_data,'e'),index(a_data,'E'))
            if (i_loc .gt. 1 .and. index(a_data,'.') .eq. 0) then
              a_data=a_data(1:i_loc-1)//'.'//a_data(i_loc:)
              write(6,*) 'inserting a decimal at ',i_loc,'   ',a_data
            end if
            read(a_data,*,iostat=i_err) r_data
          return
        end

        subroutine median(r_lvl,i_cnt,r_val,r_med)

          implicit none

          integer*4 i_cnt
          real*4 r_val(i_cnt)
          real*4 r_med
          real*4 r_lvl

          real*4 r_low
          real*4 r_hgh
          real*4 r_rng

          integer*4 i_idx
          integer*4 i_hist(0:11)

          integer*4 i
          integer*4 i_num
          integer*4 i_sum
          integer*4 i_low
          integer*4 i_hgh
          integer*4 i_itr


c           write(6,*) 'i_cnt=',i_cnt
            r_low=r_val(1)
            r_hgh=r_val(1)
            do i=2,i_cnt
               if (r_val(i) .lt. r_low) r_low=r_val(i)
               if (r_val(i) .gt. r_hgh) r_hgh=r_val(i)
            end do
c           write(6,*) 'looking for lvl: ',r_lvl
            if (r_lvl .le. 0) then
              r_med=r_low
            else if (r_lvl .ge. 1.) then
              r_med=r_hgh
            else
              i_idx=0
              i_hist(i_idx)=i_cnt
              i_itr=0
              do while(i_hist(i_idx) .gt. max(int(0.00001*i_cnt),1) .and. i_itr .lt. 10 .and.
     &             r_hgh-r_low .gt. 0)
                i_itr=i_itr+1
c               write(6,*) 'low,high = ',r_low,r_hgh
c               write(6,*) 'Loop',i_itr
                r_rng=(r_hgh-r_low)
c               write(6,*) 'rng = ',r_rng
                do i=0,11
                  i_hist(i)=0
                end do
                do i=1,i_cnt
c                  i_idx=min(max(int((10*(r_val(i)-r_low)/(r_rng))+1),0),11)
                  i_idx=int(min(max(((10*(r_val(i)-r_low)/r_rng)+1.0),0.0),11.0))
                  i_hist(i_idx)=i_hist(i_idx)+1
                end do
                i_sum=0
                i_idx=0
                do i=0,11
c                 write(6,*) 'Hist ',i,i_hist(i)
                  i_sum=i_sum+i_hist(i)
                  if (i_sum .le. i_cnt*r_lvl) i_idx=i+1
                end do
c               write(6,*) 'idx = ',i_idx
                r_low=(r_rng*(float(i_idx-1)/10))+r_low
                r_hgh=r_low+r_rng/10
              end do
              r_med = 0
              i_num = 0
              do i=1,i_cnt
                 if ((r_val(i) .ge. r_low) .and. (r_val(i) .le. r_hgh)) then
                   i_num=i_num+1
c                  write(6,*) 'idx,r_val=',i_num,r_val(i)
                   r_med = r_med+r_val(i)
                 end if
              end do
              if (i_num .gt. 0) then
                r_med=r_med/i_num
              else
                r_med=(r_low+r_hgh)/2
              end if
              i_low=0
              i_hgh=0
              do i=1,i_cnt
                 if (r_val(i) .lt. r_med) i_low=i_low+1
                 if (r_val(i) .gt. r_med) i_hgh=i_hgh+1
              end do
c             write(6,*) 'balance=',i_low,i_hgh,i_low/float(i_low+i_hgh)
            end if
          return
        end


#ifdef IO64
        integer*8 function i_getfsize(i_setunit)

        implicit none

        integer*4 i_setunit
        integer*8 i_mbytes,i_fbytes

        integer*8 ioseek64
        external ioseek64

          i_mbytes= 0 ! 1E11
          i_fbytes = ioseek64(i_setunit,i_mbytes,2)
          i_getfsize = i_fbytes+1

        end

#else
        integer*4 function i_getfsize(i_setunit)
        implicit none

        integer*4 i_setunit
        integer*4 i_mbytes,i_fbytes

        integer*4 ioseek
        external ioseek

          i_mbytes=0 ! 2147483647
          i_fbytes = ioseek(i_setunit,i_mbytes,2)
          i_getfsize = i_fbytes+1

        end
#endif

        integer function i_setvbyt(i_setvfmt)

        implicit none

        integer i_setvfmt
        integer i_bytes

          goto (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150), abs(i_setvfmt)+1
c           write (6,*) 'i_setvbyt error = ',i_bytes
            i_bytes=4
            goto 200

10        continue        ! 0 - byte (unsigned integer*1)
             i_bytes=1
             goto 200
20        continue        ! 1 - integer*1
             i_bytes=1
             goto 200
30        continue        ! 2 - integer*2
             i_bytes=2
             goto 200
40        continue        ! 3 - integer*4
             i_bytes=4
             goto 200
50        continue        ! 4 - real*4
             i_bytes=4
             goto 200
60        continue        ! 5 - real*8
             i_bytes=8
             goto 200
70        continue        ! 6 - complex magnitude
             i_bytes=8
             goto 200
80        continue        ! 7 - complex phase
             i_bytes=8
             goto 200
90        continue        ! 8 - unsigned integer*2
             i_bytes=2
             goto 200
100       continue        ! 9 - stokes11
             i_bytes=10
             goto 200
110       continue        ! 10 - Complex 2 magnitude
             i_bytes=2
             goto 200
120       continue        ! 11 - complex 2 phase
             i_bytes=2
             goto 200
130       continue        ! 12 - complex 4 magnitude
             i_bytes=4
             goto 200
140       continue        ! 13 - complex 4 phase
             i_bytes=4
             goto 200
150       continue        ! 14 - Real*4_Magnitude
             i_bytes=4
             goto 200
200       continue
          i_setvbyt = i_bytes
          return
        end

        subroutine get_ecmd(i_wkspace,i_ecnt,i_edat,i_ecmd,i_debug)

          ! This subroutine retrieves expose events from the expose event buffer
          ! If an expose event will exceed the size of the wkspace buffer, the event
          ! is divided into multiple parts and output one at a time
          !
          implicit none

          integer*4 i
          integer*4 ie
          integer*4 i_ecnt
          integer*4 i_edat(0:10,200)
          integer*4 i_ecmd(0:10)

          integer*4 i_row
          integer*4 i_ncx
          integer*4 i_nrx
          integer*4 i_scl
          integer*4 i_brow
          integer*4 i_blks
          integer*4 i_wkspace

          integer*4 i_debug

            do i=0,10
              i_ecmd(i) = i_edat(i,1)
            end do

            i_row = i_edat(4,1)
            i_ncx = i_edat(5,1)
            i_nrx = i_edat(6,1) ! number of lines in expose event
            i_scl = i_edat(7,1)
            i_brow = i_wkspace/i_ncx
            i_blks = (i_nrx-1)/i_brow
            if ((i_blks .gt. 0 .or. i_edat(8,1) .ne. i_edat(4,1)) .and. i_debug .ge. 4)
     &                write(6,*) 'Splitting expose, block ',i_blks,i_row,i_nrx
            if (i_scl .ne. 1) then
              i_ecmd(4)   = i_edat(4,1)
              i_edat(4,1) = i_edat(4,1)+min(i_brow,i_nrx)
            else
              i_ecmd(4) = i_row+max(0,i_nrx-i_brow)
            end if
            i_ecmd(6) = min(i_brow,i_nrx)
            i_ecmd(7) = i_blks

c            if (i_blks .eq. 1 .and. i_nrx .eq. i_brow) i_ecmd(7)= 0        ! From Ron M.  ! should be unnecessary with the change to the addition of a -1 in the blks calculation

            i_edat(6,1) = max(0,i_edat(6,1)-i_ecmd(6))
            if (i_edat(6,1) .le. 0) then
              i_ecnt = max(i_ecnt-1,0)
              do ie=1,i_ecnt    ! Move Items up in buffer
                do i=0,10
                  i_edat(i,ie) = i_edat(i,ie+1)
                end do
              end do
            end if

          return
        end

        real function wrap(r_value,r_wrap)

          implicit none

          real*4 r_value
          real*4 r_wrap
          real*4 r_outp

c            r_outp=r_value-r_wrap*nint((r_value/r_wrap)-0.5)
c           if (r_outp .eq. r_wrap) r_outp=0.
            r_outp = mod(r_value,r_wrap)
            if (r_value .lt. 0) r_outp=r_wrap-abs(r_outp)
c           if (r_wrap .eq. 100) write(6,*) '*** value,wrap,mod,outp: ',r_value,r_wrap,mod(r_value,r_wrap),r_outp
          wrap=r_outp
          return

        end

        subroutine vecmulti(r_a,r_b,r_c)

          implicit none

          real*8 r_a(3,3)
          real*8 r_b(3)
          real*8 r_c(3)

          r_c(1)=r_a(1,1)*r_b(1)+r_a(1,2)*r_b(2)+r_a(1,3)*r_b(3)
          r_c(2)=r_a(2,1)*r_b(1)+r_a(2,2)*r_b(2)+r_a(2,3)*r_b(3)
          r_c(3)=r_a(3,1)*r_b(1)+r_a(3,2)*r_b(2)+r_a(3,3)*r_b(3)

          return

        end


        subroutine vecscale(r_scale,r_a,r_b)

          implicit none

          real*8 r_scale
          real*8 r_a(3)
          real*8 r_b(3)

            r_b(1)=r_scale*r_a(1)
            r_b(2)=r_scale*r_a(2)
            r_b(3)=r_scale*r_a(3)

            return

        end

        subroutine vecaddit(r_a,r_b,r_c)

          implicit none

          real*8 r_a(3)
          real*8 r_b(3)
          real*8 r_c(3)

            r_c(1)=r_a(1)+r_b(1)
            r_c(2)=r_a(2)+r_b(2)
            r_c(3)=r_a(3)+r_b(3)

            return

        end


c****************************************************************

        subroutine tcnatm(r_a,r_e2,r_peg,r_atm)

c****************************************************************
c**
c**     FILE NAME: tcnatm.for
c**
c**     DATE WRITTEN:10/25/95
c**
c**     PROGRAMMER:Scott Shaffer
c**
c**     FUNCTIONAL DESCRIPTION:This routine computes the transformation
c**     matris and translation vector needed to get between radar (t,c,n)
c**     coordinates and (x,y,z) WGS-84 coordinates.
c**
c**     ROUTINES CALLED:
c**
c**     NOTES: none
c**
c**     UPDATE LOG:
c**
c*****************************************************************

        implicit none

c       INPUT VARIABLES:
        real*8 r_a                    !semimajor axis
        real*8 r_e2                   !eccentricity squared
        real*8 r_peg(3)               !peg latitude,longitude,heading

c       OUTPUT VARIABLES:
        real*8 r_atm(3,4)             !rotation matris

c       LOCAL VARIABLES:
        integer i_type
        real*8 r_hgt
        real*8 r_slt,r_clt,r_clo,r_slo,r_chg,r_shg

        real*8 rdir
        external rdir

c       DATA STATEMENTS:none

c       PROCESSING STEPS:

c       first determine the rotation matris

        r_clt = cos(r_peg(1))
        r_slt = sin(r_peg(1))
        r_clo = cos(r_peg(2))
        r_slo = sin(r_peg(2))
        r_chg = cos(r_peg(3))
        r_shg = sin(r_peg(3))

        r_atm(1,1) = - r_slo*r_shg - r_slt*r_clo*r_chg
        r_atm(1,2) =   r_slo*r_chg - r_slt*r_clo*r_shg
        r_atm(1,3) =   r_clt*r_clo
        r_atm(2,1) =   r_clo*r_shg - r_slt*r_slo*r_chg
        r_atm(2,2) = - r_clo*r_chg - r_slt*r_slo*r_shg
        r_atm(2,3) =   r_clt*r_slo
        r_atm(3,1) =   r_clt*r_chg
        r_atm(3,2) =   r_clt*r_shg
        r_atm(3,3) =   r_slt

c       find the translation vector

        i_type = 1
        r_hgt = 0.
        call latlon(r_a,r_e2,r_atm(1,4),r_peg(1),r_peg(2),r_hgt,i_type)

        return
        end

c****************************************************************
        subroutine latlon(r_a,r_e2,r_v,r_lat,r_lon,r_hgt,i_type)

c****************************************************************
c**
c**   FILE NAME: latlon.f
c**
c**   DATE WRITTEN:7/22/93
c**
c**   PROGRAMMER:Scott Hensley
c**
c**   FUNCTIONAL DESCRIPTION:This program converts a vector to
c**   lat,lon and height above the reference ellipsoid or given a
c**   lat,lon and height produces a geocentric vector.
c**
c**   ROUTINES CALLED:none
c**
c**   NOTES: none
c**
c**   UPDATE LOG:
c**
c****************************************************************

        implicit none

c     INPUT VARIABLES:
        integer i_type                   !1=lat,lon to vector,2= vector to lat,lon
        real*8 r_a                       !ellispoid semi-major axis
        real*8 r_e2                      !ellipsoid eccentricity squared
        real*8 r_v(3)                    !geocentric vector (meters)
        real*8 r_lat                     !latitude (deg -90 to 90)
        real*8 r_lon                     !longitude (deg -180 to 180)
        real*8 r_hgt                     !height above ellipsoid (meters)

c       OUTPUT VARIABLES:see input

c       LOCAL VARIABLES:
        integer i_ft
        real*8 pi,r_dtor,r_re,r_q2,r_q3,r_b,r_q
        real*8 r_p,r_tant,r_theta

c       DATA STATEMENTS:
        data pi /3.141592653589793238d0/
        data r_dtor /1.74532925199d-2/
        data i_ft /0/

C       FUNCTION STATEMENTS:

c       PROCESSING STEPS:

        if(i_type .eq. 1)then  !convert lat,lon to vector

           r_re = r_a/sqrt(1.d0 - r_e2*sin(r_lat)**2)

           r_v(1) = (r_re + r_hgt)*cos(r_lat)*cos(r_lon)
           r_v(2) = (r_re + r_hgt)*cos(r_lat)*sin(r_lon)
           r_v(3) = (r_re*(1.d0-r_e2) + r_hgt)*sin(r_lat)

        elseif(i_type .eq. 2)then  !convert vector to lat,lon

           if(i_ft .eq. 0)then
              r_q2 = 1.d0/(1.d0 - r_e2)
              r_q = sqrt(r_q2)
              r_q3 = r_q2 - 1.d0
              r_b = r_a*sqrt(1.d0 - r_e2)
           end if

           r_lon = atan2(r_v(2),r_v(1))

           r_p = sqrt(r_v(1)**2 + r_v(2)**2)
           r_tant = (r_v(3)/r_p)*r_q
           r_theta = atan(r_tant)
           r_tant = (r_v(3) + r_q3*r_b*sin(r_theta)**3)/
     +          (r_p - r_e2*r_a*cos(r_theta)**3)
           r_lat =  atan(r_tant)
           r_re = r_a/sqrt(1.d0 - r_e2*sin(r_lat)**2)
           r_hgt = r_p/cos(r_lat) - r_re

        end if

        return
        end

c****************************************************************
        subroutine sch_to_tcn(r_a,r_v,r_lat,r_lon,r_hgt,i_type)

c****************************************************************
c**
c**   FILE NAME: sch_to_tcn.f
c**
c**   DATE WRITTEN:7/22/93
c**
c**   PROGRAMMER:Scott Hensley
c**
c**   FUNCTIONAL DESCRIPTION:This program converts a vector to
c**   lat,lon and height above the reference ellipsoid or given a
c**   lat,lon and height produces a geocentric vector.
c**
c**   ROUTINES CALLED:none
c**
c**   NOTES: none
c**
c**   UPDATE LOG:
c**
c****************************************************************

        implicit none

c     INPUT VARIABLES:
        integer i_type                   !1=lat,lon to vector,2= vector to lat,lon
        real*8 r_a                       !ellispoid semi-major axis
        real*8 r_v(3)                    !geocentric vector (meters)
        real*8 r_lat                     !latitude (deg -90 to 90)
        real*8 r_lon                     !longitude (deg -180 to 180)
        real*8 r_hgt                     !height above ellipsoid (meters)

c       OUTPUT VARIABLES:see input

c       LOCAL VARIABLES:
        real*8 r_p

C       FUNCTION STATEMENTS:

c       PROCESSING STEPS:

        if(i_type .eq. 1)then  !convert lat,lon to vector

           r_v(3) = (r_a + r_hgt)*cos(r_lat)*cos(r_lon) - r_a
           r_v(1) = (r_a + r_hgt)*cos(r_lat)*sin(r_lon)
           r_v(2) = (r_a + r_hgt)*sin(r_lat)

        elseif(i_type .eq. 2)then  !convert vector to lat,lon, hgt

           r_p = sqrt(r_v(1)**2 + r_v(2)**2 + (r_v(3)+r_a)**2)
           r_lat = asin(r_v(2)/r_p)
           r_lon = atan2(r_v(1),(r_v(3)+r_a))
           r_hgt = r_p - r_a

        end if

        return
        end

c****************************************************************
c
c       Various curvature functions
c
c
c****************************************************************
c**
c**     FILE NAME: curvature.f
c**
c**     DATE WRITTEN: 12/02/93
c**
c**     PROGRAMMER:Scott Hensley
c**
c**     FUNCTIONAL DESCRIPTION: This routine computes the curvature for
c**     of various types required for ellipsoidal or spherical earth
c**     calculations.
c**
c**     ROUTINES CALLED:none
c**
c**     NOTES: none
c**
c**     UPDATE LOG:
c**
c*****************************************************************

        real*8 function  reast(r_a,r_e2,r_lat)

        implicit none
        real*8 r_a,r_e2,r_lat

        reast = r_a/sqrt(1.d0 - r_e2*sin(r_lat)**2)

        end

        real*8 function  rnorth(r_a,r_e2,r_lat)

        implicit none
        real*8 r_a,r_e2,r_lat

        rnorth = (r_a*(1.d0 - r_e2))/
     1                  (1.d0 - r_e2*sin(r_lat)**2)**(1.5d0)

        end


        real*8 function  rdir(r_a,r_e2,r_hdg,r_lat)

        implicit none
        real*8 r_a,r_e2,r_lat,r_hdg,r_re,r_rn,reast,rnorth

        r_re = reast(r_a,r_e2,r_lat)
        r_rn = rnorth(r_a,r_e2,r_lat)

        rdir = (r_re*r_rn)/(r_re*cos(r_hdg)**2 + r_rn*sin(r_hdg)**2)

        end

c****************************************************************

        subroutine utmtoll(r_a,r_e2,i_zone,a_grid,r_vec,r_lat,
     +     r_lon,i_type)

c****************************************************************
c**
c**   FILE NAME: utmtoll.f
c**
c**   DATE WRITTEN:7/22/93
c**
c**   PROGRAMMER:Scott Hensley
c**
c**   FUNCTIONAL DESCRIPTION: This routine converts between lat
c**   lon and utm coordinates for a datum determined from the input
c**   a and e2.
c**
c**   ROUTINES CALLED:none
c**
c**   NOTES: none
c**
c**   UPDATE LOG:
c**
c****************************************************************

        implicit none

c     INPUT VARIABLES:
        integer i_type                   !1=lat,lon to utm,2= utm to lat,lon
        real*8 r_a                       !ellispoid semi-major axis
        real*8 r_e2                      !ellipsoid eccentricity squared
        real*8 r_vec(2)                    !Northing,Easting(m)
        real*8 r_lat                     !latitude (deg -90 to 90)
        real*8 r_lon                     !longitude (deg -180 to 180)
        integer i_zone                   !UTM zone
        character*1 a_grid               !UTM North-South grid

c       OUTPUT VARIABLES:see input

c       LOCAL VARIABLES:
        integer i_ft,i_gi
        real*8 r_v(2)                    !Northing,Easting(m)
        real*8 pi,r_dtor
        real*8 r_ep2,r_k0,r_k
        real*8 r_fe,r_fn(2)
        real*8 r_e4,r_e6,r_n,r_t,r_t2,r_c,r_c2,r_ba
        real*8 r_a2,r_a3,r_a4,r_a5,r_a6
        real*8 r_d,r_d2,r_d3,r_d4,r_d5,r_d6
        real*8 r_lon0,r_lat1,r_m,r_m0,r_mu,r_lat0
        real*8 r_et,r_e1,r_e12,r_e13,r_e14,r_r
        character*1 a_griddes(20)

c       DATA STATEMENTS:
        data pi /3.141592653589793238d0/
        data r_dtor /1.74532925199d-2/
        data i_ft /0/
        data a_griddes /'C','D','E','F','G','H','J',
     +       'K','L','M','N','P','Q','R','S','T','U',
     +       'V','W','X'/
        data r_k0 /.9996d0/    !scale at center
        data r_lat0 /0.d0/
        data r_fe,r_fn /500000.d0,0.d0,10000000.d0/

C       FUNCTION STATEMENTS:none

c       PROCESSING STEPS:

        r_ep2 = r_e2/(1.d0 - r_e2)
        r_e4 = r_e2**2
        r_e6 = r_e2**3
        pi =  4.d0*atan(1.d0)
        r_dtor = pi/180.d0

        if (i_zone .le. 0) i_zone = int(mod(r_lon+3.d0*pi,2.d0*pi)/(r_dtor*6.d0))
     +          + 1

        if(i_type .eq. 2)then  !convert lat,lon to UTM

           i_zone = max(min(i_zone,60),1)
           r_lon0 = -pi + 6.d0*r_dtor*(i_zone-1) + 3.d0*r_dtor

           r_n = r_a/sqrt(1.d0 - r_e2*sin(r_lat)**2)
           r_t = tan(r_lat)**2
           r_t2 = r_t**2
           r_c = r_ep2*cos(r_lat)**2
           r_ba = (r_lon - r_lon0)*cos(r_lat)
           r_a2 = r_ba**2
           r_a3 = r_ba*r_a2
           r_a4 = r_ba*r_a3
           r_a5 = r_ba*r_a4
           r_a6 = r_ba*r_a5
           r_m = r_a*((1.d0-r_e2/4 - 3.d0*r_e4/64.d0 -
     +      5.d0*r_e6/256.d0)*r_lat - (3.d0*r_e2/8.d0 +
     +          3.d0*r_e4/32.d0 +
     +      45.d0*r_e6/1024.d0)*sin(2.d0*r_lat) +
     +          (15.d0*r_e4/256.d0 +
     +      45.d0*r_e6/1024.d0)*sin(4.d0*r_lat) -
     +          (35.d0*r_e6/3072.d0)*
     +      sin(6.d0*r_lat))
           r_m0 = r_a*((1.d0-r_e2/4 - 3.d0*r_e4/64.d0 -
     +      5.d0*r_e6/256.d0)*r_lat0 - (3.d0*r_e2/8.d0 +
     +          3.d0*r_e4/32.d0 +
     +      45.d0*r_e6/1024.d0)*sin(2.d0*r_lat0) +
     +          (15.d0*r_e4/256.d0 +
     +      45.d0*r_e6/1024.d0)*sin(4.d0*r_lat0) -
     +          (35.d0*r_e6/3072.d0)*
     +      sin(6.d0*r_lat0))

           r_vec(2) = r_k0*r_n*(r_ba+(1.d0-r_t+r_c)*r_a3/6.d0 +
     +       (5.d0-18.d0*r_t+r_t2+72.d0*r_c-58.d0*r_ep2)*r_a5/120.d0)
           r_vec(2) = r_vec(2) + r_fe

           r_vec(1) = r_k0*(r_m - r_m0 + r_n*tan(r_lat)*
     +       ( r_a2/2.d0 + (5.d0-r_t+9.d0*r_c+4.d0*r_c**2)*
     +        (r_a4/24.d0) + (61.d0-58.d0*r_t+r_t2+600.d0*r_c-
     +        330.d0*r_ep2)*(r_a6/720.d0) ))
           if(r_lat .ge. 0)then
              r_vec(1) = r_vec(1) + r_fn(1)
           else
              r_vec(1) = r_vec(1) + r_fn(2)
           end if

           r_k = r_k0*(1.d0+(1.d0+r_ep2*cos(r_lat)**2)*
     +          (r_vec(2)-r_fe)**2/
     +          (2.d0*(r_k0**2)*r_n**2))

           i_gi = int((r_lat/r_dtor+80.d0)/8.d0) + 1
           i_gi = max(min(i_gi,20),1)
           a_grid = a_griddes(i_gi)

        elseif(i_type .eq. 1)then  !convert UTM to lat,lon

           r_v(1) = r_vec(1)
           r_v(2) = r_vec(2)
           r_v(2) = r_v(2) - r_fe
           if(r_v(1) .ge. r_fn(2))then
              r_v(1) = r_v(1) - r_fn(2)
           end if
           r_lon0 = -pi + 6.d0*r_dtor*(i_zone-1) + 3.d0*r_dtor

           r_et = sqrt(1.d0-r_e2)
           r_e1 = (1.d0-r_et)/(1.d0+r_et)
           r_e12 = r_e1**2
           r_e13 = r_e1*r_e12
           r_e14 = r_e1*r_e13
           r_m = r_v(1)/r_k0
           r_mu = r_m/(r_a*(1.d0-r_e2/4.d0-3.d0*r_e4/64.d0-
     +          5.d0*r_e6/256.d0))
           r_lat1 = r_mu + (3.d0*r_e1/2.d0-27.d0*r_e13/32.d0)*
     +          sin(2.d0*r_mu)+
     +          (21.d0*r_e12/16.d0-55.d0*r_e14/32.d0)*sin(4.d0*r_mu)+
     +          (51.d0*r_e13/96.d0)*sin(6.d0*r_mu) +
     +          (1097.d0*r_e14/512.d0)*sin(8.d0*r_mu)

           r_n = r_a/sqrt(1.d0 - r_e2*sin(r_lat1)**2)
           r_r = (r_a*(1.d0-r_e2))/sqrt(1.d0 - r_e2*sin(r_lat1)**2)**3
           r_t = tan(r_lat1)**2
           r_t2 = r_t**2
           r_c = r_ep2*cos(r_lat1)**2
           r_c2 = r_c**2
           r_d = r_v(2)/(r_n*r_k0)
           r_d2 = r_d**2
           r_d3 = r_d2*r_d
           r_d4 = r_d3*r_d
           r_d5 = r_d4*r_d
           r_d6 = r_d5*r_d

           r_lat = r_lat1 - (r_n*tan(r_lat1)/r_r)*(r_d2/2.d0+
     +       (5.d0+3.d0*r_t+10.d0*r_c-4.d0*r_c2-9.d0*r_ep2)*
     +          r_d4/24.d0 +
     +       (61.d0+90*r_t+298.d0*r_c+45.d0*r_t2-252.d0*r_ep2-3.d0*
     +          r_c2)*
     +       (r_d6/720.d0))
           r_lon = r_lon0 + (r_d - (1.d0+2.d0*r_t+r_c)*r_d3/6.d0 +
     +       (5.d0-2.d0*r_c+28.d0*r_t-3.d0*r_c2+8.d0*r_ep2+
     +          24.d0*r_t2)*
     +       (r_d5/120.d0))/cos(r_lat1)

        end if

        end

c****************************************************************

        subroutine enutoll(r_a,r_e2,i_zone,a_grid,r_vec,r_lat,
     +     r_lon,i_type)

c****************************************************************
c**
c**   FILE NAME: enutoll.f
c**
c**   DATE WRITTEN:7/22/93
c**
c**   PROGRAMMER:Scott Hensley
c**
c**   FUNCTIONAL DESCRIPTION: This routine converts between lat
c**   lon and enu coordinates for a datum determined from the input
c**   a and e2.
c**
c**   ROUTINES CALLED:none
c**
c**   NOTES: none
c**
c**   UPDATE LOG:  added zone selection logic SJS 3/28/96
c**
c****************************************************************

        implicit none

c     INPUT VARIABLES:
        integer i_type                   !2=lat,lon to utm,1= utm to lat,lon
        real*8 r_a                       !ellispoid semi-major axis
        real*8 r_e2                      !ellipsoid eccentricity squared
        real*8 r_vec(2)                  !Northing,Easting(m)
        real*8 r_lat                     !latitude (deg -90 to 90)
        real*8 r_lon                     !longitude (deg -180 to 180)
        integer i_zone                   !UTM zone
        character*1 a_grid               !UTM North-South grid

c       OUTPUT VARIABLES:see input

c       LOCAL VARIABLES:
        integer i_ft,i_gi
        real*8 pi,r_dtor
        real*8 r_v(2)                    !Northing,Easting(m)
        real*8 r_ep2,r_k0,r_k
        real*8 r_fe,r_fn(2)
        real*8 r_e4,r_e6,r_n,r_t,r_t2,r_c,r_c2,r_ba
        real*8 r_a2,r_a3,r_a4,r_a5,r_a6
        real*8 r_d,r_d2,r_d3,r_d4,r_d5,r_d6
        real*8 r_lon0,r_lat1,r_m,r_m0,r_mu,r_lat0
        real*8 r_et,r_e1,r_e12,r_e13,r_e14,r_r
        character*1 a_griddes(20)

c       DATA STATEMENTS:
        data pi /3.141592653589793238d0/
        data r_dtor /1.74532925199d-2/
        data i_ft /0/
        data a_griddes /'C','D','E','F','G','H','J',
     +       'K','L','M','N','P','Q','R','S','T','U',
     +       'V','W','X'/
        data r_k0 /.9996d0/    !scale at center
        data r_lat0 /0.d0/
        data r_fe,r_fn /500000.d0,0.d0,10000000.d0/

C       FUNCTION STATEMENTS:none

c       PROCESSING STEPS:

        r_ep2 = r_e2/(1.d0 - r_e2)
        r_e4 = r_e2**2
        r_e6 = r_e2**3
        pi =  4.d0*atan(1.d0)
        r_dtor = pi/180.d0

        if(i_type .eq. 2)then  !convert lat,lon to UTM

           if (i_zone .le. 0) i_zone = int(mod(r_lon+3.d0*pi,2.d0*pi)/(r_dtor*6.d0))
     +          + 1

           i_zone = max(min(i_zone,60),1)
           r_lon0 = -pi + 6.d0*r_dtor*(i_zone-1) + 3.d0*r_dtor

           r_n = r_a/sqrt(1.d0 - r_e2*sin(r_lat)**2)
           r_t = tan(r_lat)**2
           r_t2 = r_t**2
           r_c = r_ep2*cos(r_lat)**2
           r_ba = (r_lon - r_lon0)*cos(r_lat)
           r_a2 = r_ba**2
           r_a3 = r_ba*r_a2
           r_a4 = r_ba*r_a3
           r_a5 = r_ba*r_a4
           r_a6 = r_ba*r_a5
           r_m = r_a*((1.d0-r_e2/4 - 3.d0*r_e4/64.d0 -
     +      5.d0*r_e6/256.d0)*r_lat - (3.d0*r_e2/8.d0 +
     +          3.d0*r_e4/32.d0 +
     +      45.d0*r_e6/1024.d0)*sin(2.d0*r_lat) +
     +          (15.d0*r_e4/256.d0 +
     +      45.d0*r_e6/1024.d0)*sin(4.d0*r_lat) -
     +          (35.d0*r_e6/3072.d0)*
     +      sin(6.d0*r_lat))
           r_m0 = r_a*((1.d0-r_e2/4 - 3.d0*r_e4/64.d0 -
     +      5.d0*r_e6/256.d0)*r_lat0 - (3.d0*r_e2/8.d0 +
     +          3.d0*r_e4/32.d0 +
     +      45.d0*r_e6/1024.d0)*sin(2.d0*r_lat0) +
     +          (15.d0*r_e4/256.d0 +
     +      45.d0*r_e6/1024.d0)*sin(4.d0*r_lat0) -
     +          (35.d0*r_e6/3072.d0)*
     +      sin(6.d0*r_lat0))

           r_vec(1) = r_k0*r_n*(r_ba+(1.d0-r_t+r_c)*r_a3/6.d0 +
     +       (5.d0-18.d0*r_t+r_t2+72.d0*r_c-58.d0*r_ep2)*r_a5/120.d0)
           r_vec(1) = r_vec(1) + r_fe

           r_vec(2) = r_k0*(r_m - r_m0 + r_n*tan(r_lat)*
     +       ( r_a2/2.d0 + (5.d0-r_t+9.d0*r_c+4.d0*r_c**2)*
     +        (r_a4/24.d0) + (61.d0-58.d0*r_t+r_t2+600.d0*r_c-
     +        330.d0*r_ep2)*(r_a6/720.d0) ))
           if(r_lat .ge. 0)then
              r_vec(2) = r_vec(2) + r_fn(1)
           else
              r_vec(2) = r_vec(2) + r_fn(2)
           end if

           r_k = r_k0*(1.d0+(1.d0+r_ep2*cos(r_lat)**2)*
     +          (r_vec(1)-r_fe)**2/
     +          (2.d0*(r_k0**2)*r_n**2))

           i_gi = int((r_lat/r_dtor+80.d0)/8.d0) + 1
           i_gi = max(min(i_gi,20),1)
           a_grid = a_griddes(i_gi)

        else if(i_type .eq. 1)then  !convert UTM to lat,lon

           r_v(1) = r_vec(1)
           r_v(2) = r_vec(2)
           r_v(1) = r_v(1) - r_fe
           if(r_v(2) .ge. r_fn(2))then
              r_v(2) = r_v(2) - r_fn(2)
           end if
           r_lon0 = -pi + 6.d0*r_dtor*(i_zone-1) + 3.d0*r_dtor

           r_et = sqrt(1.d0-r_e2)
           r_e1 = (1.d0-r_et)/(1.d0+r_et)
           r_e12 = r_e1**2
           r_e13 = r_e1*r_e12
           r_e14 = r_e1*r_e13
           r_m = r_v(2)/r_k0
           r_mu = r_m/(r_a*(1.d0-r_e2/4.d0-3.d0*r_e4/64.d0-
     +          5.d0*r_e6/256.d0))
           r_lat1 = r_mu + (3.d0*r_e1/2.d0-27.d0*r_e13/32.d0)*
     +          sin(2.d0*r_mu)+
     +          (21.d0*r_e12/16.d0-55.d0*r_e14/32.d0)*sin(4.d0*r_mu)+
     +          (51.d0*r_e13/96.d0)*sin(6.d0*r_mu) +
     +          (1097.d0*r_e14/512.d0)*sin(8.d0*r_mu)

           r_n = r_a/sqrt(1.d0 - r_e2*sin(r_lat1)**2)
           r_r = (r_a*(1.d0-r_e2))/sqrt(1.d0 - r_e2*sin(r_lat1)**2)**3
           r_t = tan(r_lat1)**2
           r_t2 = r_t**2
           r_c = r_ep2*cos(r_lat1)**2
           r_c2 = r_c**2
           r_d = r_v(1)/(r_n*r_k0)
           r_d2 = r_d**2
           r_d3 = r_d2*r_d
           r_d4 = r_d3*r_d
           r_d5 = r_d4*r_d
           r_d6 = r_d5*r_d

           r_lat = r_lat1 - (r_n*tan(r_lat1)/r_r)*(r_d2/2.d0+
     +       (5.d0+3.d0*r_t+10.d0*r_c-4.d0*r_c2-9.d0*r_ep2)*
     +          r_d4/24.d0 +
     +       (61.d0+90*r_t+298.d0*r_c+45.d0*r_t2-252.d0*r_ep2-3.d0*
     +          r_c2)*
     +       (r_d6/720.d0))
           r_lon = r_lon0 + (r_d - (1.d0+2.d0*r_t+r_c)*r_d3/6.d0 +
     +       (5.d0-2.d0*r_c+28.d0*r_t-3.d0*r_c2+8.d0*r_ep2+
     +          24.d0*r_t2)*
     +       (r_d5/120.d0))/cos(r_lat1)

        end if

        end

        subroutine invrstrn(r_atm,r_mta)
c
c  This subroutine finds the inverse of an affine transformation
c                  including the translation vector
c
          implicit none

          real*8 r_atm(3,4)
          real*8 r_mta(3,4)
          real*8 r_tmp(3)
          real*8 r_one

            r_one = -1.0

            call matinvrt(r_atm,r_mta)
            call vecmulti(r_mta,r_atm(1,4),r_tmp)
            call vecscale(r_one,r_tmp,r_mta(1,4))

            return
        end

        subroutine matinvrt(r_a,r_b)

          implicit none

          real*8 a11
          real*8 a12
          real*8 a13
          real*8 a21
          real*8 a22
          real*8 a23
          real*8 a31
          real*8 a32
          real*8 a33

          real*8 r_a(3,3)
          real*8 r_b(3,3)

          real*8 r_dd

          a11=r_a(1,1)
          a12=r_a(1,2)
          a13=r_a(1,3)
          a21=r_a(2,1)
          a22=r_a(2,2)
          a23=r_a(2,3)
          a31=r_a(3,1)
          a32=r_a(3,2)
          a33=r_a(3,3)

          r_dd=a11*(a22*a33-a23*a32)-a12*(a21*a33-a23*a31)+
     &         a13*(a21*a32-a22*a31)

          if (r_dd .ne. 0.) then
            r_b(1,1)=(a22*a33-a23*a32)/r_dd
            r_b(1,2)=(a13*a32-a12*a33)/r_dd
            r_b(1,3)=(a12*a23-a13*a22)/r_dd
            r_b(2,1)=(a23*a31-a21*a33)/r_dd
            r_b(2,2)=(a11*a33-a13*a31)/r_dd
            r_b(2,3)=(a13*a21-a11*a23)/r_dd
            r_b(3,1)=(a21*a32-a22*a31)/r_dd
            r_b(3,2)=(a12*a31-a11*a32)/r_dd
            r_b(3,3)=(a11*a22-a12*a21)/r_dd
          else
            write(6,*) 'Determinant =  0 in Subroutine matinvrt'
            r_b(1,1)=1.
            r_b(1,2)=0.
            r_b(1,3)=0.
            r_b(2,1)=0.
            r_b(2,2)=1.
            r_b(2,3)=0.
            r_b(3,1)=0.
            r_b(3,2)=0.
            r_b(3,3)=1.
          endif

          return

        end


      subroutine get_coordinates(a_setproj,r_setpegv,r_loc11,r_loc22,i_flag,i_debug,i_err)

        implicit none

        character*200 a_setproj      ! Projection name

        integer*4     i_flag
        integer*4     i_debug
        integer*4     i_err
        real*4        r_setpegv(3)   ! Peg Point
        real*4        r_row
        real*4        r_col
        real*4        r_val

        real*4        r_loc11(3)
        real*4        r_loc22(3)

        real*8        r_setpegvdble(3)   ! Peg Point
        real*8        r_loc1(3)
        real*8        r_loc2(3)
        real*8        r_loc3(3)
        real*8        r_loc4(3)
        real*8        r_rtod
        real*8        r_rad
        real*8        r_hhh
        real*8        r_lat
        real*8        r_lon

        real*8 r_pi
        real*8 r_e2
        real*8 r_a

        real*8 r_atm(3,4)
        real*8 r_mta(3,4)
        integer*4     i_zone

        character*1 a_grid               !UTM North-South grid

        integer rdflen
        external rdflen

        real*8 rdir
        external rdir

c
c     Initialize pi and conversions
c
            r_pi = 4.d0*atan(1.0d0)
            r_rtod = 180.0d0/r_pi
            r_a =  6378137.0
            r_e2 = 0.00669438
            r_setpegvdble(1) = r_setpegv(1)
            r_setpegvdble(2) = r_setpegv(2)
            r_setpegvdble(3) = r_setpegv(3)
            i_zone = 0
            r_lon=r_setpegv(2)

            r_rad = rdir(r_a,r_e2,r_setpegvdble(3),r_setpegvdble(1))

            i_err=0

            if (i_flag .eq. 1) then  ! convert row/column to lat/lon

              if (i_debug .eq. -6 .or. i_debug .ge. 6) write(6,*) 'r_peglat=',r_setpegv(1)
              if (i_debug .eq. -6 .or. i_debug .ge. 6) write(6,*) 'r_peglon=',r_setpegv(2)
              if (i_debug .eq. -6 .or. i_debug .ge. 6) write(6,*) 'r_peghdg=',r_setpegv(3)

              if (i_debug .eq. -6 .or. i_debug .ge. 6) write(6,*) 'r_rad=',r_rad

              r_loc1(1) = r_loc11(1)
              r_loc1(2) = r_loc11(2)
              r_loc1(3) = r_loc11(3)
              if (i_debug .ge. 6) write(6,*) 'r_loc1=',r_loc1

              if (a_setproj .eq. 'sch' .or. a_setproj .eq. 'SCH') then
                 r_lon=r_loc1(1)/r_rad
                 r_lat=r_loc1(2)/r_rad
                 r_hhh=r_loc1(3)
                 call tcnatm(r_a,r_e2,r_setpegvdble,r_atm)
                 call sch_to_tcn(r_rad,r_loc3,r_lat,r_lon,r_hhh,1)
                 call vecmulti(r_atm,r_loc3,r_loc4) ! convert from input xyz to output xyz
                 call vecaddit(r_atm(1,4),r_loc4,r_loc4)
                 call latlon(r_a,r_e2,r_loc4,r_lat,r_lon,r_hhh,2)
              else if (a_setproj .eq. 'scx' .or. a_setproj .eq. 'SCX') then
                 r_lon=r_loc1(1)/r_rad
                 r_lat=r_loc1(2)/r_rad
                 r_hhh=0
                 call tcnatm(r_a,r_e2,r_setpegvdble,r_atm)
                 call sch_to_tcn(r_rad,r_loc3,r_lat,r_lon,r_hhh,1)
                 call vecmulti(r_atm,r_loc3,r_loc4) ! convert from input xyz to output xyz
                 call vecaddit(r_atm(1,4),r_loc4,r_loc4)
                 call latlon(r_a,r_e2,r_loc4,r_lat,r_lon,r_hhh,2)
                 r_hhh=r_loc1(3)
              else if (a_setproj .eq. 'eqa' .or. a_setproj .eq. 'EQA') then
                 r_lat=r_loc1(1)/r_rtod
                 r_lon=r_loc1(2)/r_rtod
                 r_hhh=r_loc1(3)
              else if (a_setproj .eq. 'utm' .or. a_setproj .eq. 'UTM') then
                 call utmtoll(r_a,r_e2,i_zone,a_grid,r_loc1,r_lat,r_lon,1)
                 if (i_debug .ge. 6) write(6,*) 'i_zone=',i_zone
                 r_hhh=r_loc1(3)
              else if (a_setproj .eq. 'neu' .or. a_setproj .eq. 'NEU') then
                 call utmtoll(r_a,r_e2,i_zone,a_grid,r_loc1,r_lat,r_lon,1)
                 r_hhh=r_loc1(3)
              else if (a_setproj .eq. 'enu' .or. a_setproj .eq. 'ENU') then
                 call enutoll(r_a,r_e2,i_zone,a_grid,r_loc1,r_lat,r_lon,1)
                 r_hhh=r_loc1(3)
              else
                 i_err=1
                 if (i_debug .ge. 1) write(6,*) 'Lat/Long conversion not supported for ',a_setproj(1:max(1,rdflen(a_setproj)))
              end if
              r_loc22(1)=r_lat
              r_loc22(2)=r_lon
              r_loc22(3)=r_hhh

           else  ! convert lat/lon to row/column

              r_lat = r_loc22(1)
              r_lon = r_loc22(2)
              r_hhh = r_loc22(3)

              if (i_debug .eq. -6 .or. i_debug .ge. 6) write(6,*) 'r_lat=',r_lat
              if (i_debug .eq. -6 .or. i_debug .ge. 6) write(6,*) 'r_lon=',r_lon
              if (i_debug .eq. -6 .or. i_debug .ge. 6) write(6,*) 'r_hdg=',r_hhh

              if (i_debug .eq. -6 .or. i_debug .ge. 6) write(6,*) 'r_rad=',r_rad
              if (i_debug .eq. -6 .or. i_debug .ge. 6) write(6,*) 'Converting from lat/lon to row/column(almost) '//a_setproj

              if (a_setproj .eq. 'sch' .or. a_setproj .eq. 'SCH') then
                 call latlon(r_a,r_e2,r_loc1,r_lat,r_lon,r_hhh,1) ! convert lat/lon to wgs84 xyz
                 if (i_debug .eq. -9 .or. i_debug .ge. 9) write(6,*) 'wgs84xyz=',r_loc1
                 call tcnatm(r_a,r_e2,r_setpegvdble,r_atm)  ! compute tcn to xyz transform
                 call invrstrn(r_atm,r_mta)
                 call vecmulti(r_mta,r_loc1,r_loc2) ! convert from wgs84 xyz to tcn
                 call vecaddit(r_mta(1,4),r_loc2,r_loc2)
                 if (i_debug .eq. -6 .or. i_debug .ge. 6) write(6,*) 'tcnxyz=',r_loc2
                 call sch_to_tcn(r_rad,r_loc2,r_lat,r_lon,r_hhh,2) ! convert tcn to sch
                 r_loc11(1)=r_lon*r_rad ! Convert sch
                 r_loc11(2)=r_lat*r_rad
                 r_loc11(3)=r_hhh
              else if (a_setproj .eq. 'scx' .or. a_setproj .eq. 'SCX') then
                 r_hhh=0.0
                 call latlon(r_a,r_e2,r_loc1,r_lat,r_lon,r_hhh,1) ! convert lat/lon to wgs84 xyz
                 if (i_debug .eq. -9 .or. i_debug .ge. 9) write(6,*) 'wgs84xyz=',r_loc1
                 call tcnatm(r_a,r_e2,r_setpegvdble,r_atm)  ! compute tcn to xyz transform
c                 write(6,*) 'peg=',r_setpegv
                 call invrstrn(r_atm,r_mta)
                 call vecmulti(r_mta,r_loc1,r_loc2) ! convert from wgs84 xyz to tcn
                 call vecaddit(r_mta(1,4),r_loc2,r_loc2)
                 if (i_debug .eq. -9 .or. i_debug .ge. 9) write(6,*) 'tcnxyz=',r_loc2
                 call sch_to_tcn(r_rad,r_loc2,r_lat,r_lon,r_hhh,2) ! convert tcn to sch
                 r_loc11(1)=r_lon*r_rad ! Convert sch
                 r_loc11(2)=r_lat*r_rad
                 r_loc11(3)=r_hhh
              else if (a_setproj .eq. 'eqa' .or. a_setproj .eq. 'EQA') then
                 r_loc11(1)=r_lat*r_rtod
                 r_loc11(2)=r_lon*r_rtod
                 r_loc11(3)=r_hhh
              else
                 i_err=1
                 if (i_debug .ge. 1) write(6,*) 'Lat/Long output not supported for ',a_setproj(1:max(1,rdflen(a_setproj)))
              end if

           end if

        return
      end

      subroutine buffer_cmd(i_event,i_bdat,i_bcnt,i_base,I_BMAX,i_abort,i_debug)

      implicit none

      integer*4 I_BMAX
      integer*4 i_event(0:10)
      integer*4 i_bdat(0:10,I_BMAX)
      integer*4 i_bcnt
      integer*4 i_abort
      integer*4 i_base
      integer*4 i_debug

      integer*4 i
      integer*4 j
      integer*4 k
      integer*4 ii

c      if ((i_event(0) .eq. 0 .and. i_event(2) .eq. 0) .or. i_event(2) .eq. 9) then
        if ((i_event(0) .eq. 0 .and. i_event(2) .eq. 0) ) then
           ! do nothing
        else
          i_bcnt = min(i_bcnt+1,I_BMAX)
          do i=0,10
            i_bdat(i,i_bcnt) = i_event(i)
          end do

        end if
      return
      end


      subroutine get_colortable(a_colordir,a_dspctbl,i_dspnumt,r_dspredt,r_dspgrnt,r_dspblut,i_debug)

      implicit none

      character*(*) a_dspctbl
      character*(255) a_line
      character*(255) a_colordir
      character*(255) a_file
      integer*4     i_dspnumt
      integer*4     i_debug
      real*4        r_dspredt(0:255)
      real*4        r_dspgrnt(0:255)
      real*4        r_dspblut(0:255)

      integer*4     i_nrgb(0:3,0:256)

      integer*4     i, j, k
      integer*4     i_cnt
      integer*4     i_loc
      integer*4     ierr

      integer rdflen
      external rdflen

      if (i_debug .eq. -3 .or. i_debug .ge. 3) write(6,*) 'in get_colortable:',a_dspctbl,a_colordir
      if (a_dspctbl .eq. ' ' .or. a_dspctbl .eq. '?' .or. a_dspctbl .eq. '*') then
c         i_dspnumt=256
c         do i=0,255
c            r_dspredt(i) = 0.   ! Values of red   color table
c            r_dspgrnt(i) = 0.   ! Values of green color table
c            r_dspblut(i) = 0.   ! Values of blue  color table
c         end do
      else if (a_dspctbl .eq. 'bitmap') then
         i_dspnumt=2
         r_dspredt(0) = 0.   ! Values of red   color table
         r_dspgrnt(0) = 0.   ! Values of green color table
         r_dspblut(0) = 0.   ! Values of blue  color table
         r_dspredt(1) = 1.   ! Values of red   color table
         r_dspgrnt(1) = 1.   ! Values of green color table
         r_dspblut(1) = 1.   ! Values of blue  color table
      else if (a_dspctbl .eq. 'white') then
         i_dspnumt=2
         do i=0,i_dspnumt-1
            r_dspredt(i) = 1.0    ! Values of red   color table
            r_dspgrnt(i) = 1.0    ! Values of green color table
            r_dspblut(i) = 1.0    ! Values of blue  color table
         end do
      else if (a_dspctbl .eq. 'black') then
         i_dspnumt=2
         do i=0,i_dspnumt-1
            r_dspredt(i) = 0.0    ! Values of red   color table
            r_dspgrnt(i) = 0.0    ! Values of green color table
            r_dspblut(i) = 0.0    ! Values of blue  color table
         end do
      else if (a_dspctbl .eq. 'grey') then
         i_dspnumt=256
         do i=0,i_dspnumt-1
            r_dspredt(i) = max(0.,min(1.,i/(i_dspnumt-1.)))    ! Values of red   color table
            r_dspgrnt(i) = max(0.,min(1.,i/(i_dspnumt-1.)))    ! Values of green color table
            r_dspblut(i) = max(0.,min(1.,i/(i_dspnumt-1.)))    ! Values of blue  color table
         end do
      else if (a_dspctbl .eq. 'red') then
         i_dspnumt=256
         do i=0,i_dspnumt-1
            r_dspredt(i) = max(0.,min(1.,i/(i_dspnumt-1.)))    ! Values of red   color table
            r_dspgrnt(i) = 0    ! Values of green color table
            r_dspblut(i) = 0    ! Values of blue  color table
         end do
      else if (a_dspctbl .eq. 'green') then
         i_dspnumt=256
         do i=0,i_dspnumt-1
            r_dspredt(i) = 0    ! Values of red   color table
            r_dspgrnt(i) = max(0.,min(1.,i/(i_dspnumt-1.)))    ! Values of green color table
            r_dspblut(i) = 0    ! Values of blue  color table
         end do
      else if (a_dspctbl .eq. 'blue') then
         i_dspnumt=256
         do i=0,i_dspnumt-1
            r_dspredt(i) = 0    ! Values of red   color table
            r_dspgrnt(i) = 0    ! Values of green color table
            r_dspblut(i)=max(0.,min(1.,i/(i_dspnumt-1.)))    ! Values of blue  color table
         end do
      else if (a_dspctbl .eq. 'cmy') then
         do i=0,84
            r_dspredt(i) = i*3  ! Values of red   color table
            r_dspgrnt(i) = 255-i*3 ! Values of green color table
            r_dspblut(i) = 255  ! Values of blue  color table
         end do
         do i=0,84
            r_dspredt(i+85) = 255 ! Values of red   color table
            r_dspgrnt(i+85) = i*3 ! Values of green color table
            r_dspblut(i+85) = 255-i*3 ! Values of blue  color table
         end do
         do i=0,84
            r_dspredt(i+170) = 255-i*3 ! Values of red   color table
            r_dspgrnt(i+170) = 255 ! Values of green color table
            r_dspblut(i+170) = i*3 ! Values of blue  color table
         end do
         r_dspredt(255) = 0     ! Values of red   color table
         r_dspgrnt(255) = 255   ! Values of green color table
         r_dspblut(255) = 255   ! Values of blue  color table
         i_dspnumt=256
         do i=0,255
           r_dspredt(i)=max(0.,min(1.,r_dspredt(i)/255.))
           r_dspgrnt(i)=max(0.,min(1.,r_dspgrnt(i)/255.))
           r_dspblut(i)=max(0.,min(1.,r_dspblut(i)/255.))
         end do
      else if (a_dspctbl .eq. 'myc') then
         do i=0,84
            r_dspredt(i) = 255  ! Values of red   color table
            r_dspgrnt(i) = i*3  ! Values of green color table
            r_dspblut(i) = 255-i*3 ! Values of blue  color table
         end do
         do i=0,84
            r_dspredt(i+85) = 255-i*3 ! Values of red   color table
            r_dspgrnt(i+85) = 255 ! Values of green color table
            r_dspblut(i+85) = i*3 ! Values of blue  color table
         end do
         do i=0,84
            r_dspredt(i+170) = i*3 ! Values of red   color table
            r_dspgrnt(i+170) = 255-i*3 ! Values of green color table
            r_dspblut(i+170) = 255 ! Values of blue  color table
         end do
         r_dspredt(255) = 255   ! Values of red   color table
         r_dspgrnt(255) = 0     ! Values of green color table
         r_dspblut(255) = 255   ! Values of blue  color table
         i_dspnumt=256
         do i=0,255
           r_dspredt(i)=max(0.,min(1.,r_dspredt(i)/255.))
           r_dspgrnt(i)=max(0.,min(1.,r_dspgrnt(i)/255.))
           r_dspblut(i)=max(0.,min(1.,r_dspblut(i)/255.))
         end do
      else if (a_dspctbl .eq. 'bgw') then
         i_dspnumt=256
         i_cnt = 4
         i_nrgb(0,1) = 0
         i_nrgb(1,1) = 25
         i_nrgb(2,1) = 25
         i_nrgb(3,1) = 112
         i_nrgb(0,2) = 127
         i_nrgb(1,2) = 34
         i_nrgb(2,2) = 139
         i_nrgb(3,2) = 34
         i_nrgb(0,3) = 200
         i_nrgb(1,3) = 139
         i_nrgb(2,3) = 69
         i_nrgb(3,3) = 19
         i_nrgb(0,4) = 255
         i_nrgb(1,4) = 180
         i_nrgb(2,4) = 180
         i_nrgb(3,4) = 180
         do i=0,i_dspnumt-1
            k=0
            do j=1,i_cnt
               if (i_nrgb(0,j-1) .le. i .and. i_nrgb(0,j) .ge. i) then
                  k = j
               end if
            end do
            if (k .eq. 0) then
               if (i_debug .ge. 1) write(6,*) 'Error in bgw get_colortable'
            end if
            r_dspredt(i) = i_nrgb(1,k-1)+((i_nrgb(1,k)-i_nrgb(1,k-1))*(i
     &           -i_nrgb(0,k-1)))/(i_nrgb(0,k)-i_nrgb(0,k-1))
            r_dspgrnt(i) = i_nrgb(2,k-1)+((i_nrgb(2,k)-i_nrgb(2,k-1))*(i
     &           -i_nrgb(0,k-1)))/(i_nrgb(0,k)-i_nrgb(0,k-1))
            r_dspblut(i) = i_nrgb(3,k-1)+((i_nrgb(3,k)-i_nrgb(3,k-1))*(i
     &           -i_nrgb(0,k-1)))/(i_nrgb(0,k)-i_nrgb(0,k-1))

         end do
         do i=0,i_dspnumt-1
           r_dspredt(i)=max(0.,min(1.,r_dspredt(i)/255.))
           r_dspgrnt(i)=max(0.,min(1.,r_dspgrnt(i)/255.))
           r_dspblut(i)=max(0.,min(1.,r_dspblut(i)/255.))
         end do
      else
         i_cnt=0
         i_dspnumt=1
         i_nrgb(0,0)=0
         i_nrgb(1,0)=0
         i_nrgb(2,0)=0
         i_nrgb(3,0)=0
         if (i_debug .eq. -3 .or. i_debug .ge. 3) write(6,*) 'Loading external color table: ',a_dspctbl
         a_file=a_dspctbl
         open(unit=81,file=a_file,form='formatted',status='old',iostat=ierr)
         if (ierr .ne. 0 .and. index(a_dspctbl,'/') .eq. 0) then
           a_file=a_colordir(1:rdflen(a_colordir))//a_dspctbl
           open(unit=81,file=a_file,form='formatted',status='old',iostat=ierr)
         end if
         if (i_debug .eq. -3 .or. i_debug .ge. 3) write(6,*) 'reading color file:',a_file
         if (ierr .eq. 0) then
           do while (ierr .eq. 0 .and. i_cnt .lt. 256)
             read(81,fmt='(a)',err=900,end=900) a_line
             if (a_line(1:1) .ne. 'c' .and. a_line(1:1) .ne. '#' .and. a_line(1:1) .ne. '!' .and.
     &           a_line(1:1) .ne. '%' .and. a_line(1:1) .ne. '/' .and. a_line(1:1) .ne. 'C' ) then
               if (index(a_line,'!') .gt. 1) a_line=a_line(1:index(a_line,'!')-1)
               read(unit=a_line,fmt=*,iostat=ierr) i_nrgb(0,i_cnt),i_nrgb(1,i_cnt)
     &              ,i_nrgb(2,i_cnt),i_nrgb(3,i_cnt)
               if (ierr .eq. 0) then
                 if (i_debug .eq. -6 .or. i_debug .ge. 6) write(6,*) 'cfile:',i_nrgb(0,i_cnt),i_nrgb(1,i_cnt)
     &              ,i_nrgb(2,i_cnt),i_nrgb(3,i_cnt)
                 i_dspnumt=max(i_dspnumt,i_nrgb(0,i_cnt)+1)
                 i_cnt=i_cnt+1
               end if
             end if
           end do
           if (i_cnt .gt. 256) stop 'Error - External color table too big'
 900       continue
           close(81)
         else
           do i=0,256
              i_nrgb(0,i)=i
              i_nrgb(1,i)=i
              i_nrgb(2,i)=i
              i_nrgb(3,i)=i
           end do
           i_cnt=256
           i_dspnumt=256
           a_dspctbl=a_dspctbl(1:max(1,rdflen(a_dspctbl)))//' - not found.  Using grey'
         end if
         i_dspnumt=min(i_dspnumt,256)
         if (i_debug .ge. 4) write(6,*) 'Number of colors in file: ',i_cnt
         if (i_debug .ge. 4) write(6,*) 'Number of colors in cmap: ',i_dspnumt
         do i=0,i_dspnumt-1
            k=0
            do j=1,i_cnt
               if (i_nrgb(0,j-1) .le. i .and. i_nrgb(0,j) .ge. i) then
                  k = j
               end if
            end do
            if (k .eq. 0) then
               if (i_debug .ge. 1) write(6,*) 'Error in bgw get_colortable'
            end if
            r_dspredt(i) = i_nrgb(1,k-1)+((i_nrgb(1,k)-i_nrgb(1,k-1))*(i
     &           -i_nrgb(0,k-1)))/(i_nrgb(0,k)-i_nrgb(0,k-1))
            r_dspgrnt(i) = i_nrgb(2,k-1)+((i_nrgb(2,k)-i_nrgb(2,k-1))*(i
     &           -i_nrgb(0,k-1)))/(i_nrgb(0,k)-i_nrgb(0,k-1))
            r_dspblut(i) = i_nrgb(3,k-1)+((i_nrgb(3,k)-i_nrgb(3,k-1))*(i
     &           -i_nrgb(0,k-1)))/(i_nrgb(0,k)-i_nrgb(0,k-1))

         end do
         do i=0,i_dspnumt-1
           r_dspredt(i)=max(0.,min(1.,r_dspredt(i)/255.))
           r_dspgrnt(i)=max(0.,min(1.,r_dspgrnt(i)/255.))
           r_dspblut(i)=max(0.,min(1.,r_dspblut(i)/255.))
           if (i_debug .ge. 6) write(6,*) i,r_dspredt(i),r_dspgrnt(i),r_dspblut(i)
         end do

      end if
      return
      end


       subroutine copy_setdata(i_in,i_out,i_setunit,i_setrows,i_setcols,
     &                         a_setname,a_setfile,a_setinfo,a_setproj,
     &                         i_setshdr,i_setstlr,i_setrhdr,i_setrtlr,i_setchdr,i_setctlr,
     &                         r_setrmlt,r_setradr,r_setcmlt,r_setcadr,r_setvmlt,r_setvadr,
     &                         i_setvend,i_setvfmt,r_setvmin,r_setvmax,a_setvnul,r_setpegv)

       implicit none

       integer i_in
       integer i_out

      integer I_FMAX            ! Maximum number of data files
      parameter(I_FMAX= 6)

      integer I_CMAX            ! Maximum number of data channels
      parameter(I_CMAX=10)

c     structure / set_structure / s_set(*)
      character*200 a_setname(-I_FMAX:I_CMAX) ! Parameter name
      character*200 a_setfile(-I_FMAX:I_CMAX) ! Data filename
      character*200 a_setinfo(-I_FMAX:I_CMAX) ! Header filename
      character*200 a_setproj(-I_FMAX:I_CMAX) ! Projection name
      character*16  a_setvnul(-I_FMAX:I_CMAX) ! Hex string of null value
      integer i_setunit(-I_FMAX:I_CMAX) ! Unit number to read set
      integer i_setrows(-I_FMAX:I_CMAX) ! Number of rows in set
      integer i_setcols(-I_FMAX:I_CMAX) ! Number of columns in set
      integer i_setshdr(-I_FMAX:I_CMAX) ! Number of bytes in set header
      integer i_setstlr(-I_FMAX:I_CMAX) ! Number of bytes in set trailer
      integer i_setrhdr(-I_FMAX:I_CMAX) ! Number of bytes in row header
      integer i_setrtlr(-I_FMAX:I_CMAX) ! Number of bytes in row trailer
      integer i_setchdr(-I_FMAX:I_CMAX) ! Number of bytes in column header
      integer i_setctlr(-I_FMAX:I_CMAX) ! Number of bytes in column trailer
      integer i_setvend(-I_FMAX:I_CMAX) ! Endian flag
      integer i_setvfmt(-I_FMAX:I_CMAX) ! Method to decode columns
      real*4  r_setrmlt(-I_FMAX:I_CMAX) ! Row Scale  for set
      real*4  r_setradr(-I_FMAX:I_CMAX) ! Row Offset for set
      real*4  r_setcmlt(-I_FMAX:I_CMAX) ! Column Scale  for set
      real*4  r_setcadr(-I_FMAX:I_CMAX) ! Column Offset for set
      real*4  r_setvmlt(-I_FMAX:I_CMAX) ! Value Scale  for set
      real*4  r_setvadr(-I_FMAX:I_CMAX) ! Value Offset for set
      real*4  r_setvmin(-I_FMAX:I_CMAX) ! Minimum valid value
      real*4  r_setvmax(-I_FMAX:I_CMAX) ! Maximum valid value
      real*4  r_setpegv(3,-I_FMAX:I_CMAX) ! Maximum valid value
c     end structure

                  if (i_out .lt. -I_FMAX .or. i_out .gt. I_CMAX) write(6,*) 'i_out error in copy setdata ',i_out
                  if (i_in  .lt. -I_FMAX .or. i_in  .gt. I_CMAX) write(6,*) 'i_in error in copy setdata ',i_in

                  a_setname(i_out) = a_setname(i_in)
                  a_setfile(i_out) = a_setfile(i_in)
                  a_setinfo(i_out) = a_setinfo(i_in)
                  a_setproj(i_out) = a_setproj(i_in)
                  i_setunit(i_out) = i_setunit(i_in)
                  i_setrows(i_out) = i_setrows(i_in)
                  i_setcols(i_out) = i_setcols(i_in)
                  i_setshdr(i_out) = i_setshdr(i_in)
                  i_setstlr(i_out) = i_setstlr(i_in)
                  i_setrhdr(i_out) = i_setrhdr(i_in)
                  i_setrtlr(i_out) = i_setrtlr(i_in)
                  i_setchdr(i_out) = i_setchdr(i_in)
                  i_setctlr(i_out) = i_setctlr(i_in)
                  i_setvend(i_out) = i_setvend(i_in)
                  i_setvfmt(i_out) = i_setvfmt(i_in)
                  r_setvmlt(i_out) = r_setvmlt(i_in)
                  r_setvadr(i_out) = r_setvadr(i_in)
                  r_setvmin(i_out) = r_setvmin(i_in)
                  r_setvmax(i_out) = r_setvmax(i_in)
                  a_setvnul(i_out) = a_setvnul(i_in)

                  r_setrmlt(i_out) = r_setrmlt(i_in)
                  r_setradr(i_out) = r_setradr(i_in)

                  r_setcmlt(i_out) = r_setcmlt(i_in)
                  r_setcadr(i_out) = r_setcadr(i_in)

                  r_setvmlt(i_out) = r_setvmlt(i_in)
                  r_setvadr(i_out) = r_setvadr(i_in)
                  r_setvadr(i_out) = r_setvadr(i_in)

                  r_setpegv(1,i_out) = r_setpegv(1,i_in)
                  r_setpegv(2,i_out) = r_setpegv(2,i_in)
                  r_setpegv(3,i_out) = r_setpegv(3,i_in)


       return
           end


       subroutine copy_dspdata(i_in,i_out,
     &                         r_dspaddr,r_dspmult,r_dspwrap,r_dspexpn,r_dspcplw,r_dspcphi,
     &                         r_dspval1,r_dspval2,r_dspval3,i_dspmode,i_dspaddr,i_dspmult,i_dspmixv,
     &                         a_dspctbl,i_dspdvdc)
c     structure / dspinfo / s_dsp

       implicit none

           integer i_in
           integer i_out

      integer I_FMAX            ! Maximum number of data files
      parameter(I_FMAX= 6)

      integer I_CMAX            ! Maximum number of data channels
      parameter(I_CMAX=10)

      character*200 a_dspctbl(-I_FMAX:I_CMAX) ! Color table file
      integer i_dspcnt
      integer i_dspchnl         ! Number of sets to display
      integer i_dspaddr(-I_FMAX:I_CMAX) ! Add  auto Scale flag
      integer i_dspmult(-I_FMAX:I_CMAX) ! Mult auto Scale flag
      integer i_dspmixv(-I_FMAX:I_CMAX) ! Method to mix set (add, multiply, max, avg)
      integer i_dspmode(-I_FMAX:I_CMAX)
      integer i_dspdvdc(-I_FMAX:I_CMAX)
      real*4  r_dspcplw(-I_FMAX:I_CMAX) ! Discard if below value
      real*4  r_dspcphi(-I_FMAX:I_CMAX) ! Discard if above value
      real*4  r_dspaddr(-I_FMAX:I_CMAX) ! Shift data by value
      real*4  r_dspwrap(-I_FMAX:I_CMAX) ! Wrap data by value
      real*4  r_dspexpn(-I_FMAX:I_CMAX) ! Compress data
      real*4  r_dspmult(-I_FMAX:I_CMAX) ! Multiply data by value
      real*4  r_dspval1(-I_FMAX:I_CMAX)
      real*4  r_dspval2(-I_FMAX:I_CMAX)
      real*4  r_dspval3(-I_FMAX:I_CMAX)
c     end structure

                  if (i_out .lt. -I_FMAX .or. i_out .gt. I_CMAX) write(6,*) 'i_out error in copy dspdata ',i_out
                  if (i_in  .lt. -I_FMAX .or. i_in  .gt. I_CMAX) write(6,*) 'i_in error in copy dspdata ',i_in

                  r_dspaddr(i_out) = r_dspaddr(i_in)
                  r_dspwrap(i_out) = r_dspwrap(i_in)
                  r_dspexpn(i_out) = r_dspexpn(i_in)
                  r_dspmult(i_out) = r_dspmult(i_in)
                  r_dspcplw(i_out) = r_dspcplw(i_in)
                  r_dspcphi(i_out) = r_dspcphi(i_in)
                  r_dspmult(i_out) = r_dspmult(i_in)
                  r_dspval1(i_out) = r_dspval1(i_in)
                  r_dspval2(i_out) = r_dspval2(i_in)
                  r_dspval3(i_out) = r_dspval3(i_in)
                  i_dspmode(i_out) = i_dspmode(i_in)
                  i_dspdvdc(i_out) = i_dspdvdc(i_in)
                  i_dspaddr(i_out) = i_dspaddr(i_in)
                  i_dspmult(i_out) = i_dspmult(i_in)
                  i_dspmixv(i_out) = i_dspmixv(i_in)
                  a_dspctbl(i_out) = a_dspctbl(i_in)

       return
           end

      subroutine init_dsp(a_lcolor,i_debug)


      implicit none

      integer i_debug

      integer i_clrs
      integer i_dxi
      integer i_wxi             ! Number of windows
      integer i_wxs(0:20)       ! Size of window canvas in x direction
      integer i_wys(0:20)       ! Size of window canvas in y direction

      integer*4 i_type(0:20)
      integer*4 i_frx(0:20)
      integer*4 i_fry(0:20)

      character*80 a_labl(0:20)
      character*20 a_menu(0:5,0:9)
      character*80 a_lcolor

      integer init_gx
      external init_gx


c
c     Initialize graphics
c

      i_wxi = 0
      i_clrs=0

      i_dxi=init_gx(i_wxi,i_type,a_labl,i_wxs,i_wys,i_frx,i_fry,a_menu,
     &     a_lcolor,i_clrs,i_debug)
c      write(6,*) '0 is good from init_dsp = ',i_dxi

      return
      end

      subroutine create_dsp(a_dspname,i_winrows,i_wincols,i_winy,i_winx,a_setname
     &     ,i_set,i_dxi,i_menu,a_tname,i_close,a_lcolor,i_debug)


      implicit none

      integer I_CMAX
      parameter (I_CMAX=10)

      integer i
      integer j
      integer i_set
      character*(*) a_dspname
      character*(*) a_setname(i_set)
      integer i_winrows
      integer i_wincols
      integer i_winx
      integer i_winy
      integer i_debug
      integer i_menu
      integer i_close
      character*20 a_tname(5)

      integer i_clrs
      integer i_dxi
      integer i_wxi             ! Number of windows
      integer i_wxs(0:20)       ! Size of window canvas in x direction
      integer i_wys(0:20)       ! Size of window canvas in y direction

      integer*4 i_type(0:20)
      integer*4 i_frx(0:20)
      integer*4 i_fry(0:20)

      character*80 a_labl(0:20)
      character*20 a_menu(0:5,0:9)
      character*80 a_lcolor

      integer init_gx
      external init_gx


c
c     Initialize graphics
c
      if (i_menu .eq. 1) then
        a_menu(0,0)= 'Application'
        a_menu(1,0)= 'Spawn  ^A'
        a_menu(2,0)= 'Quit   ^Q'
        a_menu(3,0)= ' '
        a_menu(4,0)= ' '
        a_menu(5,0)= ' '
        a_menu(0,1)= 'Display'
        a_menu(1,1)= 'Open   ^D'
        a_menu(2,1)= 'Close  ^K'
        a_menu(3,1)= 'Resize ^R'
        a_menu(4,1)= ' '
        a_menu(5,1)= ' '
        a_menu(0,2)= 'Set'
        a_menu(1,2)= 'Add    ^I'
        a_menu(2,2)= 'Delete ^K'
        a_menu(3,2)= 'Modify ^M'
        a_menu(4,2)= ' '
        a_menu(5,2)= ' '
        a_menu(0,3)= 'Zoom'
        a_menu(1,3)= 'None   ^N'
        a_menu(2,3)= '+2x    ^+'
        a_menu(3,3)= '-2x    ^-'
        a_menu(4,3)= 'Other  ^Z'
        a_menu(5,3)= ' '
        a_menu(0,4)= 'Select'
        a_menu(1,4)= 'Mode'
        a_menu(2,4)= 'Import'
        a_menu(3,4)= 'Export'
        a_menu(4,4)= 'Clear'
        a_menu(5,4)= ' '
        a_menu(0,5)= 'Print'
        a_menu(1,5)= 'To Printer ^PP'
        a_menu(2,5)= 'To File    ^PF'
        a_menu(3,5)= 'Setup      ^PS'
        a_menu(4,5)= ' '
        a_menu(5,5)= ' '
        if (a_tname(1) .ne. ' ' .or. a_tname(2) .ne. ' ' .or.
     &        a_tname(3) .ne. ' ' .or. a_tname(4) .ne. ' ' .or. a_tname(5) .ne. ' ') then
          a_menu(0,6)='Tools'
        else
          a_menu(0,6)= ' '
        end if
        a_menu(1,6)= a_tname(1)
        a_menu(2,6)= a_tname(2)
        a_menu(3,6)= a_tname(3)
        a_menu(4,6)= a_tname(4)
        a_menu(5,6)= a_tname(5)
        a_menu(0,7)= ' '
        a_menu(1,7)= ' '
        a_menu(2,7)= ' '
        a_menu(3,7)= ' '
        a_menu(4,7)= ' '
        a_menu(5,7)= ' '
        a_menu(0,8)= ' '
        a_menu(1,8)= ' '
        a_menu(2,8)= ' '
        a_menu(3,8)= ' '
        a_menu(4,8)= ' '
        a_menu(5,8)= ' '
        a_menu(0,9)= 'Help'
        a_menu(1,9)= 'Reference   ^HC'
        a_menu(2,9)= 'Users Guide ^HT'
        a_menu(3,9)= ' '
        a_menu(4,9)= ' '
        a_menu(5,9)= ' '
      else
        do i=0,5
          do j=0,9
            a_menu(i,j)=' '
          end do
        end do
      end if

      i_wxi = max(4,min(i_set+1+2*i_close,I_CMAX+2))

      if (i_debug .eq. -3 .or. i_debug .ge. 3) write(6,*) 'i_wxi = ',i_wxi

      i_type(1) = 4
      do i=2,i_wxi
        i_type(i) = 6
      end do

      a_labl(0) = a_dspname
      a_labl(1) = 'Image Window'
      do i=2,i_wxi
        a_labl(i) = ' '
      end do
      if (i_close .ne. 0) a_labl(i_wxi) = 'Close'

      write(6,*) 'i_set=',i_set,I_CMAX
      do i=1,min(i_set,I_CMAX)
         if (a_setname(i) .ne. ' ') then
            a_labl(i+1)=a_setname(i)
         else
            write(a_labl(i+1),'(a,i2)') 'Set',i
         end if
      end do

      write(6,*) 'a_labl(i_wxi)=',a_labl(i_wxi)

      if (i_winx .ne. 0) then
        i_wxs(0) = i_winx
      else
        i_wxs(0) = min(i_wincols+28,800)
      end if
      if (i_winy .ne. 0) then
        i_wys(0) = i_winy
      else
        i_wys(0) = min(i_winrows+120,600)
      end if
      i_wxs(1) = i_wincols
      i_wys(1) = i_winrows
      do i=2,i_wxi
        i_wxs(i) = 50
        i_wys(i) = 50
      end do

      i_frx(0) = i_wxi-1
      i_frx(1) = i_wxi-1
      do i=2,i_wxi
        i_frx(i) = 1
      end do

      i_fry(0) = 0
      i_fry(1) = 400
      do i=2,i_wxi
        i_fry(i) = -25
      end do

      i_clrs=0

      i_dxi=init_gx(i_wxi,i_type,a_labl,i_wxs,i_wys,i_frx,i_fry,a_menu,
     &     a_lcolor,i_clrs,i_debug)
      if (i_debug .ge. 4) write(6,*) 'i_dxi = ',i_dxi

      return
      end


****************************************************************
        subroutine read_hdr(a_hdrfile,i_lsize,i_ssize,r_peg,a_type,
     &              r_str,r_spc,i_mbytes,i_dbytes,r_mmul,r_madd,
     &              r_dmul,r_dadd,i_err)

c****************************************************************
c**
c**     FILE NAME: read_hdr.f
c**
c**     DATE WRITTEN: 2/15/96
c**
c**     PROGRAMMER:Scott Shaffer
c**
c**     FUNCTIONAL DESCRIPTION: Reads some of an IFPROC header file.
c**
c**     ROUTINES CALLED:none
c**
c**     NOTES:
c**
c**
c*****************************************************************

        implicit none

c       INPUT VARIABLES:

        character*(*) a_hdrfile         !header file

c       OUTPUT VARIABLES:

        character*(*) a_type

        integer*4 i_err
        integer*4 i_lsize
        integer*4 i_ssize

        integer*4 i_mbytes
        integer*4 i_dbytes

        real*8 r_peg(3)
        real*8 r_str(2)
        real*8 r_spc(2)
        real r_mmul
        real r_madd
        real r_dmul
        real r_dadd


c       LOCAL VARIABLES:

        integer*4 i
        integer*4 j
        integer*4 i_cnt
        integer*4 i_zone
        real*8 r_atm(3,4)
        real*8 r_pi
        real*8 r_rtod
        real*8 r_mdnc(2)
        real*8 r_ddnc(2)

        character*255 a_tmp

c       FUNCTION STATEMENTS:

        integer rdflen
        external rdflen

c       DATA STATEMENTS: none

c       PROCESSING STEPS:

c
c  Initialize pi and conversions
c
        r_pi = 4.d0*atan(1.0d0)
        r_rtod = 180.0d0/r_pi

        i_err = 1
        i_cnt = 0

        write(6,*) ' '
        write(6,*) 'Opening hdr input  file: ',a_hdrfile(1:52)
        open(12,file=a_hdrfile,status='old',form='formatted',err=900)
        write(6,*) 'Reading'

        do i=1,100000
           read(12,'(a)',end=900) a_tmp
           if (a_tmp .eq. ' ') then
             ! do nothing
           else if (index(a_tmp,'Data file dimensions') .gt. 0) then
             read(a_tmp,*) i_lsize,i_ssize
             i_cnt = i_cnt + 1
           else if (index(a_tmp,'Post Spacing') .gt. 0) then
             read(a_tmp,*) r_spc
             i_cnt = i_cnt + 2
           else if (index(a_tmp,'Peg position (WGS-84)') .gt. 0) then
             read(a_tmp,*) r_peg
             r_peg(1) = r_peg(1)/r_rtod
             r_peg(2) = r_peg(2)/r_rtod
             r_peg(3) = r_peg(3)/r_rtod
             i_cnt = i_cnt + 4
           else if (index(a_tmp,'UTM Zone') .gt. 0) then
             read(a_tmp,*) i_zone
             r_peg(2)=(i_zone-0.5)*(6.d0/r_rtod)-r_pi
           else if (index(a_tmp,'Starting corner position (s,c)') .gt. 0) then
             read(a_tmp,*) r_str
             i_cnt = i_cnt + 8
           else if (index(a_tmp,'M11 M12 M13') .gt. 0) then
             read(a_tmp,*) r_atm(1,1),r_atm(1,2),r_atm(1,3)
c             i_cnt = i_cnt + 16
           else if (index(a_tmp,'M21 M22 M23') .gt. 0) then
             read(a_tmp,*) r_atm(2,1),r_atm(2,2),r_atm(2,3)
c             i_cnt = i_cnt + 32
           else if (index(a_tmp,'M31 M32 M33') .gt. 0) then
             read(a_tmp,*) r_atm(3,1),r_atm(3,2),r_atm(3,3)
c             i_cnt = i_cnt + 64
           else if (index(a_tmp,'O1 O2 O3') .gt. 0) then
             read(a_tmp,*) r_atm(1,4),r_atm(2,4),r_atm(3,4)
c             i_cnt = i_cnt + 128
           else if (index(a_tmp,'Magnitude Scale and Shift') .gt. 0) then
             read(a_tmp,*) r_mdnc
             r_mmul=r_mdnc(1)
             r_madd=r_mdnc(2)
           else if (index(a_tmp,'Elevation Scale and Shift') .gt. 0) then
             read(a_tmp,*) r_ddnc
             r_dmul=r_ddnc(1)
             r_dadd=r_ddnc(2)
             write(6,*) 'r_dm,r_da=',r_dmul,r_dadd
           else if (index(a_tmp,'Magnitude Bytes per Pixel') .gt. 0) then
             read(a_tmp,*) i_mbytes
           else if (index(a_tmp,'Elevation Bytes per Pixel') .gt. 0) then
             read(a_tmp,*) i_dbytes
             write(6,*) 'i_dbytes=',i_dbytes
           else if (index(a_tmp,'Data file type') .gt. 0) then
             a_type = a_tmp(1:max(1,index(a_tmp,';')-1))
             do j=1,rdflen(a_type)
               if (ichar(a_type(1:1)) .eq. 32 .or. ichar(a_type(1:1)) .eq. 9) a_type = a_type(2:)
             end do
           end if
        end do
        close(12)
        stop 'Error reading header file, too many lines'

900     close(12,err=910)
910     if (i_cnt .eq. 15) i_err = 0
        return
      end

      subroutine get_airsarinfo(          a_setname,
     &     a_setfile,
     &     a_setproj,
     &     i_setunit,
     &     i_setrows,
     &     i_setcols,
     &     i_setshdr,
     &     i_setstlr,
     &     i_setrhdr,
     &     i_setrtlr,
     &     i_setchdr,
     &     i_setctlr,
     &     i_setvend,
     &     i_setvfmt,
     &     r_setvmlt,
     &     r_setvadr,
     &     r_setvmin,
     &     r_setvmax,
     &     a_setvnul,
     &     r_setrmlt,
     &     r_setradr,
     &     r_setcmlt,
     &     r_setcadr,
     &     r_setpegv,
     &     a_dspctbl   )

c     structure / set_structure / s_set(-I_FMAX:I_CMAX)
      character*200 a_setname   ! Parameter name
      character*200 a_setfile   ! Data filename
      character*200 a_setinfo   ! Header filename
      character*200 a_setproj   ! Projection name
      integer i_setunit         ! Unit number to read set
      integer i_setrows         ! Number of rows in set
      integer i_setcols         ! Number of columns in set
      integer i_setshdr         ! Number of bytes in set header
      integer i_setstlr         ! Number of bytes in set trailer
      integer i_setrhdr         ! Number of bytes in row header
      integer i_setrtlr         ! Number of bytes in row trailer
      integer i_setchdr         ! Number of bytes in column header
      integer i_setctlr         ! Number of bytes in column trailer
      integer i_setvend         ! Endian flag
      integer i_setvfmt         ! Method to decode columns
      real*4  r_setrmlt         ! Row Scale  for set
      real*4  r_setradr         ! Row Offset for set
      real*4  r_setcmlt         ! Column Scale  for set
      real*4  r_setcadr         ! Column Offset for set
      real*4  r_setvmlt         ! Value Scale  for set
      real*4  r_setvadr         ! Value Offset for set
      real*4  r_setvmin         ! Minimum valid value
      real*4  r_setvmax         ! Maximum valid value
      character*16 a_setvnul    ! Invalid value
      real*4  r_setvavg         ! Average value in set
      real*4  r_setvstd         ! Standard deviation of values in set
      real*4  r_setpegv(3)      ! Peg Point
c     end structure

      character*200 a_dspctbl

        integer i
        integer i_err
        integer i_unit
        integer i_bytes

        character*50 a_string(100)

        byte b_string(5000)

        real*8 r_pi
        real*8 r_rtod
        real*4 r_temp

        integer nread


        integer initdk
        external initdk

        integer closedk
        external closedk

#ifdef IO64
      integer*8 nseek

      integer*8 ioseek64
      external ioseek64

      integer*8 i_demoff
      integer*8 i_magoff
      integer*8 i_paroff
#else
      integer*4 nseek

      integer*4 ioseek
      external ioseek

      integer*4 i_demoff
      integer*4 i_magoff
      integer*4 i_paroff
#endif

        integer ioread
        external ioread

        equivalence(a_string,b_string)

c
c  Initialize pi and conversions
c
        r_pi = 4.d0*atan(1.0d0)
        r_rtod = 180.0d0/r_pi


c          write(6,*) 'AIRSAR: ',a_setfile(1:60)
          i_unit = initdk(19,a_setfile)
c          write(6,*) 'i_unit=',i_unit
          i_demoff = 0
#ifdef IO64
          nseek   = ioseek64(i_unit,i_demoff,0)
c         write(6,*) 'nseek64=',nseek
#else
          nseek   = ioseek(i_unit,i_demoff,0)
c         write(6,*) 'nseek=',nseek
#endif
c          write(6,*) 'i_unit again = ',i_unit
          nread = ioread(i_unit,b_string(1),5000)

          i_demoff = -1
          i_magoff = -1
          i_paroff = -1
c         write(6,*) 'nread=',nread
          nread=5000
          do i=1,(nread-1)/50+1
            if (a_string(i) .eq. ' ') then
              ! do nothing
            else if (index(a_string(i),'NUMBER OF SAMPLES PER RECORD =') .gt. 0) then
              read(a_string(i)(35:),*) i_setcols
              write(6,*) ' '
              write(6,*) 'Reading AIRSAR header '
              a_setname = 'AIRSAR-MAG'
            else if (index(a_string(i),'NUMBER OF LINES IN IMAGE =') .gt. 0) then
              read(a_string(i)(35:),*) i_setrows
            else if (index(a_string(i),'NUMBER OF BYTES PER SAMPLE =') .gt. 0) then
              read(a_string(i)(35:),*) i_bytes

              if (i_bytes .eq. 0) then
                ! do nothing
              else if (i_bytes .eq. 1) then
                i_setvfmt = 0 ! 'val_frmt = BYTE'
              else if (i_bytes .eq. 2) then
                i_setvfmt = 2 ! 'val_frmt = INTEGER*2'
              else if (i_bytes .eq. 4) then
                i_setvfmt = 4 ! 'val_frmt = REAL*4'
              else if (i_bytes .eq. 8) then
                i_setvfmt = 6 ! 'val_frmt = Complex magnitude
              else if (i_bytes .eq. 10) then
                i_setvfmt = 9 ! 'val_frmt = Compressed stokes11
                a_setname = 'AIRSAR-M11'
              end if

            else if (index(a_string(i),'BYTE OFFSET OF FIRST DATA RECORD =') .gt. 0) then
              read(a_string(i)(35:),*) i_setshdr
            else if (index(a_string(i),'BYTE OFFSET OF DEM HEADER =') .gt. 0) then
              read(a_string(i)(35:),*) i_demoff
              if (i_demoff .gt. 0) a_setname = 'AIRSAR-DEM'
            else if (index(a_string(i),'BYTE OFFSET OF CALIBRATION HEADER =') .gt. 0) then
              read(a_string(i)(37:),*) i_magoff
              if (i_magoff .gt. 0) a_setname = 'AIRSAR-MAG'
            else if (index(a_string(i),'BYTE OFFSET OF PARAMETER HEADER =') .gt. 0) then
              read(a_string(i)(37:),*) i_paroff
            endif
          enddo

          if (i_demoff .ge. 0) then
#ifdef IO64
          nseek   = ioseek64(i_unit,i_demoff,0)
#else
          nseek   = ioseek(i_unit,i_demoff,0)
#endif
            nread = ioread(i_unit,b_string,4550)

            do i=1,(nread-1)/50+1
              if (a_string(i)(35:) .eq. ' ') then
                ! do nothing
              else if (index(a_string(i),'X-DIRECTION POST SPACING (M) =') .gt. 0) then
                read(a_string(i)(35:),*) r_setrmlt
              else if (index(a_string(i),'Y-DIRECTION POST SPACING (M) =') .gt. 0) then
                read(a_string(i)(35:),*) r_setcmlt
              else if (index(a_string(i),'ELEVATION INCREMENT (M) =') .gt. 0) then
                read(a_string(i)(35:),*,iostat=i_err) r_temp
                if (r_temp .ne. 0.0) r_setvmlt = r_temp
              else if (index(a_string(i),'ELEVATION OFFSET (M) =') .gt. 0) then
                read(a_string(i)(35:),*,iostat=i_err) r_setvadr
              else if (index(a_string(i),'LATITUDE OF PEG POINT =') .gt. 0) then
                read(a_string(i)(35:),*) r_setpegv(1)
                r_setpegv(1) = r_setpegv(1) / r_rtod
                a_setproj = 'sch'
              else if (index(a_string(i),'LONGITUDE OF PEG POINT =') .gt. 0) then
                read(a_string(i)(35:),*) r_setpegv(2)
                r_setpegv(2) = r_setpegv(2) / r_rtod
              else if (index(a_string(i),'HEADING AT PEG POINT (DEGREES) =') .gt. 0) then
                read(a_string(i)(35:),*) r_setpegv(3)
                r_setpegv(3) = r_setpegv(3) / r_rtod
              else if (index(a_string(i),'ALONG-TRACK OFFSET S0 (M) =') .gt. 0) then
                read(a_string(i)(35:),*) r_setradr
              else if (index(a_string(i),'CROSS-TRACK OFFSET C0 (M) =') .gt. 0) then
                read(a_string(i)(35:),*) r_setcadr
              endif
            enddo
          end if
          i_err = closedk(i_unit)

        return

      end



      subroutine get_pdsinfo(          a_setname,
     &     a_setfile,
     &     a_setproj,
     &     i_setunit,
     &     i_setrows,
     &     i_setcols,
     &     i_setshdr,
     &     i_setstlr,
     &     i_setrhdr,
     &     i_setrtlr,
     &     i_setchdr,
     &     i_setctlr,
     &     i_setvend,
     &     i_setvfmt,
     &     r_setvmlt,
     &     r_setvadr,
     &     r_setvmin,
     &     r_setvmax,
     &     a_setvnul,
     &     r_setrmlt,
     &     r_setradr,
     &     r_setcmlt,
     &     r_setcadr,
     &     r_setpegv,
     &     a_dspctbl,i_debug   )

c     structure / set_structure / s_set(-I_FMAX:I_CMAX)
      character*200 a_setname   ! Parameter name
      character*200 a_setfile   ! Data filename
      character*200 a_setinfo   ! Header filename
      character*200 a_setproj   ! Projection name
      integer i_setunit         ! Unit number to read set
      integer i_setrows         ! Number of rows in set
      integer i_setcols         ! Number of columns in set
      integer i_setshdr         ! Number of bytes in set header
      integer i_setstlr         ! Number of bytes in set trailer
      integer i_setrhdr         ! Number of bytes in row header
      integer i_setrtlr         ! Number of bytes in row trailer
      integer i_setchdr         ! Number of bytes in column header
      integer i_setctlr         ! Number of bytes in column trailer
      integer i_setvend         ! Endian flag
      integer i_setvfmt         ! Method to decode columns
      real*4  r_setrmlt         ! Row Scale  for set
      real*4  r_setradr         ! Row Offset for set
      real*4  r_setcmlt         ! Column Scale  for set
      real*4  r_setcadr         ! Column Offset for set
      real*4  r_setvmlt         ! Value Scale  for set
      real*4  r_setvadr         ! Value Offset for set
      real*4  r_setvmin         ! Minimum valid value
      real*4  r_setvmax         ! Maximum valid value
      character*16 a_setvnul    ! Invalid value
      real*4  r_setvavg         ! Average value in set
      real*4  r_setvstd         ! Standard deviation of values in set
      real*4  r_setpegv(3)      ! Peg Point
c     end structure

      character*200 a_dspctbl

        integer i
        integer i_err
        integer i_unit
        integer i_bytes
        integer i_debug

        integer i_recbytes
        integer i_label

        character*10000 a_string
        character*255 a_line
        character*255 a_key
        character*255 a_val
        character*255 a_object

        byte b_string(10000)

        real*8 r_pi
        real*8 r_rtod
        real*4 r_temp

        integer nread


        integer initdk
        external initdk

        integer closedk
        external closedk

#ifdef IO64
      integer*8 nseek

      integer*8 ioseek64
      external ioseek64

      integer*8 i_demoff
      integer*8 i_magoff
      integer*8 i_paroff
#else
      integer*4 nseek

      integer*4 ioseek
      external ioseek

      integer*4 i_demoff
      integer*4 i_magoff
      integer*4 i_paroff
#endif

        integer ioread
        external ioread


        integer rdflen
        external rdflen

        equivalence(a_string,b_string)

c
c  Initialize pi and conversions
c
        r_pi = 4.d0*atan(1.0d0)
        r_rtod = 180.0d0/r_pi


c          write(6,*) 'AIRSAR: ',a_setfile(1:60)
          i_unit = initdk(19,a_setfile)
c          write(6,*) 'i_unit=',i_unit
          i_demoff = 0
#ifdef IO64
          nseek   = ioseek64(i_unit,i_demoff,0)
c         write(6,*) 'nseek64=',nseek
#else
          nseek   = ioseek(i_unit,i_demoff,0)
c         write(6,*) 'nseek=',nseek
#endif
c          write(6,*) 'i_unit again = ',i_unit
          nread = ioread(i_unit,b_string(1),10000)

          if (a_string(1:14) .eq. 'PDS_VERSION_ID') then

            i_demoff = -1
            i_magoff = -1
            i_paroff = -1

            i_bytes = 0

            write(6,*) ' '
            write(6,*) 'Reading PDS Label  ',index(a_string,char(10)),i_debug
            do while(index(a_string,char(10)) .gt. 0)
              a_line=a_string(1:index(a_string,char(10)))
              a_key = a_line(1:max(1,index(a_line,'=')-1))
              a_val = a_line(max(1,index(a_line,'=')+1):)
              if (index(a_val,char(13)) .gt. 1) a_val = a_val(1:index(a_val,char(13))-1)
              if (index(a_val,char(10)) .gt. 1) a_val = a_val(1:index(a_val,char(10))-1)
c              write(6,*) 'length = ',rdflen(a_val)
c              do i=1,rdflen(a_val)
c                write(6,*) i,' ',ichar(a_val(i:i)),' ',a_val(i:i)
c              end do
              if (i_debug .eq. -13 .or. i_debug .ge. 13) write(6,*) 'a_line=',a_line(1:70)
              a_string=a_string(index(a_string,char(10))+1:)
              if (a_line .eq. ' ') then
                ! do nothing
              else if (a_key .eq. 'RECORD_BYTES') then
                read(a_val,*) i_recbytes
              else if (a_key .eq. 'LABEL_RECORDS') then
                read(a_val,*) i_label
              else if (a_key .eq. 'OBJECT') then
c                write(6,*) 'a_val=',a_val(1:70)
                if (a_val .eq. ' LBDR_TABLE') then
                  a_object = 'LBIDR_TABLE'
                  i_setcols = 32768
                  i_setrhdr = 1272
                  i_setvfmt = 4
                else if (a_val .eq. ' IMAGE') then
                  a_object = 'IMAGE'
                else if (a_val .eq. ' IMAGE_MAP_PROJECTION') then
                  a_object = 'IMAGE_MAP_PROJECTION'
                end if
c                write(6,*) '  Object = ',a_object(1:30),i_setcols
              else if (a_key .eq. 'END_OBJECT') then
                a_object = ' '
              else if (a_object .eq. 'LBIDR_TABLE') then
                if (a_key .eq. '  ROWS') then
                  read(a_val,*) i_setrows
                end if
              else if (a_object .eq. 'IMAGE') then
                if (a_key .eq. '  LINE_SAMPLES') then
                  read(a_val,*) i_setcols
                else if (a_key .eq. '  LINES') then
                  read(a_val,*) i_setrows
                else if (a_key .eq. '  SAMPLE_TYPE') then
c                  do i=1,25
c                    write(6,*) 'i/val=',i,ichar(a_line(32+i:32+i)),' ',a_line(32+i:32+i)
c                  end do
                  if (a_val .eq. ' ') then
                    ! do nothing
                  else if (a_val(1:19) .eq. ' "UNSIGNED INTEGER"') then
                    if (i_bytes .eq. 0) then
                      i_setvfmt = 0 ! 'val_frmt = BYTE'
                    else if (i_bytes .eq. 1) then
                      i_setvfmt = 0 ! 'val_frmt = BYTE'
                    else if (i_bytes .eq. 2) then
                      i_setvfmt = 8 ! 'val_frmt = BYTE*2'
                    else
                      i_setvfmt = 0 ! 'val_frmt = BYTE'
                    end if
                  else if (a_val(1:10) .eq. ' "INTEGER"') then
c                    write(6,*) 'INTEGER data detected ',i_bytes
                    if (i_bytes .eq. 0) then
                      i_setvfmt = 3 ! 'val_frmt = INTEGER*4'
                    else if (i_bytes .eq. 1) then
                      i_setvfmt = 1 ! 'val_frmt = INTEGER*1'
                    else if (i_bytes .eq. 2) then
                      i_setvfmt = 2 ! 'val_frmt = INTEGER*2'
                    else if (i_bytes .eq. 4) then
                      i_setvfmt = 3 ! 'val_frmt = INTEGER*4'
                    else
                      i_setvfmt = 3 ! 'val_frmt = INTEGER*4'
                    end if
c                    write(6,*) 'i_setvfmt = ',i_setvfmt
                  else if (a_val(1:10) .eq. ' "PC_REAL"') then
                    if (i_bytes .eq. 0) then
                      i_setvfmt = 4 ! 'val_frmt = REAL*4'
                    else if (i_bytes .eq. 4) then
                      i_setvfmt = 4 ! 'val_frmt = REAL*4'
                    else if (i_bytes .eq. 8) then
                      i_setvfmt = 5 ! 'val_frmt = REAL*8'
                    else
                      i_setvfmt = 4 ! 'val_frmt = REAL*4'
                    end if
                  else if (a_val(1:10) .eq. ' "COMPLEX"') then
                    i_setvfmt = 6 ! 'val_frmt = Complex magnitude
                  end if
                else if (a_key .eq. '  SAMPLE_BITS') then
                  read(a_val,*) i_bytes
                  i_bytes = i_bytes/8
                  if (i_bytes .eq. 0) then
                    ! do nothing
                  else if (i_bytes .eq. 1) then
                      if (i_setvfmt .eq. 2 .or. i_setvfmt .eq. 3) then
                      i_setvfmt = 1 ! 'val_frmt = INTEGER*1'
                    else if (i_setvfmt .eq. 4 .or. i_setvfmt .eq. 5) then
                       ! do nothing
                    end if
                  else if (i_bytes .eq. 2) then
                    if (i_setvfmt .eq. 0) then
                      i_setvfmt = 8 ! 'val_frmt = BYTE*2'
                    else if (i_setvfmt .eq. 1 .or. i_setvfmt .eq. 3) then
                      i_setvfmt = 2 ! 'val_frmt = INTEGER*2'
                    else if (i_setvfmt .eq. 4 .or. i_setvfmt .eq. 5) then
                      ! do nothing
                    end if
                  else if (i_bytes .eq. 4) then
                    if (i_setvfmt .eq. 0 .or. i_setvfmt .eq. 8) then
                      ! do nothing
                    else if (i_setvfmt .eq. 1 .or. i_setvfmt .eq. 2) then
                      i_setvfmt = 3 ! 'val_frmt = INTEGER*4'
                    else if (i_setvfmt .eq. 5) then
                      i_setvfmt = 4 ! 'val_frmt = REAL*4'
                    end if
                  else if (i_bytes .eq. 8) then
                    if (i_setvfmt .eq. 0 .or. i_setvfmt .eq. 8) then
                      ! do nothing
                    else if (i_setvfmt .eq. 1 .or. i_setvfmt .eq. 2 .or. i_setvfmt .eq. 3) then
                      ! do nothing
                    else if (i_setvfmt .eq. 4) then
                      i_setvfmt = 5 ! 'val_frmt = REAL*4'
                    end if
                  end if
                end if
              else if (a_object .eq. 'IMAGE_MAP_PROJECTION') then
                if (a_key .eq. '  MAP_SCALE') then
                  read(a_val,*) r_setrmlt
                  read(a_val,*) r_setcmlt
                else if (a_key .eq. '  OBLIQUE_PROJ_POLE_LATITUDE') then
                  read(a_val,*) r_setpegv(1)
                  r_setpegv(1) = r_setpegv(1) / r_rtod
                a_setproj = 'sch'
                else if (a_key .eq. '  OBLIQUE_PROJ_POLE_LONGITUDE') then
                  read(a_val,*) r_setpegv(2)
                  r_setpegv(2) = r_setpegv(2) / r_rtod
                else if (a_key .eq. '  OBLIQUE_PROJ_POLE_ROTATION') then
                  read(a_val,*) r_setpegv(3)
                  r_setpegv(3) = r_setpegv(3) / r_rtod
                else if (a_key .eq. '  LINE_PROJECTION_OFFSET') then
                  read(a_val,*) r_setradr
                else if (a_key .eq. '  SAMPLE_PROJECTION_OFFSET') then
                  read(a_val,*) r_setcadr
                endif
              end if
            enddo
            i_setshdr = i_recbytes*i_label
          end if
          i_err = closedk(i_unit)

        return

      end

      subroutine get_cubinfo(          a_setname,
     &     a_setfile,
     &     a_setproj,
     &     i_setunit,
     &     i_setrows,
     &     i_setcols,
     &     i_setshdr,
     &     i_setstlr,
     &     i_setrhdr,
     &     i_setrtlr,
     &     i_setchdr,
     &     i_setctlr,
     &     i_setvend,
     &     i_setvfmt,
     &     r_setvmlt,
     &     r_setvadr,
     &     r_setvmin,
     &     r_setvmax,
     &     a_setvnul,
     &     r_setrmlt,
     &     r_setradr,
     &     r_setcmlt,
     &     r_setcadr,
     &     r_setpegv,
     &     a_dspctbl,i_debug   )

c     structure / set_structure / s_set(-I_FMAX:I_CMAX)
      character*200 a_setname   ! Parameter name
      character*200 a_setfile   ! Data filename
      character*200 a_setinfo   ! Header filename
      character*200 a_setproj   ! Projection name
      integer i_setunit         ! Unit number to read set
      integer i_setrows         ! Number of rows in set
      integer i_setcols         ! Number of columns in set
      integer i_setshdr         ! Number of bytes in set header
      integer i_setstlr         ! Number of bytes in set trailer
      integer i_setrhdr         ! Number of bytes in row header
      integer i_setrtlr         ! Number of bytes in row trailer
      integer i_setchdr         ! Number of bytes in column header
      integer i_setctlr         ! Number of bytes in column trailer
      integer i_setvend         ! Endian flag
      integer i_setvfmt         ! Method to decode columns
      real*4  r_setrmlt         ! Row Scale  for set
      real*4  r_setradr         ! Row Offset for set
      real*4  r_setcmlt         ! Column Scale  for set
      real*4  r_setcadr         ! Column Offset for set
      real*4  r_setvmlt         ! Value Scale  for set
      real*4  r_setvadr         ! Value Offset for set
      real*4  r_setvmin         ! Minimum valid value
      real*4  r_setvmax         ! Maximum valid value
      character*16 a_setvnul    ! Invalid value
      real*4  r_setvavg         ! Average value in set
      real*4  r_setvstd         ! Standard deviation of values in set
      real*4  r_setpegv(3)      ! Peg Point
c     end structure

      character*200 a_dspctbl

        integer i
        integer i_err
        integer i_unit
        integer i_bytes
        integer i_debug

        integer i_recbytes
        integer i_label

        character*10000 a_string
        character*255 a_line
        character*255 a_key
        character*255 a_val
        character*255 a_object
        character*255 a_group

        integer      i_values
        character*20 a_values(20)

        integer i_band

        byte b_string(10000)

        real*8 r_pi
        real*8 r_rtod
        real*4 r_temp

        integer nread


        integer initdk
        external initdk

        integer closedk
        external closedk

#ifdef IO64
      integer*8 nseek

      integer*8 ioseek64
      external ioseek64

      integer*8 i_demoff
      integer*8 i_magoff
      integer*8 i_paroff
#else
      integer*4 nseek

      integer*4 ioseek
      external ioseek

      integer*4 i_demoff
      integer*4 i_magoff
      integer*4 i_paroff
#endif

        integer ioread
        external ioread


        integer rdflen
        external rdflen

        equivalence(a_string,b_string)

c
c  Initialize pi and conversions
c
        r_pi = 4.d0*atan(1.0d0)
        r_rtod = 180.0d0/r_pi


c          write(6,*) 'AIRSAR: ',a_setfile(1:60)
          i_unit = initdk(19,a_setfile)
c          write(6,*) 'i_unit=',i_unit
          i_demoff = 0
#ifdef IO64
          nseek   = ioseek64(i_unit,i_demoff,0)
c         write(6,*) 'nseek64=',nseek
#else
          nseek   = ioseek(i_unit,i_demoff,0)
c         write(6,*) 'nseek=',nseek
#endif
c          write(6,*) 'i_unit again = ',i_unit
          nread = ioread(i_unit,b_string(1),10000)

          if (a_string(1:12) .eq. 'CCSD3ZF00001') then

            if (a_setname .eq. ' ') a_setname = 'QUBE QUBE'

            i_demoff = -1
            i_magoff = -1
            i_paroff = -1

            i_bytes = 0

            write(6,*) ' '
            write(6,*) 'Reading CUB Label  ',index(a_string,char(10)),i_debug
            do while(index(a_string,char(10)) .gt. 0)
              a_line=a_string(1:index(a_string,char(10)))
              a_key = a_line(1:max(1,index(a_line,'=')-1))
              a_val = a_line(max(1,index(a_line,'=')+1):)
              if (index(a_val,char(13)) .gt. 1) a_val = a_val(1:index(a_val,char(13))-1)
              if (index(a_val,char(10)) .gt. 1) a_val = a_val(1:index(a_val,char(10))-1)

              if (i_debug .eq. -1001 .or. i_debug .ge. 1001) then
                write(6,*) 'length = ',rdflen(a_val)
                do i=1,rdflen(a_val)
                  write(6,*) i,' ',ichar(a_val(i:i)),' ',a_val(i:i)
                end do
              end if
              if (i_debug .eq. -13 .or. i_debug .ge. 13) write(6,*) 'a_line=',a_line(1:70)
              a_string=a_string(index(a_string,char(10))+1:)
              if (a_line .eq. ' ') then
                ! do nothing
              else if (a_key .eq. 'RECORD_BYTES') then
                read(a_val,*) i_recbytes
c                write(6,*) 'i_recbytes=',i_recbytes,'  ',a_val
c              else if (a_key .eq. 'LABEL_RECORDS') then
c                read(a_val,*) i_label
c                write(6,*) 'i_label=',i_label,'  ',a_val
              else if (a_key .eq. '^QUBE') then
                read(a_val,*) i_label
                i_label=i_label-1
c                write(6,*) 'i_label=',i_label,'  ',a_val
              else if (a_key .eq. 'OBJECT') then
c                write(6,*) 'a_val=',a_val(1:70)
                if (a_val .eq. ' QUBE') then
                  a_object = 'QUBE'
                else
                  a_object = 'UNKNOWN'
                end if
c                write(6,*) '  Object = ',a_object(1:30),i_setcols
              else if (a_key .eq. '  GROUP') then
c                write(6,*) 'a_val=',a_val(1:70)
                if (index(a_val,' IMAGE_MAP_PROJECTION') .gt. 0) then
                  a_group = 'IMAGE_MAP_PROJECTION'
                  a_setproj='EQA'
                  r_setpegv(1)=0.
                  r_setpegv(2)=0.
                  r_setpegv(3)=0.
                else
                  a_group = 'UNKNOWN'
                end if
c                write(6,*) '  Object = ',a_object(1:30),i_setcols
              else if (a_key .eq. 'END_OBJECT') then
                a_object = ' '
              else if (index(a_key,'END_GROUP') .gt. 0) then
                a_group = ' '
              else if (a_object .eq. 'QUBE') then
                if (a_key .eq. '  CORE_ITEMS') then
                  a_val=a_val(index(a_val,'(')+1:)
                  a_val=a_val(:index(a_val,')')-1)
                  call rdf_getfields(a_val,i_values,a_values)
                  read(a_values(1),*) i_setcols
                  read(a_values(2),*) i_setrows
                  read(a_values(3),*) i_band
                  if (i_band .ne. 1) write(6,*) 'Band error in Qube header ',i_band
                else if (a_key .eq. '  CORE_ITEM_BYTES') then
                  read(a_val,*) i_bytes
                else if (a_key .eq. '  CORE_ITEM_TYPE') then
c                  do i=1,25
c                    write(6,*) 'i/val=',i,ichar(a_line(32+i:32+i)),' ',a_line(32+i:32+i)
c                  end do
                  if (a_val .eq. ' ') then
                    ! do nothing
                  else if (a_val(1:19) .eq. ' UNSIGNED INTEGER') then
                    if (i_bytes .eq. 0) then
                      i_setvfmt = 0 ! 'val_frmt = BYTE'
                    else if (i_bytes .eq. 1) then
                      i_setvfmt = 0 ! 'val_frmt = BYTE'
                    else if (i_bytes .eq. 2) then
                      i_setvfmt = 8 ! 'val_frmt = BYTE*2'
                    else
                      i_setvfmt = 0 ! 'val_frmt = BYTE'
                    end if
                  else if (a_val(1:10) .eq. ' INTEGER') then
c                    write(6,*) 'INTEGER data detected ',i_bytes
                    if (i_bytes .eq. 0) then
                      i_setvfmt = 3 ! 'val_frmt = INTEGER*4'
                    else if (i_bytes .eq. 1) then
                      i_setvfmt = 1 ! 'val_frmt = INTEGER*1'
                    else if (i_bytes .eq. 2) then
                      i_setvfmt = 2 ! 'val_frmt = INTEGER*2'
                    else if (i_bytes .eq. 4) then
                      i_setvfmt = 3 ! 'val_frmt = INTEGER*4'
                    else
                      i_setvfmt = 3 ! 'val_frmt = INTEGER*4'
                    end if
c                    write(6,*) 'i_setvfmt = ',i_setvfmt
                  else if (a_val(1:10) .eq. ' PC_REAL') then
                    i_setvend=-1
                    if (i_bytes .eq. 0) then
                      i_setvfmt = 4 ! 'val_frmt = REAL*4'
                    else if (i_bytes .eq. 4) then
                      i_setvfmt = 4 ! 'val_frmt = REAL*4'
                    else if (i_bytes .eq. 8) then
                      i_setvfmt = 5 ! 'val_frmt = REAL*8'
                    else
                      i_setvfmt = 4 ! 'val_frmt = REAL*4'
                    end if
                  else if (a_val(1:10) .eq. ' SUN_REAL') then
                    i_setvend=1
                    if (i_bytes .eq. 0) then
                      i_setvfmt = 4 ! 'val_frmt = REAL*4'
                    else if (i_bytes .eq. 4) then
                      i_setvfmt = 4 ! 'val_frmt = REAL*4'
                    else if (i_bytes .eq. 8) then
                      i_setvfmt = 5 ! 'val_frmt = REAL*8'
                    else
                      i_setvfmt = 4 ! 'val_frmt = REAL*4'
                    end if
                  else if (a_val(1:10) .eq. ' COMPLEX') then
                    i_setvfmt = 6 ! 'val_frmt = Complex magnitude
                  end if
                else if (a_key .eq. '  SAMPLE_BITS') then
                  read(a_val,*) i_bytes
                  i_bytes = i_bytes/8
                  if (i_bytes .eq. 0) then
                    ! do nothing
                  else if (i_bytes .eq. 1) then
                      if (i_setvfmt .eq. 2 .or. i_setvfmt .eq. 3) then
                      i_setvfmt = 1 ! 'val_frmt = INTEGER*1'
                    else if (i_setvfmt .eq. 4 .or. i_setvfmt .eq. 5) then
                       ! do nothing
                    end if
                  else if (i_bytes .eq. 2) then
                    if (i_setvfmt .eq. 0) then
                      i_setvfmt = 8 ! 'val_frmt = BYTE*2'
                    else if (i_setvfmt .eq. 1 .or. i_setvfmt .eq. 3) then
                      i_setvfmt = 2 ! 'val_frmt = INTEGER*2'
                    else if (i_setvfmt .eq. 4 .or. i_setvfmt .eq. 5) then
                      ! do nothing
                    end if
                  else if (i_bytes .eq. 4) then
                    if (i_setvfmt .eq. 0 .or. i_setvfmt .eq. 8) then
                      ! do nothing
                    else if (i_setvfmt .eq. 1 .or. i_setvfmt .eq. 2) then
                      i_setvfmt = 3 ! 'val_frmt = INTEGER*4'
                    else if (i_setvfmt .eq. 5) then
                      i_setvfmt = 4 ! 'val_frmt = REAL*4'
                    end if
                  else if (i_bytes .eq. 8) then
                    if (i_setvfmt .eq. 0 .or. i_setvfmt .eq. 8) then
                      ! do nothing
                    else if (i_setvfmt .eq. 1 .or. i_setvfmt .eq. 2 .or. i_setvfmt .eq. 3) then
                      ! do nothing
                    else if (i_setvfmt .eq. 4) then
                      i_setvfmt = 5 ! 'val_frmt = REAL*4'
                    end if
                  end if
                else if (a_group .eq. 'IMAGE_MAP_PROJECTION') then
                  if (index(a_key,'MAP_SCALE') .gt. 0) then
                    read(a_val,*) r_setrmlt
                    read(a_val,*) r_setcmlt
                  else if (index(a_key,'LINE_PROJECTION_OFFSET') .gt. 0) then
                    read(a_val,*) r_setradr
                  else if (index(a_key,'SAMPLE_PROJECTION_OFFSET') .gt. 0) then
                    read(a_val,*) r_setcadr
                  endif
                end if
              end if
            enddo
            i_setshdr = i_recbytes*i_label
          end if
          i_err = closedk(i_unit)

        return

      end


****************************************************************
      subroutine get_setinfo(          a_setname,
     &     a_setinfo,
     &     a_setproj,
     &     i_setunit,
     &     i_setrows,
     &     i_setcols,
     &     i_setshdr,
     &     i_setstlr,
     &     i_setrhdr,
     &     i_setrtlr,
     &     i_setchdr,
     &     i_setctlr,
     &     i_setvend,
     &     i_setvfmt,
     &     r_setvmlt,
     &     r_setvadr,
     &     r_setvmin,
     &     r_setvmax,
     &     a_setvnul,
     &     r_setrmlt,
     &     r_setradr,
     &     r_setcmlt,
     &     r_setcadr,
     &     r_setpegv,
     &     r_dspaddr,
     &     r_dspmult,
     &     r_dspwrap,
     &     r_dspexpn,
     &     r_dspcplw,
     &     r_dspcphi,
     &     r_dspval1,
     &     r_dspval2,
     &     r_dspval3,
     &     i_dspmode,
     &     i_dspaddr,
     &     i_dspmult,
     &     i_dspmixv,
     &     i_dspdvdc,
     &     a_dspctbl   )

      implicit none

c     INPUT VARIABLES:

c     OUTPUT VARIABLES:

c     structure / set_structure / s_set(-I_FMAX:I_CMAX)
      character*200 a_setname   ! Parameter name
      character*200 a_setfile   ! Data filename
      character*200 a_setinfo   ! Header filename
      character*200 a_setproj   ! Projection name
      integer i_setunit         ! Unit number to read set
      integer i_setrows         ! Number of rows in set
      integer i_setcols         ! Number of columns in set
      integer i_setshdr         ! Number of bytes in set header
      integer i_setstlr         ! Number of bytes in set trailer
      integer i_setrhdr         ! Number of bytes in row header
      integer i_setrtlr         ! Number of bytes in row trailer
      integer i_setchdr         ! Number of bytes in column header
      integer i_setctlr         ! Number of bytes in column trailer
      integer i_setvend         ! Endian flag
      integer i_setvfmt         ! Method to decode columns
      real*4  r_setrmlt         ! Row Scale  for set
      real*4  r_setradr         ! Row Offset for set
      real*4  r_setcmlt         ! Column Scale  for set
      real*4  r_setcadr         ! Column Offset for set
      real*4  r_setvmlt         ! Value Scale  for set
      real*4  r_setvadr         ! Value Offset for set
      real*4  r_setvmin         ! Minimum valid value
      real*4  r_setvmax         ! Maximum valid value
      character*16 a_setvnul    ! Invalid value
      real*4  r_setvavg         ! Average value in set
      real*4  r_setvstd         ! Standard deviation of values in set
      real*4  r_setpegv(3)      ! Peg Point
c     end structure

c     structure / dspinfo / s_dsp
          character*200 a_dspctbl ! Color table file
          integer i_dspcnt
          integer i_dspchnl         ! Number of sets to display
          integer i_dspaddr ! Add  auto Scale flag
          integer i_dspmult ! Mult auto Scale flag
          integer i_dspmixv ! Method to mix set (add, multiply, max, avg)
          integer i_dspnumt ! Number of entries in color table
          integer i_dspmode
          integer i_dspdvdc
          integer i_dspactv(0:5)
c          real*4  r_dspredt(0:255) ! Values of red color table
c          real*4  r_dspgrnt(0:255) ! Values of green color table
c          real*4  r_dspblut(0:255) ! Values of blue color table
          real*4  r_dspcplw ! Discard if below value
          real*4  r_dspcphi ! Discard if above value
          real*4  r_dspexpn ! Exponent to raise data
          real*4  r_dspaddr ! Shift data by value
          real*4  r_dspwrap ! Wrap data by value
          real*4  r_dspmult ! Multiply data by value
          real*4  r_dspvmin ! Min value to display
          real*4  r_dspvmax ! Max value to display
          real*4  r_dspval1
          real*4  r_dspval2
          real*4  r_dspval3
c     end structure

c     LOCAL VARIABLES:

      integer*4 i
      integer*4 j
      integer*4 i_cnt
      integer*4 i_oper
      integer*4 i_set
      integer*4 i_stat
      integer*4 i_flg
      integer*4 i_indx

      character*255 a_tmp
      character*255 a_set
      character*255 a_key
      character*255 a_keyword
      character*255 a_valword
      character*255 a_value

c     FUNCTION STATEMENTS:

      character*320 rdfdata
      external rdfdata

      integer rdflen
      external rdflen

      integer rdfnum
      external rdfnum

      integer rdferr
      external rdferr

      integer rdfmap
      external rdfmap

      character*320 rdfdimn
      external rdfdimn

      character*320 rdfvalu
      external rdfvalu

      character*320 rdfunit
      external rdfunit

      character*320 rdfcmnt
      external rdfcmnt

      character*320 rdfelem
      external rdfelem

      character*320 rdfoper
      external rdfoper

      character*320 rdfint
      external rdfint

      character*320 rdfreal
      external rdfreal

      character*320 rdfdble
      external rdfdble

      character*40 rdflower
      external rdflower

      character*50 rdfversion
      external rdfversion

      integer*4 i_CnvrtFmt
      external i_CnvrtFmt

      call rdf_init('ERROR_SCREEN=OFF')

c     write(6,*) ' '
c     write(6,*) rdfversion()
c     write(6,*) ' '
c     write(6,*) ' '

      call rdf_clear()
      call rdf_read(a_setinfo)
      call rdf_init('ERROR_SCREEN=ON')


      if (a_setname .ne. ' ') then
         a_key = a_setname(1:rdflen(a_setname))//'.'
      else
         a_key = ' '
      end if

      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'set_name',i_indx
     &     ,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfdata(a_key(1:max(1,rdflen(a_key)))//'set_name','&')
         if (a_value .ne. ' ') a_setname = a_value
      end if
      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'set_rows',i_indx
     &     ,i_flg)
c     type *,a_key(1:max(1,rdflen(a_key)))//'set_rows:',i_indx,i_flg
      if (i_flg .eq. 1) then
         a_value=rdfdata(a_key(1:max(1,rdflen(a_key)))//'set_rows'
     &        ,'pixels')
         if (a_value .ne. ' ')  read(unit=a_value,fmt='(i10)') i_setrows
      end if
      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'set_cols',i_indx
     &     ,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfdata(a_key(1:max(1,rdflen(a_key)))//'set_cols'
     &        ,'pixels')
         if (a_value .ne. ' ') read(unit=a_value,fmt='(i10)') i_setcols !@#&%  change fmt
      end if
      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'set_hddr',i_indx
     &     ,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfdata(a_key(1:max(1,rdflen(a_key)))//'set_hddr'
     &        ,'bytes')
         if (a_value .ne. ' ') read(unit=a_value,fmt='(i10)') i_setshdr
      end if
      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'set_tail',i_indx
     &     ,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfdata(a_key(1:max(1,rdflen(a_key)))//'set_tail'
     &        ,'bytes')
         if (a_value .ne. ' ') read(unit=a_value,fmt='(i10)') i_setstlr
      end if
      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'row_hddr',i_indx
     &     ,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfdata(a_key(1:max(1,rdflen(a_key)))//'row_hddr'
     &        ,'bytes')
         if (a_value .ne. ' ') read(unit=a_value,fmt='(i10)') i_setrhdr
      end if
      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'row_tail',i_indx
     &     ,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfdata(a_key(1:max(1,rdflen(a_key)))//'row_tail'
     &        ,'bytes')
         if (a_value .ne. ' ') read(unit=a_value,fmt='(i10)') i_setrtlr
      end if
      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'col_hddr',i_indx
     &     ,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfdata(a_key(1:max(1,rdflen(a_key)))//'col_hddr'
     &        ,'bytes')
         if (a_value .ne. ' ') read(unit=a_value,fmt='(i10)') i_setchdr
      end if
      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'col_tail',i_indx
     &     ,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfdata(a_key(1:max(1,rdflen(a_key)))//'col_tail'
     &        ,'bytes')
         if (a_value .ne. ' ') read(unit=a_value,fmt='(i10)') i_setctlr
      end if
      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'val_endi',i_indx
     &     ,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfdata(a_key(1:max(1,rdflen(a_key)))//'val_endi'
     &        ,'bytes')
         if (a_value .eq. ' ') then
           ! do nothing
         else if (rdflower(a_value) .eq. 'little endian') then
            i_setvend = -1
         else if (rdflower(a_value) .eq. 'little_endian') then
            i_setvend = -1
         else if (rdflower(a_value) .eq. 'big endian'   ) then
            i_setvend = 1
         else if (rdflower(a_value) .eq. 'big_endian'   ) then
            i_setvend = 1
         else if (rdflower(a_value) .eq. 'byte swap'    ) then
            i_setvend = -i_setvend
         else if (rdflower(a_value) .eq. 'byte_swap'    ) then
            i_setvend = -i_setvend
         end if
      end if
      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'val_frmt',i_indx
     &     ,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfdata(a_key(1:max(1,rdflen(a_key)))//'val_frmt','&')
         if (a_value .ne. ' ') i_setvfmt = i_CnvrtFmt(a_value)
      end if
      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'row_mult',i_indx
     &     ,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfdata(a_key(1:max(1,rdflen(a_key)))//'row_mult',' ')
         if (a_value .ne. ' ') read(unit=a_value,fmt='(f15.4)')
     &        r_setrmlt
      end if
      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'row_addr',i_indx
     &     ,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfdata(a_key(1:max(1,rdflen(a_key)))//'row_addr',' ')
         if (a_value .ne. ' ') read(unit=a_value,fmt='(f15.4)')
     &        r_setradr
      end if
      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'col_mult',i_indx
     &     ,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfdata(a_key(1:max(1,rdflen(a_key)))//'col_mult',' ')
         if (a_value .ne. ' ') read(unit=a_value,fmt='(f15.4)')
     &        r_setcmlt
      end if
      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'col_addr',i_indx
     &     ,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfdata(a_key(1:max(1,rdflen(a_key)))//'col_addr',' ')
         if (a_value .ne. ' ') read(unit=a_value,fmt='(f15.4)')
     &        r_setcadr
      end if
      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'val_mult',i_indx ,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfdata(a_key(1:max(1,rdflen(a_key)))//'val_mult',' ')
c         write(6,*) 'val_mult=',a_value
         if (a_value .ne. ' ') read(unit=a_value,fmt='(f15.4)') r_setvmlt
      end if
      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'val_addr',i_indx,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfdata(a_key(1:max(1,rdflen(a_key)))//'val_addr',' ')
         if (index(a_value,'.') .eq. 0) a_value=a_value(1:max(1,rdflen(a_value)))//'.'
         if (a_value .ne. ' ') read(unit=a_value,fmt='(f15.4)') r_setvadr
c         write(6,*) 'r_setvadr=',a_value
      end if
      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'val_minv',i_indx ,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfdata(a_key(1:max(1,rdflen(a_key)))//'val_minv',' ')
         if (index(a_value,'.') .eq. 0) a_value=a_value(1:max(1,rdflen(a_value)))//'.'
         if (a_value .ne. ' ') read(unit=a_value,fmt='(f15.4)') r_setvmin
      end if
      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'val_maxv',i_indx,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfdata(a_key(1:max(1,rdflen(a_key)))//'val_maxv',' ')
         if (index(a_value,'.') .eq. 0) a_value=a_value(1:max(1,rdflen(a_value)))//'.'
         if (a_value .ne. ' ') read(unit=a_value,fmt='(f15.4)')r_setvmax
      end if
      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'val_null',i_indx,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfdata(a_key(1:max(1,rdflen(a_key)))//'val_null',' ')
         if (a_value .ne. ' ') a_setvnul = a_value
      end if
      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'set_plat',i_indx,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfdata(a_key(1:max(1,rdflen(a_key)))//'set_plat','rad')
         if (a_value .ne. ' ') read(unit=a_value,fmt='(f15.4)') r_setpegv(1)
      end if
      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'set_plon',i_indx,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfdata(a_key(1:max(1,rdflen(a_key)))//'set_plon','rad')
         if (a_value .ne. ' ') read(unit=a_value,fmt='(f15.4)') r_setpegv(2)
      end if
      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'set_phdg',i_indx,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfdata(a_key(1:max(1,rdflen(a_key)))//'set_phdg','rad')
         if (a_value .ne. ' ') read(unit=a_value,fmt='(f15.4)') r_setpegv(3)
      end if
      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'set_pegv',i_indx,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfdata(a_key(1:max(1,rdflen(a_key)))//'set_phdg','rad')
         if (a_value .ne. ' ') read(unit=a_value,fmt='(3f15.4)') r_setpegv
      end if
      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'set_proj',i_indx,i_flg)
      if (i_flg .eq. 1) then
         a_setproj=rdfvalu(a_key(1:max(1,rdflen(a_key)))//'set_proj')
      end if


      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'dsp_cmap',i_indx,i_flg)
      if (i_flg .eq. 1) then
         a_dspctbl=rdfvalu(a_key(1:max(1,rdflen(a_key)))//'dsp_cmap')
      end if

      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'dsp_mode',i_indx,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfvalu(a_key(1:max(1,rdflen(a_key)))//'dsp_mode')
c         write(6,*) 'Mode = ','*'//a_value//'*'
         if (a_value .eq. ' ') then
           ! do nothing
         else if (rdflower(a_value) .eq. 'range') then
           i_dspmode = 1
c           write(6,*) 'setting mode to 1'
         else if (rdflower(a_value) .eq. 'sdev') then
           i_dspmode = 2
         else if (rdflower(a_value) .eq. 'per' .or. rdflower(a_value) .eq. 'percent') then
           i_dspmode = 3
         else if (rdflower(a_value) .eq. 'norm' .or. rdflower(a_value) .eq. 'normal') then
           i_dspmode = 4
         else if (rdflower(a_value) .eq. 'cw' .or. rdflower(a_value) .eq. 'charlie') then
           i_dspmode = 5
         else if (rdflower(a_value) .eq. 'wrap') then
           i_dspmode = 6
         end if
      end if

      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'dsp_wrap',i_indx,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfvalu(a_key(1:max(1,rdflen(a_key)))//'dsp_wrap')
         if (a_value .ne. ' ') read(unit=a_value,fmt='(f15.4)') r_dspwrap
      end if

      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'dsp_addr',i_indx,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfvalu(a_key(1:max(1,rdflen(a_key)))//'dsp_addr')
         if (a_value .ne. ' ') i_dspaddr=0
         if (a_value .ne. ' ') read(unit=a_value,fmt='(f15.4)') r_dspaddr
      end if

      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'dsp_mult',i_indx,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfvalu(a_key(1:max(1,rdflen(a_key)))//'dsp_mult')
         if (a_value .ne. ' ') i_dspmult=0
         if (a_value .ne. ' ') read(unit=a_value,fmt='(f15.4)') r_dspmult
      end if

      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'dsp_fact',i_indx,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfvalu(a_key(1:max(1,rdflen(a_key)))//'dsp_fact')
         if (a_value .ne. ' ') read(unit=a_value,fmt='(f15.4)') r_dspval1
      end if

      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'dsp_expn',i_indx,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfvalu(a_key(1:max(1,rdflen(a_key)))//'dsp_expn')
         if (a_value .ne. ' ') read(unit=a_value,fmt='(f15.4)') r_dspexpn
      end if

      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'dsp_expn',i_indx,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfvalu(a_key(1:max(1,rdflen(a_key)))//'dsp_expn')
         if (a_value .ne. ' ') read(unit=a_value,fmt='(f15.4)') r_dspexpn
      end if

      call rdf_index(a_key(1:max(1,rdflen(a_key)))//'dsp_dvdc',i_indx,i_flg)
      if (i_flg .eq. 1) then
         a_value=rdfvalu(a_key(1:max(1,rdflen(a_key)))//'dsp_dvdc')
         if (rdflower(a_value) .eq. 'y' .or. rdflower(a_value) .eq. 'yes' .or.
     &       rdflower(a_value) .eq. 't' .or. rdflower(a_value) .eq. 'true' .or.
     &       a_value .eq. '1') then
           i_dspdvdc=1
         else
           i_dspdvdc=0
         end if
      end if

      return
      end


****************************************************************
      subroutine put_setinfo(          a_setname,
     &     a_setinfo,
     &     a_setproj,
     &     i_setunit,
     &     i_setrows,
     &     i_setcols,
     &     i_setshdr,
     &     i_setstlr,
     &     i_setrhdr,
     &     i_setrtlr,
     &     i_setchdr,
     &     i_setctlr,
     &     i_setvend,
     &     i_setvfmt,
     &     r_setvmlt,
     &     r_setvadr,
     &     r_setvmin,
     &     r_setvmax,
     &     a_setvnul,
     &     r_setrmlt,
     &     r_setradr,
     &     r_setcmlt,
     &     r_setcadr,
     &     r_setpegv   )

      implicit none

c     INPUT VARIABLES:

c     OUTPUT VARIABLES:

c     structure / set_structure / s_set(-I_FMAX:I_CMAX)
      character*200 a_setname   ! Parameter name
      character*200 a_setfile   ! Data filename
      character*200 a_setinfo   ! Header filename
      character*200 a_setproj   ! Projection name
      integer i_setunit         ! Unit number to read set
      integer i_setrows         ! Number of rows in set
      integer i_setcols         ! Number of columns in set
      integer i_setshdr         ! Number of bytes in set header
      integer i_setstlr         ! Number of bytes in set trailer
      integer i_setrhdr         ! Number of bytes in row header
      integer i_setrtlr         ! Number of bytes in row trailer
      integer i_setchdr         ! Number of bytes in column header
      integer i_setctlr         ! Number of bytes in column trailer
      integer i_setvend         ! Endian flag
      integer i_setvfmt         ! Method to decode columns
      real*4  r_setrmlt         ! Row Scale  for set
      real*4  r_setradr         ! Row Offset for set
      real*4  r_setcmlt         ! Column Scale  for set
      real*4  r_setcadr         ! Column Offset for set
      real*4  r_setvmlt         ! Value Scale  for set
      real*4  r_setvadr         ! Value Offset for set
      real*4  r_setvmin         ! Minimum valid value
      real*4  r_setvmax         ! Maximum valid value
      character*16 a_setvnul    ! Invalid value
      real*4  r_setvavg         ! Average value in set
      real*4  r_setvstd         ! Standard deviation of values in set
      real*4  r_setpegv(3)      ! Peg Point
c     end structure


c     LOCAL VARIABLES:

      integer*4 i
      integer*4 j
      integer*4 i_cnt
      integer*4 i_oper
      integer*4 i_set
      integer*4 i_stat
      integer*4 i_flg
      integer*4 i_indx

      character*255 a_tmp
      character*255 a_set
      character*255 a_key
      character*255 a_keyword
      character*255 a_valword
      character*255 a_value

      character*255 a_data

c     FUNCTION STATEMENTS:

      integer rdflen
      external rdflen

      integer rdfnum
      external rdfnum

      integer rdferr
      external rdferr

      integer rdfmap
      external rdfmap

      integer*4 i_CnvrtFmt
      external i_CnvrtFmt

      call rdf_init(' ')

      call rdf_clear()

      write(a_data,'(a,a)') 'set_name =',a_setname
      call rdf_append(a_data)
      write(6,*) 'set_rows = ',i_setrows
      write(a_data,fmt=*) 'set_rows =',i_setrows
      write(6,*) 'a_data=',a_data
      call rdf_append(a_data)
      write(a_data,fmt=*) 'set_cols =',i_setcols
      call rdf_append(a_data)
      write(a_data,fmt=*) 'set_hddr =',i_setshdr
      call rdf_append(a_data)
      write(a_data,fmt=*) 'set_tail =',i_setstlr
      call rdf_append(a_data)
      write(a_data,fmt=*) 'row_hddr =',i_setrhdr
      call rdf_append(a_data)
      write(a_data,fmt=*) 'row_tail =',i_setrtlr
      call rdf_append(a_data)
      write(a_data,fmt=*) 'col_hddr =',i_setchdr
      call rdf_append(a_data)
      write(a_data,fmt=*) 'col_tail =',i_setctlr
      call rdf_append(a_data)
      if (i_setvend .eq. -1) then
        write(a_data,fmt=*) 'val_endi = LITTLE ENDIAN'
      else
        write(a_data,fmt=*) 'val_endi = BIG ENDIAN'
      end if
      call rdf_append(a_data)
      If (i_setvfmt .eq. -1) then
        ! do nothing
      else if (i_setvfmt .eq. 0) then
        write(a_data,fmt=*) 'val_frmt = BYTE'
      else if (i_setvfmt .eq. 1) then
        write(a_data,fmt=*) 'val_frmt = INTEGER*1'
      else if (i_setvfmt .eq. 2) then
        write(a_data,fmt=*) 'val_frmt = INTEGER*2'
      else if (i_setvfmt .eq. 3) then
        write(a_data,fmt=*) 'val_frmt = INTEGER*4'
      else if (i_setvfmt .eq. 4) then
        write(a_data,fmt=*) 'val_frmt = REAL*4'
      else if (i_setvfmt .eq. 5) then
        write(a_data,fmt=*) 'val_frmt = REAL*8'
      else if (i_setvfmt .eq. 6) then
        write(a_data,fmt=*) 'val_frmt = COMPLEX_MAGNITUDE'
      else if (i_setvfmt .eq. 7) then
        write(a_data,fmt=*) 'val_frmt = COMPLEX_PHASE'
      else if (i_setvfmt .eq. 8) then
        write(a_data,fmt=*) 'val_frmt = BYTE*2'
      else if (i_setvfmt .eq. 9) then
        write(a_data,fmt=*) 'val_frmt = COMPRESSED_STOKES'
      else if (i_setvfmt .eq. 10) then
        write(a_data,fmt=*) 'val_frmt = COMPLEX*2_MAGNITUDE'
      else if (i_setvfmt .eq. 11) then
        write(a_data,fmt=*) 'val_frmt = COMPLEX*2_PHASE'
      else if (i_setvfmt .eq. 12) then
        write(a_data,fmt=*) 'val_frmt = REAL*4_MAGNITUDE'
      else
        write(6,*) 'ERROR IN PUT_SETINFO'
      end if
      call rdf_append(a_data)
      write(a_data,fmt=*) 'row_mult =',r_setrmlt
      call rdf_append(a_data)
      write(a_data,fmt=*) 'row_addr =',r_setradr
      call rdf_append(a_data)
      write(a_data,fmt=*) 'col_mult =',r_setcmlt
      call rdf_append(a_data)
      write(a_data,fmt=*) 'col_addr =',r_setcadr
      call rdf_append(a_data)
      write(a_data,fmt=*) 'val_mult =',r_setvmlt
      call rdf_append(a_data)
      write(a_data,fmt=*) 'val_addr =',r_setvadr
      call rdf_append(a_data)
      write(a_data,fmt=*) 'val_minv =',r_setvmin
      call rdf_append(a_data)
      write(a_data,fmt=*) 'val_maxv =',r_setvmax
      call rdf_append(a_data)
      write(a_data,'(a,a)') 'val_null =',a_setvnul
      call rdf_append(a_data)
      write(a_data,fmt=*) 'set_plat =',r_setpegv(1)
      call rdf_append(a_data)
      write(a_data,fmt=*) 'set_plon =',r_setpegv(2)
      call rdf_append(a_data)
      write(a_data,fmt=*) 'set_phdg =',r_setpegv(3)
      call rdf_append(a_data)
      write(a_data,'(a,a)') 'set_proj =',a_setproj
      call rdf_append(a_data)

      call rdf_write(a_setinfo)
      return
      end

      integer function i_CnvrtFmt(a_fmt)

      implicit none

      character*(*) a_fmt
      integer i_fmt

      character*20 rdfupper
      external rdfupper

      if (a_fmt .eq. ' ') then
         i_fmt = -1
      else if (rdfupper(a_fmt) .eq. 'BYTE*1' .or. a_fmt .eq. 'BYTE') then
         i_fmt = 0
      else if (rdfupper(a_fmt) .eq. 'INTEGER*1') then
         i_fmt = 1
      else if (rdfupper(a_fmt) .eq. 'INTEGER*2') then
         i_fmt = 2
      else if (rdfupper(a_fmt) .eq. 'INTEGER*4') then
         i_fmt = 3
      else if (rdfupper(a_fmt) .eq. 'REAL*4') then
         i_fmt = 4
      else if (rdfupper(a_fmt) .eq. 'REAL*8') then
         i_fmt = 5
      else if (rdfupper(a_fmt) .eq. 'COMPLEX_MAGNITUDE' .or. rdfupper(a_fmt) .eq. 'COMPLEX*8_MAGNITUDE') then
         i_fmt = 6
      else if (rdfupper(a_fmt) .eq. 'COMPLEX_PHASE' .or. rdfupper(a_fmt) .eq. 'COMPLEX*8_PHASE') then
         i_fmt = 7
      else if (rdfupper(a_fmt) .eq. 'BYTE*2') then
         i_fmt = 8
      else if (rdfupper(a_fmt) .eq. 'COMPRESSED_STOKES' .or. a_fmt .eq. 'STOKES11') then
         i_fmt = 9
      else if (rdfupper(a_fmt) .eq. 'COMPLEX*2_MAGNITUDE') then
         i_fmt = 10
      else if (rdfupper(a_fmt) .eq. 'COMPLEX*2_PHASE') then
         i_fmt = 11
      else if (rdfupper(a_fmt) .eq. 'COMPLEX*4_MAGNITUDE') then
         i_fmt = 12
      else if (rdfupper(a_fmt) .eq. 'COMPLEX*4_PHASE') then
         i_fmt = 13
      else if (rdfupper(a_fmt) .eq. 'REAL*4_MAGNITUDE') then
         i_fmt = 14
      else
         i_fmt = -1
      endif
      i_CnvrtFmt = i_fmt
      end

      subroutine write_greeting()

      implicit none

        write(6,*) ' '
        write(6,*) ' '
        write(6,*) ' '
        write(6,*) 'Usage: mdx file1 '
        write(6,*) '       mdx file1 -x xval -y yval '
        write(6,*) '       mdx file1 -x xval -y yval file2 -zval '
        write(6,*) '       mdx file1 -set setname1 -x xval -y yval -set setname2 -z zval '
        write(6,*) '       mdx file1 -x xval -set setname1 -y yval -set setname2 -z zval'
        write(6,*) ' '
        write(6,*) 'Rules for using flags:'
        write(6,*) ' '
        write(6,*) '   Flags specified before the first filename are used as default for '
        write(6,*) '      all following files.'
        write(6,*) '   Flags specified after a filename but before any set names are used '
        write(6,*) '      as the default for all the sets in that file.'
        write(6,*) '   Flags specified after a set name only apply to that set.'
        write(6,*) '   In general, flags that are capitalized don''t require an argument,'
        write(6,*) '      flags in lower case do.'
        write(6,*) ' '
        write(6,*) 'Unobvious features:'
        write(6,*) ' '
        write(6,*) '  To activate one desired set, left-click on that sets selector button'
        write(6,*) ' '
        write(6,*) '  To toggle a set on or off, middle-click on that sets selector button'
        write(6,*) ' '
        write(6,*) '  To bring up a menu of set parameters, right-click on the set selector button'
        write(6,*) ' '
        write(6,*) '  To get an xmgrace display of a sets color bar, hold the shift key and click '
        write(6,*) '   on the set selector button '
        write(6,*) ' '
        write(6,*) '  To center the display on a pixel that isnt currently visible, click on the location'
        write(6,*) '   bar just above the image and enter the pixel row/column when asked'
        write(6,*) ' '
        write(6,*) '  To center the display on a latitude/longitude, hold the shift key down and click on '
        write(6,*) '   the locationbar just above the image and enter the lat/long when asked'
        write(6,*) ' '
        write(6,*) '  To center the display on a particular visable pixel, middle-click on that pixel'
        write(6,*) ' '
        write(6,*) '  To center other displays on a particular visable pixel, hold the shift key down'
        write(6,*) '   and middle-click on that pixel'
        write(6,*) ' '
        write(6,*) 'Flags:'
        write(6,*) ' '
        write(6,*) '-cols, -columns, -s, or -samples    = Number of samples per line'
        write(6,*) '-rows, -l, or -lines                = Number of lines in file'
        write(6,*) '-shdr  =  Size of Header (in bytes) at top of file'
        write(6,*) '-rhdr  =  Size of header (in bytes) at start of each line'
        write(6,*) '-chdr  =  Size of header (in bytes) at start of each sample'
        write(6,*) '-stlr  =  Size of trailer (in bytes at the end of each set'
        write(6,*) '-rtlr  =  Size of trailer (in bytes) at the end of each line'
        write(6,*) '-ctlr  =  Size of trailer (in bytes) at the end of each sample'
        write(6,*) '-rmlt  =  Multiplier to convert image row number to an engineering unit'
        write(6,*) '-radr  =  Offset to convert image row number to an engineering unit'
        write(6,*) '-cmlt  =  Multiplier to convert image column number to an engineering unit'
        write(6,*) '-cadr  =  Offset to convert image column number to an engineering unit'
        write(6,*) '-vmlt  =  Multiplier to convert image data to an engineering unit'
        write(6,*) '-vadr  =  Offset to convert image data to an engineering unit'
        write(6,*) '-plat  =  Peg Latitude'
        write(6,*) '-plon  =  Peg Longitude'
        write(6,*) '-phdr  =  Peg heading'
        write(6,*) '-proj  =  Projection name'
        write(6,*) ' '
        write(6,*) '-min, -vmin, or -minval  =  Minimum valid value (in engineering units)'
        write(6,*) '-max, -vmax, or -maxval  =  Maximum valid value (in engineering units)'
        write(6,*) ' '
        write(6,*) '-e, -exp = Exponent that data is raised to after scaling between 0 and 1'
        write(6,*) '-addr, -a, or -daddr  = Offset to shift color table in display'
        write(6,*) '-mult, -m, or -dmult  = Scaler to stretch color table in display'
        write(6,*) '-cws, -cw, -charlie   = Scale factor in CW mode'
        write(6,*) '-wrap, or -d          = Wrap value for display'
        write(6,*) '-fact, or -f          = Sets number of standard deviations to display across color table'
        write(6,*) '-per, percent, or -p  = percent of data that is clipped in the display'
        write(6,*) '-clpmin, or -minclp   = Minimum value before clipping during display'
        write(6,*) '-clpmax, or -maxclp   = Maximum value before clipping during display'
        write(6,*) ' '
        write(6,*) '-row    = row of display center on startup'
        write(6,*) '-col    = column of display center on startup'
        write(6,*) '-lat    = latitude of display center on startup'
        write(6,*) '-lon    = longitude of display center on startup'
        write(6,*) ' '
        write(6,*) '-active  = The following 1s and 0s set the on/off status of the sets at startup'
        write(6,*) '-z, or -zoom  =  Initial zoom of display'
        write(6,*) '-pz, or -pzoom  =  Zoom factor for printing to file'
        write(6,*) '-vx =  x dimension of initial display window'
        write(6,*) '-vy =  y dimension of initial display window'
        write(6,*) '-mix  =  Sets how to combine sets +, x are options'
        write(6,*) '-cmap, or -ctable  =  Name of color table to use'
        write(6,*) '-nc, -null_color, or -cnull  =  RGB color value to use for null data'
        write(6,*) '-emod  =  Number of rows to read before checking if window update (def=10)'
        write(6,*) '-debug =  Sets debug level (def=2) '
        write(6,*) '-workdir = working directory for out.ppm '
        write(6,*) '-colordir = default directory for color tables'
        write(6,*) ' '
        write(6,*) '-h        = Specifies header file name'
        write(6,*) '-maghdr   = Name of .hdr file to be used for a magnitude file (must be after set name)'
        write(6,*) '-dtehdr   = Name of .hdr file to be used for a height file (must be after set name)'
        write(6,*) '-pts or -points  = Filename of input selection points to overlay on display'
        write(6,*) ' '
        write(6,*) '-pcpad     = Number of pixel in column direction to reduce print size by'
        write(6,*) '-prpad     = Number of pixel in row direction to reduce print size by'
        write(6,*) ' '
        write(6,*) '-col, -cpos or -c   =  jump to specified column at start up'
        write(6,*) '-row, -rpos or -r   =  jump to specified row at start up'
        write(6,*) '-lat, or -latitude  =  jump to specified latitude at start up'
        write(6,*) '-lon, or -longitude =  jump to specified longitude at start up'
        write(6,*) ' '
        write(6,*) 'Display Mode Stuff '
        write(6,*) '-STD  = Sets display scaling to Standard deviation mode with factor at 2'
        write(6,*) '-PER  = Sets display scaling to Percentage mode with percent set to 90%'
        write(6,*) '-CW   = Sets display scaling to Charlie Warner mode with factor at 1'
        write(6,*) '-WRAP = Sets display scaling to wrap mode with a modules of Pi'
        write(6,*) '-ON   = Turns set on at startup (default)'
        write(6,*) '-OFF  = Turns set off at startup'
        write(6,*) ' '
        write(6,*) '-P, -ponly = No display, only create ppm file of sets'
        write(6,*) '-D, -dvdc, -dc, -dx or -slope = Slope of channel in column direction'
        write(6,*) '-LE, -le, -little = little endian'
        write(6,*) '-BE, -be, -big    = big endian'
        write(6,*) '-BS, -bs, -bswap  = byte swapped from default machine format'
        write(6,*) '-NM               = turns off main menu'
        write(6,*) '-C   -CLOSE       = enables close button in bottom right corner'
        write(6,*) '-NC  -NOCLOSE     = disables close button in bottom right corner'
        write(6,*) ' '
        write(6,*) 'File definition shortcuts '
        write(6,*) '-b1, or -byte       =  Unsigned byte file'
        write(6,*) '-b2, or -byte2      =  Unsigned 2-byte integer file'
        write(6,*) '-i1, or -integer*1  =  Signed byte file'
        write(6,*) '-i2, or -integer*2  =  Signed 2-byte integer file'
        write(6,*) '-i4, or -integer*4  =  Signed 4-byte integer file'
        write(6,*) '-r4, or -real*4     =  IEEE 4-byte Float file'
        write(6,*) '-c2, or -complex*2  =  Complex*2  (mag and phase sets)'
        write(6,*) '-c8, or -complex*8  =  Complex*8  (mag and phase sets)'
        write(6,*) '-c8mag, or -cmag    =  Magnitude portion of a c8 file only'
        write(6,*) '-c8pha, or -cpha    =  Phase portion of a c8 file only'
        write(6,*) '-c2mag              =  Magnitude portion of a c2 file only'
        write(6,*) '-c2pha,             =  Phase portion of a c2 file only'
        write(6,*) '-rmg                =  RMG file (mag and dte sets)'
        write(6,*) '-vfmt or -val_frmt  =  Character string indicating format (i.e. real*4)'
        write(6,*) ' '
        write(6,*) ' '
        write(6,*) 'Please forward any comments or suggestions '
        write(6,*) 'regarding mdx to:  Scott.Shaffer@jpl.nasa.gov    '
        write(6,*) ' '
        write(6,*) ' '
        write(6,*) ' '
        write(6,*) ' '
        write(6,*) ' '
        write(6,*) ' '
      return
      end

      subroutine readdat(i_setunit,   ! This version uses ioseek/read
     &     i_setrows,
     &     i_setcols,
     &     i_setshdr,
     &     i_setstlr,
     &     i_setrhdr,
     &     i_setrtlr,
     &     i_setchdr,
     &     i_setctlr,
     &     i_setvend,
     &     i_setvfmt,
     &     r_setvmlt,
     &     r_setvadr,
     &     r_setvmin,
     &     r_setvmax,
     &     b_setvnul,
     &     i_row,i_col,i_num,r_data,i_data,readfunc,i_err)

      implicit none

      integer i
      integer i_err
      integer i_num
      integer i_row
      integer i_col
      integer i_pos

      integer i_ll
      integer i_hh


      integer i_colsize
      integer i_rowsize

      integer i_setunit         ! Unit number to read set
      integer i_setrows         ! Number of rows in set
      integer i_setcols         ! Number of columns in set
      integer i_setshdr         ! Number of bytes in set header
      integer i_setstlr         ! Number of bytes in set trailer
      integer i_setrhdr         ! Number of bytes in row header
      integer i_setrtlr         ! Number of bytes in row trailer
      integer i_setchdr         ! Number of bytes in column header
      integer i_setctlr         ! Number of bytes in column trailer
      integer i_setvend         ! Endian flag
      integer i_setvfmt         ! Method to decode columns
      real*4  r_setvmlt         ! Value Scale  for set
      real*4  r_setvadr         ! Value Offset for set
      real*4  r_setvmin         ! Minimum valid value
      real*4  r_setvmax         ! Maximum valid value
      byte    b_setvnul(0:16)   ! Invalid value

      integer i_numxx

      integer nread
      integer ioread
      external ioread

#ifdef IO64
      integer*8 i_strtc
      integer*8 i_stopc
      integer*8 nseek

      integer*8 ioseek64
      external ioseek64

      integer*8 i_eight
      external i_eight

      integer*8 readfunc
      external readfunc
#else
      integer*4 i_strtc
      integer*4 i_stopc
      integer*4 nseek

      integer*4 ioseek
      external ioseek

      integer*4 readfunc
      external readfunc
#endif

      real*4 r_cnvrtdat
      external r_cnvrtdat

      real r_data(0:i_num-1)
      integer i_data(0:i_num-1)
      byte b_data(0:400000)

      real r_data2(0:100000)
      integer i_data2(0:100000)

      integer i_checknul
      external i_checknul

      integer i_setvbyt
      external i_setvbyt

      byte b_tmp(4)
      real*4 r_tmp
      equivalence(b_tmp,r_tmp)

      if (i_num .gt. 100000) stop 'Error - i_num too big in readdat'

      if (i_row .ge. 0 .and. i_row .lt. i_setrows .and.
     &     i_col+i_num-1 .ge. 0 .and. i_col .lt. i_setcols) then
         i_colsize = i_setchdr + i_setctlr + i_setvbyt(i_setvfmt)
         i_rowsize = i_setrhdr + i_setrtlr + i_colsize*i_setcols
c     write(6,*) 'i_colsize=',i_colsize
c     write(6,*) 'i_rowsize=',i_rowsize

#ifdef IO64
         i_strtc = i_setshdr + i_setrhdr + i_setchdr + (i_row)*i_eight(i_rowsize)
     &                + (max(0,i_col*i_colsize))
         i_stopc = i_strtc + min(i_num,i_setcols-i_col)*i_colsize
         if (i_setunit .gt. 0) then
           nseek   = ioseek64(i_setunit,i_strtc,0)
           if (nseek .ne. i_strtc) write(6,*) 'nseek<>i_strtc ',nseek,i_strtc

           i_numxx = i_stopc - i_strtc
           nread = ioread(i_setunit,b_data(max(0,-i_colsize*i_col)),i_numxx)
         else
           i_numxx = i_stopc - i_strtc
           nread = readfunc(0,-i_setunit,i_strtc,i_numxx,b_data(max(0,-i_colsize*i_col)))

         end if
#else
         i_strtc = i_setshdr + i_setrhdr + i_setchdr + (i_row)*(i_rowsize)
     &                + (max(0,i_col*i_colsize))
         i_stopc = i_strtc + min(i_num,i_setcols-i_col)*i_colsize
         if (i_setunit .gt. 0) then
           nseek   = ioseek(i_setunit,i_strtc,0)
           if (nseek .ne. i_strtc) write(6,*) 'nseek<>i_strtc ',nseek,i_strtc

           i_numxx = i_stopc - i_strtc
           nread = ioread(i_setunit,b_data(max(0,-i_colsize*i_col)),i_numxx)
         else
           i_numxx = i_stopc - i_strtc
           nread = readfunc(0,-i_setunit,i_strtc,i_numxx,b_data(max(0,-i_colsize*i_col)))

         end if
#endif
         if (nread .ne. i_numxx) write(6,*) 'nread<>i_numxx ',nread,i_numxx

         do i = 0, i_num-1
            if (i+i_col .ge. 0 .and. i+i_col .lt. i_setcols) then
               i_pos = i*i_colsize
               if (i_checknul(i_setvbyt(i_setvfmt),b_data(i_pos),b_setvnul) .eq. 0) ! Data not flagged as bad
     &              then
                  r_data(i) = r_cnvrtdat(i_setvfmt,i_setvend,b_data(i_pos))
                  if (r_data(i) .eq. r_data(i)) then                                ! Check if valid number
                    r_data(i) = r_setvmlt*r_data(i)+r_setvadr
                    i_data(i) = 0
                    if (r_setvmax .gt. r_setvmin) then ! check for bad data outside range
                       if (r_data(i) .lt. r_setvmin .or. r_data(i) .gt. r_setvmax) then   ! bad data
                          i_data(i) = 1
                       end if
                    else  ! Check for bad data within range
                       if (r_data(i) .le. r_setvmin .and. r_data(i) .ge. r_setvmax) then ! bad data
                          i_data(i) = 1
                       end if
                    end if
                  else              ! NaN or something
                    i_data(i) = 5
                  end if
               else
                  r_data(i) = 0
                  i_data(i) = 2
               end if
            else
               r_data(i) = 0
               i_data(i) = 3
            end if
c     write(6,*) 'r_data=',r_data(i),i,i_row,i_col,i_strtc,i_numxx
         end do
         if (i_setvfmt .lt. 0) then
           do i = 0, i_num-1
             r_data2(i) = r_data(i)
             i_data2(i) = i_data(i)
           end do
           do i = 0, i_num-1
             i_ll=max(0,i-1)
             i_hh=min(i_num-1,i+1)
             if (i_data2(i_ll) .eq. 0 .and. i_data2(i_hh) .eq. 0) then
               i_data(i)=0
               r_data(i)=(r_data2(i_hh)-r_data2(i_ll))/(i_hh-i_ll)
             else
               i_data(i)=1
               r_data(i)=0
             end if
           end do
         end if
      else
         do i=0,i_num-1
            r_data(i) = 0
            i_data(i) = 3
         end do
      end if
      return
      end

#ifdef IO64
      integer*8 function i_eight(i_value4)

        implicit none

        integer*4 i_value4

        i_eight=i_value4

        return

      end
#else
      integer*4 function i_eight(i_value4)

        implicit none

        integer*4 i_value4

        i_eight=i_value4

        return

      end
#endif

      real*4 function r_cnvrtdat(i_fmt,i_end,b_data)

      implicit none

      integer*4 i
      integer*4 i_fmt
      integer*4 i_end
      byte b_data(16)

      byte      b_value(16)
      integer*2 i_value2(8)
      integer*4 i_value4(4)
      real*4    r_value4(4)
      real*8    r_value8(2)
      real*4    r_val

      real*8    r_realval
      real*8    r_imagval
      equivalence(b_value,i_value2)
      equivalence(b_value,i_value4)
      equivalence(b_value,r_value4)
      equivalence(b_value,r_value8)

      r_val=0
      goto (10,20,30,40,50,60,70,80,90,100,110,120,130,140,150), abs(i_fmt)+1
         stop 'Format not recognized in r_cnvrtdat'

10    continue ! byte
         r_val = b_data(1)
         if (r_val .lt. 0.) r_val = r_val + 256
      goto 200

20    continue ! integer*1
         r_val = b_data(1)
         if (r_val .gt. 127.) r_val = r_val - 256
      goto 200

30    continue ! integer*2
         if (i_end .gt. 0) then
           b_value(1) = b_data(1)
           b_value(2) = b_data(2)
         else
           b_value(1) = b_data(2)
           b_value(2) = b_data(1)
         end if
         r_val = i_value2(1)
      goto 200

40    continue ! integer*4
         if (i_end .gt. 0) then
           b_value(1) = b_data(1)
           b_value(2) = b_data(2)
           b_value(3) = b_data(3)
           b_value(4) = b_data(4)
         else
           b_value(1) = b_data(4)
           b_value(2) = b_data(3)
           b_value(3) = b_data(2)
           b_value(4) = b_data(1)
         end if
         r_val = i_value4(1)
      goto 200

50    continue ! Real*4
         if (i_end .gt. 0) then
           b_value(1) = b_data(1)
           b_value(2) = b_data(2)
           b_value(3) = b_data(3)
           b_value(4) = b_data(4)
         else
           b_value(1) = b_data(4)
           b_value(2) = b_data(3)
           b_value(3) = b_data(2)
           b_value(4) = b_data(1)
         end if
         r_val = r_value4(1)
      goto 200

60    continue ! Real*8
         if (i_end .gt. 0) then
           do i=1,8
              b_value(i) = b_data(i)
           end do
         else
           do i=1,8
             b_value(i) = b_data(9-i)
           end do
         end if
         r_val = r_value8(1)
      goto 200

70    continue ! Complex*8 Magnitude
         if (i_end .gt. 0) then
            do i=1,8
               b_value(i) = b_data(i)
            end do
         else
            do i=1,4
               b_value(i) = b_data(5-i)
               b_value(4+i) = b_data(9-i)
            end do
         end if
         r_realval = r_value4(1)
         r_imagval = r_value4(2)
         r_val = sqrt(r_realval**2+r_imagval**2)
      goto 200

80    continue ! Complex*8 Phase
         if (i_end .gt. 0) then
            do i=1,8
               b_value(i) = b_data(i)
            end do
         else
            do i=1,4
               b_value(i) = b_data(5-i)
               b_value(4+i) = b_data(9-i)
            end do
         end if
         if (r_value4(2) .eq. 0.0 .and. r_value4(1) .eq. 0.0) then
           r_val=0.0
         else
           r_val = atan2(r_value4(2),r_value4(1))
         end if
      goto 200

90    continue ! unsigned integer*2
         i_value2(1) = 0
         if (i_end .gt. 0) then
           b_value(1) = b_data(1)
           b_value(2) = b_data(2)
         else
           b_value(1) = b_data(2)
           b_value(2) = b_data(1)
         end if
         if (i_value2(1) .ge.0) then
           r_val = i_value2(1)
         else
           r_val = i_value2(1)+65536
         end if
      goto 200

100    continue ! Stokes11
         i_value2(1) = b_data(1)
         if (i_value2(1) .ge. 128) i_value2(1) = i_value2(1)-256
         i_value2(2) = b_data(2)
         if (i_value2(2) .ge. 128) i_value2(2) = i_value2(2)-256
         r_val = ((float(int(i_value2(2)))/254.0) + 1.5) * 2.**(i_value2(1))
      goto 200
110   continue ! Complex*2 Magnitude
         r_value4(1) = b_data(1)
         r_value4(2) = b_data(2)
         r_val = sqrt(r_value4(1)**2+r_value4(2)**2)
      goto 200

120    continue ! Complex*2 Phase
         r_value4(1) = b_data(1)
         r_value4(2) = b_data(2)
         r_val = atan2(r_value4(2),r_value4(1))
      goto 200

130   continue ! Complex*4 Magnitude
         if (i_end .gt. 0) then
            do i=1,4
               b_value(i) = b_data(i)
            end do
         else
            do i=1,2
               b_value(i) = b_data(3-i)
               b_value(2+i) = b_data(5-i)
            end do
         end if

         r_val = sqrt(float(int(i_value2(1)))**2+float(int(i_value2(2)))**2)
      goto 200

140   continue ! Complex*4 Phase
         if (i_end .gt. 0) then
            do i=1,4
               b_value(i) = b_data(i)
            end do
         else
            do i=1,2
               b_value(i) = b_data(3-i)
               b_value(2+i) = b_data(5-i)
            end do
         end if
         r_val = atan2(float(int(i_value2(2))),float(int(i_value2(1))))
      goto 200

150    continue ! Real*4_Magnitude
         if (i_end .gt. 0) then
           b_value(1) = b_data(1)
           b_value(2) = b_data(2)
           b_value(3) = b_data(3)
           b_value(4) = b_data(4)
         else
           b_value(1) = b_data(4)
           b_value(2) = b_data(3)
           b_value(3) = b_data(2)
           b_value(4) = b_data(1)
         end if
         r_val = abs(r_value4(1))
      goto 200




200   continue
      r_cnvrtdat = r_val
      return
      end


      real*4 function r_cnvrtdat_old(i_fmt,b_data)

      implicit none

      integer*4 i
      integer*4 i_fmt
      byte b_data(16)

      byte      b_value(16)
      integer*2 i_value2(8)
      integer*4 i_value4(4)
      real*4    r_value4(4)
      real*4    r_value8(2)
      real*4    r_val
      equivalence(b_value,i_value2)
      equivalence(b_value,i_value4)
      equivalence(b_value,r_value4)
      equivalence(b_value,r_value8)

      r_val=0
      if (i_fmt .lt. 0) then
         stop 'Format not recognized'
      else if (i_fmt .eq. 0) then  ! byte
         r_value4(1) = b_data(1)
         if (r_value4(1) .lt. 0.) r_value4(1) = r_value4(1) + 256
         r_val = r_value4(1)
      else if (i_fmt .eq. 1) then  ! integer*1
         r_value4(1) = b_data(1)
         if (r_value4(1) .gt. 127.) r_value4(1) = r_value4(1) - 256
         r_val = r_value4(1)
      else if (i_fmt .eq. 2) then  ! integer*2
         do i=1,2
            b_value(i) = b_data(i)
         end do
         r_val = i_value2(1)
      else if (i_fmt .eq. 3) then  ! integer*4
         do i=1,4
            b_value(i) = b_data(i)
         end do
         r_val = i_value4(1)
      else if (i_fmt .eq. 4) then  ! Real*4
         do i=1,4
            b_value(i) = b_data(i)
         end do
         r_val = r_value4(1)
      else if (i_fmt .eq. 5) then  ! Real*8
         do i=1,8
            b_value(i) = b_data(i)
         end do
         r_val = r_value8(1)
      else if (i_fmt .eq. 6) then  ! Complex Magnitude
         do i=1,8
            b_value(i) = b_data(i)
         end do
         r_val = sqrt(r_value4(1)**2+r_value4(2)**2)
      else if (i_fmt .eq. 7) then  ! Complex Phase
         do i=1,8
            b_value(i) = b_data(i)
         end do
         r_val = atan2(r_value4(2),r_value4(1))
      else
         write(6,*) 'Fmt = ',i_fmt
         stop 'Format not recognized in r_cnvrtdat'
      end if
      r_cnvrtdat_old = r_val
      return
      end



      integer*4 function i_checknul(i_byt,b_data,b_vnul)

      implicit none

      integer*4 i
      integer*4 i_byt
      integer*4 i_flg
      byte b_data(16)
      byte b_vnul(0:16)
      integer i_vnul

      if (b_vnul(0) .eq. 0) then
         i_flg = 0
      else
         i_flg = 1
c         write(6,*) 'b_vnul(0)=',b_vnul(0)
         i_vnul=b_vnul(0)
         if (i_vnul .lt. 0) i_vnul=i_vnul+256
         do i=1,min(i_byt,int(b_vnul(0)))
c            write(6,*) b_data(i),b_vnul(i)
            if (b_data(i) .ne. b_vnul(i)) i_flg=0
         end do
      end if
      i_checknul = i_flg
      return
      end
