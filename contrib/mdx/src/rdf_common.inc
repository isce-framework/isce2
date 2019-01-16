c     PARAMETER STATEMENTS:
      integer I_PARAMS
      parameter(I_PARAMS = 100)

      integer I_MCPF
      parameter(I_MCPF = 120)

      integer i_nums
      integer i_pntr
      character*120 a_dsets(I_PARAMS)
      character*120 a_prfxs(I_PARAMS)
      character*120 a_sufxs(I_PARAMS)
      character*120 a_strts(I_PARAMS)
      character*120 a_matks(I_PARAMS)
      character*120 a_keyws(I_PARAMS)
      character*120 a_units(I_PARAMS)
      character*120 a_dimns(I_PARAMS)
      character*120 a_elems(I_PARAMS)
      character*120 a_opers(I_PARAMS)
      character*120 a_cmnts(I_PARAMS)
      character*120 a_valus(I_PARAMS)
      common /params/ i_pntr,i_nums,a_dsets,a_prfxs,a_sufxs,a_strts,a_matks,
     &                a_keyws,a_units,a_dimns,a_elems,a_opers,a_valus,a_cmnts

      integer i_errflag(3)
	  integer i_error
      character*120 a_errfile
      character*120 a_error(I_PARAMS)
      common /errmsg/ i_errflag,i_error,a_error,a_errfile

      integer i_fsizes(10)
      integer i_delflag(4)
      character*120 a_intfmt
      character*120 a_realfmt
      character*120 a_dblefmt
      character*120 a_cmdl(0:2)
      character*120 a_version
      common /inital/ i_fsizes,i_delflag,a_intfmt,a_realfmt,a_dblefmt,a_cmdl,a_version

      integer i_prelen
      integer i_suflen
      character*120 a_prfx
      character*120 a_sufx
      character*120 a_prefix
      character*120 a_suffix
      common /indata/ a_prfx,a_sufx,a_prefix,a_suffix,i_prelen,i_suflen

      integer i_stack
      character*120 a_stack(10)
      common /stack/ i_stack,a_stack

