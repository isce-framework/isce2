c****************************************************************

      character*(*) function rdfversion()

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      call rdf_trace('RDFVERSION')

      rdfversion = a_version

      call rdf_trace(' ')
      return

      end

c****************************************************************

      subroutine rdf_init(a_data)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      character*(*) a_data
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i
      integer i_lun
      integer i_iostat
      integer i_tabs(10)

      integer i_val
      character*320 a_vals(100)

      character*320 a_keyw
      character*320 a_valu
      character*320 a_unit
      character*320 a_dimn
      character*320 a_elem
      character*320 a_oper
      character*320 a_cmnt
      character*320 a_errtmp

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer*4 rdflen
      external rdflen

      character*320 rdftrim
      external rdftrim

      character*320 rdfupper
      external rdfupper

c     DATA STATEMENTS:

      data i_errflag / 1, 0, 0    /
      data i_error   / 0          /
      data a_errfile / 'message'  /
      data i_fsizes  / 40, 10,  6,  4,  4, 11,  3, 0, 0, 0/
      data i_prelen  /  0         /
      data i_suflen  /  0         /
      data i_stack   /  0         /
      data a_prefix  /  ' '       /
      data a_suffix  /  ' '       /
      data a_prfx    /  ' '       /
      data a_sufx    /  ' '       /
      data a_intfmt  / 'i'        /
      data a_realfmt / 'f'        /
      data a_dblefmt / '*'        /
      data a_cmdl(0) / '!'        /
      data a_cmdl(1) / ';'        /
      data a_cmdl(2) / ' '        /
      data i_delflag / 0, 0, 0, 0 /
      data a_version /'<< RDF_READER  Version 30.0    30-September-1999 >>'/

c     PROCESSING STEPS:

      call rdf_trace('RDF_INIT')
         if (a_data .ne. ' ') then
           call rdf_parse(a_data,a_keyw,a_unit,a_dimn,a_elem,a_oper,a_valu,a_cmnt)

           a_keyw = rdfupper(a_keyw)
           if (a_keyw .eq. ' ') then
              call rdf_error('Command field blank. ')
           else if (a_keyw .eq. 'ERRFILE') then
              write(6,*) 'Error file = ',a_valu(1:max(1,rdflen(a_valu)))
              if (rdfupper(a_errfile) .eq. 'SCREEN') then
                i_errflag(1) = 1
                i_errflag(2) = 0
                i_errflag(3) = 0
                a_errfile = ' '
              else if (rdfupper(a_errfile) .eq. 'MESSAGE') then
                i_errflag(1) = 0
                i_errflag(2) = 1
                i_errflag(3) = 0
                a_errfile = ' '
              else
                i_errflag(1) = 0
                i_errflag(2) = 0
                i_errflag(3) = 1
                a_errfile = a_valu
              endif
           else if (a_keyw .eq. 'ERROR_SCREEN') then
              if (rdfupper(a_valu) .eq. 'ON') then
                i_errflag(1) = 1
              else
                i_errflag(1) = 0
              endif
           else if (a_keyw .eq. 'ERROR_BUFFER') then
              if (rdfupper(a_valu) .eq. 'ON') then
                i_errflag(2) = 1
              else
                i_errflag(2) = 0
              endif
           else if (a_keyw .eq. 'ERROR_OUTPUT') then
              if (a_valu .eq. ' ' .or. rdfupper(a_valu) .eq. 'OFF') then
                i_errflag(3) = 0
                a_errfile = ' '
              else
                i_errflag(3) = 1
                a_errfile = a_valu
              endif
           else if (a_keyw .eq. 'COMMENT') then
              do i=1,3
                a_cmdl(i-1) = ' '
              end do
              call rdf_parse(a_data,a_keyw,a_unit,a_dimn,a_elem,a_oper,a_valu,a_cmnt)
              call rdf_getfields(a_valu,i_val,a_vals)
              do i=1,3
                if (i .le. i_val) then
                  a_cmdl(i-1) = a_vals(i)
                else
                  a_cmdl(i-1) = ' '
                end if
              end do
           else if (a_keyw .eq. 'COMMENT0') then
              a_cmdl(0) = a_valu
           else if (a_keyw .eq. 'COMMENT1') then
              a_cmdl(1) = a_valu
           else if (a_keyw .eq. 'COMMENT2') then
              a_cmdl(2) = a_valu
           else if (a_keyw .eq. 'COMMENT_DELIMITOR_SUPPRESS') then
              if (rdfupper(a_valu) .eq. 'ON') then
                i_delflag(1) = 1
              else
                i_delflag(1) = 0
              endif
           else if (a_keyw .eq. 'TABS') then
              read(a_valu,fmt=*,iostat=i_iostat) (i_tabs(i),i=1,7)
              if (i_iostat .ne. 0) then
                 a_errtmp = 'Unable to parse tab command. '// a_data(1:max(1,rdflen(a_data)))
                 call rdf_error(a_errtmp)
              endif
              write(6,*) 'tabs = ',(i_tabs(i),i=1,7)
              i_fsizes(1) = i_tabs(1)
              do i = 2,7
                i_fsizes(i) = i_tabs(i) - i_tabs(i-1)
              enddo
              write(6,*) 'fields = ',(i_fsizes(i),i=1,7)
           else if (a_keyw .eq. 'KEYWORD FIELD SIZE') then
              read(a_valu,fmt=*,iostat=i_iostat) i_fsizes(1)
              if (i_iostat .ne. 0) then
                 a_errtmp = 'Unable to parse keyword field size. '//a_data(1:max(1,rdflen(a_data)))
                 call rdf_error(a_errtmp)
              endif
           else if (a_keyw .eq. 'UNIT FIELD SIZE') then
              read(a_valu,fmt=*,iostat=i_iostat) i_fsizes(2)
              if (i_iostat .ne. 0) then
                 a_errtmp = 'Unable to parse unit field size. '//a_data(1:max(1,rdflen(a_data)))
                call rdf_error(a_errtmp)
              endif
           else if (a_keyw .eq. 'DIMENSION FIELD SIZE') then
              read(a_valu,fmt=*,iostat=i_iostat) i_fsizes(3)
              if (i_iostat .ne. 0) then
                 a_errtmp = 'Unable to parse dimension field size. '//a_data(1:max(1,rdflen(a_data)))
                 call rdf_error(a_errtmp)
              endif
           else if (a_keyw .eq. 'ELEMENT FIELD SIZE') then
              read(a_valu,fmt=*,iostat=i_iostat) i_fsizes(4)
              if (i_iostat .ne. 0) then
                 a_errtmp = 'Unable to parse element field size. '//a_data(1:max(1,rdflen(a_data)))
                 call rdf_error(a_errtmp)
              endif
           else if (a_keyw .eq. 'OPERATOR FIELD SIZE') then
              read(a_valu,fmt=*,iostat=i_iostat) i_fsizes(5)
              if (i_iostat .ne. 0) then
                 a_errtmp = 'Unable to parse operator field size. '//a_data(1:max(1,rdflen(a_data)))
                 call rdf_error(a_errtmp)
              endif
           else if (a_keyw .eq. 'VALUE FIELD SIZE') then
              read(a_valu,fmt=*,iostat=i_iostat) i_fsizes(6)
              if (i_iostat .ne. 0) then
                 a_errtmp = 'Unable to parse value field size. '//a_data(1:max(1,rdflen(a_data)))
                 call rdf_error(a_errtmp) 
              endif
           else if (a_keyw .eq. 'COMMENT FIELD SIZE') then
              read(a_valu,fmt=*,iostat=i_iostat) i_fsizes(7)
              if (i_iostat .ne. 0) then
                 a_errtmp = 'Unable to parse comment field size. '// a_data(1:max(1,rdflen(a_data)))
                 call rdf_error(a_errtmp)
              endif
           else if (a_keyw .eq. 'INTEGER FORMAT') then
              a_intfmt = a_valu
c              if (index(rdfupper(a_intfmt),'I') .eq. 0) then
c                call rdf_error('Unable to parse integer format. '//
c     &             a_data(1:max(1,rdflen(a_data))))
c              endif
           else if (a_keyw .eq. 'REAL FORMAT') then
              a_realfmt = a_valu
c              if (index(rdfupper(a_realfmt),'F') .eq. 0) then
c                call rdf_error('Unable to parse real format. '//
c     &             a_data(1:max(1,rdflen(a_data))))
c              endif
           else if (a_keyw .eq. 'DOUBLE FORMAT') then
              a_dblefmt = a_valu
c              if (index(rdfupper(a_dblefmt),'F') .eq. 0) then
c                call rdf_error('Unable to parse dble format. '//
c     &             a_data(1:max(1,rdflen(a_data))))
c              endif
           else
              a_errtmp = 'Command not recognized. '// a_data(1:max(1,rdflen(a_data)))
              call rdf_error(a_errtmp)
           endif
         endif
         call rdf_trace(' ')
         return
         end

c****************************************************************

      subroutine rdf_read(a_rdfname)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
c**   
c**   ROUTINES CALLED:
c**     rdf_merge
c**   
c**   NOTES: 
c**     rdf_merge actually reads the file.  rdf_read is a special case where
c**     you zero out all of the existing data loading into memory
c**   
c**   UPDATE LOG:
c**   
c**   Date Changed        Reason Changed                  CR # and Version #
c**   ------------       ----------------                 -----------------
c**   
c*****************************************************************

      implicit none

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      character*(*) a_rdfname
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

      data i_nums /0/

c     FUNCTION_STATEMENTS:

c     PROCESSING STEPS:

      call rdf_trace('RDF_READ')
      i_nums = 0                                ! zeros out all loaded data fields
      i_pntr = 0

      call rdf_merge(a_rdfname)

      call rdf_trace(' ')
      return
      end

c****************************************************************

      subroutine rdf_clear()

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      integer i
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

c     PROCESSING STEPS:

      call rdf_trace('RDF_CLEAR')
      do i=1,i_nums
         a_dsets(i) = ' '
         a_matks(i) = ' '
         a_strts(i) = ' '
         a_prfxs(i) = ' '
         a_sufxs(i) = ' '
         a_keyws(i) = ' '
         a_units(i) = ' '
         a_dimns(i) = ' '
         a_elems(i) = ' '
         a_opers(i) = ' '
         a_valus(i) = ' '
         a_cmnts(i) = ' '
      enddo


      i_nums = 0
      i_pntr = 0

      call rdf_trace(' ')
      return
      end  
      
c****************************************************************

      subroutine rdf_num(i_num)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      integer i_num
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

c     PROCESSING STEPS:

      call rdf_trace('RDF_NUM')
      i_num = i_nums
c      i_pntr = i_nums

      call rdf_trace(' ')
      return
      end  
      
c****************************************************************

      integer*4 function rdfnum()

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

c     PROCESSING STEPS:

      call rdf_trace('RDFNUM')
      i_pntr = i_nums
      rdfnum = i_nums

      call rdf_trace(' ')
      return
      end  
      
c****************************************************************

      subroutine rdf_insert(a_data)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      character*(*) a_data
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i
      integer i_flg
      integer i_indx

      integer i_loc
      integer i_indxx
      integer i_iostat

      character*320 a_keyw
      character*320 a_valu
      character*320 a_unit
      character*320 a_dimn
      character*320 a_elem
      character*320 a_oper
      character*320 a_cmnt

      character*320 a_kkkk
      character*320 a_dset
      character*320 a_errtmp

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer rdflen
      external rdflen

      character*320 rdftrim
      external rdftrim

      character*320 rdfupper
      external rdfupper

      character*320 rdfcullsp
      external rdfcullsp

      character*320 rdfint1
      external rdfint1

c     PROCESSING STEPS:

      call rdf_trace('RDF_INSERT')
      if (i_pntr .eq. 0) then
         i_indx=1
      else
         i_indx=i_pntr
      endif

      call rdf_parse(a_data,a_keyw,a_unit,a_dimn,a_elem,a_oper,a_valu,a_cmnt)

      if (a_keyw .ne. ' ') call rdf_index(a_keyw,i_indxx,i_flg)
c      if (i_flg .gt. 0) then
c         call rdf_error('Parameter already exists.  '//
c     &           a_keyw(1:max(rdflen(a_keyw),1)))
c      else

      if (.true.) then

        if (i_nums .ge. I_PARAMS) then
           a_errtmp = 'RDF Buffer full, unable to insert parameter.  '//
     &             a_keyw(1:max(rdflen(a_keyw),1))
           call rdf_error(a_errtmp)
        else if (i_indx .lt. 1 .or. i_indx .gt. i_nums+1) then
           a_errtmp = 'Index not within valid range 1 to i_nums+1.  '//
     &             a_keyw(1:max(rdflen(a_keyw),1))//' '//rdfint1(i_indx)
           call rdf_error(a_errtmp)
        else

           i_loc = index(a_keyw,':')
           if (i_loc .gt. 0) then
              a_kkkk = rdftrim(a_keyw(i_loc+1:))
              if (i_loc .gt. 1) then
                 a_dset = rdftrim(a_keyw(1:i_loc-1))
              else
                 a_dset = ' '
              endif
           else
              a_kkkk = rdftrim(a_keyw)
              a_dset = ' '
           endif

           if (rdfupper(a_kkkk) .eq. 'PREFIX') then
              a_prfx = a_valu
              a_prefix = a_prfx
              call rdf_unquote(a_prefix,i_prelen)
           else if (rdfupper(a_kkkk) .eq. 'SUFFIX') then
              a_sufx = a_valu
              a_suffix = a_sufx
              call rdf_unquote(a_suffix,i_suflen)
           else
              do i=i_nums,i_indx,-1
            
                 a_dsets(i+1) = a_dsets(i)
                 a_matks(i+1) = a_matks(i)
                 a_strts(i+1) = a_strts(i)
                 a_prfxs(i+1) = a_prfxs(i)
                 a_sufxs(i+1) = a_sufxs(i)
                 a_keyws(i+1) = a_keyws(i)
                 a_valus(i+1) = a_valus(i)
                 a_units(i+1) = a_units(i)
                 a_dimns(i+1) = a_dimns(i)
                 a_elems(i+1) = a_elems(i)
                 a_opers(i+1) = a_opers(i)
                 a_cmnts(i+1) = a_cmnts(i)

              enddo

              i_nums = i_nums + 1

              a_dsets(i_indx) = a_dset
              a_strts(i_indx) = ' '
              a_keyws(i_indx) = a_kkkk
              a_valus(i_indx) = a_valu
              a_units(i_indx) = a_unit
              a_dimns(i_indx) = a_dimn
              a_elems(i_indx) = a_elem
              a_opers(i_indx) = a_oper
              a_cmnts(i_indx) = a_cmnt

              if (a_keyws(i_indx) .ne. ' ') then
                 a_prfxs(i_indx) = a_prfx
                 a_sufxs(i_indx) = a_sufx
                 if (i_prelen .gt. 0) then
                    a_matks(i_indx) = rdfupper(rdfcullsp(rdftrim(a_prefix(1:i_prelen)//a_keyws(i_indx))))
                 else
                    a_matks(i_indx) = rdfupper(rdfcullsp(rdftrim(a_keyws(i_indx))))
                 endif
                 a_matks(i_indx) = a_matks(i_indx)(1:rdflen(a_matks(i_indx)))//rdfupper(rdfcullsp(a_suffix))
              else
                 a_prfxs(i_indx) = ' '
                 a_sufxs(i_indx) = ' '
                 a_matks(i_indx) = ' '
              endif
           endif

           i_pntr = 0
           if (a_keyw .ne. ' ') call rdf_index(a_keyw,i_indxx,i_flg)

           i_pntr = i_indx

        endif

      endif

      call rdf_trace(' ')

      return

      end

c****************************************************************

      subroutine rdf_append(a_data)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      integer i
      integer i_flg

      character*(*) a_data
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i_loc
      integer i_lun
      integer i_indx
      integer i_indxx
      integer i_iostat

      character*320 a_keyw
      character*320 a_valu
      character*320 a_unit
      character*320 a_dimn
      character*320 a_elem
      character*320 a_oper
      character*320 a_cmnt

      character*320 a_kkkk
      character*320 a_dset
      character*320 a_errtmp

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer rdflen
      external rdflen

      character*320 rdftrim
      external rdftrim

      character*320 rdfupper
      external rdfupper

      character*320 rdfcullsp
      external rdfcullsp

      character*320 rdfint1
      external rdfint1

c     PROCESSING STEPS:

      call rdf_trace('RDF_APPEND')
      if (i_pntr .eq. 0) then
         i_indx=i_nums
      else
         i_indx=i_pntr
      endif

      call rdf_parse(a_data,a_keyw,a_unit,a_dimn,a_elem,a_oper,a_valu,a_cmnt)

      i_flg = 0
      if (a_keyw .ne. ' ') call rdf_index(a_keyw,i_indxx,i_flg)
      if (i_flg .gt. 0) then
         a_errtmp = 'Parameter already exists.  '//
     &           a_keyw(1:max(rdflen(a_keyw),1))
         call rdf_error(a_errtmp)
      else

        if (i_nums .ge. I_PARAMS) then
           a_errtmp = 'Buffer full, unable to insert parameter.  '//
     &             a_keyw(1:max(rdflen(a_keyw),1))
           call rdf_error(a_errtmp)
        else if (i_indx .lt. 0 .or. i_indx .gt. i_nums) then
           a_errtmp = 'Index not within valid range 1 to i_nums+1.  '//
     &             a_keyw(1:max(rdflen(a_keyw),1))//' '//rdfint1(i_indx)
           call rdf_error(a_errtmp)
        else

           i_loc = index(a_keyw,':')
           if (i_loc .gt. 0) then
              a_kkkk = rdftrim(a_keyw(i_loc+1:))
              if (i_loc .gt. 1) then
                 a_dset = rdftrim(a_keyw(1:i_loc-1))
              else
                 a_dset = ' '
              endif
           else
              a_kkkk = rdftrim(a_keyw)
              a_dset = ' '
           endif

           if (rdfupper(a_kkkk) .eq. 'PREFIX') then
              a_prfx = a_valu
              a_prefix = a_prfx
              call rdf_unquote(a_prefix,i_prelen)
           else if (rdfupper(a_kkkk) .eq. 'SUFFIX') then
              a_sufx = a_valu
              a_suffix = a_sufx
              call rdf_unquote(a_suffix,i_suflen)
           else
              do i=i_nums,i_indx+1,-1
            
                 a_dsets(i+1) = a_dsets(i)
                 a_matks(i+1) = a_matks(i)
                 a_strts(i+1) = a_strts(i)
                 a_prfxs(i+1) = a_prfxs(i)
                 a_sufxs(i+1) = a_sufxs(i)
                 a_keyws(i+1) = a_keyws(i)
                 a_valus(i+1) = a_valus(i)
                 a_units(i+1) = a_units(i)
                 a_dimns(i+1) = a_dimns(i)
                 a_elems(i+1) = a_elems(i)
                 a_opers(i+1) = a_opers(i)
                 a_cmnts(i+1) = a_cmnts(i)

              enddo

              i_nums = i_nums+1

              a_dsets(i_indx+1) = a_dset
              a_strts(i_indx+1) = ' '
              a_keyws(i_indx+1) = a_kkkk
              a_valus(i_indx+1) = a_valu
              a_units(i_indx+1) = a_unit
              a_dimns(i_indx+1) = a_dimn
              a_elems(i_indx+1) = a_elem
              a_opers(i_indx+1) = a_oper
              a_cmnts(i_indx+1) = a_cmnt

              if (a_keyws(i_indx+1) .ne. ' ') then
                 a_prfxs(i_indx+1) = a_prfx
                 a_sufxs(i_indx+1) = a_sufx
                 if (i_prelen .gt. 0) then
                    a_matks(i_indx+1) = rdfupper(rdfcullsp(rdftrim(a_prefix(1:i_prelen)//a_keyws(i_indx+1))))
                 else
                    a_matks(i_indx+1) = rdfupper(rdfcullsp(rdftrim(a_keyws(i_indx+1))))
                 endif
                 a_matks(i_indx+1) = a_matks(i_indx+1)(1:rdflen(a_matks(i_indx+1)))//rdfupper(rdfcullsp(a_suffix))
              else
                 a_prfxs(i_indx+1) = ' '
                 a_sufxs(i_indx+1) = ' '
                 a_matks(i_indx+1) = ' '
              endif
           endif

           i_pntr = 0
           if (a_keyw .ne. ' ') call rdf_index(a_keyw,i_indxx,i_flg)

           i_pntr = i_indx+1

        endif

      endif

      call rdf_trace(' ')

      return

      end

c****************************************************************

      subroutine rdf_insertcols(a_keyw,a_valu,a_unit,a_dimn,a_elem,a_oper,a_cmnt)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      character*(*) a_keyw
      character*(*) a_valu
      character*(*) a_unit
      character*(*) a_dimn
      character*(*) a_elem
      character*(*) a_oper
      character*(*) a_cmnt
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i 
      integer i_flg
      integer i_loc
      integer i_lun
      integer i_indx
      integer i_indxx

      character*320 a_kkkk
      character*320 a_dset
      character*320 a_errtmp

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer rdflen
      external rdflen

      character*320 rdftrim
      external rdftrim

      character*320 rdfupper
      external rdfupper

      character*320 rdfcullsp
      external rdfcullsp

      character*320 rdfint1
      external rdfint1

c     PROCESSING STEPS:

      call rdf_trace('RDF_INSERTCOLS')
      if (i_pntr .eq. 0) then
         i_indx=1
      else
         i_indx=i_pntr
      endif

      if (i_nums .ge. I_PARAMS) then
         a_errtmp = 'Buffer full, unable to insert parameter.  '//
     &           a_keyw(1:max(rdflen(a_keyw),1))
         call rdf_error(a_errtmp)
      else if (i_indx .lt. 1 .or. i_indx .gt. i_nums+1) then
         a_errtmp = 'Index not within valid range 1 to i_nums+1.  '//
     &           a_keyw(1:max(rdflen(a_keyw),1))//' '//rdfint1(i_indx)
         call rdf_error(a_errtmp)
      else

         i_loc = index(a_keyw,':')
         if (i_loc .gt. 0) then
            a_kkkk = rdftrim(a_keyw(i_loc+1:))
            if (i_loc .gt. 1) then
               a_dset = rdfupper(rdfcullsp(rdftrim(a_keyw(1:i_loc-1))))
            else
               a_dset = ' '
            endif
         else
            a_kkkk = rdftrim(a_keyw)
            a_dset = ' '
         endif

         do i=i_nums,i_indx,-1
            
            a_dsets(i+1) = a_dsets(i)
            a_matks(i+1) = a_matks(i)
            a_strts(i+1) = a_strts(i)
            a_prfxs(i+1) = a_prfxs(i)
            a_sufxs(i+1) = a_sufxs(i)
            a_keyws(i+1) = a_keyws(i)
            a_valus(i+1) = a_valus(i)
            a_units(i+1) = a_units(i)
            a_dimns(i+1) = a_dimns(i)
            a_elems(i+1) = a_elems(i)
            a_opers(i+1) = a_opers(i)
            a_cmnts(i+1) = a_cmnts(i)

         enddo
         i_nums = i_nums + 1
         a_dsets(i_indx) = a_dset
         a_strts(i_indx) = ' '
         a_keyws(i_indx) = a_kkkk
         a_valus(i_indx) = a_valu
         a_units(i_indx) = a_unit
         a_dimns(i_indx) = a_dimn
         a_elems(i_indx) = a_elem
         a_opers(i_indx) = a_oper
         a_cmnts(i_indx) = a_cmnt

         if (a_keyws(i_indx) .ne. ' ') then
           a_prfxs(i_indx) = a_prfx
           a_sufxs(i_indx) = a_sufx
           if (i_prelen .gt. 0) then
              a_matks(i_indx) = rdfupper(rdfcullsp(rdftrim(a_prefix(1:i_prelen)//a_keyws(i_indx))))
           else
              a_matks(i_indx) = rdfupper(rdfcullsp(rdftrim(a_keyws(i_indx))))
           endif
           a_matks(i_indx) = a_matks(i_indx)(1:rdflen(a_matks(i_indx)))//rdfupper(rdfcullsp(a_suffix))
         else
           a_prfxs(i_indx) = ' '
           a_sufxs(i_indx) = ' '
           a_matks(i_indx) = ' '
         endif

         i_pntr = 0
         if (a_keyw .ne. ' ') call rdf_index(a_keyw,i_indxx,i_flg)

         i_pntr = i_indx
      endif

      call rdf_trace(' ')
      return

      end

c****************************************************************

      subroutine rdf_appendcols(a_keyw,a_valu,a_unit,a_dimn,a_elem,a_oper,a_cmnt)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      character*(*) a_keyw
      character*(*) a_valu
      character*(*) a_unit
      character*(*) a_dimn
      character*(*) a_elem
      character*(*) a_oper
      character*(*) a_cmnt
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i
      integer i_flg
      integer i_loc
      integer i_lun
      integer i_indx
      integer i_indxx


      character*320 a_kkkk
      character*320 a_dset
      character*320 a_errtmp

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer*4 rdflen
      external rdflen

      character*320 rdftrim
      external rdftrim

      character*320 rdfint1
      external rdfint1

      character*320 rdfupper
      external rdfupper

      character*320 rdfcullsp
      external rdfcullsp

c     PROCESSING STEPS:

      call rdf_trace('RDF_APPENDCOLS')
      if (i_pntr .eq. 0) then
         i_indx=i_nums
      else
         i_indx=i_pntr
      endif


      if (i_nums .ge. I_PARAMS) then
         a_errtmp = 'Buffer full, unable to insert parameter.  '//
     &           a_keyw(1:max(rdflen(a_keyw),1))
         call rdf_error(a_errtmp)
      else if (i_indx .lt. 0 .or. i_indx .gt. i_nums) then
         a_errtmp = 'Index not within valid range 1 to i_nums+1.  '//
     &           a_keyw(1:max(rdflen(a_keyw),1))//' '//rdfint1(i_indx-1)
         call rdf_error(a_errtmp)
      else

         i_loc = index(a_keyw,':')
         if (i_loc .gt. 0) then
            a_kkkk = rdftrim(a_keyw(i_loc+1:))
            if (i_loc .gt. 1) then
               a_dset = rdfupper(rdfcullsp(rdftrim(a_keyw(1:i_loc-1))))
            else
               a_dset = ' '
            endif
         else
            a_kkkk = rdftrim(a_keyw)
            a_dset = ' '
         endif

         if (rdfupper(a_kkkk) .eq. 'PREFIX') then
            a_prfx = a_valu
            a_prefix = a_prfx
            call rdf_unquote(a_prefix,i_prelen)
         else if (rdfupper(a_kkkk) .eq. 'SUFFIX') then
            a_sufx = a_valu
            a_suffix = a_sufx
            call rdf_unquote(a_suffix,i_suflen)
         else
           do i=i_nums,i_indx+1,-1
            
              a_dsets(i+1) = a_dsets(i)
              a_strts(i+1) = a_strts(i)
              a_prfxs(i+1) = a_prfxs(i)
              a_sufxs(i+1) = a_sufxs(i)
              a_keyws(i+1) = a_keyws(i)
              a_valus(i+1) = a_valus(i)
              a_units(i+1) = a_units(i)
              a_dimns(i+1) = a_dimns(i)
              a_elems(i+1) = a_elems(i)
              a_opers(i+1) = a_opers(i)
              a_cmnts(i+1) = a_cmnts(i)

           enddo
           a_dsets(i_indx+1) = a_dset
           a_strts(i_indx+1) = ' '
           a_keyws(i_indx+1) = a_kkkk
           a_valus(i_indx+1) = a_valu
           a_units(i_indx+1) = a_unit
           a_dimns(i_indx+1) = a_dimn
           a_elems(i_indx+1) = a_elem
           a_opers(i_indx+1) = a_oper
           a_cmnts(i_indx+1) = a_cmnt
           if (a_keyws(i_indx+1) .ne. ' ') then
             a_prfxs(i_indx+1) = a_prfx
             a_sufxs(i_indx+1) = a_sufx
             if (i_prelen .gt. 0) then
                a_matks(i_indx+1) = rdfupper(rdfcullsp(rdftrim(a_prefix(1:i_prelen)//a_keyws(i_indx+1))))
             else
                a_matks(i_indx+1) = rdfupper(rdfcullsp(rdftrim(a_keyws(i_indx+1))))
             endif
             a_matks(i_indx+1) = a_matks(i_indx+1)(1:rdflen(a_matks(i_indx+1)))//rdfupper(rdfcullsp(a_suffix))
           else
             a_prfxs(i_indx+1) = ' '
             a_sufxs(i_indx+1) = ' '
             a_matks(i_indx+1) = ' '
           endif

           i_pntr = 0
           if (a_keyw .ne. ' ') call rdf_index(a_keyw,i_indxx,i_flg)

           i_pntr = i_indx+1
           i_nums = i_nums + 1
         endif
      endif

      call rdf_trace(' ')
      return

      end

c****************************************************************

      subroutine rdf_entercols(i_indx,a_keyw,a_valu,a_unit,a_dimn,a_elem,a_oper,a_cmnt)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      integer i
      integer i_indx

      character*(*) a_keyw
      character*(*) a_valu
      character*(*) a_unit
      character*(*) a_dimn
      character*(*) a_elem
      character*(*) a_oper
      character*(*) a_cmnt
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i_flg
      integer i_loc
      integer i_lun
      integer i_indxx
      integer i_indxxx

      character*320 a_kkkk
      character*320 a_dset
      character*320 a_errtmp

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer rdflen
      external rdflen

      character*320 rdftrim
      external rdftrim

      character*320 rdfupper
      external rdfupper

      character*320 rdfcullsp
      external rdfcullsp

      character*320 rdfint1
      external rdfint1

c     PROCESSING STEPS:

      call rdf_trace('RDF_ENTERCOLS')
      if (i_indx .eq. 0) then
         i_indxx=i_pntr
      else
         i_indxx=i_indx
      endif

      if (i_nums .ge. I_PARAMS) then
         a_errtmp = 'Buffer full, unable to insert parameter.  '//
     &           a_keyw(1:max(rdflen(a_keyw),1))
         call rdf_error(a_errtmp)
      else if (i_indxx .lt. 1 .or. i_indxx .gt. i_nums+1) then
         a_errtmp = 'Index not within valid range 1 to i_nums+1.  '//
     &           a_keyw(1:max(rdflen(a_keyw),1))//' '//rdfint1(i_indxx)
         call rdf_error(a_errtmp)
      else

         i_loc = index(a_keyw,':')
         if (i_loc .gt. 0) then
            a_kkkk = rdftrim(a_keyw(i_loc+1:))
            if (i_loc .gt. 1) then
               a_dset = rdfupper(rdfcullsp(rdftrim(a_keyw(1:i_loc-1))))
            else
               a_dset = ' '
            endif
         else
            a_kkkk = rdftrim(a_keyw)
            a_dset = ' '
         endif

         do i=i_nums,i_indxx,-1
            
            a_dsets(i+1) = a_dsets(i)
            a_strts(i+1) = a_strts(i)
            a_prfxs(i+1) = a_prfxs(i)
            a_sufxs(i+1) = a_sufxs(i)
            a_keyws(i+1) = a_keyws(i)
            a_valus(i+1) = a_valus(i)
            a_units(i+1) = a_units(i)
            a_dimns(i+1) = a_dimns(i)
            a_elems(i+1) = a_elems(i)
            a_opers(i+1) = a_opers(i)
            a_cmnts(i+1) = a_cmnts(i)

         enddo
         i_nums = i_nums + 1
         a_dsets(i_indxx) = a_dset
         a_strts(i_indxx) = ' '
         a_keyws(i_indxx) = a_kkkk
         a_valus(i_indxx) = a_valu
         a_units(i_indxx) = a_unit
         a_dimns(i_indxx) = a_dimn
         a_elems(i_indxx) = a_elem
         a_opers(i_indxx) = a_oper
         a_cmnts(i_indxx) = a_cmnt
         if (a_keyws(i_indxx) .ne. ' ') then
           a_prfxs(i_indxx) = a_prfx
           a_sufxs(i_indxx) = a_sufx
           if (i_prelen .gt. 0) then
              a_matks(i_indxx) = rdfupper(rdfcullsp(rdftrim(a_prefix(1:i_prelen)//a_keyws(i_indxx))))
           else
              a_matks(i_indxx) = rdfupper(rdfcullsp(rdftrim(a_keyws(i_indxx))))
           endif
           a_matks(i_indxx) = a_matks(i_indxx)(1:rdflen(a_matks(i_indxx)))//rdfupper(rdfcullsp(a_suffix))
         else
           a_prfxs(i_indxx) = ' '
           a_sufxs(i_indxx) = ' '
           a_matks(i_indxx) = ' '
         endif

         i_pntr = 0
         if (a_keyw .ne. ' ') call rdf_index(a_keyw,i_indxxx,i_flg)

        i_pntr = i_indxx
      endif

      call rdf_trace(' ')
      return

      end

c****************************************************************

      subroutine rdf_view(i_indx,a_data)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      integer i_indx
      
c     OUTPUT VARIABLES:

      character*(*) a_data

c     LOCAL VARIABLES:

      integer i_lun

      character*320 a_keyw
      character*320 a_valu
      character*320 a_unit
      character*320 a_dimn
      character*320 a_elem
      character*320 a_oper
      character*320 a_cmnt
      character*320 a_errtmp


c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer*4 rdflen
      external rdflen

      character*320 rdfint1
      external rdfint1

c     PROCESSING STEPS:

      call rdf_trace('RDF_VIEW')
      i_pntr = max(min(i_indx,i_nums),0)
      if (i_indx .ge. 1 .and. i_indx .le. i_nums) then

         if (a_dsets(i_indx) .eq. ' ') then
            a_keyw = a_matks(i_indx)
         else
            a_keyw = a_dsets(i_indx)(1:rdflen(a_dsets(i_indx)))//':'//a_matks(i_indx)
         endif
         a_valu = a_valus(i_indx)
         a_unit = a_units(i_indx)
         a_dimn = a_dimns(i_indx)
         a_elem = a_elems(i_indx)
         a_oper = a_opers(i_indx)
         a_cmnt = a_cmnts(i_indx)
c         type *,'a_keyw =',a_keyw(1:max(rdflen(a_keyw),1)),rdflen(a_keyw)
c         type *,'a_unit =',a_unit(1:max(rdflen(a_unit),1)),rdflen(a_unit)
c         type *,'a_dimn =',a_dimn(1:max(rdflen(a_dimn),1)),rdflen(a_dimn)
c         type *,'a_elem =',a_elem(1:max(rdflen(a_elem),1)),rdflen(a_elem)
c         type *,'a_oper =',a_oper(1:max(rdflen(a_oper),1)),rdflen(a_oper)
c         type *,'a_valu =',a_valu(1:max(rdflen(a_valu),1)),rdflen(a_valu)
c         type *,'a_cmnt =',a_cmnt(1:max(rdflen(a_cmnt),1)),rdflen(a_cmnt)
         call rdf_unparse(a_data,a_keyw,a_unit,a_dimn,a_elem,a_oper,a_valu,a_cmnt)
c         type *,'a_data =',a_data(1:max(rdflen(a_data),1)),rdflen(a_data)

      else
         a_valu = ' '
         a_unit = ' '
         a_dimn = ' '
         a_elem = ' '
         a_oper = ' '
         a_cmnt = ' '
         if (i_indx .ne. 0) then
            a_errtmp = 'Requested buffer entry does not contain valid data. '
     &          //rdfint1(i_indx)
            call rdf_error(a_errtmp)
         endif
         a_data = ' '
      endif

      call rdf_trace(' ')
      return

      end

c****************************************************************

      subroutine rdf_viewcols(i_indx,a_keyw,a_valu,a_unit,a_dimn,a_elem,a_oper,a_cmnt)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      integer i_indx
      
c     OUTPUT VARIABLES:

      character*(*) a_keyw
      character*(*) a_valu
      character*(*) a_unit
      character*(*) a_dimn
      character*(*) a_elem
      character*(*) a_oper
      character*(*) a_cmnt
      character*320 a_errtmp


c     LOCAL VARIABLES:

      integer i_lun

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer*4 rdflen
      external rdflen

      character*320 rdfint1
      external rdfint1

c     PROCESSING STEPS:

      call rdf_trace('RDF_VIEWCOLS')
      i_pntr = max(min(i_indx,i_nums),0)
      if (i_indx .ge. 1 .and. i_indx .le. i_nums) then

         if (a_dsets(i_indx) .eq. ' ') then
            a_keyw = a_keyws(i_indx)
         else
            a_keyw = a_dsets(i_indx)(1:rdflen(a_dsets(i_indx)))//':'//a_keyws(i_indx)
         endif
         a_valu = a_valus(i_indx)
         a_unit = a_units(i_indx)
         a_dimn = a_dimns(i_indx)
         a_elem = a_elems(i_indx)
         a_oper = a_opers(i_indx)
         a_cmnt = a_cmnts(i_indx)
c         i_pntr = i_indx

      else
         a_valu = ' '
         a_unit = ' '
         a_dimn = ' '
         a_elem = ' '
         a_oper = ' '
         a_cmnt = ' '
         if (i_indx .ne. 0) then 
            a_errtmp = 'Requested buffer entry does not contain valid data. '
     &          //rdfint1(i_indx)
            call rdf_error(a_errtmp)
         endif
      endif

      call rdf_trace(' ')
      return

      end

c****************************************************************

      subroutine rdf_find(a_keyw,a_data)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      character*(*) a_keyw
      character*320 a_valu
      character*320 a_unit
      character*320 a_dimn
      character*320 a_elem
      character*320 a_oper
      character*320 a_cmnt
      character*320 a_errtmp

      character*(*) a_data
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i_indx
      integer i_flg
      integer i_lun

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer*4 rdflen
      external rdflen

      character*320 rdfint1
      external rdfint1

c     PROCESSING STEPS:

      call rdf_trace('RDF_FIND')
      call rdf_index(a_keyw,i_indx,i_flg)

      if (i_flg .ge. 1) then
          a_valu = a_valus(i_indx)
          a_unit = a_units(i_indx)
          a_dimn = a_dimns(i_indx)
          a_elem = a_elems(i_indx)
          a_oper = a_opers(i_indx)
          a_cmnt = a_cmnts(i_indx)
          call rdf_unparse(a_data,a_keyw,a_unit,a_dimn,a_elem,a_oper,a_valu,a_cmnt)
      endif

      if (i_flg .eq. 0) then                                               ! Data not found
         a_errtmp = 'Keyword not found. '//a_keyw(1:max(min(rdflen(a_keyw),150),2))
         call rdf_error(a_errtmp)
      else if (i_flg .ge. 2) then
         a_errtmp = 'Multiple matching keywords found, returning last one found. '//
     &                 a_keyw(1:max(min(rdflen(a_keyw),150),2))//'  '//rdfint1(i_flg)
         call rdf_error(a_errtmp)
      endif
      call rdf_trace(' ')
      return

      end

c****************************************************************

      subroutine rdf_findcols(a_keyw,a_valu,a_unit,a_dimn,a_elem,a_oper,a_cmnt)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      character*(*) a_keyw
      character*(*) a_valu
      character*(*) a_unit
      character*(*) a_dimn
      character*(*) a_elem
      character*(*) a_oper
      character*(*) a_cmnt
      character*320 a_errtmp
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i_indx
      integer i_flg
      integer i_lun

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer*4 rdflen
      external rdflen

      character*320 rdfint1
      external rdfint1

c     PROCESSING STEPS:

      call rdf_trace('RDF_FINDCOLS')
      call rdf_index(a_keyw,i_indx,i_flg)

      if (i_flg .eq. 1) then
          a_valu = a_valus(i_indx)
          a_unit = a_units(i_indx)
          a_dimn = a_dimns(i_indx)
          a_elem = a_elems(i_indx)
          a_oper = a_opers(i_indx)
          a_cmnt = a_cmnts(i_indx)
      endif

      if (i_flg .eq. 0) then                                               ! Data not found
         a_errtmp = 'Keyword not found. '//a_keyw(1:max(min(rdflen(a_keyw),150),2))
         call rdf_error(a_errtmp)
      else if (i_flg .ge. 2) then
         a_errtmp = 'Multiple matching keywords found, returning last one found. '//
     &                 a_keyw(1:max(min(rdflen(a_keyw),150),2))//'  '//rdfint1(i_flg)
         call rdf_error(a_errtmp)
      endif
      call rdf_trace(' ')
      return

      end

c****************************************************************

      subroutine rdf_remove(a_keyw)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      character*(*) a_keyw
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i
      integer i_flg
      integer i_indx

      character*320 a_kkkk
      character*320 a_dset
      character*320 a_errtmp

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer*4 rdflen
      external rdflen

c     PROCESSING STEPS:

      call rdf_trace('RDF_REMOVE')
      call rdf_index(a_keyw,i_indx,i_flg)
      if (i_flg .eq. 0) then
         a_errtmp = 'Keyword not found. '//
     &        a_keyw(1:max(min(rdflen(a_keyw),150),2))
         call rdf_error(a_errtmp)
      else
        if (i_flg .gt. 1) then
           a_errtmp = 'Multiple Keywords found.  Deleting last occurance.  '//
     &        a_keyw(1:max(min(rdflen(a_keyw),150),2))
          call rdf_error(a_errtmp)
        endif
        i_pntr = i_indx
        do i = i_indx+1,i_nums
          a_dsets(i-1) = a_dsets(i)
          a_matks(i-1) = a_matks(i)
          a_strts(i-1) = a_strts(i)
          a_prfxs(i-1) = a_prfxs(i)
          a_sufxs(i-1) = a_sufxs(i)
          a_keyws(i-1) = a_keyws(i)
          a_valus(i-1) = a_valus(i)
          a_units(i-1) = a_units(i) 
          a_dimns(i-1) = a_dimns(i) 
          a_elems(i-1) = a_elems(i) 
          a_opers(i-1) = a_opers(i)
          a_cmnts(i-1) = a_cmnts(i)
        enddo
      endif
      i_nums = i_nums - 1

      call rdf_trace(' ')
      return

      end

c****************************************************************

      subroutine rdf_update(a_data)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      character*(*) a_data
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      character*320 a_keyw
      character*320 a_valu
      character*320 a_unit
      character*320 a_dimn
      character*320 a_elem
      character*320 a_oper
      character*320 a_cmnt
      character*320 a_errtmp
      
      integer i_flg
      integer i_indx
      integer i_lun
      integer i_iostat

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer*4 rdflen
      external rdflen

c     PROCESSING STEPS:


      call rdf_trace('RDF_UPDATE')
      call rdf_unparse(a_data,a_keyw,a_unit,a_dimn,a_elem,a_oper,a_valu,a_cmnt)
      call rdf_index(a_keyw,i_indx,i_flg)

      if (i_flg .ge. 1) then
         a_valus(i_indx) = a_valu
         a_units(i_indx) = a_unit 
         a_dimns(i_indx) = a_dimn
         a_elems(i_indx) = a_elem
         a_opers(i_indx) = a_oper
         a_cmnts(i_indx) = a_cmnt
      endif

      if (i_flg .eq. 0) then
         if (i_nums .lt. I_PARAMS) then
            a_errtmp = 'Keyword not found, inserting at end.  '//
     &           a_keyw(1:max(min(rdflen(a_keyw),150),2))
            call rdf_error(a_errtmp)
            call rdf_insertcols(a_keyw,a_valu,a_unit,a_dimn,a_elem,a_oper,a_cmnt)
         else
            a_errtmp = 'Buffer Full, cannot add parameter '//
     &           a_keyw(1:max(min(rdflen(a_keyw),150),2))
            call rdf_error(a_errtmp)
         endif
      endif

      call rdf_trace(' ')
      return

      end

c****************************************************************

      subroutine rdf_updatecols(a_keyw,a_unit,a_dimn,a_elem,a_oper,a_cmnt,a_valu)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      character*(*) a_keyw
      character*(*) a_valu
      character*(*) a_unit
      character*(*) a_dimn
      character*(*) a_elem
      character*(*) a_oper
      character*(*) a_cmnt
      character*320 a_errtmp
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i_flg
      integer i_indx
      integer i_lun
      integer i_iostat

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer*4 rdflen
      external rdflen

c     PROCESSING STEPS:

      call rdf_trace('RDF_UPDATECOLS')
      call rdf_index(a_keyw,i_indx,i_flg)

      if (i_flg .ge. 1) then
         a_valus(i_indx) = a_valu
         a_units(i_indx) = a_unit 
         a_dimns(i_indx) = a_dimn
         a_elems(i_indx) = a_elem
         a_opers(i_indx) = a_oper
         a_cmnts(i_indx) = a_cmnt
      endif

      if (i_flg .eq. 0) then
         if (i_nums .lt. I_PARAMS) then
            a_errtmp = 'Keyword not found, inserting at end.  '//
     &           a_keyw(1:max(min(rdflen(a_keyw),150),2))
            call rdf_error(a_errtmp)
            call rdf_insertcols(a_keyw,a_valu,a_unit,a_dimn,a_elem,a_oper,a_cmnt)
         else
            a_errtmp = 'Buffer Full, cannot add parameter '//
     &           a_keyw(1:max(min(rdflen(a_keyw),150),2))
            call rdf_error(a_errtmp)
         endif
      endif

      call rdf_trace(' ')
      return

      end

c****************************************************************

      subroutine rdf_index(a_keyw,i_indx,i_flg)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      character*(*) a_keyw
      
c     OUTPUT VARIABLES:

      integer i_indx
      integer i_flg

c     LOCAL VARIABLES:

      integer i
      integer i_loc
      integer i_ocr
      integer i_ocl
      integer i_cnt

      integer i_stat

      character*320 a_kkkk
      character*320 a_dset

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer*4 rdflen
      external rdflen

      character*320 rdfupper
      external rdfupper

      character*320 rdftrim
      external rdftrim

      character*320 rdfcullsp
      external rdfcullsp

      data i_ocl / 0/
      save i_ocl

      data i_cnt / 0/
      save i_cnt

c     PROCESSING STEPS:

      call rdf_trace('RDF_INDEX')
      i_loc = index(a_keyw,':')
      if (i_loc .gt. 0) then
         a_kkkk = rdfupper(rdfcullsp(rdftrim(a_keyw(i_loc+1:))))
         if (i_loc .gt. 1) then
            a_dset = rdfupper(rdfcullsp(rdftrim(a_keyw(1:i_loc-1))))
         else
            a_dset = ' '
         endif
      else
         a_kkkk = rdfupper(rdfcullsp(rdftrim(a_keyw)))
         a_dset = ' '
      endif

      i_loc = index(a_kkkk,';')
      if (i_loc .gt. 0) then
         read(a_kkkk(i_loc+1:),'(i10)',iostat=i_stat) i_ocr
         if (i_stat .ne. 0) call rdf_error('Error reading i_ocr')
         if (i_loc .gt. 1) then
            a_kkkk = a_kkkk(1:i_loc-1)
         else
            a_kkkk = ' '
         endif
      else
         i_ocr = 0
      endif

      i_flg  = 0
      i_indx = 0

c      type *,'a_kkkk=',a_kkkk(1:max(1,rdflen(a_kkkk)))
c      type *,'i_ocr =',i_ocr,i_ocl
      if (a_kkkk .ne. ' ') then
      if (i_pntr .ge. 1 .and. i_pntr .le. i_nums) then
        if (a_kkkk .eq. a_matks(i_pntr) .and.
     &       (a_dset .eq. a_dsets(i_pntr) .or. a_dset .eq. ' ') .and.
     &       ((i_ocr .eq. 0 .and. i_cnt .eq. 1).or. (i_ocr .eq. i_ocl)) ) then ! Found a match
           i_indx = i_pntr
           if (i_ocr .eq. 0) then
             i_flg = i_cnt
           else
             i_flg = 1
           endif
           call rdf_trace(' ')
           return
        endif
      endif

      i_pntr = 0
      i_ocl = 0
      i_cnt = 0
      i_flg = 0
      do i = 1,i_nums
            if (a_kkkk .eq. a_matks(i) .and.
     &           (a_dset .eq. a_dsets(i) .or. a_dset .eq. ' ') ) then ! Found a match
               i_cnt = i_cnt + 1
c               type *,'a_kkkk=a_matks(i)',i_cnt,'  ',a_matks(i)(1:max(1,rdflen(a_matks(i))))
               if (i_ocr .eq. i_cnt .or. i_ocr .eq. 0) then
                 i_flg = i_flg + 1
                 i_indx = i
                 i_pntr = i
                 i_ocl = i_cnt
               endif
            endif
      enddo
      endif

c      type *,'i_flg=',i_flg
      call rdf_trace(' ')
      return

      end

c****************************************************************

      integer*4 function rdfindx(a_keyw)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      character*(*) a_keyw
      character*320 a_errtmp

c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i_flg
      integer i_indx
      integer i_lun

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer*4 rdflen
      external rdflen

      character*320 rdfint1
      external rdfint1

c     PROCESSING STEPS:

      call rdf_trace('RDFINDX')
      call rdf_index(a_keyw,i_indx,i_flg)

      if (i_flg .eq. 0) then                                               ! Data not found
         a_errtmp = 'Keyword not found. '//a_keyw(1:max(min(rdflen(a_keyw),150),2))
         call rdf_error(a_errtmp)
      else if (i_flg .ge. 2) then
         a_errtmp = 'Multiple matching keywords found, returning index of last. '//
     &                 a_keyw(1:max(min(rdflen(a_keyw),150),2))//'  '//rdfint1(i_flg)
         call rdf_error(a_errtmp)
      endif

      rdfindx = i_indx

      call rdf_trace(' ')
      return

      end

c****************************************************************

      character*(*) function rdfvalu(a_keyw)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      character*(*) a_keyw
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i_flg
      integer i_indx
      integer i_lun
      integer i_iostat

      character*320 a_valu
      character*320 a_data
      character*320 a_errtmp

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer rdflen
      external rdflen

      character*320 rdfint1
      external rdfint1

c     PROCESSING STEPS:

      call rdf_trace('RDFVALU')
      a_valu = ' '

      call rdf_index(a_keyw,i_indx,i_flg)

      if (i_flg .eq. 1) then
          a_valu = a_valus(i_indx)
      else if (i_flg .eq. 0) then                                               ! Data not found
         a_errtmp = 'Keyword not found. '//a_keyw(1:max(min(rdflen(a_keyw),150),2))
         call rdf_error(a_errtmp)
          a_valu = ' '
      else if (i_flg .ge. 2) then
         a_errtmp = 'Multiple matching keywords found, returning index of last. '//
     &                 a_keyw(1:max(min(rdflen(a_keyw),150),2))//'  '//rdfint1(i_flg)
         call rdf_error(a_errtmp)
          a_valu = a_valus(i_indx)
      endif

      rdfvalu = a_valu

      call rdf_trace(' ')
      return

      end

c****************************************************************

      character*(*) function rdfunit(a_keyw)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      character*(*) a_keyw
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i_flg
      integer i_indx
      integer i_lun

      character*320 a_unit
      character*320 a_errtmp

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer rdflen
      external rdflen

      character*320 rdfint1
      external rdfint1

c     PROCESSING STEPS:

      call rdf_trace('RDFUNIT')
      a_unit = ' '

      call rdf_index(a_keyw,i_indx,i_flg)

      if (i_flg .eq. 1) then
          a_unit = a_units(i_indx)
      else if (i_flg .eq. 0) then                                               ! Data not found
         a_errtmp = 'Keyword not found. '//a_keyw(1:max(min(rdflen(a_keyw),150),2))
         call rdf_error(a_errtmp)
      else if (i_flg .ge. 2) then
         a_errtmp = 'Multiple matching keywords found, returning index of last. '//
     &                 a_keyw(1:max(min(rdflen(a_keyw),150),2))//'  '//rdfint1(i_flg)
         call rdf_error(a_errtmp)
      endif

      rdfunit = a_unit

      call rdf_trace(' ')
      return

      end

c****************************************************************

      character*(*) function rdfdimn(a_keyw)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      character*(*) a_keyw
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i_flg
      integer i_indx
      integer i_lun

      character*320 a_dimn
      character*320 a_errtmp

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer rdflen
      external rdflen

      character*320 rdfint1
      external rdfint1

c     PROCESSING STEPS:

      call rdf_trace('RDFDIMN')
      a_dimn = ' '

      call rdf_index(a_keyw,i_indx,i_flg)

      if (i_flg .eq. 1) then
          a_dimn = a_dimns(i_indx)
      else if (i_flg .eq. 0) then                                               ! Data not found
         a_errtmp = 'Keyword not found. '//a_keyw(1:max(min(rdflen(a_keyw),150),2))
         call rdf_error(a_errtmp)
      else if (i_flg .ge. 2) then
         a_errtmp = 'Multiple matching keywords found, returning index of last. '//
     &                 a_keyw(1:max(min(rdflen(a_keyw),150),2))//'  '//rdfint1(i_flg)
         call rdf_error(a_errtmp)
      endif

      rdfdimn = a_dimn

      call rdf_trace(' ')
      return

      end

c****************************************************************

      character*(*) function rdfelem(a_keyw)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      character*(*) a_keyw
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i_flg
      integer i_indx
      integer i_lun

      character*320 a_elem
      character*320 a_errtmp

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer rdflen
      external rdflen

      character*320 rdfint1
      external rdfint1

c     PROCESSING STEPS:

      call rdf_trace('RDFELEM')
      a_elem = ' '

      call rdf_index(a_keyw,i_indx,i_flg)

      if (i_flg .eq. 1) then
          a_elem = a_elems(i_indx)
      else if (i_flg .eq. 0) then                                               ! Data not found
         a_errtmp = 'Keyword not found. '//a_keyw(1:max(min(rdflen(a_keyw),150),2))
         call rdf_error(a_errtmp)
      else if (i_flg .ge. 2) then
         a_errtmp = 'Multiple matching keywords found, returning index of last. '//
     &                 a_keyw(1:max(min(rdflen(a_keyw),150),2))//'  '//rdfint1(i_flg)
         call rdf_error(a_errtmp)
      endif

      rdfelem = a_elem

      call rdf_trace(' ')
      return

      end

c****************************************************************

      character*(*) function rdfoper(a_keyw)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      character*(*) a_keyw
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i_flg
      integer i_indx
      integer i_lun

      character*320 a_oper
      character*320 a_errtmp

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

      integer rdflen
      external rdflen

      character*320 rdfint1
      external rdfint1

c     FUNCTION_STATEMENTS:

c     PROCESSING STEPS:

      call rdf_trace('RDFOPER')
      a_oper = ' '

      call rdf_index(a_keyw,i_indx,i_flg)

      if (i_flg .eq. 1) then
          a_oper = a_opers(i_indx)
      else if (i_flg .eq. 0) then                                               ! Data not found
         a_errtmp = 'Keyword not found. '//a_keyw(1:max(min(rdflen(a_keyw),150),2))
         call rdf_error(a_errtmp)
      else if (i_flg .ge. 2) then
         a_errtmp = 'Multiple matching keywords found, returning index of last. '//
     &                 a_keyw(1:max(min(rdflen(a_keyw),150),2))//'  '//rdfint1(i_flg)
         call rdf_error(a_errtmp)
      endif

      rdfoper = a_oper

      call rdf_trace(' ')
      return

      end

c****************************************************************

      character*(*) function rdfcmnt(a_keyw)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      character*(*) a_keyw
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i_flg
      integer i_indx
      integer i_lun

      character*320 a_cmnt
      character*320 a_errtmp

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer rdflen
      external rdflen

      character*320 rdfint1
      external rdfint1

c     PROCESSING STEPS:

      call rdf_trace('RDFCMNT')
      a_cmnt = ' '

      call rdf_index(a_keyw,i_indx,i_flg)

      if (i_flg .eq. 1) then
          a_cmnt = a_cmnts(i_indx)
      else if (i_flg .eq. 0) then                                               ! Data not found
         a_errtmp = 'Keyword not found. '//a_keyw(1:max(min(rdflen(a_keyw),150),1))
         call rdf_error(a_errtmp)
      else if (i_flg .ge. 2) then
         a_errtmp = 'Multiple matching keywords found, returning index of last. '//
     &                 a_keyw(1:max(min(rdflen(a_keyw),150),2))//'  '//rdfint1(i_flg)
         call rdf_error(a_errtmp)
      endif

      rdfcmnt = a_cmnt

      call rdf_trace(' ')
      return

      end

c****************************************************************

      character*(*) function rdfval(a_keyw,a_unit)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
c**      This routine is just to maintain backward compatibility 
c**      with older versions of rdf_reader.  Should use rdfdata.
c**   
c**   ROUTINES CALLED:
c**      rdfdata
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

      character*(*) a_keyw
      character*(*) a_unit
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      character*320 rdfdata
      external rdfdata

c     PROCESSING STEPS:

      call rdf_trace('RDFVAL')
      rdfval = rdfdata(a_keyw,a_unit)

      call rdf_trace(' ')
      return
      end


c****************************************************************

      character*(*) function rdfdata(a_keyw,a_ounit)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      character*(*) a_keyw
      character*(*) a_ounit
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i_flg
      integer i_indx
      integer i_lun

      character*320 a_valu
      character*320 a_unit
      character*320 a_dimn
      character*320 a_elem
      character*320 a_oper
      character*320 a_cmnt
      character*320 a_errtmp

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer rdflen
      external rdflen

      character*320 rdfint1
      external rdfint1

c     PROCESSING STEPS:

      call rdf_trace('RDFDATA')
      a_valu = ' '
      a_unit = ' '
      a_dimn = ' '
      a_elem = ' '
      a_oper = ' '
      a_cmnt = ' '

      call rdf_index(a_keyw,i_indx,i_flg)

      if (i_flg .eq. 1) then
          a_valu = a_valus(i_indx)
          a_unit = a_units(i_indx)
          a_dimn = a_dimns(i_indx)
          a_elem = a_elems(i_indx)
          a_oper = a_opers(i_indx)
          a_cmnt = a_cmnts(i_indx)
      endif

      if (i_flg .eq. 0) then                                               ! Data not found
         a_errtmp = 'Keyword not found. '//a_keyw(1:max(min(rdflen(a_keyw),150),2))
         call rdf_error(a_errtmp)
      else if (i_flg .ge. 2) then
         a_errtmp = 'Multiple matching keywords found, returning index of last. '//
     &                 a_keyw(1:max(min(rdflen(a_keyw),150),2))//'  '//rdfint1(i_flg)
         call rdf_error(a_errtmp)
      else
         call rdf_cnvrt(a_ounit,a_unit,a_valu)
      endif

      rdfdata = a_valu

      call rdf_trace(' ')
      return

      end

c****************************************************************

      subroutine rdf_cnvrt(a_ounit,a_unit,a_valu)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      character*(*) a_ounit
      character*(*) a_unit
      character*(*) a_valu
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i
      integer ii
      integer i_stat
      integer i_type
      integer i_uinp
      integer i_uout
      integer i_lun
      integer i_iostat

      integer i_val
      real*8  r_val

      character*320 a_uinp(100)
      character*320 a_uout(100)
      character*320 a_vals(100)
      character*320 a_fmt
      character*320 a_errtmp

      real*8 r_addit1
      real*8 r_addit2
      real*8 r_scale1
      real*8 r_scale2

      real*8       r_cnv(20,20,2)
      integer      i_cnv(20)
      character*20 a_cnv(20,20)

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer*4 rdflen
      external rdflen

      character*320 rdflower
      external rdflower

      character*320 rdftrim
      external rdftrim

c     DATA STATEMENTS:

      data i_cnv(1) /9/								! length
      data a_cnv(1,1) /'nm'/, r_cnv(1,1,1) /1.e-9/,       r_cnv(1,1,2) /0./
      data a_cnv(1,2) /'um'/, r_cnv(1,2,1) /1.e-6/,       r_cnv(1,2,2) /0./
      data a_cnv(1,3) /'mm'/, r_cnv(1,3,1) /1.e-3/,       r_cnv(1,3,2) /0./
      data a_cnv(1,4) /'cm'/, r_cnv(1,4,1) /1.e-2/,       r_cnv(1,4,2) /0./
      data a_cnv(1,5) /'m' /, r_cnv(1,5,1) /1.0  /,       r_cnv(1,5,2) /0./
      data a_cnv(1,6) /'km'/, r_cnv(1,6,1) /1.e+3/,       r_cnv(1,6,2) /0./
      data a_cnv(1,7) /'in'/, r_cnv(1,7,1) /2.54e-2/,     r_cnv(1,7,2) /0./
      data a_cnv(1,8) /'ft'/, r_cnv(1,8,1) /3.048e-1/,    r_cnv(1,8,2) /0./
      data a_cnv(1,9) /'mi'/, r_cnv(1,9,1) /1.609344e3/,  r_cnv(1,9,2) /0./

      data i_cnv(2) /7/								! area
      data a_cnv(2,1) /'mm*mm'/, r_cnv(2,1,1) /1.e-6/,        r_cnv(2,1,2) /0./
      data a_cnv(2,2) /'cm*cm'/, r_cnv(2,2,1) /1.e-4/,        r_cnv(2,2,2) /0./
      data a_cnv(2,3) /'m*m' /,  r_cnv(2,3,1) /1.0  /,        r_cnv(2,3,2) /0./
      data a_cnv(2,4) /'km*km'/, r_cnv(2,4,1) /1.e+6/,        r_cnv(2,4,2) /0./
      data a_cnv(2,5) /'in*in'/, r_cnv(2,5,1) /6.4516e-4/,    r_cnv(2,5,2) /0./
      data a_cnv(2,6) /'ft*ft'/, r_cnv(2,6,1) /9.290304e-2/,  r_cnv(2,6,2) /0./
      data a_cnv(2,7) /'mi*mi'/, r_cnv(2,7,1) /2.58995511e6/, r_cnv(2,7,2) /0./

      data i_cnv(3) /7/								! time
      data a_cnv(3,1) /'ns'/, r_cnv(3,1,1) /1.e-9/,  r_cnv(3,1,2) /0./
      data a_cnv(3,2) /'us'/, r_cnv(3,2,1) /1.e-6/,  r_cnv(3,2,2) /0./
      data a_cnv(3,3) /'ms'/, r_cnv(3,3,1) /1.e-3/,  r_cnv(3,3,2) /0./
      data a_cnv(3,4) /'s' /, r_cnv(3,4,1) /1.0/,    r_cnv(3,4,2) /0./
      data a_cnv(3,5) /'min'/,r_cnv(3,5,1) /6.0e1/,  r_cnv(3,5,2) /0./
      data a_cnv(3,6) /'hr' /,r_cnv(3,6,1) /3.6e3/,  r_cnv(3,6,2) /0./
      data a_cnv(3,7) /'day'/,r_cnv(3,7,1) /8.64e4/, r_cnv(3,7,2) /0./

      data i_cnv(4) /6/								! velocity
      data a_cnv(4,1) /'cm/s'/,  r_cnv(4,1,1) /1.e-2/,         r_cnv(4,1,2) /0./
      data a_cnv(4,2) /'m/s'/,   r_cnv(4,2,1) /1.0/,           r_cnv(4,2,2) /0./
      data a_cnv(4,3) /'km/s'/,  r_cnv(4,3,1) /1.e3/,          r_cnv(4,3,2) /0./
      data a_cnv(4,4) /'km/hr'/, r_cnv(4,4,1) /2.77777778e-1/, r_cnv(4,4,2) /0./
      data a_cnv(4,5) /'ft/s'/,  r_cnv(4,5,1) /3.04878e-1/,    r_cnv(4,5,2) /0./
      data a_cnv(4,6) /'mi/hr'/, r_cnv(4,6,1) /4.4704e-1/,     r_cnv(4,6,2) /0./

      data i_cnv(5) /5/								! power
      data a_cnv(5,1) /'mw'/, r_cnv(5,1,1) /1.e-3/,  r_cnv(5,1,2) /0./
      data a_cnv(5,2) /'w'/,  r_cnv(5,2,1) /1.0/,    r_cnv(5,2,2) /0./
      data a_cnv(5,3) /'kw'/, r_cnv(5,3,1) /1.e3/,   r_cnv(5,3,2) /0./
      data a_cnv(5,4) /'dbm'/,r_cnv(5,4,1) /1.e-3/,  r_cnv(5,4,2) /0./
      data a_cnv(5,5) /'dbw'/,r_cnv(5,5,1) /1.0/,    r_cnv(5,5,2) /0./

      data i_cnv(6) /4/								! frequency
      data a_cnv(6,1) /'hz'/, r_cnv(6,1,1) /1.0/,    r_cnv(6,1,2) /0./
      data a_cnv(6,2) /'khz'/,r_cnv(6,2,1) /1.0e3/,  r_cnv(6,2,2) /0./
      data a_cnv(6,3) /'mhz'/,r_cnv(6,3,1) /1.0e6/,  r_cnv(6,3,2) /0./
      data a_cnv(6,4) /'ghz'/,r_cnv(6,4,1) /1.0e9/,  r_cnv(6,4,2) /0./

      data i_cnv(7) /3/								! angle
      data a_cnv(7,1) /'deg'/,r_cnv(7,1,1) /1.0/,         r_cnv(7,1,2) /0./
      data a_cnv(7,2) /'rad'/,r_cnv(7,2,1) /57.29577951/, r_cnv(7,2,2) /0./
      data a_cnv(7,3) /'arc'/,r_cnv(7,3,1) /0.000277778/, r_cnv(7,3,2) /0./

      data i_cnv(8) /7/								! data
      data a_cnv(8,1) /'bits'/,  r_cnv(8,1,1) /1./,        r_cnv(8,1,2) /0./
      data a_cnv(8,2) /'kbits'/, r_cnv(8,2,1) /1.e3/,      r_cnv(8,2,2) /0./
      data a_cnv(8,3) /'mbits'/, r_cnv(8,3,1) /1.e6/,      r_cnv(8,3,2) /0./
      data a_cnv(8,4) /'bytes'/, r_cnv(8,4,1) /8./,        r_cnv(8,4,2) /0./
      data a_cnv(8,5) /'kbytes'/,r_cnv(8,5,1) /8320./,     r_cnv(8,5,2) /0./
      data a_cnv(8,6) /'mbytes'/,r_cnv(8,6,1) /8388608./,  r_cnv(8,6,2) /0./
      data a_cnv(8,7) /'words'/, r_cnv(8,7,1) /32./,       r_cnv(8,7,2) /0./

      data i_cnv(9) /7/								! data rate
      data a_cnv(9,1) /'bits/s'/,  r_cnv(9,1,1) /1./,        r_cnv(9,1,2) /0./
      data a_cnv(9,2) /'kbits/s'/, r_cnv(9,2,1) /1.e3/,      r_cnv(9,2,2) /0./
      data a_cnv(9,3) /'mbits/s'/, r_cnv(9,3,1) /1.e6/,      r_cnv(9,3,2) /0./
      data a_cnv(9,4) /'bytes/s'/, r_cnv(9,4,1) /8./,        r_cnv(9,4,2) /0./
      data a_cnv(9,5) /'kbytes/s'/,r_cnv(9,5,1) /8320./,     r_cnv(9,5,2) /0./
      data a_cnv(9,6) /'mbytes/s'/,r_cnv(9,6,1) /8388608./,  r_cnv(9,6,2) /0./
      data a_cnv(9,7) /'baud'/,    r_cnv(9,7,1) /1./,        r_cnv(9,7,2) /0./

      data i_cnv(10) /3/								! temperature
      data a_cnv(10,1) /'deg c'/,r_cnv(10,1,1) /1.0/,      r_cnv(10,1,2) /0.0/
      data a_cnv(10,2) /'deg k'/,r_cnv(10,2,1) /1.0/,      r_cnv(10,2,2) /273.0/
      data a_cnv(10,3) /'deg f'/,r_cnv(10,3,1) /0.555556/, r_cnv(10,3,2) /-32/

      data i_cnv(11) /2/								! ratio
      data a_cnv(11,1) /'-'/, r_cnv(11,1,1) /1.0/,  r_cnv(11,1,2) /0.0/
      data a_cnv(11,2) /'db'/,r_cnv(11,2,1) /1.0/,  r_cnv(11,2,2) /0.0/

      data i_cnv(12) /2/								! fringe rate
      data a_cnv(12,1) /'deg/m'/,r_cnv(12,1,1) /1.0/        ,  r_cnv(12,1,2) /0.0/
      data a_cnv(12,2) /'rad/m'/,r_cnv(12,2,1) /57.29577951/,  r_cnv(12,2,2) /0.0/

      save i_cnv,r_cnv,a_cnv

c     PROCESSING STEPS:

      if (a_valu  .eq. ' ') return
	  
      if (a_unit  .eq. ' ') return
      if (a_ounit .eq. ' ') return

      if (a_unit  .eq. '&') return
      if (a_ounit .eq. '&') return

      if (a_unit  .eq. '?') return
      if (a_ounit .eq. '?') return

      call rdf_trace('RDF_CNVRT')
      i_uinp = 1
      a_uinp(1) = ' '
      do i=1,rdflen(a_unit)
         if (a_unit(i:i) .eq. ',') then
            i_uinp = i_uinp + 1
            a_uinp(i_uinp) = ' '
         else
            a_uinp(i_uinp)(rdflen(a_uinp(i_uinp))+1:) = rdflower(a_unit(i:i))
         endif
      enddo
      i_uout = 1
      a_uout(1) = ' '
      do i=1,rdflen(a_ounit)
         if (a_ounit(i:i) .eq. ',') then
            i_uout = i_uout + 1
            a_uout(i_uout) = ' '
         else
            a_uout(i_uout)(rdflen(a_uout(i_uout))+1:) = rdflower(a_ounit(i:i))
         endif
      enddo
      if (i_uinp .ne. i_uout .and. i_uinp .gt. 1 .and. i_uout .gt. 1) then
         a_errtmp = 'Number of units input not equal to number of units output. '//
     &                 a_unit(1:max(min(rdflen(a_unit),150),2))//'  '//a_ounit(1:max(min(rdflen(a_ounit),150),2))
         call rdf_error(a_errtmp)
         call rdf_trace(' ')
         return
      endif

      call rdf_getfields(a_valu,i_val,a_vals)

      if (i_uinp .eq. 1 .and. i_val .gt. 1) then
         do ii = 2,i_val
            a_uinp(ii) = a_uinp(1)
         enddo
         i_uinp = i_val
      endif
      if (i_uout .eq. 1 .and. i_val .gt. 1) then
         do ii = 2,i_val
            a_uout(ii) = a_uout(1)
         enddo
         i_uout = i_val
      endif
      do ii = i_uinp+1,i_val
         a_uinp(ii) = ' '
      enddo
      do ii = i_uout+1,i_val
         a_uout(ii) = ' '
      enddo

      do ii = 1,i_val


         if ((a_uinp(ii) .ne. ' ' .and. a_uinp(ii) .ne. '&') .and.
     &       (a_uout(ii) .ne. ' ' .and. a_uout(ii) .ne. '&')) then
		 
            i_stat=0
            if (a_uinp(ii) .ne. a_uout(ii) ) then
               do i_type = 1,12
                  if (i_stat .eq. 0) then
                     r_scale1 = 0.
                     r_scale2 = 0.
                     do i=1,i_cnv(i_type)
                        if (a_uinp(ii) .eq. a_cnv(i_type,i)) then
                           r_scale1 = r_cnv(i_type,i,1)
                           r_addit1 = r_cnv(i_type,i,2)
                        endif
                        if (a_uout(ii) .eq. a_cnv(i_type,i)) then
                           r_scale2 = r_cnv(i_type,i,1)
                           r_addit2 = r_cnv(i_type,i,2)
                        endif
                     enddo
                     if (r_scale1 .ne. 0. .and. r_scale2 .ne. 0.) then
                        read(a_vals(ii),*,iostat=i_iostat) r_val
                        if (i_iostat .eq. 0) then
                           if (index(a_uinp(ii),'db') .gt. 0) r_val = 10.0**(r_val/10.)
                           r_val = (r_val+r_addit1)*r_scale1/r_scale2 - r_addit2
                           if (index(a_uout(ii),'db') .gt. 0) r_val = 10.0*dlog10(r_val) 
                           if (a_dblefmt .eq. '*') then
                             write(a_vals(ii),fmt=*,iostat=i_iostat) r_val
                           else
                             a_fmt='('//a_dblefmt(1:max(1,rdflen(a_dblefmt)))//')'
                             write(a_vals(ii),fmt=a_fmt,iostat=i_iostat) r_val
                           endif
                           if (i_iostat .ne. 0 ) write(6,*) 'Internal write error ',i_iostat,r_val,a_vals(ii)
                           a_vals(ii) = rdftrim(a_vals(ii))
                           i_stat = 1
                        else
                           i_stat = 2
                        endif
                     endif
                  endif
               enddo
               if (i_stat .ne. 1) then
                  a_errtmp = 'Unit conversion error '//
     &                a_uinp(ii)(1:max(1,rdflen(a_uinp(ii))))//' > '//a_uout(ii)(1:max(1,rdflen(a_uout(ii))))//
     &                '  val:'//a_vals(ii)
                  call rdf_error(a_errtmp)
               endif
            endif
         endif
      enddo

      a_valu=' '
      do ii=1,i_val
         if (rdflen(a_valu) .eq. 0) then
            a_valu=a_vals(ii)
         else
            a_valu=a_valu(:rdflen(a_valu))//' '//a_vals(ii)
         endif
      enddo
c      write(6,*) a_valu(1:max(1,rdflen(a_valu)))

      call rdf_trace(' ')
      return
      end


c****************************************************************

      integer*4 function rdferr(a_err)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:
      
c     OUTPUT VARIABLES:

      character*(*) a_err

c     LOCAL VARIABLES:

      integer i
      integer i_err

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

c     PROCESSING STEPS:

      call rdf_trace('RDFERR')
      i_err = max(i_error,0)
      if (i_error .gt. 0) then
        a_err = a_error(1)      
        do i = 1,i_error-1
          a_error(i) = a_error(i+1)
        enddo
        i_error = i_error - 1
      else
        a_err = ' '
        i_error = 0
      endif

      rdferr = i_err
      call rdf_trace(' ')
      return
      end

c****************************************************************

      character*(*) function rdftrim(a_input)     

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

      character*(*) a_input
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      character*320 a_value

      integer i
      integer i_len

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer rdflen
      external rdflen

c     PROCESSING STEPS:

      i_len=len(a_input)
      i_len = rdflen(a_input)
      call rdf_trace('RDFTRIM')
      a_value = a_input
      if (i_len .gt. 0) then
         if (i_len .gt. 320) then
            write(6,*) 'String rdflen exceeds 320 in rdftrim ',i_len
            write(6,*) a_input
         endif
         i = 1
         do while ((i .lt. i_len) .and. 
     &        (a_value(i:i) .eq. char(32) .or. a_value(i:i) .eq. char(9))) 
            i = i + 1
         enddo
         a_value = a_value(i:)
         i_len = i_len - i + 1
         do while ((i_len .gt. 1) .and. 
     &        (a_value(i_len:i_len) .eq. char(32) .or. a_value(i_len:i_len) .eq. char(9))) 
            i_len = i_len - 1
         enddo
         a_value = a_value(1:i_len)
         if (a_value(1:1) .eq. char(9)) a_value = a_value(2:)
      endif
      rdftrim = a_value
      call rdf_trace(' ')
      return
      end 


c****************************************************************

      character*(*) function rdfcullsp(a_temp)
 
c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

      integer i_pos
      integer i_len
      character*(*) a_temp
      character*320 a_temp2
      character*320 a_string
      integer*4 rdflen
      external rdflen

      call rdf_trace('RDFCULLSP')
      a_string=a_temp						! replace tabs with spaces
c      type *,'a_string=',a_string(1:max(1,rdflen(a_string)))
      i_pos = index(a_string,char(9))
      do while (i_pos .ne. 0)
        a_string(i_pos:i_pos) = ' '
c        type *,'a_string=',a_string(1:max(1,rdflen(a_string))),i_pos
        i_pos = index(a_string,char(9))
      end do

c      type *,' '
      i_len = rdflen(a_string)
      i_pos = index(a_string,'  ')			!  convert multiple spaces to single spaces
      do while (i_pos .ne. 0 .and. i_pos .lt. rdflen(a_string))
        a_string=a_string(:i_pos)//a_string(i_pos+2:)
c        type *,'a_string=',a_string(1:max(1,rdflen(a_string))),i_pos
        i_len = i_len-1
        i_pos = index(a_string,'  ')
      end do

      a_temp2 = a_string  ! (1:max(1,rdflen(a_string)))
      rdfcullsp = a_temp2
      call rdf_trace(' ')
      return
      end



c****************************************************************

      character*(*) function rdflower(a_inpval)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

      character*(*) a_inpval
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      character*320 a_outval

      integer i
      integer i_len

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer rdflen
      external rdflen

c     PROCESSING STEPS:

      call rdf_trace('RDFLOWER')
      i_len = rdflen(a_inpval)
      a_outval = ' '
      do i=1,i_len
         if (ichar(a_inpval(i:i)) .ge. 65 .and. ichar(a_inpval(i:i)) .le. 90 ) then
            a_outval(i:i) = char(ichar(a_inpval(i:i))+32)
         else
            a_outval(i:i) = a_inpval(i:i)
         endif
      enddo
      rdflower=a_outval
      call rdf_trace(' ')
      return
      end
      

c****************************************************************

      character*(*) function rdfupper(a_inpval)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

      character*(*) a_inpval
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      character*320 a_outval

      integer i
      integer i_len

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer rdflen
      external rdflen

c     PROCESSING STEPS:

      call rdf_trace('RDFUPPER')
      i_len = rdflen(a_inpval)
      a_outval = ' '
      do i=1,i_len
         if (ichar(a_inpval(i:i)) .ge. 97 .and. ichar(a_inpval(i:i)) .le. 122 ) then
            a_outval(i:i) = char(ichar(a_inpval(i:i))-32)
         else
            a_outval(i:i) = a_inpval(i:i)
         endif
      enddo
      rdfupper=a_outval
      call rdf_trace(' ')
      return
      end
      
c****************************************************************

      character*(*) function rdfint(i_num,i_data)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      integer i_num
      integer i_data(*)
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i

      character*320 a_fmt
      character*320 a_outval

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer rdflen
      external rdflen

c     PROCESSING STEPS:

      call rdf_trace('RDFINT')
      if (a_intfmt .eq. '*') then
        write(unit=a_outval,fmt=*) (i_data(i),i=1,i_num)
      else
cbjs  The below line would produce a format string a_fmt="( 2i)"
cbjs  which is a syntactic error since the 'i' does not have
cbjs  a width specified.  ifort, f95, and pgf95 did not reject it.
cbjs  However, it was rejected by g95 and gfortran.
cbjs  f95 treated the 'i' as 'i0'.  The others treated it as 'i12'.
cbjs  Modification will force a '0' for the field width
cbjs  causing a_fmt="( 2i0)"   (when i_num=2)
c       write(a_fmt,'(a,i2,a,a)') '(',i_num,a_intfmt(1:max(rdflen(a_intfmt),1)),')'
        write(a_fmt,'(a,i2,a,"0",a)') '(',i_num,a_intfmt(1:max(rdflen(a_intfmt),1)),')'
        write(unit=a_outval,fmt=a_fmt) (i_data(i),i=1,i_num)
      endif
      rdfint=a_outval
      call rdf_trace(' ')
      return
      end
      

c****************************************************************

      character*(*) function rdfint1(i_data)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

      integer i_data
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      character*320 a_outval

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer rdflen
      external rdflen

c     PROCESSING STEPS:

      call rdf_trace('RDFINT1')
      write(a_outval,*) i_data
      rdfint1=a_outval
      call rdf_trace(' ')
      return
      end
      

c****************************************************************

      character*(*) function rdfint2(i_data1,i_data2)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

      integer i_data1
      integer i_data2
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      character*320 a_outval

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer rdflen
      external rdflen

c     PROCESSING STEPS:

      call rdf_trace('RDFINT2')
      write(a_outval,*) i_data1,i_data2
      rdfint2=a_outval
      call rdf_trace(' ')
      return
      end
      
c****************************************************************

      character*(*) function rdfint3(i_data1,i_data2,i_data3)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

      integer i_data1
      integer i_data2
      integer i_data3
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      character*320 a_outval

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer rdflen
      external rdflen

c     PROCESSING STEPS:

      call rdf_trace('RDFINT3')
      write(a_outval,*) i_data1,i_data2,i_data3
      rdfint3=a_outval
      call rdf_trace(' ')
      return
      end
      

c****************************************************************

      character*(*) function rdfreal(i_num,r_data)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      integer*4 i_num
      real*4 r_data(*)
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i

      character*320 a_fmt
      character*320 a_outval

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer rdflen
      external rdflen

c     PROCESSING STEPS:
      
      call rdf_trace('RDFREAL')
      if (a_realfmt .eq. '*') then
        write(unit=a_outval,fmt=*) (r_data(i),i=1,i_num)
      else
        write(a_fmt,'(a,i2,a,a)') '(',i_num,a_realfmt(1:max(rdflen(a_realfmt),1)),')'
        write(unit=a_outval,fmt=a_fmt) (r_data(i),i=1,i_num)
      endif
      rdfreal=a_outval
      call rdf_trace(' ')
      return

      end

c****************************************************************

      character*(*) function rdfreal1(r_data)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

      real*4 r_data
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      character*320 a_outval

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer rdflen
      external rdflen

c     PROCESSING STEPS:

      call rdf_trace('RDFREAL1')
      write(a_outval,*) r_data
      rdfreal1=a_outval
      call rdf_trace(' ')
      return
      end
      
c****************************************************************

      character*(*) function rdfreal2(r_data1,r_data2)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

      real*4 r_data1,r_data2
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      character*320 a_outval

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer rdflen
      external rdflen

c     PROCESSING STEPS:

      call rdf_trace('RDFREAL2')
      write(a_outval,*) r_data1,r_data2
      rdfreal2=a_outval
      call rdf_trace(' ')
      return
      end

c****************************************************************

      character*(*) function rdfreal3(r_data1,r_data2,r_data3)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

      real*4 r_data1,r_data2,r_data3
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      character*320 a_outval

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer rdflen
      external rdflen

c     PROCESSING STEPS:
      
      call rdf_trace('RDFREAL3')
      write(a_outval,*) r_data1,r_data2,r_data3
      rdfreal3=a_outval
      call rdf_trace(' ')
      return

      end

c****************************************************************

      character*(*) function rdfdble(i_num,r_data)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      integer*4 i_num
      real*8 r_data(*)
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i 

      character*320 a_fmt
      character*320 a_outval

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer rdflen
      external rdflen

c     PROCESSING STEPS:
      
      call rdf_trace('RDFDBLE')
      if (a_dblefmt .eq. '*') then
        write(unit=a_outval,fmt=*) (r_data(i),i=1,i_num)
      else
        write(a_fmt,'(a,i2,a,a)') '(',i_num,'('//a_dblefmt(1:max(rdflen(a_dblefmt),1)),',1x))'
        write(unit=a_outval,fmt=a_fmt) (r_data(i),i=1,i_num)
      endif
      rdfdble=a_outval
      call rdf_trace(' ')
      return

      end

c****************************************************************

      character*(*) function rdfdble1(r_data)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

      real*8 r_data
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      character*320 a_outval

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer rdflen
      external rdflen

c     PROCESSING STEPS:

      call rdf_trace('RDFDBLE1')
      write(a_outval,*) r_data
      rdfdble1=a_outval
      call rdf_trace(' ')
      return
      end
      
c****************************************************************

      character*(*) function rdfdble2(r_data1,r_data2)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

      real*8 r_data1,r_data2
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      character*320 a_outval

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer rdflen
      external rdflen

c     PROCESSING STEPS:

      call rdf_trace('RDFDBLE2')
      write(a_outval,*) r_data1,r_data2
      rdfdble2=a_outval
      call rdf_trace(' ')
      return
      end

c****************************************************************

      character*(*) function rdfdble3(r_data1,r_data2,r_data3)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

      real*8 r_data1,r_data2,r_data3
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      character*320 a_outval

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer rdflen
      external rdflen

c     PROCESSING STEPS:
      
      call rdf_trace('RDFDBLE3')
      write(a_outval,*) r_data1,r_data2,r_data3
      rdfdble3=a_outval
      call rdf_trace(' ')
      return

      end

c****************************************************************

      integer*4 function rdflen(a_string)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: This function returns the position 
c**   of the last none blank character in the string. 
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
 
      character*(*) a_string
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i_len

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

c     PROCESSING STEPS:

      call rdf_trace('RDFLEN')
      i_len=len(a_string)
      do while(i_len .gt. 0 .and. (a_string(i_len:i_len) .eq. ' ' .or. 
     &     ichar(a_string(i_len:i_len)) .eq. 0))
         i_len=i_len-1
c         write(6,*) i_len,' ',ichar(a_string(i_len:i_len)),' ',a_string(i_len:i_len)
      enddo

      rdflen=i_len
      call rdf_trace(' ')
      return

      end

c****************************************************************

      character*(*) function rdfquote(a_string)

c****************************************************************
c**     
c**   FILE NAME: rdf_reader.f
c**     
c**   DATE WRITTEN: 15-Sept-1997
c**     
c**   PROGRAMMER: Scott Shaffer
c**     
c**   FUNCTIONAL DESCRIPTION: 
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

      character*(*) a_string
	
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i_string

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

C     FUNCTION_STATEMENTS:

      integer*4 rdflen
      external rdflen

c     PROCESSING STEPS:

      call rdf_trace('RDFQUOTE')
      i_string = rdflen(a_string)
      rdfquote = '"'//a_string(1:i_string)//'"'
      call rdf_trace(' ')
      return

      end


c****************************************************************

      character*(*) function rdfunquote(a_string)

c****************************************************************
c**     
c**   FILE NAME: rdf_reader.f
c**     
c**   DATE WRITTEN: 15-Sept-1997
c**     
c**   PROGRAMMER: Scott Shaffer
c**     
c**   FUNCTIONAL DESCRIPTION: 
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

      character*(*) a_string
	
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i_string

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

C     FUNCTION_STATEMENTS:

c     PROCESSING STEPS:

      call rdf_trace('UNRDFQUOTE')
      call rdf_unquote(a_string,i_string)
      rdfunquote = a_string
      call rdf_trace(' ')
      return

      end


c****************************************************************

      subroutine rdf_unquote(a_string,i_string)

c****************************************************************
c**     
c**   FILE NAME: rdf_reader.f
c**     
c**   DATE WRITTEN: 15-Sept-1997
c**     
c**   PROGRAMMER: Scott Shaffer
c**     
c**   FUNCTIONAL DESCRIPTION: 
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

      character*(*) a_string
	
c     OUTPUT VARIABLES:

      integer i_string

c     LOCAL VARIABLES:

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

C     FUNCTION_STATEMENTS:

      integer*4 rdflen
      external rdflen

c     PROCESSING STEPS:

      call rdf_trace('RDF_UNQUOTE')
      i_string = rdflen(a_string)
      if (i_string .gt. 1) then
         if (a_string(1:1) .eq. '"' .and. a_string(i_string:i_string) .eq. '"' ) then
            if (i_string .eq. 2) then
               a_string = ' '
            else
               a_string = a_string(2:i_string-1)
            endif
            i_string = i_string-2
         endif
      endif
      call rdf_trace(' ')
      return

      end


c****************************************************************

      integer*4 function rdfmap(i,j,k)

c****************************************************************
c**     
c**   FILE NAME: rdf_reader.f
c**     
c**   DATE WRITTEN: 15-Sept-1997
c**     
c**   PROGRAMMER: Scott Shaffer
c**     
c**   FUNCTIONAL DESCRIPTION: 
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

      integer i
      integer j
      integer k
	
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

C     FUNCTION_STATEMENTS:

c     PROCESSING STEPS:

      call rdf_trace('RDF_MAP')
      if (k .eq. 0) then
        rdfmap = 0
      else if (k .eq. 1) then
        rdfmap = i
      else if (k .eq. 2) then
        rdfmap = j
      else
        rdfmap = 0
      endif
      call rdf_trace(' ')
      return

      end


c****************************************************************

      subroutine rdf_indices(a_dimn,i_dimn,i_strt,i_stop,i_order)

c****************************************************************
c**     
c**   FILE NAME: rdf_reader.f
c**     
c**   DATE WRITTEN: 15-Sept-1997
c**     
c**   PROGRAMMER: Scott Shaffer
c**     
c**   FUNCTIONAL DESCRIPTION: 
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

      character*(*) a_dimn
	
c     OUTPUT VARIABLES:

      integer i_dimn
      integer i_order(20)
      integer i_strt(20)
      integer i_stop(20)

c     LOCAL VARIABLES:

      integer i
      integer i_pos
      integer i_stat
      integer i_fields

      character*320 a_fields(100)
     
c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

C     FUNCTION_STATEMENTS:

      integer*4 rdflen
      external rdflen

c     PROCESSING STEPS:

      call rdf_trace('RDF_INDICES')
      call rdf_getfields(a_dimn,i_fields,a_fields)

      do i=1,i_fields
         i_pos = index(a_fields(i),'-')
         if (i_pos .gt. 0) then
            if (i_pos .gt. 1) then
               read(a_fields(i)(1:i_pos-1),'(i10)',iostat=i_stat) i_order(i)
               if (i_stat .ne. 0) then
                  write(6,   *) '*** RDF ERROR ***  Cannot parse indices order field ',a_fields(i)(1:i_pos-1)
                  i_order(i) = 1
               endif
            else 
               i_order(i) = i
            endif
            a_fields(i) = a_fields(i)(i_pos+1:)
         else
            i_order(i) = i
         endif
         i_pos = index(a_fields(i),':')
         if (i_pos .gt. 0) then
            if (i_pos .gt. 1) then
               read(a_fields(i)(1:i_pos-1),'(i10)',iostat=i_stat) i_strt(i)
               if (i_stat .ne. 0) then
                  write(6,   *) '*** RDF ERROR ***  Cannot parse indices start field ',a_fields(i)(1:i_pos-1)
                  i_strt(i) = 1
               endif
            else 
               i_strt(i) = 1
            endif
            a_fields(i) = a_fields(i)(i_pos+1:)
         else
            i_strt(i) = 1
         endif
         i_pos=max(1,rdflen(a_fields(i)))                 ! inserted for Vax compatibility
         read(unit=a_fields(i)(1:i_pos),fmt='(i10)',iostat=i_stat) i_stop(i)
         if (i_stat .ne. 0) then
            write(6,   *) '*** RDF ERROR ***  Cannot parse indices stop field: ',rdflen(a_fields(i)),':',
     &            a_fields(i)(1:max(1,rdflen(a_fields(i))))
            i_stop(i) = i_strt(i)
         endif
      enddo
      i_dimn = i_fields
      call rdf_trace(' ')
      return

      end


c****************************************************************

      subroutine rdf_getfields(a_string,i_values,a_values)

c****************************************************************
c**     
c**   FILE NAME: rdf_reader.f
c**     
c**   DATE WRITTEN: 15-Sept-1997
c**     
c**   PROGRAMMER: Scott Shaffer
c**     
c**   FUNCTIONAL DESCRIPTION: 
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

      character*(*) a_string
	
c     OUTPUT VARIABLES:

      character*(*) a_values(*)
      integer i_values

c     LOCAL VARIABLES:

      integer i
      integer i_on
      integer i_cnt
      integer i_quote

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

C     FUNCTION_STATEMENTS:

c     PROCESSING STEPS:

      call rdf_trace('RDF_GETFIELDS')
      i_on = 0
      i_cnt = 0
      i_values = 0
      i_quote = 0
      do i=1,len(a_string)
         if (i_quote .eq. 1 .or. (
     &        a_string(i:i) .ne. ' ' .and. 
     &        a_string(i:i) .ne. ',' .and.
     &        a_string(i:i) .ne. char(9)) ) then
            if (i_on .eq. 0) then
               i_on = 1
               i_cnt = 0
               i_values=min(i_values+1,100)
               a_values(i_values)=' '
            endif
            if (a_string(i:i) .eq. '"') then
               i_quote=1-i_quote
            endif
            i_cnt = i_cnt+1
            a_values(i_values)(i_cnt:i_cnt) = a_string(i:i)
         else 
            if (i_quote .eq. 0) then
               i_on = 0
               i_cnt = 0
            endif
         endif
      enddo
      call rdf_trace(' ')
      return

      end



c****************************************************************

      subroutine rdf_parse(a_data,a_keyw,a_unit,a_dimn,a_elem,a_oper,a_valu,a_cmnt)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      integer i
      character*(*) a_data
      
c     OUTPUT VARIABLES:

      character*(*) a_keyw
      character*(*) a_valu
      character*(*) a_unit
      character*(*) a_dimn
      character*(*) a_elem
      character*(*) a_oper
      character*(*) a_cmnt
      character*320 a_errtmp

c     LOCAL VARIABLES:

      integer i_type
      integer i_keyw
      integer i_valu
      integer i_unit
      integer i_dimn
      integer i_elem
      integer i_oper
      integer i_cmnt

      integer i_lun
      integer i_iostat

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer*4 rdflen
      external rdflen

      character*320 rdftrim
      external rdftrim

c     PROCESSING STEPS:

      call rdf_trace('RDF_PARSE')
         a_keyw = ' '
         a_valu = ' '
         a_oper = ' '
         a_unit = ' '
         a_dimn = ' '
         a_elem = ' '
         a_cmnt = ' '
         i_keyw = 0
         i_valu = 0
         i_oper = 0
         i_unit = 0
         i_elem = 0
         i_dimn = 0
         i_cmnt = 0

         i_type = 1

         do i=1,rdflen(a_data)
            if (i_type .eq. 0) then
               i_cmnt = i_cmnt + 1
               if (i_cmnt .le. I_MCPF) a_cmnt(i_cmnt:i_cmnt) = a_data(i:i)
            else if (a_data(i:i) .eq. a_cmdl(0) .and. a_cmdl(0) .ne. ' ') then
               i_type = 0
            else if (a_data(i:i) .eq. a_cmdl(1) .and. a_cmdl(1) .ne. ' ') then
               i_type = 0
            else if (a_data(i:i) .eq. a_cmdl(2) .and. a_cmdl(2) .ne. ' ') then
               i_type = 0
            else if (i_type .eq. 10) then
               i_valu = i_valu + 1
               if (i_valu .le. I_MCPF) then
                  a_valu(i_valu:i_valu) = a_data(i:i)
               else if (i_valu .eq. I_MCPF+1) then
                  a_errtmp = '*** WARNING ***   RDF_PARSE - Value field exceeds max characters per line. '//
     &                 a_cmnt
                  call rdf_error(a_errtmp)
               endif
            else if (a_data(i:i) .eq. '(' ) then
               i_type = 2
            else if (a_data(i:i) .eq. ')' ) then
               i_type = 1
            else if (a_data(i:i) .eq. '[' ) then
               i_type = 3
            else if (a_data(i:i) .eq. ']' ) then
               i_type = 1
            else if (a_data(i:i) .eq. '{' ) then
               i_type = 4
            else if (a_data(i:i) .eq. '}' ) then
               i_type = 1
            else if (a_data(i:i) .eq. '=' ) then
               i_type = 10
               a_oper = '='
            else if (a_data(i:i) .eq. '<' ) then
               i_type = 10
               a_oper = '<'
            else if (a_data(i:i) .eq. '>' ) then
               i_type = 10
               a_oper = '>'
            else if (i_type .eq. 1) then
               i_keyw = i_keyw + 1
               if (i_keyw .le. I_MCPF) a_keyw(i_keyw:i_keyw) = (a_data(i:i))
            else if (i_type .eq. 2) then
               i_unit = i_unit + 1
               if (i_unit .le. I_MCPF) a_unit(i_unit:i_unit) = (a_data(i:i))
            else if (i_type .eq. 3) then
               i_dimn = i_dimn + 1
               if (i_dimn .le. I_MCPF) a_dimn(i_dimn:i_dimn) = (a_data(i:i))
            else if (i_type .eq. 4) then
               i_elem = i_elem + 1
               if (i_elem .le. I_MCPF) a_elem(i_elem:i_elem) = (a_data(i:i))
            endif
         enddo

         if (i_cmnt .eq. I_MCPF+1) then
            a_errtmp = '*** WARNING ***   Comment field exceeds max characters per line. '//
     &           a_cmnt
            call rdf_error(a_errtmp)
         endif
         if (i_keyw .eq. I_MCPF+1) then
            a_errtmp = 'Keyword field exceeds max characters per line. '//
     &           a_cmnt
            call rdf_error(a_errtmp)
         endif
         if (i_unit .eq. I_MCPF+1) then
            a_errtmp = 'Unit field exceeds max characters per line. '//
     &           a_unit
            call rdf_error(a_errtmp)
         endif
         if (i_dimn .eq. I_MCPF+1) then
            a_errtmp = 'Dimension field exceeds max characters per line. '//
     &           a_dimn
            call rdf_error(a_errtmp)
         endif
         if (i_elem .eq. I_MCPF+1) then
            a_errtmp = 'Element field exceeds max characters per line. '//
     &           a_elem
            call rdf_error(a_errtmp)
         endif
         a_keyw = rdftrim(a_keyw)
         a_valu = rdftrim(a_valu)
         a_unit = rdftrim(a_unit)
         a_dimn = rdftrim(a_dimn)
         a_elem = rdftrim(a_elem)
         a_oper = rdftrim(a_oper)

      call rdf_trace(' ')
      return
      end  

c****************************************************************

      subroutine rdf_unparse(a_data,a_keyw,a_unit,a_dimn,a_elem,a_oper,a_valu,a_cmnt)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      character*(*) a_keyw
      character*(*) a_valu
      character*(*) a_unit
      character*(*) a_dimn
      character*(*) a_elem
      character*(*) a_oper
      character*(*) a_cmnt
      
c     OUTPUT VARIABLES:

      character*(*) a_data

c     LOCAL VARIABLES:

      integer i
      integer i_tabs(10)

      integer i_keyw
      integer i_valu
      integer i_unit
      integer i_dimn
      integer i_elem
      integer i_oper
      integer i_cmnt

      character*320 a_ktemp
      character*320 a_otemp
      character*320 a_vtemp
      character*320 a_ctemp
      character*320 a_utemp
      character*320 a_dtemp
      character*320 a_etemp
      character*320 a_cdel

c     COMMON BLOCKS

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer*4 rdflen
      external rdflen

c     PROCESSING STEPS:

      call rdf_trace('RDF_UNPARSE')
            if (a_keyw .eq. ' ' .and. a_unit .eq. ' ' .and. 
     &           a_valu .eq. ' ' .and. a_oper .eq. ' ') then
               if (a_cmnt .eq. ' ') then
                  a_data = ' '
               else
                  a_cdel  = a_cmdl(0)
c                  if (a_cdel .eq. ' ') a_cdel = '!'
c                   a_data =  a_cdel(1:max(rdflen(a_cdel),1))//' '//a_cmnt(1:rdflen(a_cmnt))
                  if (a_cdel .eq. ' ') then
                     a_data = ' '
                  else
                    a_data =  a_cdel(1:max(rdflen(a_cdel),1))//' '//a_cmnt(1:rdflen(a_cmnt))
                  endif
               endif
            else

               a_cdel  = a_cmdl(0)
c               if (a_cdel .eq. ' ') a_cdel = '!'
               if (a_cmnt .eq. ' ' .and. i_delflag(1) .eq. 1) a_cdel = ' '

               a_ktemp = a_keyw
               a_otemp = a_oper
               a_vtemp = a_valu

               a_utemp = ' '
               a_dtemp = ' '
               a_etemp = ' '
               if (a_cdel .eq. ' ') then
                 a_ctemp = ' '
               else
                 a_ctemp =  a_cdel(1:max(rdflen(a_cdel),1))//' '//a_cmnt(1:max(rdflen(a_cmnt),1))
               endif
               if (a_unit .ne. ' ') a_utemp =  '('//a_unit(1:max(rdflen(a_unit),1))//')'
               if (a_dimn .ne. ' ') a_dtemp =  '['//a_dimn(1:max(rdflen(a_dimn),1))//']'
               if (a_elem .ne. ' ') a_etemp =  '{'//a_elem(1:max(rdflen(a_elem),1))//'}'

               i_tabs(1) = i_fsizes(1)
               do i = 2,7
                 i_tabs(i) = i_tabs(i-1) + i_fsizes(i)
               enddo

               i_keyw = min(max(rdflen(a_ktemp) + 1, i_tabs(1) ),320)
               i_unit = min(max(rdflen(a_utemp) + 1, i_tabs(2) - i_keyw),320)
               i_dimn = min(max(rdflen(a_dtemp) + 1, i_tabs(3) - i_unit - i_keyw),320)
               i_elem = min(max(rdflen(a_etemp) + 1, i_tabs(4) - i_dimn - i_unit - i_keyw),320)
               i_oper = min(max(rdflen(a_otemp) + 1, i_tabs(5) - i_elem - i_dimn - i_unit - i_keyw),320)
               i_valu = min(max(rdflen(a_vtemp) + 1, i_tabs(6) - i_oper - i_elem - i_dimn - i_unit - i_keyw),320)
               i_cmnt = min(max(rdflen(a_ctemp) + 1, i_tabs(7) - i_valu - i_oper - i_elem - i_dimn - i_unit - i_keyw),320)
               a_data = a_ktemp(1:i_keyw)//a_utemp(1:i_unit)//a_dtemp(1:i_dimn)//a_etemp(1:i_elem)//
     &                  a_otemp(1:i_oper)//a_vtemp(1:i_valu)//a_ctemp(1:i_cmnt)
            endif

         call rdf_trace(' ')
         return
         end


c****************************************************************

      subroutine rdf_trace(a_routine)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: 
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

c     INCLUDE FILES

      include 'rdf_common.inc'

c     INPUT VARIABLES:

      character*(*) a_routine

c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i
      integer i_setup

c     COMMON BLOCKS

c     EQUIVALENCE STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer*4 rdflen
      external rdflen

c     DATA STATEMENTS:
      
      data i_setup /0/

      save i_setup

c     PROCESSING STEPS:

         if (i_setup .eq. 0) then
           i_stack = 0
           i_setup = 1
         endif

        if (a_routine .ne. ' ') then
          i_stack = i_stack+1
          if (i_stack .gt. 0 .and. i_stack .le. 10) a_stack(i_stack) = a_routine
c          type *,'TRACE  IN: i_stack=',i_stack,'  ',a_stack(i_stack)
        else
c          type *,'TRACE OUT: i_stack=',i_stack,'  ',a_stack(i_stack)
          if (i_stack .gt. 0 .and. i_stack .le. 10) a_stack(i_stack) = ' '
          i_stack = max(i_stack - 1, 0)
        endif

      return
      end


c The following is a commented out version of the include file that must accompany the source code

cc     PARAMETER STATEMENTS:
c      integer I_PARAMS
c      parameter(I_PARAMS = 500)
c
c      integer I_MCPF
c      parameter(I_MCPF = 320)
c
c      integer i_nums
c      integer i_pntr
c      character*320 a_dsets(I_PARAMS)
c      character*320 a_prfxs(I_PARAMS)
c      character*320 a_sufxs(I_PARAMS)
c      character*320 a_strts(I_PARAMS)
c      character*320 a_matks(I_PARAMS)
c      character*320 a_keyws(I_PARAMS)
c      character*320 a_units(I_PARAMS)
c      character*320 a_dimns(I_PARAMS)
c      character*320 a_elems(I_PARAMS)
c      character*320 a_opers(I_PARAMS)
c      character*320 a_cmnts(I_PARAMS)
c      character*320 a_valus(I_PARAMS)
c      common /params/ i_pntr,i_nums,a_dsets,a_prfxs,a_sufxs,a_strts,a_matks,
c     &                a_keyws,a_units,a_dimns,a_elems,a_opers,a_valus,a_cmnts
c
c      integer i_errflag(3)
c      integer i_error
c      character*320 a_error(I_PARAMS)
c      character*320 a_errfile
c      common /errmsg/ i_errflag,i_error,a_error,a_errfile
c
c      integer i_fsizes(10)
c      character*320 a_intfmt
c      character*320 a_realfmt
c      character*320 a_dblefmt
c      common /inital/ i_fsizes,a_intfmt,a_realfmt,a_dblefmt
c
c      integer i_prelen
c      integer i_suflen
c      character*320 a_prfx
c      character*320 a_sufx
c      character*320 a_prefix
c      character*320 a_suffix
c      common /indata/ a_prfx,a_sufx,a_prefix,a_suffix,i_prelen,i_suflen

c 3456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012
c        1         2         3         4         5         6         7         8         9       100       110       120       130

