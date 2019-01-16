c****************************************************************

      subroutine rdf_error(a_message)

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
c**     rdf_error performs the internal error handeling for rdf reader 
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

      character*(*) a_message
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i_lun
      integer i_setup
      integer i_iostat
      character*320 a_output

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer*4 rdflen
      external rdflen

c     DATA STATEMENTS:

      data i_setup /0/

      save i_setup

c     PROCESSING STEPS:

         if (i_setup .eq. 0) then
           i_error = 0
           i_setup = 1
         endif

         if (i_stack .eq. 1) then
            a_output = '*** RDF ERROR ***'//
     &            '  in '//a_stack(i_stack)(1:max(1,rdflen(a_stack(i_stack))))//
     &            '  -  '//a_message(1:max(1,rdflen(a_message)))
         else
            a_output = '*** RDF ERROR ***'//
     &            '  in '//a_stack(i_stack)(1:max(1,rdflen(a_stack(i_stack))))//
     &            '  -  '//a_message(1:max(1,rdflen(a_message)))//
     &            '   Entry: '//a_stack(1)(1:max(1,rdflen(a_stack(1))))
         endif

         if (i_errflag(1) .ne. 0) then                             ! Write to screen
              write(6,'(a)') a_output(1:max(1,rdflen(a_output)))
         endif

         if (i_errflag(2) .ne. 0) then                             ! Write to Error Buffer
           i_error = min(i_error+1,I_PARAMS)
           a_error(i_error) = a_output(1:max(1,rdflen(a_output)))
         endif

         if (i_errflag(3) .ne. 0) then                             ! Write to Error Log
           call rdf_getlun(i_lun)
           open(i_lun,file=a_errfile,status='unknown',form='formatted',
     &         iostat=i_iostat)
           if (i_iostat .eq. 0) then
             write(i_lun,'(a)',iostat=i_iostat) a_output(1:max(1,rdflen(a_output)))
             if (i_iostat .ne. 0) then
                 write(6,*) '*** RDF ERROR ***  in RDF_ERROR  -  Unable to write to Error file: ',
     &          a_errfile(1:max(rdflen(a_errfile),1))
               write(6,*) '                            Re-directing error messages to screen'
               write(6,'(a)') a_output(1:max(1,rdflen(a_output)))
             endif
             close(i_lun)
           else
             write(6,*) '*** RDF ERROR ***  in RDF_ERROR  -  Unable to Open Error file: ',
     &          a_errfile(1:max(rdflen(a_errfile),1)) 
             write(6,*) '                            Re-directing error messages to screen'
             write(6,*) a_output(1:max(1,rdflen(a_output)))
           endif
         endif

         return

       end

c****************************************************************

      subroutine rdf_merge(a_rdfname)

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

      character*(*) a_rdfname
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i
      integer i_num
      integer i_loc

      integer i_lun
      integer i_stat
      integer i_done

      integer i_cont
      integer i_data

      integer i_val
      character*320 a_vals(100)

      character*320 a_file
      character*320 a_dset
      character*320 a_line
      character*320 a_data
      
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

      integer rdflen
      external rdflen

      character*320 rdfupper
      external rdfupper

      character*320 rdfcullsp
      external rdfcullsp

      character*320 rdftrim
      external rdftrim

c     PROCESSING STEPS:
      
      call rdf_trace('RDF_MERGE')
      i_pntr = 0

      call rdf_getlun(i_lun)                                             ! find a free unit number to read file
      if (i_lun .eq. 0) then
         call rdf_error('Unable to allocate unit number')
         call rdf_trace(' ')
         return
      endif

      i_loc = index(a_rdfname,':')
      if (i_loc .gt. 0) then
         a_file = a_rdfname(i_loc+1:)
         if (i_loc .gt. 1) then
            a_dset = rdfupper(rdfcullsp(rdftrim(a_rdfname(1:i_loc-1))))
         else
            a_dset = ' '
         endif
      else
         a_file = a_rdfname
         a_dset = ' '
      endif

      open(unit=i_lun,file=a_file(1:rdflen(a_file)),status='old',form='formatted',
     &     iostat=i_stat)
c     &     iostat=i_stat,readonly)
      if (i_stat .ne. 0) then
         a_errtmp = 'Unable to open rdf file: '//a_file(1:min(max(rdflen(a_file),1),120))
         call rdf_error(a_errtmp)
         call rdf_trace(' ')
         return
      endif
      write(6,'(1x,a,a)') 'Reading from: ',a_file(1:max(rdflen(a_file),1))

      a_prfx = ' '
      a_sufx = ' '
      a_prefix = ' '
      a_suffix = ' '
      i_prelen = 0
      i_suflen = 0

      i_done = 0
      do while(i_done .eq. 0 .and. i_nums .lt. I_PARAMS)

         a_data = ' '
         i_data = 0
         i_cont = 0
         do while(i_cont .eq. 0)
            read(i_lun,'(a)',iostat=i_stat) a_line
            if (i_data .eq. 0) then
               a_data = rdftrim(a_line)
            else
               a_data(i_data+1:) = rdftrim(a_line)
               if (i_data+rdflen(rdftrim(a_line)) .gt. I_MCPF) then
                  a_errtmp = 'Data field exceeds max characters per line. '//
     &                        a_data(1:max(1,rdflen(a_data)))
                  call rdf_error(a_errtmp)
               endif  
            endif
            i_data = rdflen(a_data)
            if (i_data .eq. 0) then
               i_cont = 1
            else if (ichar(a_data(i_data:i_data)) .ne. 92 ) then ! check for '\' (backslach)
               i_cont = 1
            else
               i_data = i_data-1
            endif
         enddo
         if (i_stat .ne. 0) then
            a_data = ' '
            i_done = 1
         else

            call rdf_parse(a_data,a_keyw,a_unit,a_dimn,a_elem,a_oper,a_valu,a_cmnt)

            a_dsets(i_nums+1) = rdftrim(a_dset)
            a_keyws(i_nums+1) = rdftrim(a_keyw)
            a_units(i_nums+1) = rdftrim(a_unit)
            a_dimns(i_nums+1) = rdftrim(a_dimn)
            a_elems(i_nums+1) = rdftrim(a_elem)
            a_opers(i_nums+1) = rdftrim(a_oper)
            a_valus(i_nums+1) = rdftrim(a_valu)
            a_cmnts(i_nums+1) = rdftrim(a_cmnt)

            if (rdfupper(a_keyws(i_nums+1)) .eq. 'PREFIX') then
               a_prfx = a_valus(i_nums+1)
               a_prefix = a_prfx
               call rdf_unquote(a_prefix,i_prelen)
            else if (rdfupper(a_keyws(i_nums+1)) .eq. 'SUFFIX') then
               a_sufx = a_valus(i_nums+1)
               a_suffix = a_sufx
               call rdf_unquote(a_suffix,i_suflen)
            else if (rdfupper(a_keyws(i_nums+1)) .eq. 'COMMENT') then
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
               a_cmdl(0) = rdftrim(a_valus(i_nums+1))
            else if (rdfupper(a_keyws(i_nums+1)) .eq. 'END_RDF_DATA') then
               a_data = ' '
               i_done = 1
            else
               i_nums = i_nums+1
               if (a_keyws(i_nums) .ne. ' ') then
                  a_prfxs(i_nums) = a_prfx
                  a_sufxs(i_nums) = a_sufx
                  if (i_prelen .gt. 0) then
                     a_matks(i_nums) = rdfupper(rdfcullsp(rdftrim(a_prefix(1:i_prelen)//a_keyws(i_nums))))
                  else
                     a_matks(i_nums) = rdfupper(rdfcullsp(rdftrim(a_keyws(i_nums))))
                  endif
                  a_matks(i_nums) = a_matks(i_nums)(1:rdflen(a_matks(i_nums)))//rdfupper(rdfcullsp(a_suffix))
               else
                  a_matks(i_nums) = ' '
               endif
            endif
         endif
      enddo

      close(i_lun)

      if (i_nums .eq. I_PARAMS) 
     &    write(6,*) 'Internal buffer full, may not have read all data'
      i_num = i_nums

      call rdf_trace(' ')
      return
      end  

c****************************************************************

      subroutine top_read(a_rdfname)

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

      character*(*) a_rdfname
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i_num

      integer i
      integer i_len
      integer i_lun
      integer i_stat
      integer i_done
      integer i_type

      integer i_keyws
      integer i_valus
      integer i_units
      integer i_opers
      integer i_cmnts

      character*320 a_data

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

c     PROCESSING STEPS:

      i_pntr = 0

      call rdf_getlun(i_lun)
      if (i_lun .le. 10) stop 'Error tring to get logical unit number'

      write(6,*) ' '
      write(6,'(1x,a,a)') 'Reading from: ',a_rdfname(1:max(rdflen(a_rdfname),1))
c      open(unit=i_lun,file=a_rdfname,status='old',form='formatted',iostat=i_stat,readonly)
      open(unit=i_lun,file=a_rdfname,status='old',form='formatted',iostat=i_stat)
      if (i_stat .ne. 0) write(6,   *) 'i_lun  = ',i_lun
      if (i_stat .ne. 0) write(6,   *) 'i_stat = ',i_stat
      if (i_stat .ne. 0) stop 'Error opening RDF file'

      i_nums = 0
      i_done = 0
      do while(i_done .eq. 0)

         a_dsets(i_nums+1) = ' '
         a_matks(i_nums+1) = ' '
         a_strts(i_nums+1) = ' '
         a_prfxs(i_nums+1) = ' '
         a_sufxs(i_nums+1) = ' '
         a_keyws(i_nums+1) = ' '
         a_valus(i_nums+1) = ' '
         a_opers(i_nums+1) = ' '
         a_units(i_nums+1) = ' '
         a_dimns(i_nums+1) = ' '
         a_elems(i_nums+1) = ' '
         a_cmnts(i_nums+1) = ' '
         i_keyws = 0
         i_valus = 0
         i_opers = 0
         i_units = 0
         i_cmnts = 0
         read(i_lun,'(a)',iostat=i_stat) a_data
         if (i_stat .ne. 0) then
            i_len = 0
            a_data = ' '
            i_done = 1
         else
            i_len = rdflen(a_data)
         endif

         i_type = 1
c         write(6,   *) 'i_len=',i_len
         do i=1,i_len
            if (i_type .eq. 0) then
               i_cmnts = i_cmnts + 1
               a_cmnts(i_nums+1)(i_cmnts:i_cmnts) = a_data(i:i)
            else if (a_data(i:i) .eq. '(' ) then
               i_type = 10
            else if (a_data(i:i) .eq. ')' ) then
               i_type = 2
            else if (a_data(i:i) .eq. '=' ) then
               i_type = 2
               a_opers(i_nums+1) = '='
            else if (a_data(i:i) .eq. '<' ) then
               i_type = 2
               a_opers(i_nums+1) = '<'
            else if (a_data(i:i) .eq. '>' ) then
               i_type = 2
               a_opers(i_nums+1) = '>'
            else if (a_data(i:i) .eq. ';' ) then
               i_type = 2
               a_opers(i_nums+1) = '='
            else if (a_data(i:i) .eq. '#' ) then
               i_type = 0
            else if (a_data(i:i) .eq. '!' ) then
               i_type = 0
            else
               if (i_type .eq. 2) then
                  i_keyws = i_keyws + 1
                  a_keyws(i_nums+1)(i_keyws:i_keyws) = (a_data(i:i)) ! rdfupper(a_data(i:i))
               else if (i_type .eq. 10) then
                  i_units = i_units + 1
                  a_units(i_nums+1)(i_units:i_units) = (a_data(i:i)) ! rdfupper(a_data(i:i))
               else if (i_type .eq. 1) then
                  i_valus = i_valus + 1
                  a_valus(i_nums+1)(i_valus:i_valus) = a_data(i:i)
               endif
            endif
         enddo

c     if (a_opers(i_nums+1) .ne. ' ') then
         i_nums = i_nums+1
         a_keyws(i_nums) = rdftrim(a_keyws(i_nums))
         a_valus(i_nums) = rdftrim(a_valus(i_nums))
         a_units(i_nums) = rdftrim(a_units(i_nums))
         a_opers(i_nums) = rdftrim(a_opers(i_nums))
         a_matks(i_nums) = rdfupper(rdfcullsp(a_keyws(i_nums)))
c     endif

      enddo

      close(i_lun)

      i_num = i_nums

      return
      end  
      
c****************************************************************

      subroutine rdf_write(a_rdfname)

c****************************************************************
c**   
c**   FILE NAME: rdf_write.f
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

      character*(*) a_rdfname
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i
      integer i_loc
      integer i_lun
      integer i_stat

      integer i_iostat

      character*320 a_file
      character*320 a_dset
      character*320 a_lpre
      character*320 a_lsuf

      character*320 a_data
      character*320 a_errtmp

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer rdflen
      external rdflen

      character*320 rdfupper
      external rdfupper

      character*320 rdftrim
      external rdftrim

      character*320 rdfint2
      external rdfint2


c     PROCESSING STEPS:

      call rdf_trace('RDF_WRITE')
      call rdf_getlun(i_lun)
      if (i_lun .eq. 0) then
         call rdf_error('Unable to allocate unit number')
         call rdf_trace(' ')
         return
      endif

      i_loc = index(a_rdfname,':')
      if (i_loc .gt. 0) then
         a_file = a_rdfname(i_loc+1:)
         if (i_loc .gt. 1) then
            a_dset = rdfupper(rdftrim(a_rdfname(1:i_loc-1)))
         else
            a_dset = ' '
         endif
      else
         a_file = a_rdfname
         a_dset = ' '
      endif

      write(6,*) ' '
      open(unit=i_lun,file=a_file,status='unknown',form='formatted',iostat=i_stat)
      if (i_stat .ne. 0) then
        a_errtmp = 'Unable to open rdf file: '//
     &      a_file(1:min(max(rdflen(a_file),1),120))//'  lun,iostat = '//rdfint2(i_lun,i_stat)
        call rdf_error(a_errtmp)
        call rdf_trace(' ')
        return
      endif
      write(6,*) 'Writing to:   ',a_file(1:min(max(rdflen(a_file),1),150))

      a_lpre = ' '
      a_lsuf = ' '
      do i = 1,i_nums
         if (a_dset .eq. ' ' .or. a_dset .eq. a_dsets(i) ) then
            if (a_keyws(i) .ne. ' ' .and. a_prfxs(i) .ne. a_lpre) then
              a_lpre = a_prfxs(i)
c              type *,'a_prfxs = ',rdflen(a_prfxs(i)),' ',a_prfxs(i)
              a_data=' '

c              type *,'a_data = ',rdflen(a_data),' ',a_data
              call rdf_unparse(a_data,'PREFIX ',   ' ',   ' ',   ' ',   '=',a_prfxs(i),' ')
c              type *,'a_data = ',rdflen(a_data),' ',a_data
              write(i_lun,'(a)',iostat=i_stat) a_data(1:max(1,rdflen(a_data)))
              if (i_stat .ne. 0) then
                 a_errtmp = 'Unable to write to file. '//
     &              a_data(1:min(max(1,rdflen(a_data)),120))
                 call rdf_error(a_errtmp)
              endif
            endif

            if (a_keyws(i) .ne. ' ' .and. a_sufxs(i) .ne. a_lsuf) then
              a_lsuf = a_sufxs(i)
              call rdf_unparse(a_data,'SUFFIX',' ',' ',' ','=',a_sufxs(i),' ')
              write(i_lun,'(a)',iostat=i_stat) a_data(1:max(1,rdflen(a_data)))
              if (i_stat .ne. 0) then
                 a_errtmp = 'Unable to write to file. '//
     &              a_data(1:min(max(1,rdflen(a_data)),120))
                 call rdf_error(a_errtmp)
              endif
            endif

            call rdf_unparse(a_data,a_keyws(i),a_units(i),a_dimns(i),a_elems(i),a_opers(i),a_valus(i),a_cmnts(i))
            write(i_lun,'(a)',iostat=i_stat) a_data(1:max(1,rdflen(a_data)))
            if (i_stat .ne. 0) then
                 a_errtmp = 'Unable to write to file. '//
     &              a_data(1:min(max(1,rdflen(a_data)),120))
                 call rdf_error(a_errtmp)
            endif
         endif
      enddo

      close(i_lun)

      call rdf_trace(' ')
      return
      end  


c****************************************************************

      subroutine top_write(a_rdfname)

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

      character*(*) a_rdfname
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i
      integer i_lun
      integer i_stat
      integer i_keyws
      integer i_valus
      integer i_units
      integer i_opers
      integer i_cmnts
      integer i_iostat

      character*320 a_temp,a_otmp, a_errtmp

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

      integer rdflen
      external rdflen

c     PROCESSING STEPS:

      call rdf_trace('TOP_WRITE')
      call rdf_getlun(i_lun)
      if (i_lun .eq. 0) then
         call rdf_error('Unable to allocate unit number')
         call rdf_trace(' ')
         return
      endif

      write(6,*) ' '
      write(6,*) 'Writing to:   ',a_rdfname(1:max(rdflen(a_rdfname),1))
      open(unit=i_lun,file=a_rdfname,status='unknown',form='formatted',iostat=i_stat)
      if (i_stat .ne. 0) then
         a_errtmp = 'Unable to open rdf file: '//
     &      a_rdfname(1:min(max(rdflen(a_rdfname),1),120))
         call rdf_error(a_errtmp)
         call rdf_trace(' ')
         return
      endif

      do i = 1,i_nums
         if (a_keyws(i) .eq. ' ' .and. a_units(i) .eq. ' ' .and. 
     &        a_valus(i) .eq. ' ' .and. a_opers(i) .eq. ' ') then
            if (a_cmnts(i) .eq. ' ') then
               write(i_lun,*) ' '
            else
               write(i_lun,'(a)') '#'//a_cmnts(i)(1:rdflen(a_cmnts(i)))
            endif
         else
            a_otmp = a_opers(i)
            if (a_otmp .eq. '=') a_otmp=';'
            if (a_units(i) .eq. ' ') then
               i_valus = min(max(rdflen(a_valus(i)) + 1, 55),320)
               i_opers = min(max(rdflen(a_opers(i)) + 1, 57 - i_valus),320)
               i_keyws = min(max(rdflen(a_valus(i)) + 1, 78 - i_opers - i_valus),320)
               i_cmnts = min(max(rdflen(a_cmnts(i)) + 2, 80 - i_valus - i_opers - i_keyws),320)
               if (a_cmnts(i) .eq. ' ') then
                  write(i_lun,'(4a)',iostat=i_stat) a_valus(i)(1:i_valus),a_otmp(1:i_opers),
     &                 a_keyws(i)(1:i_keyws)
               else
                  write(i_lun,'(4a)',iostat=i_stat) a_valus(i)(1:i_valus),a_otmp(1:i_opers),
     &                 a_keyws(i)(1:i_keyws),'# '//a_cmnts(i)(1:i_cmnts)
               endif
            else
               i_valus = min(max(rdflen(a_valus(i)) + 1, 55),320)
               i_opers = min(max(rdflen(a_opers(i)) + 1, 57 - i_valus),320)
               i_keyws = min(max(rdflen(a_valus(i)) + 1, 70 - i_opers - i_valus),320)
               a_temp = '('//a_units(i)(1:rdflen(a_units(i)))//')'
               i_units = min(max(rdflen(a_temp) + 1,     73 - i_keyws - i_opers - i_valus),320)
               i_cmnts = min(max(rdflen(a_cmnts(i)) + 2, 80 - i_valus - i_opers - i_units - i_keyws),320)
               if (a_cmnts(i) .eq. ' ') then
                  write(i_lun,'(5a)',iostat=i_stat) a_valus(i)(1:i_valus),a_otmp(1:i_opers),a_keyws(i)(1:i_keyws),
     &                 a_valus(i)(1:i_valus),a_temp(1:i_units)
               else
                  write(i_lun,'(6a)',iostat=i_stat) a_valus(i)(1:i_valus),a_otmp(1:i_opers),a_keyws(i)(1:i_keyws),
     &                 a_valus(i)(1:i_valus),a_temp(1:i_units),'# '//a_cmnts(i)(1:i_cmnts)
               endif
            endif
            if (i_stat .ne. 0) then
               a_errtmp = 'Unable to write to file. '//
     &         a_keyws(i)(1:min(max(rdflen(a_keyws(i)),1),150))
               call rdf_error(a_errtmp)
            endif
         endif
      enddo

      close(i_lun)

      call rdf_trace(' ')
      return
      end  
      

c****************************************************************

      subroutine rdf_getlun(i_lun)

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

      integer i_lun
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      logical l_open

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

c     PROCESSING STEPS:

      call rdf_trace('RDF_GETLUN')
      i_lun=10
      l_open = .true.
      do while(i_lun .lt. 99 .and. l_open)
         i_lun = i_lun + 1
         inquire(unit=i_lun,opened=l_open)
      enddo
      
      if (i_lun .ge. 99) i_lun = 0

      call rdf_trace(' ')
      return
      end

