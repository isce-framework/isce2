c     replacement routines for fortran intrinsics
c        getarg
c        iargc
c     to operate on a string representing the command line
c
c     extra spaces are ignored
c
c     used to facilitate rapid conversion of
c     fortran PROGRAMs to SUBROUTINEs that
c     that can be bound to python
c
      subroutine getarg_roi(command_line,n,val)
      character*(*) command_line
      character*(*) val

      integer command_len

      val='NotSetYet'

      iarg_count=0
      command_len=len(command_line)

      ival_start=1
      ival_end=0

      in_space=1
      do 10 i=1,command_len
         if((in_space.eq.1).and.(' '.ne.command_line(i:i)))then
            in_space=0
            iarg_count=iarg_count+1
            if(iarg_count.eq.n)then
               ival_start=i
            endif
         endif
         if((in_space.eq.0).and.(' '.eq.command_line(i:i)))then
            in_space=1
            if(iarg_count.eq.n)then
               ival_end=i-1
c              should break out of loop here
            endif
         endif
 10   continue
      if((in_space.eq.0).and.(iarg_count.eq.n))then
         ival_end=i-1
      endif
      val=command_line(ival_start:ival_end)
      end
c
c
c
      function iargc_roi(command_line)
      character*(*) command_line

      integer command_len

      iarg_count=0

      command_len=len(command_line)

      in_space=1
      do 10 i=1,command_len
         if((in_space.eq.1).and.(' '.ne.command_line(i:i)))then
            in_space=0
            iarg_count=iarg_count+1
         endif
         if((in_space.eq.0).and.(' '.eq.command_line(i:i)))then
            in_space=1
         endif
 10   continue
      iargc_roi=iarg_count
      end
