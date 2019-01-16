!c****************************************************************

      subroutine residues(interf, b_PatchTrees, phase, nr_start, nr_end, naz_start, naz_end, iz, jz, nres)


!c****************************************************************
!c**     
!c**   FILE NAME: residues.f
!c**     
!c**   DATE WRITTEN: 28-Feb-98
!c**     
!c**   PROGRAMMER: Charles Werner
!c**     
!c**   FUNCTIONAL DESCRIPTION:  calculate wrapped phase and find residues 
!c**     
!c**   UPDATE LOG:
!c**
!c**   Date Changed        Reason Changed          CR # and Version #
!c**   ------------       ----------------         ------------------
!c**
!c*****************************************************************

      use icuState
      implicit none


!c    INPUT VARIABLES:

      complex*8 interf(0:infp%i_rsamps-1,0:infp%i_azbufsize-1)	!interferogram
      integer*1 b_PatchTrees(0:infp%i_rsamps-1,0:infp%i_azbufsize-1)	!unwrapping flags
      real*4 phase(0:infp%i_rsamps-1,0:infp%i_azbufsize-1)	!phase modulo 2PI
      integer*4 nr_start,nr_end			!starting and ending range samples
      integer*4 naz_start, naz_end		!starting and ending azimuth lines

!c    OUTPUT VARIABLES
      integer*4 iz(0:*),jz(0:*)	!lists of residues
      integer*4 nres				!number of residues

!c     LOCAL VARIABLES:
 
      integer*4 i,j,k				!loop counters

c$doacross local(i,j),
c$&        share(nr_start,nr_end,naz_start,naz_end,phase,interf) 
      do i = nr_start, nr_end
        do j = naz_start, naz_end		!calculate wrapped phase data 
           if(abs(interf(i,j)) .ne. 0.0) then
             phase(i,j) = atan2(aimag(interf(i,j)), real(interf(i,j)))
           else
             phase(i,j) = 0.0 
           end if
        end do
      end do

c$doacross local(i,j,k),
c$&        share(nr_start,nr_end,naz_start,naz_end,phase,b_PatchTrees)   
      do i = nr_start, nr_end-1			!calculate residues
        do j = naz_start, naz_end-1 
          k = nint((phase(i+1,j)-phase(i,j))/TWO_PI) + nint((phase(i+1,j+1)-phase(i+1,j))/TWO_PI)
     $      + nint((phase(i,j+1)-phase(i+1,j+1))/TWO_PI) + nint((phase(i,j)-phase(i,j+1))/TWO_PI)
             if(k .gt. 0) then
	       b_PatchTrees(i,j) = IOR(PLUS,b_PatchTrees(i,j))	!positive residue
 	     end if		 
             if(k .lt. 0) then
	       b_PatchTrees(i,j) = IOR(MINUS,b_PatchTrees(i,j))	!negative residue
 	     end if		 
         end do
      end do
 
      nres = 0					!initialize residue counter
      do i = nr_start, nr_end-1			!calculate residues
        do j = naz_start, naz_end-1 
 	  if(IAND(b_PatchTrees(i,j),CHARGE).ne.0)then
            iz(nres) = i			!save the residue location in the list
            jz(nres) = j
            nres=nres+1  
	  end if		 
        end do
      end do
      end
