!c****************************************************************

      subroutine grass(phase, iseed, jseed, trees, nr_start, nr_end, 
     $  naz_start, naz_end, r_unw, i_unw_ctr)

!c****************************************************************
!c**     
!c**   FILE NAME: grass.f
!c**     
!c**   DATE WRITTEN: 6/30/97
!c**     
!c**   PROGRAMMER: Charles Werner, Paul Rosen, Scott Hensley
!c**     
!c**   FUNCTIONAL DESCRIPTION: This routine grows the grass after the 
!c**   residues and trees have been generated. This means actually
!c**   unwrapping the phase.
!c**     
!c**   ROUTINES CALLED:
!c**     
!c**   UPDATE LOG:
!c**
!c**   Date Changed        Reason Changed          CR # and Version #
!c**   ------------       ----------------         ------------------
!c**    28-Oct-97   incorrect count of points unwrapped	   
!c**    11-Nov-97   removed connected components array      
!c**    19-Jan-98   updated program format                  
!c**
!c*****************************************************************
      use icuState
      implicit none

!c     INPUT VARIABLES:

      real*4    phase(0:infp%i_rsamps-1,0:infp%i_azbufsize-1)	!interferogram phase modulo 2PI
      integer*1 trees(0:infp%i_rsamps-1,0:infp%i_azbufsize-1) !unwrapping trees, neutrons, residues, low correlation
      integer*4 iseed, jseed			!starting seed point for phase unwrapping
      integer*4 nr_start, nr_end		!starting and ending range sample in the interferogram array
      integer*4 naz_start, naz_end		!starting and ending azimuth line

!c     OUTPUT VARIABLES:

      real*4 r_unw(0:infp%i_rsamps-1,0:infp%i_azbufsize-1)	!unwrapped phase
      integer*4 i_unw_ctr			!number of points unwrapped

c!     LOCAL VARIABLES:
	
      integer*4 ii(0:MAX_GRASS-1,0:1),jj(0:MAX_GRASS-1,0:1)	!ping-pong lists of the perimeter of the growing region
!c      integer*4, dimension(:,:), allocatable :: ii,jj !ping-pong lists of the perimeter of the growing region
      integer*4 nn(0:1)				!array that contains lengths of ping-pong lists
      integer*4 isearch(0:3),jsearch(0:3)
      integer*4 i,j,k,l,m
      integer*4 i1,j1
      integer*4 nunw				!counter of the number of points unwrapped 
      integer*4 igsz
      real*4    p1				!phase of point on the perimeter of the growing region


!c     PROCESSING STEPS:
 
      igsz = MAX_GRASS

      isearch(0) = 1				!offsets to adjacent samples for growing the grass
      jsearch(0) = 0
      isearch(1) = 0
      jsearch(1) = 1
      isearch(2) = -1
      jsearch(2) = 0
      isearch(3) = 0
      jsearch(3) = -1

      ii(0,0) = iseed				!initial element of list 0
      jj(0,0) = jseed
      nn(0) = 1					!initial length of list 0
      nn(1) = 0					!initial length of list 1
      m=0					!initialize ping-pong list pointer
 
      r_unw(iseed,jseed) = phase(iseed,jseed)	!initialize output unwrapped phase value
      trees(iseed,jseed) = IOR(trees(iseed,jseed), LAWN)
      nunw = 1					!initialize counter of unwrapped points					
     
      do while(nn(m) .ne. 0)			!continue until list empty

        nn(1-m) = 0				!initialize length of the new list

        do k=0, nn(m)-1				!grow all elements of the current list
          i = ii(k,m)
          j = jj(k,m)
          p1 = r_unw(i,j)				!phase of current point on the perimeter
 
          do l=0,3				!search in all 4 directions
            i1 = i + isearch(l)			!look in the search direction
            j1 = j + jsearch(l)

            if((i1 .lt. nr_start) .or. (i1 .gt. nr_end)) goto 20	!test if candidate pixel outside of bounds
            if((j1 .lt. naz_start) .or. (j1 .gt. naz_end)) goto 20 

            if(IAND(trees(i1,j1),LAWN) .eq. LAWN) goto 20		!check if already unwrapped
            if(IAND(trees(i1,j1),LCORR) .eq. LCORR) goto 20		!check if below CORR threshold

            r_unw(i1,j1) = phase(i1,j1) + 
     &              TWO_PI_SP*nint((p1 - phase(i1,j1))/TWO_PI_SP)	!unwrap the phase
            nunw = nunw + 1						!increment counter of unwrapped pixels
            trees(i1,j1) = IOR(trees(i1,j1),LAWN)			!mark pixel on the lawn
            if (IAND(trees(i1,j1),CUT) .eq. CUT) goto 20		!do not add to list if pixel on a CUT

            if(nn(1-m) .lt. (MAX_GRASS-1)) then				!check length of new list
              ii(nn(1-m), 1-m) = i1					!add current element to the new list
              jj(nn(1-m), 1-m) = j1
              nn(1-m) = nn(1-m) + 1 					!increment new list pointer  
!c            else
!c              write(6,*) 'WARNING GRASS: Length of ping-pong lists exceeds list size allocation'
            endif

 20          continue 

           end do	!loop on search directions 

 40        continue

         end do 	!loop on current list elements
         m = 1-m	!switch to other list (ping-pong)

      end do		!grow while current list not empty

      if (nunw .eq. 1)then
         trees(iseed,jseed) = IAND(trees(iseed,jseed),NOT(LAWN))
         r_unw(iseed,jseed) = 0.0 !reset phase of unwrapped points
         nunw = 0   
      else  
         i_unw_ctr = nunw       !return number of points unwrapped
      endif 

      return
      end         

