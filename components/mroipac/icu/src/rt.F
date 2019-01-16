!c****************************************************************

      subroutine rt(trees, iz, jz, nres, nr_start, nr_end, naz_start, 
     &              naz_end, nsets, nres_chrg)

!c****************************************************************
!c**     
!c**   FILE NAME: rt.f
!c**     
!c**   DATE WRITTEN: 19-Jan-98
!c**     
!c**   PROGRAMMER: Charles Werner
!c**     
!c**   FUNCTIONAL DESCRIPTION: generates random connection trees 
!c**   between residues in the trees array. The list of residues 
!c**   is traversed in random order to generate multiple realizations
!c**   of the tree network. 
!c**     
!c**   ROUTINES CALLED: bermuda
!c**
!c**   NOTES: Note that the ilist,jlist,iz,jz arrays are integer*2 arrays
!c**   to conserve memory. If patches larger than 32768x32768 are needed
!c**   then these arrays must be changed to integer arrays which will
!c**   double the memory requirements. 
!c**     
!c**   UPDATE LOG:
!c**
!c**   Date Changed        Reason Changed          CR # and Version #
!c**   ------------       ----------------         ------------------
!c**    19-Jan-98       updated program format       
!c**
!c*****************************************************************
    
      use icuState
      implicit none

      real*4 RATIO              !ratio of width to height of ellipsoidal search
      parameter(RATIO = 1.0)

!c     INPUT VARIABLES:

      integer*1 trees(0:infp%i_rsamps-1,0:infp%i_azbufsize-1)   !unwrapping flags
      integer*4 iz(0:*),jz(0:*)                         !lists of residues - limits patches to 32 k by 32 k
      integer*4 nres                    !number of residues in the patch
      integer*4 nr_start,nr_end         !starting and ending range samples
      integer*4 naz_start, naz_end      !starting and ending azimuth lines
      integer*4 nsets                   !number of sets of trees                                        
      integer*4 nres_chrg               !residual tree charge

!c     LOCAL VARIABLES:
 
c      integer*2 ilist(0:LIST_SZ_TREES-1),jlist(0:LIST_SZ_TREES-1)      !list of locations for residues and neutrons in a tree
      integer*4, dimension (:),allocatable :: ilist,jlist,lists  !list of locations for residues and neutrons in a tree
      integer*4 s_tab(0:2, 0:(4*MBL*MBL + 4*MBL-1))                     !precomputed search table 

      integer*4 i,j,ll                  !loop counters
      integer*4 i5,j5                   !tree location temps
      integer*4 ichg                    !tree charge
      integer*4 nres1                   !number of residues remaining in the list
      integer*4 bx                      !current box size
      integer*4 ip,iend                 !pointers to the present residue list element, and the end of the list
      integer*4 n                       !index to list of search coordinates
      integer*4 m                       !index for generation of cuts
      integer*4 i1,j1                   !location of current residue
      integer*4 i3,j3                   !edge position when cutting to edge
      integer*4 i2,j2                   !location of current search location
      integer*4 i4,j4                   !cut pixel locations
      integer*4 bflag                   !flag used to check if a cut to the border possible
      integer*4 residual                !residual charge
      integer*4 nps                     !number of points in the spiral search table
      integer*4 kk                      !loop index for generation of branch cuts
      integer*4 iset                    !tree set loop counter
      integer*4 idum                    !random number seed
      integer*4 ipz                     !pointer into list of residues
      integer*4 itsz                    !sizeof tree list list

      integer*4 bermuda                 !function used to generate search table
      real*4 ran1                       !random number generator from Numerical Recipes
      external ran1      

!c     PROCESSING STEPS:
                                                
      itsz = infp%i_rsamps*infp%i_azbufsize/MEM_TREES_FACTOR
      allocate (ilist(0:itsz-1))
      allocate (jlist(0:itsz-1))
      allocate (lists(0:itsz-1))

      idum = -1                         !initialize random number generator on the first call
      nps = bermuda(RATIO, s_tab)       !generate elliptical spiral search table
     
      do iset=1, nsets                  !loop over the number of tree realizations

        nres1 = nres-1                  !reset counter of available residues
        residual = 0                    !reset sum of residual phases
        if(iset .gt. 1) then            !if not the first time, unmark residues
          do i = nr_start, nr_end
            do j = naz_start, naz_end   !unmark visited residues 
                trees(i,j) = IAND(trees(i,j),NOT(VISIT)) !unmark all residues as unvisited and start again
            end do
          end do
        endif
!c        write(6,'(1x,a)')"RT: generating random GZW trees"
            
        do while(nres1 .ge. 0)

          ipz = ran1(idum)*nres1
          i = iz(ipz)                   !get the random point
          j = jz(ipz)
          iz(ipz) = iz(nres1)           !get the replacement residue from the end of the list
          jz(ipz) = jz(nres1)           !new tree only if unvisited charge present
          iz(nres1) = i                 !get the replacement residue from the end of the list
          jz(nres1) = j                 !new tree only if unvisited charge present
          nres1 = nres1-1               !decrement size of available residue list
          if( (IAND(trees(i,j),CHARGE) .eq. 0) .or. (IAND(trees(i,j),VISIT) .ne. 0) )then
             goto 60    
          endif

          trees(i,j) = IOR(trees(i,j),VISIT)            !mark this charge as visited immediately
          trees(i,j) = IOR(trees(i,j), TWIG)            !mark this charge as on the current tree, this is the root
          ilist(0) = i                                  !first element of the list of charges on the tree
          jlist(0) = j
          iend = 1                                      !initialize pointer to first empty list element
          if (IAND(trees(i,j),PLUS) .eq. 1) then
            ichg = 1                                    !initialize value of tree charge
          else 
            ichg = -1
          endif
          do bx = 1, MBL                                !size of search region loop
            ip = 0                                      !initialize pointer to the top of the list of tree elements (twigs)

            do while (ip .lt. iend)
              i1 = ilist(ip)                            !i1,j1 are the column, row of the current residue                               
              j1 = jlist(ip)
              bflag = 0                                 !initialize border flag
              n = 0                                     !initialize pointer for list of search coordinates

              do while (s_tab(0,n) .le. bx)             !search over the search region for another residue or neutron 
                                                        !to make twigs
                i2 = i1 + s_tab(1,n)                    !current search location 
                j2 = j1 + s_tab(2,n) 
                n = n+1                                 !increment search table index

                if ((j2 .lt. naz_start) .or. (j2 .gt. naz_end))then     !out of bound, cut to top or bottom
                  if(i2 .eq. i1) then
                    j3 = max(j2, naz_start)             !do not cut outside array bounds
                    j3 = min(j3, naz_end)
                    kk = abs(j3-j1)                     !make a vertical cut
                    if(kk .eq. 0) then
                      trees(i1,j3) = IOR(trees(i1,j3), CUT)
                    else
                      do m=0, kk
                        j4 = j1 + (j3-j1)*m/kk
                        trees(i1,j4) = IOR(trees(i1,j4),CUT)
                      end do
                    endif
                    ichg = 0                            !discharge the tree
                    goto 40             
                  else 
                    goto 20                             !not vertical
                  endif
                endif
 
               if ((i2 .lt. nr_start) .or. (i2 .gt. nr_end))then !out of bounds, cut to right or left edge
                 if (j2 .eq. j1) then
                    i3 = max(i2, nr_start)              !do not cut outside array bounds
                    i3 = min(i3, nr_end)
                    kk = abs(i3-i1)                     !make a horizontal cut
                    if( kk .eq. 0) then
                      trees(i3,j1) = IOR(trees(i3,j1), CUT)
                    else
                      do m=0, kk
                        i4 = i1 + (i3-i1)*m/kk 
                        trees(i4,j1) = IOR(trees(i4,j1),CUT)
                      end do
                    endif
                    ichg = 0                            !discharge the tree
                    goto 40             
                  else 
                    goto 20                             !not horizontal
                  endif                                 
               endif                                    !end of test for branch cut to border

c      test if not part of current tree and if either a charge or neutron

                if ((IAND(trees(i2,j2),TWIG).eq.0) .and. 
     $             ( (IAND(trees(i2,j2),CHARGE).ne.0) .or. (IAND(trees(i2,j2),NEUTRON) .ne. 0))) then
                    
                  if (IAND(trees(i2,j2),VISIT) .eq. 0) then     !check if unvisited and a charge
                    if (IAND(trees(i2,j2),PLUS) .ne. 0) then 
                      ichg = ichg + 1                           !new value of tree charge
                    endif
                    if (IAND(trees(i2,j2),MINUS) .ne. 0) then
                      ichg = ichg - 1
                    endif
                    trees(i2,j2) = IOR(trees(i2,j2), VISIT)
                  endif

                  trees(i2,j2) = IOR(trees(i2,j2), TWIG)        !mark as twig in the current tree
                  ilist(iend) = i2                      !add location to list of charges and neutrons in this tree
                  jlist(iend) = j2
                  iend = iend + 1                       !increment pointer for end of charge and neutron list
 
                  if (iend .ge. itsz) then      !check if list of charges has exceeded its limit        
!c                     write(6,*) "WARNING RAN_TREES: list of residues has reached maximum size:",itsz
                     do ll = 1 , iend
                        lists(ll) = ilist(ll)
                     end do
                     deallocate (ilist)
                     itsz = itsz + infp%i_rsamps*infp%i_azbufsize/MEM_TREES_FACTOR
                     allocate(ilist(0:itsz-1))
                     do ll = 1 , iend
                        ilist(ll) = lists(ll)
                     end do
                     do ll = 1 , iend
                        lists(ll) = jlist(ll)
                     end do
                     deallocate (jlist)
                     allocate(jlist(0:itsz-1))
                     do ll = 1 , iend
                        jlist(ll) = lists(ll)
                     end do
                     deallocate (lists)
                     allocate(lists(0:itsz-1))
                  endif

                  kk = max(abs(i1-i2), abs(j1-j2))      !make the branch cut
 
                  if(kk .ne. 0) then                    !prevent cut to current residue
                    do m=0, kk
                      i4 = i1+(i2-i1)*m/kk 
                      j4 = j1+(j2-j1)*m/kk
                      trees(i4,j4) = IOR(trees(i4,j4),CUT)
                    end do
                  endif 
                  if (ichg .eq. 0)then
                     goto 40                    !if tree discharged, unmark residues 
                  endif
                                        !and search for new tree root
                endif                           !end of test for twigs (neutrons or charges) 

20              continue
              end do                            !end of spiral scan loop
              ip = ip +1                        !pick the next element (charge or neutron) off the list
            end do                              !end of loop over list of elements in the current tree
          end do                                !end of loop over box size

40        continue
          do m=0, iend-1                        !unmark all twigs on the current tree
             i5 = ilist(m)
             j5 = jlist(m)
             trees(i5,j5) = IAND(trees(i5,j5),NOT( TWIG))
          end do
          residual = residual + ichg            !sum up residual charge

60        continue
        end do                                  !end of scan loop for new unvisited charges
      end do                                    !end of loop over number of sets of trees
      nres_chrg = residual                      !return net residual charge

      deallocate(ilist)
      deallocate(jlist)
      deallocate(lists)

      end 
          
