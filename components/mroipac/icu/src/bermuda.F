!c****************************************************************

      integer*4 function bermuda(ratio, s_tab)

       use icuState
       implicit none

!c     INPUT VARIABLES:

       real*4 ratio				!ratio of width to length of the search area

!c     OUTPUT VARIABLES:

       integer*4 s_tab(0:2, 0:4*MBL*MBL + 4*MBL-1)	!precomputed search table array
						!s_tab(0,*) contains the radius
						!s_tab(1,*) contains the range offsets
                                                !s_tab(2,*) contains the azimuth offsets
!c     LOCAL VARIABLES:

       real*4 rat2				!square of ratio
       real*4 r2max				!square of current ellipsoid radius
       real*4 dist2
       integer*4 i1,j1

       integer*1 gf(0:2*MBL, 0:2*MBL)		!byte array used to generate search table

       integer i,j,ir				!loop indices
       integer nps				!number of points in the search table

!c     PROCESSING STEPS:
       
       do i=0, 2*MBL				!initialize byte mask array used to determine
         do j=0, 2*MBL				!if points within the ellipse
           gf(i,j) = 0
         end do
       end do
       
       rat2 = ratio*ratio			!square of ratio of ellipsoid height to  width
       if(ratio .lt. 1.0)rat2 = 1./rat2		!must be greater than 1.
       nps=0					!initialize number of points in the earch table
    
       do ir=1, MBL				!loop over radius 
         r2max = ir*ir				!current square of radius
         do i = -MBL, MBL			!scan over elements of the enclosing square rectangle
           do j = -MBL, MBL
             if ((i .eq. 0) .and. (j .eq. 0))goto 100
             if(ratio .lt. 1.0) then
                dist2 = i*i + rat2*j*j		!make sure that the ellipsoid stays with in the box
             else 
                dist2 = rat2*i*i + j*j
             end if

             if(dist2 .le. r2max) then			!test if within the ellipse
               i1 = i + MBL				!coordinates in the mask array of point inside ellipse
               j1 = j + MBL

               if(IAND(gf(i1,j1), LAWN) .eq. 0) then	!test if marked in the mask array
                 gf(i1,j1) = IOR(gf(i1,j1), LAWN)	!if not, add to list of points in the search table
                 s_tab(0,nps) = ir			!record the radius in the search table
                 s_tab(1,nps) = i			!range offset	
                 s_tab(2,nps) = j			!azimuth offset
                 nps = nps+1				!increment counter of points in the search table
               end if

             end if
 100         continue

           end do					!search through mask array
         end do

       end do						!increment radius
       bermuda = nps					!return number of points in the search table
       return
       end

      real*4 function ran1(idum)			!Numerical Recipes random number generator (0.<= x < 1.0)
      INTEGER*4 idum,IA,IM,IQ,IR,NTAB,NDIV
      REAL*4  AM,EPS,RNMX
      PARAMETER (IA=16807,IM=2147483647,AM=1./IM,IQ=127773,IR=2836)
      PARAMETER (NTAB=32,NDIV=1+(IM-1)/NTAB,EPS=1.2e-7,RNMX=1.-EPS)
      INTEGER j,k,iv(NTAB),iy
      DATA iv /NTAB*0/, iy /0/
      if (idum.le.0.or.iy.eq.0) then
        idum=max(-idum,1)
        do 11 j=NTAB+8,1,-1
          k=idum/IQ
          idum=IA*(idum-k*IQ)-IR*k
          if (idum.lt.0) idum=idum+IM
          if (j.le.NTAB) iv(j)=idum
11      continue
        iy=iv(1)
      endif
      k=idum/IQ
      idum=IA*(idum-k*IQ)-IR*k
      if (idum.lt.0) idum=idum+IM
      j=1+iy/NDIV
      iy=iv(j)
      iv(j)=idum
      ran1=min(AM*iy,RNMX)
      return
      END
