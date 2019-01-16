c****************************************************************

c     Program correct_geoid_i2_srtm 

c****************************************************************
c**     
c**   FILE NAME: correct_geoid_i2_srtm.f
c**     
c**   DATE WRITTEN: 11/10/2000
c**   (editted extensively by Elaine Chapin 10/Oct/2002)
c**     
c**   PROGRAMMER: Scott Hensley
c**     
c**   FUNCTIONAL DESCRIPTION: This program will take a file with
c**   ellipsoid heights in a DTED projection and correct for 
c**   the geoid.
c**     
c**   ROUTINES CALLED: geoid_hgt
c**     
c**   NOTES: 
c**
c**   1.) As a one time only exception angles in this program are in
c**   DEGREES not radians.
c**
c**   2.) User should reference the EGM96 Geoid file with full path
c**   in the data statement from the protected directory where the
c**   the Harmonic Coefficient File is located.
c**
c**   3.) Only point with height values greater than hgtnull a 
c**   parameter are corrected. This value should be set to the 
c**   maximal null value designator.
c**     
c**   to compile on the SGI:
c**   f77 -bytereclen -extend_source -o correct_geoid_i2_srtm  
c**    correct_geoid_i2_srtm.f rdf_reader_sub.f
c**
c**   UPDATE LOG:
c**
c**   Date Changed        Reason Changed                  CR # and Version #
c**   ------------       ----------------                 -----------------
c**
c**   5/March/2004       added generation of mdx file     -Elaine Chapin
c**   2/June/2004        made lat/longs double to fix     -Elaine Chapin
c**                      hdr write problem
c**   29/June/2004       added checking for too many samps -Elaine Chapin
c**     
c*****************************************************************

      subroutine correct_geoid_i2_srtm(inAccessor,outAccessor) 
      
      use correct_geoid_i2_srtmState
      implicit none

c     INCLUDE FILES:

c     PARAMETER STATEMENTS:

      integer GEOID_BYTES       !bytes in geoid harmonics file
      parameter(GEOID_BYTES=24)

      !use allocateble
      !integer MAXSAMPS
      !parameter(MAXSAMPS=20000)

      integer MAXGRID
      parameter(MAXGRID=2000)

      real*4 r_geoidsample
      parameter(r_geoidsample=.1)

      real r_inhgtnull
      parameter(r_inhgtnull=-1000.)
      real r_outhgtnull
      parameter(r_outhgtnull=-10000.)

c     INPUT VARIABLES:
	
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      !character*120 a_infile,a_outfile,a_string,a_geoidfile
      integer*8 inAccessor,outAccessor
      character*20000 MESSAGE
      integer i_outfile,i_geoidlat,i_geoidlon,i,j,i_geoidunit,i_eof
      integer i_lat,i_lon,ierr,iargc,i_input
      integer*2, allocatable :: i_indata(:)
      !real*8 d_clat,d_clon
      !real*8 d_dlat,d_dlon
      real*4 r_pad,r_u,r_t
      real*4, allocatable :: r_indata(:),r_outdata(:)
      real*4 r_latmax,r_latmin,r_lonmax,r_lonmin,r_geoid_cor
      !real*4 r_latgrid(MAXGRID),r_longrid(MAXGRID)
      real*4, allocatable :: r_latgrid(:),r_longrid(:)
      !real*8 r_lat,r_lon,r_geoidsamples(MAXGRID,MAXGRID),pi,r_dtor,r_rtod
      real*8 r_lat,r_lon,pi,r_dtor,r_rtod
      real*8, allocatable ::r_geoidsamples(:,:)
      integer*4 i_bytes
      real*8 d_temp(2)
c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c      data a_geoidfile /'/u/erda0/sh/EGM96/egm96geoid.dat'/

c     FUNCTION STATEMENTS:


c     SAVE STATEMENTS:

c     PROCESSING STEPS:
      write(MESSAGE,*) ' '
      call write_out(stdWriter,MESSAGE)
      write(MESSAGE,'(a)') '    << Geoid Correction I2 SRTM>> '
      call write_out(stdWriter,MESSAGE)

      write(MESSAGE,*) ' '
      call write_out(stdWriter,MESSAGE)

      write(MESSAGE,'(a)') 'Jet Propulsion Laboratory - Radar Science and Engineering '
      call write_out(stdWriter,MESSAGE)

      write(MESSAGE,*) ' '
      call write_out(stdWriter,MESSAGE)

    
      allocate(r_geoidsamples(MAXGRID,MAXGRID))
      allocate(r_latgrid(MAXGRID))
      allocate(r_longrid(MAXGRID))

      allocate(i_indata(i_samples))
      allocate(r_indata(i_samples))
      allocate(r_outdata(i_samples))
      i_input = 1
      pi = 4.d0*atan(1.d0)
      r_dtor = pi/180.d0
      r_rtod = 180.d0/pi
      if(d_clon .gt. 180.d0)then
         d_clon = d_clon - 360.d0
      endif
    
      !note that d_dlat,d_dlon are in arcsec
      d_dlat = abs(d_dlat)
      d_dlon = abs(d_dlon)
      
      i_geoidunit = 15
      open(i_geoidunit,file=a_geoidfile,form='unformatted',access='direct',recl=GEOID_BYTES,iostat=ierr)



c     determine the min,max latitude and longitude for points in the file

      r_pad = 1.5*r_geoidsample

      r_latmax = d_clat + r_pad
      r_latmin = d_clat - i_numlines*d_dlat - r_pad

c     extra logic required for working around 180 deg longtitude

      r_lonmin = d_clon - r_pad 
      r_lonmax = d_clon + i_samples*d_dlon + r_pad

      i_geoidlat = nint((r_latmax - r_latmin)/r_geoidsample)
      i_geoidlon = nint((r_lonmax - r_lonmin)/r_geoidsample)

c     sample the geoid at points within this region at r_geoidsample  degree intervals - this will be
c     used for bilinear interpolation later

      write(MESSAGE,*) ' '
      call write_out(stdWriter,MESSAGE)

      write(MESSAGE,'(a,i5,a,i5)') 'Sampling Geoid at grid points -  Longitude Samples: ',i_geoidlon,' Latitude Lines: ',i_geoidlat
      call write_out(stdWriter,MESSAGE)

      do i=1,i_geoidlat
         
         r_lat = (r_latmax - (i-1)*r_geoidsample)
         r_latgrid(i) = r_lat
         r_lat = r_lat*r_dtor

         do j=1,i_geoidlon

            r_lon = (r_lonmin + (j-1)*r_geoidsample)
            r_longrid(j) = r_lon
            r_lon = r_lon*r_dtor
            
            call geoid_hgt(i_geoidunit,r_lat,r_lon,r_geoidsamples(j,i))

         enddo

      enddo
      write(MESSAGE,'(a,4(1x,f6.2))') 'Corner Geoid Heights (m) = ',
     +  r_geoidsamples(i_geoidlon,i_geoidlat),
     +  r_geoidsamples(         1,i_geoidlat),
     +  r_geoidsamples(         1,         1),
     +  r_geoidsamples(i_geoidlon,         1)
      call write_out(stdWriter,MESSAGE)

c     now correct heights to the local geoid height

      write(MESSAGE,*) ' '
      call write_out(stdWriter,MESSAGE)

      write(MESSAGE,'(a)') 'Correcting data to geoid height...'
      call write_out(stdWriter,MESSAGE)

      write(MESSAGE,*) ' '
      call write_out(stdWriter,MESSAGE)
      do i=1,i_numlines

         if(mod(i,512) .eq. 0)then
            write(MESSAGE,'(a,x,i7)') 'At line: ',i
            call write_out(stdWriter,MESSAGE)

         endif

c     read in data

        !use a caster from the image api that reads in i2 and castes
        !into r4  
        call getLineSequential(inAccessor,r_indata,i_eof)        
!         if(i_input .eq. 1)then
!            read(12,rec=i) (i_indata(j),j=1,i_samples)
!            do j=1,i_samples
!               r_indata(j) = i_indata(j)
!           enddo
!         else
!            do j=1,i_samples
!               r_indata(j) = 0.0
!            enddo
!         endif

         r_lat = d_clat - (i-1)*d_dlat

c     latitude bilinear data coefficients

         i_lat = (r_latmax - r_lat)/r_geoidsample + 1

         r_u = (r_lat - r_latgrid(i_lat))/(r_latgrid(i_lat+1) - r_latgrid(i_lat))

         do j=1,i_samples

            r_lon = d_clon + (j-1)*d_dlon 

c     longitude bilinear data coefficients

            i_lon = (r_lon - r_lonmin)/r_geoidsample + 1

c     bilinear interpolation

            r_t = (r_lon - r_longrid(i_lon))/(r_longrid(i_lon+1) - r_longrid(i_lon))

            r_geoid_cor = (1.-r_t)*(1.-r_u)*r_geoidsamples(i_lon,i_lat) + r_u*(1.-r_t)*r_geoidsamples(i_lon,i_lat+1) + 
     +           r_t*(1.-r_u)*r_geoidsamples(i_lon+1,i_lat) + r_u*r_t*r_geoidsamples(i_lon+1,i_lat+1)

c     correct the data for the geoid

            if(r_indata(j) .gt. r_inhgtnull)then
               !jng remove rounding off to allow below meter precision
               !the image caster should take care of possible rounding
               !r_outdata(j) = nint(r_indata(j) - r_geoid_cor*i_input*i_sign)
               r_outdata(j) = (r_indata(j) - r_geoid_cor*i_input*i_sign)
            else
               r_outdata(j) = (1.0-nullIsWater) * r_outhgtnull - nullIsWater * r_geoid_cor*i_input*i_sign
            endif

         enddo
     
!         if(index(a_outfile,'OVERWRITE') .eq. 0)then
!            write(i_outfile,rec=i) (r_outdata(j),j=1,i_samples)
!         else
!            do j=1,i_samples
!               i_indata(j) = nint(r_outdata(j))
!            enddo
!            write(i_outfile,rec=i) (i_indata(j),j=1,i_samples)
!         endif
        !use a caster from the image api that writes out r4 and castes
        !into i2  
      call setLineSequential(outAccessor,r_outdata)
      enddo
      close(i_geoidunit)

      deallocate(r_geoidsamples)
      deallocate(r_latgrid)
      deallocate(r_longrid)
      deallocate(i_indata)
      deallocate(r_indata)
      deallocate(r_outdata)
      end


c****************************************************************

      subroutine geoid_hgt(i_geoidunit,r_lat,r_lon,r_h)

c****************************************************************
c**     
c**   FILE NAME: geoid_hgt.f
c**     
c**   DATE WRITTEN: 9/01/97
c**     
c**   PROGRAMMER: Scott Hensley
c**     
c**   FUNCTIONAL DESCRIPTION: This program is taken from NIMA and
c**   cleaned up somewhat for ease of use in a number of applications.
c**
c**   This program is designed for the calculation of a geoid undulation
c**   at a point whose latitude and longitude is specified. The program
c**   is designed to use the potential coefficient model egm96 and a
c**   set of spherical harmonic coefficients of a correction term.
c**   The correction term is composed of several different components
c**   the primary one being the conversion of a height anomaly to a geoid
c**   undulation. The principles of this procedure were initially
c**   described in the paper: 
c**
c**   "Use of potential coefficient models for geoid
c**   undulation determination using a spherical harmonic representation
c**   of the height anomaly/geoid undulation difference" by R.H. Rapp,
c**   Journal of Geodesy, 1996.
c**
c**   This program is designed to be used with the constants of egm96
c**   and those of the wgs84(g873) system. The undulation will refer to
c**   the WGS84 ellipsoid. Specific details on the undulation computation 
c**   will be found in the joint project report describing the development 
c**   of EGM96. his program is a modification of the program described in the
c**   following report:
c**
c**   "A fortran program for the computation of gravimetric quantities from
c**   high degree spherical harmonic expansions", Richard H. Rapp,
c**   Report 334, Department of Geodetic Science and Surveying, The Ohio
c**   State University, Columbus, 1982
c**
c**     
c**   ROUTINES CALLED:
c**     
c**   NOTES: 
c**
c**   dimensions of p,q,hc,hs must be at least ((maxn+1)*(maxn+2))/2,
c**   dimensions of sinml,cosml,scrap must be at least maxn,
c**   where maxn is maximum order of computation
c**   the current dimensions are set for a maximum degree of 360
c**     
c**   UPDATE LOG:
c**
c**   Date Changed        Reason Changed                  CR # and Version #
c**   ------------       ----------------                 -----------------
c**     
c*****************************************************************

      implicit none

c     PARAMETERS:

      integer MAXSIZ,MAXORDER
      parameter(MAXSIZ=65341,MAXORDER=361)
      integer NUM_BYTES,NUM_READ
      parameter(NUM_BYTES=24,NUM_READ=65341)

c     INPUT VARIABLES:

      integer i_geoidunit
      real*8 r_lat
      real*8 r_lon
	
c     OUTPUT VARIABLES:

      real*8 r_h

c     LOCAL VARIABLES:

      integer l,n,m,ig,nmax,iflag,ir,k,i,j,loc,i_first

      real*8  p(MAXSIZ),scrap(MAXORDER),rleg(MAXORDER),dleg(MAXORDER)
      real*8  rlnn(MAXORDER),sinml(MAXORDER),cosml(MAXORDER)
      real*8 hc(65341),hs(MAXSIZ),cc(MAXSIZ),cs(MAXSIZ)
      real*8 t1,t2,f,flatl,flat,flon,rlat1,rlat,rlon,ht,rad,gr,re,u,haco

      integer i_rec

c     COMMON BLOCKS:

      real*8 gm,ae,omega,rf,j2,j4,j6,j8,j10,e2,geqt,kg
      common /ellipdata/ gm,ae,omega,rf,j2,j4,j6,j8,j10,e2,geqt,kg

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

      data gm /.3986004418d15/       !gm in units of m**3/s**2
      data ae /6378137.0d0/          !semi-major axis in m
      data e2 /.00669437999013d0/    !eccentrcity squared 
      data rf /298.257223563d0/      !flattening
      data omega /7.292115d-5/       !spin rate rad/sec
      data j2 / 0.108262982131d-2 /  !potential coefficients
      data j4 / -.237091120053d-05/ 
      data j6 / 0.608346498882d-8/  
      data j8 / -0.142681087920d-10/
      data j10 / 0.121439275882d-13/
      data geqt / 9.7803253359d0 /    !equatorial gravity
      data kg   / .00193185265246d0/  !some constant

      data rad /57.29577951308232d0/
      data ht /0.0d0/
      data i_first /0/

c     SAVE STATEMENTS:

      save i_first,rad,ht,nmax,rleg,dleg,sinml,cosml,rlnn,hc,hs,cc,cs

C     FUNCTION STATEMENTS:

c     PROCESSING STEPS:

      flat = r_lat*rad
      flon = r_lon*rad
      if(flon .lt. 0)then
         flon = flon + 360.d0
      endif

      if(i_first .eq. 0)then
         
         i_first = 1
         
         nmax = MAXORDER - 1
         
         l = MAXSIZ
         
         do i=1,l
            cc(i)=0.0d0
            cs(i)=0.0d0
         enddo
         
c     the correction coefficients are now read in

         do i_rec=1,NUM_READ
            read(i_geoidunit,rec=i_rec) n,m,t1,t2

            ig = (n*(n+1))/2 + m + 1
            cc(ig) = t1
            cs(ig) = t2
         enddo

c     the potential coefficients are now read in and the reference
c     even degree zonal harmonic coefficients removed to degree 6
         
         call dhcsin(i_geoidunit,nmax,f,hc,hs)
         
c     setting iflag=1 prevents legendre function derivatives being taken
c     in subroutine legfdn

         iflag = 1

      endif

      ir = 0
      k = nmax + 1
      flatl = 91.0d0
         
c     compute the geocentric latitude,geocentric radius,normal gravity

      call radgra(flat,flon,ht,rlat,gr,re)
      
      if(flatl .ne. flat)then 
         rlat1 = rlat
         rlat = 1.5707963267948966d0 - rlat
         flatl = flat
         do j=1,k
            m = j-1
            call legfdn(m,rlat,rleg,dleg,nmax,ir,rlnn,iflag)
            do i =j,k
               n = i - 1
               loc = (n*(n+1))/2+m+1
               p(loc) = rleg(i)
            enddo
         enddo
      endif
      
      rlon = flon/rad
      
      call dscml (rlon,nmax,sinml,cosml)

      call hundu(u,nmax,p,hc,hs,sinml,cosml,gr,re,rlat1,cc,cs,haco)
      

c     u is the geoid undulation from the egm96 potential coefficient model
c     including the height anomaly to geoid undulation correction term
c     and a correction term to have the undulations refer to the
c     wgs84 ellipsoid. the geoid undulation unit is meters.
      
      r_h = u
      
      end

c****************************************************************

      subroutine hundu(undu,nmax,p,hc,hs,sinml,cosml,gr,re,ang,cc,
     +     cs,haco)

c****************************************************************
c**     
c**   FILE NAME: geoid_hgt.f
c**     
c**   DATE WRITTEN: Sometime in 1996.
c**     
c**   PROGRAMMER: NIMA
c**     
c**   FUNCTIONAL DESCRIPTION: Generate height undulations
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

c     PARAMETERS:

      integer MAXSIZ
      parameter(MAXSIZ=65341)

c     INPUT VARIABLES:

      integer nmax
      real*8 p(MAXSIZ)
      real*8 hc(MAXSIZ)
      real*8 hs(MAXSIZ)
      real*8 cc(MAXSIZ)
      real*8 cs(MAXSIZ)
      real*8 sinml(MAXSIZ)
      real*8 cosml(MAXSIZ)
      real*8 re
      real*8 ang
      real*8 haco
      real*8 gr
	
c     OUTPUT VARIABLES:

      real*8 undu

c     LOCAL VARIABLES:

      real*8 a,b,ar,arn,sum,sum2,sumc,tempc,temp,ac
      integer k,n,m

c     COMMON BLOCKS:

      real*8 gm,ae,omega,rf,j2,j4,j6,j8,j10,e2,geqt,kg
      common /ellipdata/ gm,ae,omega,rf,j2,j4,j6,j8,j10,e2,geqt,kg

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

C     FUNCTION STATEMENTS:

c     PROCESSING STEPS:

      ar = ae/re
      arn = ar
      ac = 0.0
      a = 0.0
      b = 0.0
      k = 3
      

      do n=2,nmax
         arn = arn*ar
         k = k+1
         sum = p(k)*hc(k)
         sumc = p(k)*cc(k)
         sum2 = 0.0
         do m =1,n
            k = k+1
            tempc = cc(k)*cosml(m)+cs(k)*sinml(m)
            temp = hc(k)*cosml(m)+hs(k)*sinml(m)
            sumc = sumc+p(k)*tempc
            sum = sum+p(k)*temp
         enddo
         ac = ac+sumc
         a = a+sum*arn
      enddo

      ac = ac+cc(1)+p(2)*cc(2)+p(3)*(cc(3)*cosml(1)+cs(3)*sinml(1))
      haco = ac/100.d0
      undu = a*gm/(gr*re)

c     add haco to convert height anomaly on the ellipsoid to the undulation
c     add -0.53m to make undulation refer to the wgs84 ellipsoid.

      undu = undu + haco - 0.53d0

      end

c****************************************************************

      subroutine dscml(rlon,nmax,sinml,cosml)

c****************************************************************
c**     
c**   FILE NAME: geoid_hgt.f
c**     
c**   DATE WRITTEN: Sometime in 96
c**     
c**   PROGRAMMER: Scott Hensley
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

c     PARAMETERS

      integer MAXSIZ
      parameter(MAXSIZ=361)

c     INPUT VARIABLES:

      integer nmax
      real*8 rlon
      
c     OUTPUT VARIABLES:

      real*8 sinml(MAXSIZ)
      real*8 cosml(MAXSIZ)

c     LOCAL VARIABLES:

      integer m
      real*8 a,b

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

C     FUNCTION STATEMENTS:

c     PROCESSING STEPS:


      a = dsin(rlon)
      b = dcos(rlon)
      sinml(1) = a
      cosml(1) = b
      sinml(2) = 2.0*b*a
      cosml(2) = 2.0*b*b - 1.d0
      
      do  m=3,nmax
         sinml(m) = 2.d0*b*sinml(m-1)-sinml(m-2)
         cosml(m) = 2.d0*b*cosml(m-1)-cosml(m-2)
      enddo

      
      end

c****************************************************************

      subroutine dhcsin(i_geoidunit,nmax,f,hc,hs)

c****************************************************************
c**     
c**   FILE NAME: geoid_hgt.f
c**     
c**   DATE WRITTEN: Sometime in 1996
c**     
c**   PROGRAMMER: NIMA
c**     
c**   FUNCTIONAL DESCRIPTION: 
c**   The even degree zonal coefficients given below were computed for the
c**   wgs84(g873) system of constants and are identical to those values
c**   used in the nima gridding procedure. Computed using subroutine
c**   grs written by N.K. Pavlis
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

c     PARAMETER:

      integer MAXSIZ
      parameter(MAXSIZ=65341)

      integer START_READ,END_READ
      parameter(START_READ=65342,END_READ=131062)

c     INPUT VARIABLES:

      integer i_geoidunit
      integer nmax
      real*8 f
	
c     OUTPUT VARIABLES:

      real*8 hc(MAXSIZ),hs(MAXSIZ)      

c     LOCAL VARIABLES:

      integer i_rec
      integer k,m,n
      real*8 c,s,ec,es

c     COMMON BLOCKS:

      real*8 gm,ae,omega,rf,j2,j4,j6,j8,j10,e2,geqt,kg
      common /ellipdata/ gm,ae,omega,rf,j2,j4,j6,j8,j10,e2,geqt,kg

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

C     FUNCTION STATEMENTS:

c     PROCESSING STEPS:

      m = ((nmax+1)*(nmax+2))/2
      do  n=1,m
         hc(n)=0.0
         hs(n)=0.0
      enddo

      do i_rec=START_READ,END_READ
         read(i_geoidunit,rec=i_rec) n,m,c,s
         n = (n*(n+1))/2 + m + 1
         hc(n) = c
         hs(n) = s
      enddo

 3    hc(4) = hc(4) + j2/dsqrt(5.d0)
      hc(11) = hc(11) + j4/3.0d0
      hc(22) = hc(22) + j6/dsqrt(13.d0)
      hc(37) = hc(37) + j8/dsqrt(17.d0)
      hc(56) = hc(56) + j10/dsqrt(21.d0)


      end

c****************************************************************

      subroutine legfdn(m,theta,rleg,dleg,nmx,ir,rlnn,iflag)

c****************************************************************
c**     
c**   FILE NAME: geoid_hgt.f
c**     
c**   DATE WRITTEN: Sometime in 1996
c**     
c**   PROGRAMMER: NIMA
c**     
c**   FUNCTIONAL DESCRIPTION: 
c**
c**            This subroutine computes  all normalized legendre function
c**            in "rleg" and their derivatives in "dleg". Order is always
c**            m , and colatitude is always theta  (radians). Maximum deg
c**            is  nmx  . All calculations in double precision.
c**            ir  must be set to zero before the first call to this sub.
c**            The dimensions of arrays  rleg, dleg, and rlnn  must be
c**            at least equal to  nmx+1  .
c**
c**            This program does not compute derivatives at the poles .
c**
c**            If    iflag = 1  , only the legendre functions are
c**            computed.
c**
c**      original programmer :Oscar L. Colombo, Dept. of Geodetic Science
c**      The Ohio State University, August 1980 . 
c**
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

c      PARAMETERS:
      
      integer MAXSIZ
      parameter(MAXSIZ=361)

c     INPUT VARIABLES:

      integer ir
      integer m
      integer nmx
      integer iflag
      real*8 theta
	
c     OUTPUT VARIABLES:

      real*8 rleg(MAXSIZ),dleg(MAXSIZ),rlnn(MAXSIZ)

c     LOCAL VARIABLES:

      integer m1,m2,m3,nmx1,nmx2p,n2,n1,n
      real*8 drts(1300),dirt(1300),cothet,sithet,sithi,rln1,rln

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     SAVE STATEMENTS:

      save

C     FUNCTION STATEMENTS:

c     PROCESSING STEPS:

      nmx1 = nmx + 1
      nmx2p = 2*nmx + 1
      m1 = m + 1
      m2 = m + 2
      m3 = m + 3


      if(ir .ne. 1)then 
         ir = 1
         do n = 1,nmx2p
            drts(n) = dsqrt(n*1.d0)
            dirt(n) = 1.d0/drts(n)
         enddo 
      endif
      cothet = dcos(theta)
      sithet = dsin(theta)

      if(iflag .ne. 1 .and. theta .ne. 0.d0)then
         sithi = 1.d0/sithet
      endif

c     compute the legendre functions 

      rlnn(1) = 1.d0
      rlnn(2) = sithet*drts(3)

      do n1 = 3,m1
         n = n1-1
         n2 = 2*n
         rlnn(n1) = drts(n2+1)*dirt(n2)*sithet*rlnn(n1-1)
      enddo


      if(m .le. 1)then
         if(m .eq. 0)then
            rleg(1) = 1.d0
            rleg(2) = cothet*drts(3)
         else
            rleg(2) = rlnn(2)
            rleg(3) = drts(5)*cothet*rleg(2)
         endif
      endif

      rleg(m1) = rlnn(m1)
      if(m2 .le. nmx1)then 
          rleg(m2) = drts(m1*2+1)*cothet*rleg(m1)
          if(m3 .le. nmx1)then 
             do  n1 = m3,nmx1
                 n = n1 - 1
                 if(.not.((m.eq.0 .and. n .lt. 2) .or. (m .eq. 1 .and. n .lt. 3)))then
                     n2 = 2*n
                     rleg(n1) = drts(n2+1)*dirt(n+m)*dirt(n-m)*(drts(n2-1)*cothet*rleg(n1-1)-drts(n+m-1)*drts(n-m-1)*dirt(n2-3)*rleg(n1-2))
                 endif
            enddo
          endif
      endif  
      
      if(iflag .eq. 1)then
         return
      endif

c     derivatives

      if(sithet .eq. 0.d0)then
         write(6,'(a)') ' *** legfdn  does not compute derivatives at the poles'
         return
      endif

c     compute all the derivatives of the legendre functions

      rlnn(1) = 0.d0
      rln = rlnn(2)
      rlnn(2) = drts(3)*cothet

      do n1 = 3, m1
         n = n1-1
         n2 = 2*n
         rln1 = rlnn(n1)
         rlnn(n1) = drts(n2+1)*dirt(n2)*(sithet*rlnn(n)+cothet*rln)
         rln = rln1
      enddo

      dleg(m1) = rlnn(m1)
      if(m2 .gt. nmx1)then
         return
      endif

      do n1 = m2,nmx1
         n = n1-1
         n2 = n*2
         dleg(n1) = sithi*(n*rleg(n1)*cothet-drts(n-m)*drts(n+m)*
     +        drts(n2+1)*dirt(n2-1)*rleg(n))
      enddo

      end

c****************************************************************

      subroutine radgra(flat,flon,ht,rlat,gr,re)

c****************************************************************
c**     
c**   FILE NAME: geoid_hgt.f
c**     
c**   DATE WRITTEN: Sometime in 1996
c**     
c**   PROGRAMMER: NIMA
c**     
c**   FUNCTIONAL DESCRIPTION: This subroutine computes geocentric distance 
c**   to the point, the geocentric latitude,and an approximate value of normal 
c**   gravity at the point based the constants of the wgs84 (g873) 
c**   system are used.
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

      real*8 flat,flon
      real*8 ht

c     OUTPUT VARIABLES:

      real*8 rlat
      real*8 gr,re

c     LOCAL VARIABLES:

      real*8 n,flatr,flonr,t1,t2,x,y,z,rad

c     COMMON BLOCKS:

      real*8 gm,ae,omega,rf,j2,j4,j6,j8,j10,e2,geqt,kg
      common /ellipdata/ gm,ae,omega,rf,j2,j4,j6,j8,j10,e2,geqt,kg

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

      data rad /57.29577951308232d0/

C     FUNCTION STATEMENTS:

c     PROCESSING STEPS:

      flatr = flat/rad
      flonr = flon/rad
      t1 = dsin(flatr)**2
      n = ae/dsqrt(1.d0 - e2*t1)
      t2 = (n + ht)*dcos(flatr)
      x = t2*dcos(flonr)
      y = t2*dsin(flonr)
      z = (n*(1.-e2) + ht)*dsin(flatr)
      n = ae/dsqrt(1.d0 - e2*t1)

c compute the geocentric radius

      re = dsqrt(x**2+y**2+z**2)

c compute the geocentric latitude

      rlat = datan(z/dsqrt(x**2 + y**2))

c compute normal gravity:units are m/sec**2

      gr = geqt*(1.d0 + kg*t1)/dsqrt(1.d0 - e2*t1)

      end  





