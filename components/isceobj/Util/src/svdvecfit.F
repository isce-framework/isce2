c****************************************************************
      subroutine svdvecfit(i_mp,i_rd,i_fp,r_vecin,r_vobs,r_cov,
     +     i_np,r_a,r_at2,r_u,r_v,r_w,r_chisq,l_chisq)

c****************************************************************
c**   
c**   FILE NAME: svdvecfit.f
c**   
c**   DATE WRITTEN: 01/02/95 
c**   
c**   PROGRAMMER: Scott Hensley
c**   
c**   FUNCTIONAL DESCRIPTION: This routine does a least squares fit 
c**   to a vector valued observation least squares problem.
c**   
c**   ROUTINES CALLED: gaussj,svbksb,svdcmp,funcs
c**   
c**   NOTES: funcs is a user supplied function giving the jacobian
c**   of the observation parameters wrt to fit parameters. This routine
c**   is a generalization of Numerical Recipes svdfit. Note that this
c**   routine can also be used in a nonlinear least squares procedure
c**   by iterating properly.
c**
c**   Solves the least problem 
c**
c**             T   -1     -1     T   -1 
c**    A = (AMAT COV  AMAT)  (AMAT COV  )VOBS 
c**
c**    where AMAT is the jacobain of the observations vs parameters,
c**    COV is the covriance matrix of observations
c**    and VOBS is the vector of observations. 
c**
c**    r_a should be passed in with current best estimate of values
c**   
c**   UPDATE LOG: 
c**         
c**  4/17/95 - Reversed order of r_vecin, r_vobs, and r_cov    SJS
c**            revmoved r_vt, cleaned up parameter list
c**   
c*****************************************************************

      implicit none

c     PARAMETERS:
      integer I_NPE                  !number of parameters to estimate = i_np
      integer I_RDE                  !number of observations per point = i_rd 
      real*8 R_TOL,R_LAMBDA
      parameter(I_NPE=7)
      parameter(I_RDE=2)
      parameter(R_TOL=1.0d-20)
      parameter (R_LAMBDA=1.d0)

c     INPUT VARIABLES:
      integer i_mp                   !number of input points
      integer i_rd                   !number of observations each point
      integer i_fp                   !number of input parameters to func
      integer i_np                   !number of parameters to solve for

      real*8  r_vecin(i_fp,i_mp) 	 !vector values for func 
      real*8  r_vobs(i_rd,i_mp) 	 !vector of observations
      real*8  r_cov(i_rd,i_rd,i_mp)  !covariance matrix of observation
      real*8  r_chisq(i_rd,0:i_mp) 	 !chisq for solution and fit vs observation 
      real*8  r_a(i_np)   			 !solution to least squares
                                     !for each point 
      logical l_chisq                !evaluate the chisq for this fit
      
c     OUTPUT VARIABLES:
      real*8 r_at2(i_np)             !delta to add to previous solution
      real*8 r_u(i_np,i_np)          !svd matrix, orthogonal matrix
      real*8 r_v(i_np,i_np)          !svd matrix, orthogonal matrix
      real*8 r_w(i_np)               !svd matrix, diagonal matrix

c     LOCAL VARIABLES:
      integer i,j,k,i_pts
      real*8  r_covtemp(I_RDE,I_RDE)
      real*8  r_am(I_NPE,I_RDE)
      real*8  r_amat(I_RDE,I_NPE)
      real*8  r_ptot(I_NPE)
      real*8  r_wmax,r_thres,r_b(I_RDE,1),r_chird(I_RDE)

      integer i_paramest(I_NPE),i_usedata(I_RDE)
      common/funcom3/i_paramest,i_usedata

c     DATA STATEMENTS:

C     FUNCTION STATEMENTS:

c     PROCESSING STEPS:

c     init some arrays

c      write(*,*)  ' '
c      write(*,*)  'Inside SVDVECFIT'
c      write(*,*)  ' '

      if (i_rd .ne. I_RDE) then 
         write(*,*) 'ERROR - i_rd not equal to I_RDE in SVDVECFIT'
         stop
      end if
      if (i_np .ne. I_NPE) then
         write(*,*) 'ERROR - i_np not equal to I_NPE in SVDVECFIT'
         stop
      end if

      do i=1,i_np
         do j=1,i_np
            r_u(i,j) = 0.0
         enddo
         r_ptot(i) = 0.0
      enddo

c     loop over the input points

      do i_pts=1,i_mp

c         write(*,*)  'i_pts = ',i_pts

c     invert the covariance matrix of the observation

         do i=1,i_rd
            do j=1,i_rd
               r_covtemp(i,j) = r_cov(i,j,i_pts)
            enddo
         enddo

         call gaussj(r_covtemp,i_rd,i_rd,r_b,1,1)

c     get the required jacobian matrix

         call funcs(i_pts,i_rd,i_fp,r_vecin(1,i_pts),i_np,r_a,r_amat)

c         do i=1,i_rd
c            do j=1,i_np
c               write(*,*)  'i,j,r_amat = ',i,j,r_amat(i,j)
c            enddo
c         enddo

c     multiply amat transpose by the inverse cov matrix

         do i=1,i_np
            do j=1,i_rd
               r_am(i,j) = 0.0
               do k=1,i_rd
                  r_am(i,j) = r_am(i,j) + r_amat(k,i)*r_covtemp(k,j)
               enddo
            enddo
         enddo

c         do i=1,i_np
c            do j=1,i_rd
c               write(*,*)  'i,j,r_am = ',i,j,r_am(i,j)
c            enddo
c         enddo

c     multiply am by amat

         do i=1,i_np
            do j=1,i_np
               do k=1,i_rd
                  r_u(i,j) = r_u(i,j) + r_am(i,k)*r_amat(k,j)
               enddo
            enddo
         enddo

c     multilpy am by vobs


c         write(*,*)  'r_vobs,i_pts = ',i_pts,r_vobs(1,i_pts),r_vobs(2,i_pts)
         do i=1,i_np
            do k=1,i_rd
               r_ptot(i) = r_ptot(i) + r_am(i,k)*r_vobs(k,i_pts)
            enddo
         enddo
         
      enddo   !i_pts

c     find the SVD of the r_u matrix

c         do i=1,i_np
c            do j=1,i_np
c               write(*,*)  'i,j,r_u = ',i,j,r_u(i,j)
c            enddo
c         enddo

      call svdcmp(r_u,i_np,i_np,i_np,i_np,r_w,r_v)

c         do i=1,i_np
c            do j=1,i_np
c               write(*,*)  'i,j,r_u,r_v = ',i,j,r_u(i,j),r_v(i,j)
c            enddo
c         enddo

c         do i=1,i_np
c            write(*,*)  'w = ',i,r_w(i)
c         enddo

c     kill off all the singular values

      r_wmax = 0.0
      do i=1,i_np
         if(r_w(i) .gt. r_wmax)then
            r_wmax = r_w(i)
         endif
      enddo
      r_thres = r_wmax*R_TOL
c      write(*,*)  'r_thres = ',r_thres

      do i=1,i_np
         if(r_w(i) .lt. r_thres)then
            r_w(i) = 0.0
         endif
      enddo

c      do i=1,i_np
c         write(*,*)  'w = ',i,r_w(i)
c      enddo

c     use the svbksb routine to solve for the desired parameters

      call svbksb(r_u,r_w,r_v,i_np,i_np,i_np,i_np,r_ptot,r_at2)

c     update the r_a vector

      do i=1,i_np
         r_at2(i) = -r_at2(i)*i_paramest(i)
         r_a(i) = r_at2(i)/R_LAMBDA + r_a(i)
c         write(*,*) 'a=',i,r_a(i),r_at2(i)
      enddo

c     evaluate the chisq array (linearized version)

      if(l_chisq)then

c     loop over data points


         do i=1,i_rd
            r_chird(i) = 0.
         enddo
         r_chisq(1,0) = 0.0
         do i=1,i_mp

            call funcs(i,i_rd,i_fp,r_vecin(1,i),i_np,r_a,r_amat)

            do j=1,i_rd
               r_chisq(j,i) = 0.0
               do k=1,i_np
                  r_chisq(j,i) = r_chisq(j,i) + r_amat(j,k)*r_at2(k)
               enddo
c               write(*,*)  'r_chisq = ',i,j,r_chisq(j,i),r_vobs(j,i)
               r_chisq(j,i) = r_covtemp(j,j)*(r_chisq(j,i) - 
     +              r_vobs(j,i))**2
               r_chisq(1,0) = r_chisq(1,0) + r_chisq(j,i)
               r_chird(j) = r_chird(j) + r_chisq(j,i)
            enddo

         enddo                  !i_pts loop for chisq

         r_chisq(1,0) = sqrt(r_chisq(1,0)/(2.*i_mp))
         write(*,*)  'r_chisq = ',r_chisq(1,0),sqrt(r_chird(1)/i_mp),sqrt(r_chird(2)/i_mp)

      endif
      
      end  

c******************************************************************************

      SUBROUTINE gaussj(a,n,np,b,m,mp)
      INTEGER m,mp,n,np,NMAX
      REAL*8 a(np,np),b(np,mp)
      PARAMETER (NMAX=50)
      INTEGER i,icol,irow,j,k,l,ll,indxc(NMAX),indxr(NMAX),ipiv(NMAX)
      REAL*8 big,dum,pivinv
      do 11 j=1,n
        ipiv(j)=0
11    continue
      do 22 i=1,n
        big=0.
        do 13 j=1,n
          if(ipiv(j).ne.1)then
            do 12 k=1,n
              if (ipiv(k).eq.0) then
                if (abs(a(j,k)).ge.big)then
                  big=abs(a(j,k))
                  irow=j
                  icol=k
                endif
              else if (ipiv(k).gt.1) then
                pause 'singular matrix in gaussj'
              endif
12          continue
          endif
13      continue
        ipiv(icol)=ipiv(icol)+1
        if (irow.ne.icol) then
          do 14 l=1,n
            dum=a(irow,l)
            a(irow,l)=a(icol,l)
            a(icol,l)=dum
14        continue
          do 15 l=1,m
            dum=b(irow,l)
            b(irow,l)=b(icol,l)
            b(icol,l)=dum
15        continue
        endif
        indxr(i)=irow
        indxc(i)=icol
        if (a(icol,icol).eq.0.) pause 'singular matrix in gaussj'
        pivinv=1./a(icol,icol)
        a(icol,icol)=1.
        do 16 l=1,n
          a(icol,l)=a(icol,l)*pivinv
16      continue
        do 17 l=1,m
          b(icol,l)=b(icol,l)*pivinv
17      continue
        do 21 ll=1,n
          if(ll.ne.icol)then
            dum=a(ll,icol)
            a(ll,icol)=0.
            do 18 l=1,n
              a(ll,l)=a(ll,l)-a(icol,l)*dum
18          continue
            do 19 l=1,m
              b(ll,l)=b(ll,l)-b(icol,l)*dum
19          continue
          endif
21      continue
22    continue
      do 24 l=n,1,-1
        if(indxr(l).ne.indxc(l))then
          do 23 k=1,n
            dum=a(k,indxr(l))
            a(k,indxr(l))=a(k,indxc(l))
            a(k,indxc(l))=dum
23        continue
        endif
24    continue
      return
      END

      SUBROUTINE svdcmp(a,m,n,mp,np,w,v)
      INTEGER m,mp,n,np,NMAX
      REAL*8 a(mp,np),v(np,np),w(np)
      PARAMETER (NMAX=500)
      INTEGER i,its,j,jj,k,l,nm
      REAL*8 anorm,c,f,g,h,s,scale,x,y,z,rv1(NMAX),pythag
      real*8 r_one

      g=0.0
      r_one = 1.d0
      scale=0.0
      anorm=0.0
      do 25 i=1,n
        l=i+1
        rv1(i)=scale*g
        g=0.0
        s=0.0
        scale=0.0
        if(i.le.m)then
          do 11 k=i,m
            scale=scale+abs(a(k,i))
11        continue
          if(scale.ne.0.0)then
            do 12 k=i,m
              a(k,i)=a(k,i)/scale
              s=s+a(k,i)*a(k,i)
12          continue
            f=a(i,i)
            g=-sign(sqrt(s),f)
            h=f*g-s
            a(i,i)=f-g
            do 15 j=l,n
              s=0.0
              do 13 k=i,m
                s=s+a(k,i)*a(k,j)
13            continue
              f=s/h
              do 14 k=i,m
                a(k,j)=a(k,j)+f*a(k,i)
14            continue
15          continue
            do 16 k=i,m
              a(k,i)=scale*a(k,i)
16          continue
          endif
        endif
        w(i)=scale *g
        g=0.0
        s=0.0
        scale=0.0
        if((i.le.m).and.(i.ne.n))then
          do 17 k=l,n
            scale=scale+abs(a(i,k))
17        continue
          if(scale.ne.0.0)then
            do 18 k=l,n
              a(i,k)=a(i,k)/scale
              s=s+a(i,k)*a(i,k)
18          continue
            f=a(i,l)
            g=-sign(sqrt(s),f)
            h=f*g-s
            a(i,l)=f-g
            do 19 k=l,n
              rv1(k)=a(i,k)/h
19          continue
            do 23 j=l,m
              s=0.0
              do 21 k=l,n
                s=s+a(j,k)*a(i,k)
21            continue
              do 22 k=l,n
                a(j,k)=a(j,k)+s*rv1(k)
22            continue
23          continue
            do 24 k=l,n
              a(i,k)=scale*a(i,k)
24          continue
          endif
        endif
        anorm=max(anorm,(abs(w(i))+abs(rv1(i))))
25    continue
      do 32 i=n,1,-1
        if(i.lt.n)then
          if(g.ne.0.0)then
            do 26 j=l,n
              v(j,i)=(a(i,j)/a(i,l))/g
26          continue
            do 29 j=l,n
              s=0.0
              do 27 k=l,n
                s=s+a(i,k)*v(k,j)
27            continue
              do 28 k=l,n
                v(k,j)=v(k,j)+s*v(k,i)
28            continue
29          continue
          endif
          do 31 j=l,n
            v(i,j)=0.0
            v(j,i)=0.0
31        continue
        endif
        v(i,i)=1.0
        g=rv1(i)
        l=i
32    continue
      do 39 i=min(m,n),1,-1
        l=i+1
        g=w(i)
        do 33 j=l,n
          a(i,j)=0.0
33      continue
        if(g.ne.0.0)then
          g=1.0/g
          do 36 j=l,n
            s=0.0
            do 34 k=l,m
              s=s+a(k,i)*a(k,j)
34          continue
            f=(s/a(i,i))*g
            do 35 k=i,m
              a(k,j)=a(k,j)+f*a(k,i)
35          continue
36        continue
          do 37 j=i,m
            a(j,i)=a(j,i)*g
37        continue
        else
          do 38 j= i,m
            a(j,i)=0.0
38        continue
        endif
        a(i,i)=a(i,i)+1.0
39    continue
      do 49 k=n,1,-1
        do 48 its=1,30
          do 41 l=k,1,-1
            nm=l-1
            if((abs(rv1(l))+anorm).eq.anorm)  goto 2
            if((abs(w(nm))+anorm).eq.anorm)  goto 1
41        continue
1         c=0.0
          s=1.0
          do 43 i=l,k
            f=s*rv1(i)
            rv1(i)=c*rv1(i)
            if((abs(f)+anorm).eq.anorm) goto 2
            g=w(i)
            h=pythag(f,g)
            w(i)=h
            h=1.0/h
            c= (g*h)
            s=-(f*h)
            do 42 j=1,m
              y=a(j,nm)
              z=a(j,i)
              a(j,nm)=(y*c)+(z*s)
              a(j,i)=-(y*s)+(z*c)
42          continue
43        continue
2         z=w(k)
          if(l.eq.k)then
            if(z.lt.0.0)then
              w(k)=-z
              do 44 j=1,n
                v(j,k)=-v(j,k)
44            continue
            endif
            goto 3
          endif
          if(its.eq.30) pause 'no convergence in svdcmp'
          x=w(l)
          nm=k-1
          y=w(nm)
          g=rv1(nm)
          h=rv1(k)
          f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y)
          g=pythag(f,r_one)
          f=((x-z)*(x+z)+h*((y/(f+sign(g,f)))-h))/x
          c=1.0
          s=1.0
          do 47 j=l,nm
            i=j+1
            g=rv1(i)
            y=w(i)
            h=s*g
            g=c*g
            z=pythag(f,h)
            rv1(j)=z
            c=f/z
            s=h/z
            f= (x*c)+(g*s)
            g=-(x*s)+(g*c)
            h=y*s
            y=y*c
            do 45 jj=1,n
              x=v(jj,j)
              z=v(jj,i)
              v(jj,j)= (x*c)+(z*s)
              v(jj,i)=-(x*s)+(z*c)
45          continue
            z=pythag(f,h)
            w(j)=z
            if(z.ne.0.0)then
              z=1.0/z
              c=f*z
              s=h*z
            endif
            f= (c*g)+(s*y)
            x=-(s*g)+(c*y)
            do 46 jj=1,m
              y=a(jj,j)
              z=a(jj,i)
              a(jj,j)= (y*c)+(z*s)
              a(jj,i)=-(y*s)+(z*c)
46          continue
47        continue
          rv1(l)=0.0
          rv1(k)=f
          w(k)=x
48      continue
3       continue
49    continue
      return
      END

      REAL*8 FUNCTION pythag(a,b)
      REAL*8 a,b
      REAL*8 absa,absb
      absa=abs(a)
      absb=abs(b)
      if(absa.gt.absb)then
        pythag=absa*sqrt(1.d0+(absb/absa)**2)
      else
        if(absb.eq.0.)then
          pythag=0.
        else
          pythag=absb*sqrt(1.d0+(absa/absb)**2)
        endif
      endif
      return
      END

      SUBROUTINE svbksb(u,w,v,m,n,mp,np,b,x)
      INTEGER m,mp,n,np,NMAX
      REAL*8 b(mp),u(mp,np),v(np,np),w(np),x(np)
      PARAMETER (NMAX=500)
      INTEGER i,j,jj
      REAL*8 s,tmp(NMAX)
      do 12 j=1,n
        s=0.
        if(w(j).ne.0.)then
          do 11 i=1,m
            s=s+u(i,j)*b(i)
11        continue
          s=s/w(j)
        endif
        tmp(j)=s
12    continue
      do 14 j=1,n
        s=0.
        do 13 jj=1,n
          s=s+v(j,jj)*tmp(jj)
13      continue
        x(j)=s
14    continue
      return
      END

      SUBROUTINE svdvar(v,ma,np,w,cvm,ncvm)
      INTEGER ma,ncvm,np,MMAX
      REAL*8 cvm(ncvm,ncvm),v(np,np),w(np)
      PARAMETER (MMAX=20)
      INTEGER i,j,k
      REAL*8 sum,wti(MMAX)
      do 11 i=1,ma
        wti(i)=0.
        if(w(i).ne.0.) wti(i)=1.d0/(w(i)*w(i))
11    continue
      do 14 i=1,ma
        do 13 j=1,i
          sum=0.
          do 12 k=1,ma
            sum=sum+v(i,k)*v(j,k)*wti(k)
12        continue
          cvm(i,j)=sum
          cvm(j,i)=sum
13      continue
14    continue
      return
      END

c Modify Numerical Recipes program moment.f to compute only 
c standard deviation and allow double precision
      SUBROUTINE moment(data,p,sdev)
      Implicit None
      INTEGER p
      REAL*8 adev,ave,curt,sdev,skew,var,data(p)
      INTEGER j
      REAL*8 t,s,ep
      if(p.le.1)pause 'p must be at least 2 in moment'
      s=0.0d0
      do 11 j=1,p
        s=s+data(j)
11    continue
      ave=s/p
      adev=0.0d0
      var=0.0d0
      skew=0.0d0
      curt=0.0d0
      ep=0.
      do 12 j=1,p
        s=data(j)-ave
        t=s*s
        var=var+t
12    continue
      adev=adev/p
      var=(var-ep**2/p)/(p-1)
      sdev=sqrt(var)
      return
      END

c This program is used to find the rotation matrix from the affine matrix
      SUBROUTINE qrdcmp(a,n,np,c,d,sing)
      INTEGER n,np
      REAL*8 a(np,np),c(n),d(n)
      LOGICAL sing
      INTEGER i,j,k
      REAL*8 scale,sigma,sum,tau
      sing=.false.
      scale=0.
      do 17 k=1,n-1
        do 11 i=k,n
          scale=max(scale,abs(a(i,k)))
11      continue
        if(scale.eq.0.)then
          sing=.true.
          c(k)=0.
          d(k)=0.
        else
          do 12 i=k,n
            a(i,k)=a(i,k)/scale
12        continue
          sum=0.
          do 13 i=k,n
            sum=sum+a(i,k)**2
13        continue
          sigma=sign(sqrt(sum),a(k,k))
          a(k,k)=a(k,k)+sigma
          c(k)=sigma*a(k,k)
          d(k)=-scale*sigma
          do 16 j=k+1,n
            sum=0.
            do 14 i=k,n
              sum=sum+a(i,k)*a(i,j)
14          continue
            tau=sum/c(k)
            do 15 i=k,n
              a(i,j)=a(i,j)-tau*a(i,k)
15          continue
16        continue
        endif
17    continue
      d(n)=a(n,n)
      if(d(n).eq.0.)sing=.true.
      return
      END
C  (C) Copr. 1986-92 Numerical Recipes Software $23#1yR.3Z9.
