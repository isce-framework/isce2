      subroutine svdfit(x,y,z,sig,ndata,a,ma,u,v,w,mp,np,chisq)
      implicit real*8 (a-h,o-z)
      parameter(nmax=327680,mmax=10,tol=1.e-12)
      dimension x(ndata),y(ndata),z(ndata),sig(ndata),a(ma),v(np,np),
     *    u(mp,np),w(np),b(nmax),afunc(mmax)
c      type *,'evaluating basis functions...'
      do 12 i=1,ndata
        call poly_funcs(x(i),y(i),afunc,ma)
        tmp=1./sig(i)
        do 11 j=1,ma
          u(i,j)=afunc(j)*tmp
11      continue
        b(i)=z(i)*tmp
12    continue
c      type *,'SVD...'
      call svdcmp(u,ndata,ma,mp,np,w,v)
      wmax=0.
      do 13 j=1,ma
        if(w(j).gt.wmax)wmax=w(j)
13    continue
      thresh=tol*wmax
c       type *,'eigen value threshold',thresh
      do 14 j=1,ma
c       type *,j,w(j)
        if(w(j).lt.thresh)w(j)=0.
14    continue
c      type *,'calculating coefficients...'
      call svbksb(u,w,v,ndata,ma,mp,np,b,a)
      chisq=0.
c      type *,'evaluating chi square...'
      do 16 i=1,ndata
        call poly_funcs(x(i),y(i),afunc,ma)
        sum=0.
        do 15 j=1,ma
          sum=sum+a(j)*afunc(j)
15      continue
        chisq=chisq+((z(i)-sum)/sig(i))**2
16    continue
      return
      end


        subroutine doppler(n_ra,l1,l2,image1,f_d,dbuf)

        use fortranUtils

        implicit none
        integer       n_ra
        complex*8     image1(N_RA,*)
        integer*4     ia,ir,i,j,jj,l1,l2
        real*4        wgth
        real*4        f_est
        real*4        f_d(N_RA)
        complex*8     dbuf(N_RA)
        integer*4     rinc
        real*8 pi 

        write(6,*) ' '
        write(6,*) ' doppler estimation as a function of range :'

        pi = getPi()

        rinc = nint(float(n_ra)/n_ra)

cc  Doppler estimation

        do i = 1,n_ra
           dbuf(i) = (0.0,0.0)
        enddo
        do ia=l1+1,l2-1
c         wgth = abs(sin(pi*ia/float(2*(l2-l1))))
          wgth = 1.0
          do ir = rinc+2,n_ra-2,rinc
            jj = ir/rinc
            do j = ir-rinc+1-2,ir-rinc+1+2
            dbuf(jj) = dbuf(jj)
     2                + wgth*image1(j,ia)*conjg(image1(j,ia-1))
            enddo       ! j-loop
          enddo ! ir-loop
        enddo   ! ia-loop

c Doppler ambiguity resolution 

        do jj = rinc+2,n_ra-2
          f_est  = atan2(aimag(dbuf(jj)),real(dbuf(jj)))/(2.d0*pi)
          if(jj .ne. rinc+2)then
             if(abs(f_est-f_d(jj-1)) .gt. .5)then
                f_est = f_est + sign(1.0,f_d(jj-1)-f_est)
             endif
          endif
          f_d(jj)= f_est
        end do
        f_d(1) = f_d(3)
        f_d(2) = f_d(3)
        f_d(n_ra-1) = f_d(n_ra-2)
        f_d(n_ra) = f_d(n_ra-2)

        return
        end
        
      subroutine covsrt(covar,ncvm,ma,lista,mfit)
      implicit real*8 (a-h,o-z)
      dimension covar(ncvm,ncvm),lista(mfit)
      do 12 j=1,ma-1
        do 11 i=j+1,ma
          covar(i,j)=0.
11      continue
12    continue
      do 14 i=1,mfit-1
        do 13 j=i+1,mfit
          if(lista(j).gt.lista(i)) then
            covar(lista(j),lista(i))=covar(i,j)
          else
            covar(lista(i),lista(j))=covar(i,j)
          endif
13      continue
14    continue
      swap=covar(1,1)
      do 15 j=1,ma
        covar(1,j)=covar(j,j)
        covar(j,j)=0.
15    continue
      covar(lista(1),lista(1))=swap
      do 16 j=2,mfit
        covar(lista(j),lista(j))=covar(1,j)
16    continue
      do 18 j=2,ma
        do 17 i=1,j-1
          covar(i,j)=covar(j,i)
17      continue
18    continue
      return
      end
