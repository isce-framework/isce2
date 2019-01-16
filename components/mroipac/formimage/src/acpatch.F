      subroutine ACpatch(trans,nnn,nl,r0,delr,
     1     wavl,vel,fd,fdd,fddd,fdddd,prf,ht,re,
     2     gm,r_platvel,r_platacc,i_lrl,
     3     npfin,a2,a4,slope,inter,ideskew,na_valid)
      
      implicit none
      REAL*8       PI,PI2
      integer nnn,nl,ideskew,npfin,na_valid
      real*4 vel, prf,a2,a4, y2, phi, dx, veleff
      real*8 r0, delr,wavl,fd,fdd,fddd, fdddd, ht,slope,inter
      real*8 r(nl), phase, az, re, gm, th, thaz, sinsqref, acc, dot
      real*8 r_platvel(3), r_platacc(3), r_lookvec(3), r_vdotl, r_adotl,r_veln
      integer*4 i, j, k, i_lrl
      integer*4 n, nfc, nf0
      complex*8 trans(nnn,nl),ref(nnn)
      real*4 t, scl
      real*4 a2p, y(nl),f0(nl),f_rate(nl), sinsq
      integer*4  np(nl)
      
!c note: - on y1 because chirp is conjugated below

!c both ref and trans are forward transformed and need scaling down

        pi=4.d0*atan2(1.d0,1.d0)
        pi2=2.d0*pi

      scl = 1./float(nnn)**2

      a2p = a2
      if(ideskew .eq. 0) a2p = 0.

      dx = vel/prf

      acc = gm/(re+ht)**2

      !!      call norm(r_platvel,r_veln)
      r_veln = sqrt(r_platvel(1)**2 + r_platvel(2)**2 + r_platvel(3)**2)
      
      do i = 1 , nl
         r(i) = r0 + float(i-1)*delr
         f0(i) = fd + ( fdd + ( fddd+fdddd*r(i) ) *r(i) )*r(i)
         th=dacos(((ht+re)**2+r(i)*r(i)-re**2)/(2.d0*r(i)*(re+ht)))
         if(i_lrl .eq. 0) then
            sinsqref = f0(i) * wavl/(2.d0*vel*sqrt(re/(re+ht))*sin(th))
            f_rate(i) = (2.d0/wavl)*(acc*cos(th)+((vel*sinsqref)**2-vel**2)/r(i))
            veleff = sqrt(abs(acc*cos(th)*r(i)+(vel*sinsqref)**2-vel**2))
         else
c replace with an even more exact expression for chirp rate
            thaz = asin(((wavl*f0(i)/(2.d0*sin(th)))+(r_platvel(3)/tan(th)))/
     $           sqrt(r_platvel(1)**2+r_platvel(2)**2))-i_lrl*atan(r_platvel(2)/r_platvel(1))
            r_lookvec(1) =  sin(th)*sin(thaz)
            r_lookvec(2) =  sin(th)*cos(thaz)*i_lrl
            r_lookvec(3) = -cos(th)

!!            r_vdotl = dot(r_lookvec,r_platvel)
!!            r_adotl = dot(r_lookvec,r_platacc)

            r_vdotl = r_lookvec(1)*r_platvel(1) + r_lookvec(2)*r_platvel(2) + r_lookvec(3)*r_platvel(3)

            r_adotl = r_lookvec(1)*r_platacc(1) + r_lookvec(2)*r_platacc(2) + r_lookvec(3) * r_platacc(3)
            
            f_rate(i) = 2.d0*(r_adotl + (r_vdotl**2 - r_veln**2)/r(i))/(wavl)
            veleff    = sqrt(-(r_adotl*r(i) + r_vdotl**2 - r_veln**2))
         end if
         np(i) = int(r(i)*a4)/2
         phi = 0.
         if(ht .lt. r(i)) phi = acos(ht/r(i))
         az = slope * r(i) + inter
         y2 = pi2 * az / float(nnn)
         sinsq = wavl*f0(i)/2./veleff
c         y(i) =  r(i) * a2p * sinsq + y2
         if(ideskew.eq.1) then
            a2p = -pi2*prf/float(nnn)/veleff
            y(i) =  r(i) * a2p * sinsq + y2
         else
            y(i) = y2
        endif

!c  zero out ref
         do  j = 1, nnn
            ref(j) = cmplx(0.,0.)
         end do
!c  create reference function
!c         phase = pi * f0(i)**2 /f_rate(i)
         phase = 0.d0
         ref(1) = cmplx(cos(phase),sin(phase))*scl
         do  j = 1, np(i)
            t = float(j)/prf
            phase = pi * f_rate(i)*t*t + pi2*f0(i)*t
            ref(j+1) = cmplx(cos(phase),sin(phase))*scl
            phase = pi * f_rate(i)*t*t - pi2*f0(i)*t
            ref(-j+nnn+1) = cmplx(cos(phase),sin(phase))*scl
         end do

!c  transform the reference
         call cfft1d_jpl(nnn,ref,-1)

!c  multiply the reference by the data
         n = nint(f0(i)/prf)
         nf0 = nnn*(f0(i)-n*prf)/prf
         nfc = nf0 + nnn/2
         if(nfc .gt. nnn) nfc = nfc - nnn
         
         phase = - y(i) * nf0
         do k = 1, nfc
            trans(k,i)=trans(k,i)*conjg(ref(k))*cmplx(cos(phase),sin(phase))
            phase = phase + y(i)
         end do
           
         phase = - y(i) * (nf0+1)
         do k = nnn, nfc+1,-1
            trans(k,i)=trans(k,i)*conjg(ref(k))*cmplx(cos(phase),sin(phase))
            phase = phase - y(i)
         end do
!c  inverse transform the product
         call cfft1d_jpl(nnn,trans(1,i),1)

      end do

      return
      end subroutine acpatch
