      SUBROUTINE LFIT(X,Y,SIG,NDATA,A,MA,LISTA,MFIT,COVAR,NCVM,CHISQ)
      PARAMETER (MMAX=50)
      implicit real*8 (a-h,o-z)
      real*8 x(*)
      real*8 sig(*),y(*) 
      DIMENSION A(MA),LISTA(MA),
     *    COVAR(NCVM,NCVM),BETA(MMAX),AFUNC(MMAX)

      KK=MFIT+1
      DO 12 J=1,MA
        IHIT=0
        DO 11 K=1,MFIT
          IF (LISTA(K).EQ.J) IHIT=IHIT+1
11      CONTINUE
        IF (IHIT.EQ.0) THEN
          LISTA(KK)=J
          KK=KK+1
        ELSE IF (IHIT.GT.1) THEN
          PAUSE 'Improper set in LISTA'
        ENDIF
12    CONTINUE
      IF (KK.NE.(MA+1)) PAUSE 'Improper set in LISTA'
      DO 14 J=1,MFIT
        DO 13 K=1,MFIT
          COVAR(J,K)=0.
13      CONTINUE
        BETA(J)=0.
14    CONTINUE
      DO 18 I=1,NDATA
        CALL FUNCS(X(I),AFUNC,MA)
        YM=Y(I)
        IF(MFIT.LT.MA) THEN
          DO 15 J=MFIT+1,MA
            YM=YM-A(LISTA(J))*AFUNC(LISTA(J))
15        CONTINUE
        ENDIF
        SIG2I=1./SIG(I)**2
        DO 17 J=1,MFIT
          WT=AFUNC(LISTA(J))*SIG2I
          DO 16 K=1,J
            COVAR(J,K)=COVAR(J,K)+WT*AFUNC(LISTA(K))
16        CONTINUE
          BETA(J)=BETA(J)+YM*WT
17      CONTINUE
18    CONTINUE
      IF (MFIT.GT.1) THEN
        DO 21 J=2,MFIT
          DO 19 K=1,J-1
            COVAR(K,J)=COVAR(J,K)
19        CONTINUE
21      CONTINUE
      ENDIF
      CALL GAUSSJ(COVAR,MFIT,NCVM,BETA,1,1)
      DO 22 J=1,MFIT
        A(LISTA(J))=BETA(J)
22    CONTINUE
      CHISQ=0.
      DO 24 I=1,NDATA
        CALL FUNCS(X(I),AFUNC,MA)
        SUM=0.
        DO 23 J=1,MA
          SUM=SUM+A(J)*AFUNC(J)
23      CONTINUE
        CHISQ=CHISQ+((Y(I)-SUM)/SIG(I))**2
24    CONTINUE
c      CALL COVSRT(COVAR,NCVM,MA,LISTA,MFIT)
      RETURN
      END






















