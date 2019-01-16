#include "image_sio.h"
#include "lib_functions.h"

void polyfit(double *T, double *Y, double *C, int *Mp, int *Np)

/* T   - array of independent variable of length M - input */
/* Y   - array of   dependent variable of length M - input */
/* C   - array of polynomial coefficients length N - output */
/*       FORTRAN callable */
   
{
	double **A, *B;
	int i,j,k;
        int M,N;

	M=*Mp;
	N=*Np;
	if(N > M) {
	  printf(" underdetermined system \n");
	  exit(-1);
	}

/* malloc the memory for A, and B */

	if((A=(double **) malloc(N*sizeof(double *))) == NULL){
	  fprintf(stderr,"Sorry, couldn't allocate memory for A-matrix.\n");
	  exit(-1);
	}
	for(i=0;i<N;i++){
	  if((A[i]=(double *) malloc(N*sizeof(double)))==NULL){
	  fprintf(stderr,"sorry, couldn't allocate memory for A-matrix.\n");
	  exit(-1);
          }
	}
	if((B=(double *) malloc(N*sizeof(double))) == NULL){
	  fprintf(stderr,"Sorry, couldn't allocate memory for B-vector.\n");
	  exit(-1);
	}

/* zero all the arrays */

  	for (i = 0; i < N; i++) {
	  B[i]=0.0;
      	  C[i]=0.0;
      	  for (j = 0; j < N; j++) A[i][j]=0.0;
	}


/* set up A and B for polynomial fit of order N */

  	for (j = 0; j < N; j++) {
      	  for (k = 0; k < M; k++) {
		B[j]=B[j]+Y[k]* pow (T[k], j);
	  }
	}

  	for (i = 0; i < N; i++) {
      	  for (j = 0; j < N; j++) {
	    for (k = 0; k < M; k++) A[i][j]=A[i][j] + pow (T[k], j + i);
	  }
	}

	gauss_jordan(A,B,C,&N);  /* solve the equations */

	for(i=0;i<N;i++) {
		free(A[i]);
	}
	free(A);
	free(B);
}

void gauss_jordan(double **A, double *B, double *X, int *Np) 
{
/* routine for solving an N by N system of linear equations B = A*X 
   using Gaussian elimination with back substitution.
   FORTRAN callable */

	double temp, factor, sum;
	int m,u,p;
	int j,k,l,N,N0;
        N=*Np;
        N0=N-1;
	for (k = 0; k < N; k++) {
	  m = k;
	  for (l = m + 1; l < N; l++) {
	    if (*(*(A+k)+m) != 0.0) {
	      factor =*(*(A+k)+l) / *(*(A+k)+m);
/* perform row operation on A */
	      for (j = 0; j < N; j++) *(*(A+j)+l) = *(*(A+j)+l) - factor * (*(*(A+j)+m));
/* perform row operation on B */
	      *(B+l) = *(B+l) - factor * (*(B+m));
	    }
	  }
	  for (j = 0; j < N; j++) {
	    temp = *(*(A+j)+k);
	    *(*(A+j)+k) = *(*(A+j)+m);
	    *(*(A+j)+m) = temp;
	  }
	  temp = *(B+k);
	  *(B+k) = *(B+m);
	  *(B+m) = temp; 
	}
/* back substitute to construct solution vector X */
	*(X+N0) = *(B+N0) / (*(*(A+N0)+N0));
	for (p = 0; p < N; p++) {
	  sum = 0.0;
	  for (u = N0 - p + 1; u < N; u++)
	    sum = sum + (*(*(A+u)+N0-p)) * (*(X+u));
	    *(X+N0-p) = (*(B+N0-p) - sum) / (*(*(A+N0-p)+N0-p));
	}
}
