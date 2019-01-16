#include "linalg3.h"
#include <stdio.h>
#include <math.h>

void zeroMat(double a[3][3])
{
    int i,j;
    for(j=0;j<3;j++)
        for(i=0;i<3;i++)
            a[j][i] = 0.0;
}

void zeroVec(double a[3])
{
    int i;
    for(i=0;i<3;i++)
        a[i] = 0.0;
}

void eye(double a[3][3])
{
    int i;
    zeroMat(a);
    for(i=0; i<3; i++)
        a[i][i] = 1.0;
}

//C-matrices should be treated as treated as transposed.
void printMat(double a[3][3])
{
    int i,j;
    printf("Matrix: \n");
    for(i=0;i<3;i++)
    {
        for(j=0;j<3;j++)
            printf("%f\t",a[i][j]);

        printf("\n");
    }
}

void printVec(double a[3])
{
    int i;
    printf("Vector: \n");
    for(i=0; i<3; i++)
        printf("%f\t", a[i]);

    printf("\n");
}

void printScalar(double k)
{
    printf("Scalar : \n");
    printf("%f \n", k);
}

int main()
{
    double A[3][3];
    double B[3][3];
    double C[3][3];
    double c[3];
    double d[3];
    double e[3];
    double k1,k2,k3;

    //Init
    zeroMat(A);
    zeroMat(B);
    zeroVec(c);
    zeroVec(d);
    zeroVec(e);

    //Testing cross
    printf("Testing Cross_C \n");
    printf("Test 1\n");
    c[0]=1.0; c[1]=0.0; c[2]=0.0;
    d[0]=0.0; d[1]=1.0; d[2]=0.0;
    cross_C(c,d,e);
    printVec(c);
    printf("  x  \n");
    printVec(d);
    printf(" = \n");
    printVec(e);

    printf("Test 2 \n");
    c[0]=0.0; c[1]=0.0; c[2]=1.0;
    d[0]=0.0; d[1]=1.0; d[2]=0.0;
    cross_C(c,d,e);
    printVec(c);
    printf("  .cross.  \n");
    printVec(d);
    printf(" = \n");
    printVec(e); 


    printf("Testing dot_C \n");
    printf("Test 1\n");
    c[0]=0.0; c[1]=0.0; c[2]=1.0;
    d[0]=0.0; d[1]=1.0; d[2]=0.0;
    k1 = dot_C(c,d);
    printVec(c);
    printf(" .dot. \n");
    printVec(d);
    printf(" = \n");
    printScalar(k1);

    printf("Test 2\n");
    c[0]=1.0; c[1]=1.0; c[2]=1.0;
    d[0]=3.0; d[1]=1.0; d[2]=2.0;
    k1 = dot_C(c,d);
    printVec(c);
    printf(" .dot. \n");
    printVec(d);
    printf(" = \n");
    printScalar(k1);

    printf("Testing norm_C\n");
    printf("Test 1\n");
    c[0] = 0.0; c[1] = -1.0; c[2]=0.0;
    k1 = norm_C(c);
    printVec(c);
    printf(".norm. = \n");
    printScalar(k1);
   
    printf("Test 2\n");
    c[0] = 1.0; c[1] = -1.0; c[2]=-1.0;
    k1 = norm_C(c);
    printVec(c);
    printf(".norm. = \n");
    printScalar(k1);

    printf("Testing unitvec_C\n");
    printf("Test 1\n");
    c[0]=1.0; c[1]=-1.0; c[2]=-1.0;
    unitvec_C(c,d);
    printVec(c);
    printf(".unit. = \n");
    printVec(d);

    printf("Test 2\n");
    c[0]=1.0; c[1]=0.0; c[2]=0.0;
    unitvec_C(c,d);
    printVec(c);
    printf(".unit. = \n");
    printVec(d);

    printf("Testing tranmat_C \n");
    printf("Test 1 \n");
    A[0][0]=0.; A[0][1] = 1.; A[0][2] = 2.;
    A[1][0]=3.; A[1][1] = 4.; A[1][2] = 5.;
    A[2][0]=6.; A[2][1] = 7.; A[2][2] = 8.;
    tranmat_C(A,B);
    printMat(A);
    printf(" .trans. = \n");
    printMat(B);

    printf("Testing matmat_C \n");
    printf("Test 1 \n");
    A[0][0]=0.; A[0][1] = 1.; A[0][2] = 2.;
    A[1][0]=3.; A[1][1] = 4.; A[1][2] = 5.;
    A[2][0]=6.; A[2][1] = 7.; A[2][2] = 8.;

    eye(B);
    matmat_C(A,B,C);
    printMat(A);
    printf(" .mul. \n");
    printMat(B);
    printf(" =  \n");
    printMat(C);

    printf("Test 2 \n");
    A[0][0]=0.; A[0][1] = 1.; A[0][2] = 2.;
    A[1][0]=3.; A[1][1] = 4.; A[1][2] = 5.;
    A[2][0]=6.; A[2][1] = 7.; A[2][2] = 8.;

    B[0][0]=1.; B[0][1]=1.; B[0][2]=1.;
    B[1][0]=0.; B[1][1]=1.; B[1][2]=1.;
    B[2][0]=0.; B[2][1]=0.; B[2][2]=1.;

    matmat_C(A,B,C);
    printMat(A);
    printf(" .mul. \n");
    printMat(B);
    printf(" =  \n");
    printMat(C);

    printf("Testing matvec_C \n");
    printf("Test 1 \n");
    A[0][0]=0.; A[0][1] = 1.; A[0][2] = 2.;
    A[1][0]=3.; A[1][1] = 4.; A[1][2] = 5.;
    A[2][0]=6.; A[2][1] = 7.; A[2][2] = 8.;

    c[0] = 0.0; c[1] = 0.0; c[2] = 1.0;
    matvec_C(A, c, d);
    printMat(A);
    printf(" .mul. \n");
    printVec(c);
    printf(" = \n");
    printVec(d);

    printf("Test 2 \n");
    A[0][0]=0.; A[0][1] = 1.; A[0][2] = 2.;
    A[1][0]=3.; A[1][1] = 4.; A[1][2] = 5.;
    A[2][0]=6.; A[2][1] = 7.; A[2][2] = 8.;

    c[0] = 1.0; c[1] = 0.0; c[2] = -1.0;
    matvec_C(A, c, d);
    printMat(A);
    printf(" .mul. \n");
    printVec(c);
    printf(" = \n");
    printVec(d);


    return 0;
}

