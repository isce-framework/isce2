#include <stdio.h>

void orbithermite_(double *x, double *v,double *t,double *time,double *xx,double *vv);

int
orbitHermite_C(double *x, double *v, double *t, double * ptime, double *xx, double *vv)
{
	int i,j,k;
    double h[4],hdot[4],f0[4],f1[4],g0[4],g1[4],sum,product,time;
    int n1,n2;
    time = (*ptime);
    n1 = 4;
    n2 = 3;
    sum = 0;
    product = 0;
    printf("did it \n");
    for(i = 0; i < n1; ++i)
    {
        h[i] = 0;
        hdot[i] = 0;
        f0[i] = 0;
        f1[i] = 0;
        g0[i] = 0;
        g1[i] = 0;
    }
    for(i = 0; i < n1; ++i)
    {
        f1[i] = time - t[i];
        sum = 0;
        for(j = 0; j < n1; ++j)
        {
            if(i != j )
            {
                sum += 1./(t[i] - t[j]);
            }
        }
        f0[i] = 1 - 2.0*(time - t[i])*sum;

    }
    for(i = 0; i < n1; ++i)
    {
        product = 1;
        for(k = 0; k < n1; ++k)
        {
            if(k != i)
            {
                product *= (time - t[k])/(t[i] - t[k]);
            }
        }
        h[i] = product;
        sum = 0;
        for(j = 0; j < n1; ++j)
        {
            product = 1;
            for(k = 0; k < n1; ++k)
            {
                if((k != i) && (k != j))
                {
                    product *= (time - t[k])/(t[i] - t[k]);
                }
            }
            if(j != i)
            {
                sum += 1.0/(t[i] - t[j])*product;
            }
        }
        hdot[i] = sum;
    }
    for(i = 0; i < n1; ++i)
    {
        g1[i] = h[i] + 2*(time - t[i])*hdot[i];
        sum = 0;
        for(j = 0; j < n1; ++j)
        {
            if(i != j)
            {
                sum += 1.0/(t[i] - t[j]);
            }
        }
        g0[i] = 2*(f0[i]*hdot[i] - h[i]*sum);
    }
    for(k = 0; k < n2; ++k)
    {
        sum = 0;
        for(i = 0; i < n1; ++i)
        {
            sum += (x[k+ i*n2]*f0[i] + v[k + i*n2]*f1[i])*h[i]*h[i];
        }
        xx[k] = sum;
        sum = 0;
        for(i = 0; i < n1; ++i)
        {
            sum += (x[k+ i*n2]*g0[i] + v[k + i*n2]*g1[i])*h[i];
        }
        vv[k] = sum;
    }

    // x and v are in row major order, which is OK since the matrices expected by orbithermite_() are the transpose
    // of what you would expect
	// xx and vv are the outputs
    /*
    for(i = 0; i < 3*4; ++i)
    {
        printf("%f %f %d \n",x[i],v[i],i);
    } 
    exit(1);
	orbithermite_(x,v,t,ptime,xx,vv);
    */
	return 0;
}
