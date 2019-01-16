#include <stdio.h>


void 
orbitHermite(double x[][3], double v[][3], double *t, double time, double *xx, double *vv)
{
    int i,j,k;
    double h[4],hdot[4],f0[4],f1[4],g0[4],g1[4],sum,product;
    int n1,n2;
    n1 = 4;
    n2 = 3;
    sum = 0;
    product = 0;
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
        sum = 0.0;
        for(j = 0; j < n1; ++j)
        {
            if(i != j )
            {
                sum += 1.0/(t[i] - t[j]);
            }
        }
        f0[i] = 1.0 - 2.0*(time - t[i])*sum;

    }
    for(i = 0; i < n1; ++i)
    {
        product = 1.0;
        for(k = 0; k < n1; ++k)
        {
            if(k != i)
            {
                product *= (time - t[k])/(t[i] - t[k]);
            }
        }
        h[i] = product;
        sum = 0.0;
        for(j = 0; j < n1; ++j)
        {
            product = 1.0;
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
        g1[i] = h[i] + 2.0*(time - t[i])*hdot[i];
        sum = 0.0;
        for(j = 0; j < n1; ++j)
        {
            if(i != j)
            {
                sum += 1.0/(t[i] - t[j]);
            }
        }
        g0[i] = 2.0*(f0[i]*hdot[i] - h[i]*sum);
    }
    for(k = 0; k < n2; ++k)
    {
        sum = 0.0;
        for(i = 0; i < n1; ++i)
        {
            sum += (x[i][k]*f0[i] + v[i][k]*f1[i])*h[i]*h[i];
        }
        xx[k] = sum;
        sum = 0.0;
        for(i = 0; i < n1; ++i)
        {
            sum += (x[i][k]*g0[i] + v[i][k]*g1[i])*h[i];
        }
        vv[k] = sum;
    }
}
