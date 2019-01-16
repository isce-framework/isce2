//////////////////////////////////////
// Cunren Liang, NASA JPL/Caltech
// Copyright 2017
//////////////////////////////////////


#include "resamp.h"

// complex operations
fcomplex cmul(fcomplex a, fcomplex b)
{
  fcomplex c;
  c.re=a.re*b.re-a.im*b.im;
  c.im=a.im*b.re+a.re*b.im;
  return c;
}

fcomplex cconj(fcomplex z)
{
  fcomplex c;
  c.re=z.re;
  c.im = -z.im;
  return c;
}

fcomplex cadd(fcomplex a, fcomplex b)
{
	fcomplex c;
	c.re=a.re+b.re;
	c.im=a.im+b.im;
	return c;
}

float xcabs(fcomplex z)
{
  float x,y,ans,temp;
  x=fabs(z.re);
  y=fabs(z.im);
  if (x == 0.0)
    ans=y;
  else if (y == 0.0)
    ans=x;
  else if (x > y) {
    temp=y/x;
    ans=x*sqrt(1.0+temp*temp);
  } else {
    temp=x/y;
    ans=y*sqrt(1.0+temp*temp);
  }
  return ans;
}

float cphs(fcomplex z){
  float ans;
  
  if(z.re == 0.0 && z.im == 0.0)
    ans = 0.0;
  else
    ans = atan2(z.im, z.re);

  return ans;
//it seems that there is no need to add the if clause
//do a test:
//  printf("%12.4f, %12.4f, %12.4f, %12.4f, %12.4f\n", \
//    atan2(0.0, 1.0), atan2(1.0, 0.0), atan2(0.0, -1.0), atan2(-1.0, 0.0), atan2(0.0, 0.0));
//output:
//      0.0000,       1.5708,       3.1416,      -1.5708,       0.0000
}




