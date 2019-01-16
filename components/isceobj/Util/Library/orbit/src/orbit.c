#include "orbit.h"

//Local functions
double quadInterpolate(double *x, double *y, double xintp);
void orbitHermite(double x[][3], double v[][3], double *t, double time, double *xx, double *vv);


cOrbit* createOrbit(int nvec, int basis)
{

    cOrbit *newObj = (cOrbit*) malloc(sizeof(cOrbit));
    if(newObj == NULL)
    {
        printf("Not enough memory for orbit object");
    }

    initOrbit(newObj, nvec, basis);
    return newObj;
}

void initOrbit(cOrbit* orb, int nvec, int basis)
{
    orb->nVectors = nvec;
    orb->basis = basis;

    orb->UTCtime = (double*) malloc(sizeof(double)*nvec);
    if(orb->UTCtime == NULL)
    {
        printf("Not enough memory for orbit UTC times");
    }

    orb->position = (double*) malloc(sizeof(double)*nvec*3);
    if(orb->position == NULL)
    {
        printf("Not enough memory for orbit positions");
    }

    orb->velocity = (double*) malloc(sizeof(double)*nvec*3);
    if(orb->velocity == NULL)
    {
        printf("Not enough memory for orbit velocity");
    }

}

void cleanOrbit(cOrbit *orb)
{

    if (orb->UTCtime != NULL)
    {
        free((char*) (orb->UTCtime));
    }

    if (orb->position != NULL)
    {
        free((char*) (orb->position));
    }

    if (orb->velocity != NULL)
    {
        free((char*) (orb->velocity));
    }

    strcpy(orb->yyyymmdd, "");
    orb->nVectors = 0;
}

void deleteOrbit(cOrbit *orb)
{
     cleanOrbit(orb);
     free((char*) orb);
}


void getPositionVelocity(cOrbit* orb, double tintp, double *pos, double *vel)
{
    if (orb->basis == WGS84_ORBIT)
        interpolateWGS84Orbit(orb, tintp, pos, vel);
    else
        interpolateSCHOrbit(orb, tintp, pos, vel);
}

void setStateVector(cOrbit* orb, int index, double t, double *pos, double *vel)
{
    int i;
    if((index >= orb->nVectors) || (index < 0))
    {
        printf("Trying to set state vector %d out of %d\n",index, orb->nVectors);
        exit(1);
    }

    orb->UTCtime[index] = t;
    for(i=0;i<3;i++)
        orb->position[3*index+i] = pos[i];

    for(i=0;i<3;i++)
        orb->velocity[3*index+i] = vel[i];

}

void getStateVector(cOrbit* orb, int index, double *t, double *pos, double *vel)
{
    int i;
    if((index >= orb->nVectors) || (index < 0))
    {
        printf("Trying to get state vector %d out of %d \n", index, orb->nVectors);
        exit(1);
    }

    *(t) = orb->UTCtime[index];
    for(i=0;i<3;i++)
        pos[i] = orb->position[3*index+i];

    for(i=0;i<3;i++)
        vel[i] = orb->velocity[3*index+i];

}

int interpolateSCHOrbit(cOrbit* orb, double tintp, double *opos, double *ovel)
{

    int i, j, k;

    double pos[2][3];
    double vel[2][3];
    double t[2];
    double frac,num,den;

    if( orb->nVectors < 2)
    {
        printf("Need atleast 2 state vectors for SCH orbit interpolation. \n");
        return 1;
    }

    if((tintp < orb->UTCtime[0]) || (tintp > orb->UTCtime[orb->nVectors - 1]))
    {
        printf("Requested epoch outside orbit state vector span. \n");
        return 1;
    }

    opos[0] = 0.0;
    opos[1] = 0.0;
    opos[2] = 0.0;
    ovel[0] = 0.0;
    ovel[1] = 0.0;
    ovel[2] = 0.0;

    for(i=0; i<orb->nVectors; i++)
    {
        frac = 1.0;
        getStateVector(orb,i,t,pos[0],vel[0]);
        for(j=0;j<orb->nVectors; j++)
        {
            if (i==j)
                continue;
            getStateVector(orb,j,t+1,pos[1],vel[1]);
            num = t[1]-tintp;
            den = t[1]-t[0];
            
            frac *= num/den;
        }

        for(k=0;k<3;k++)
        {
            opos[k] += frac*pos[0][k];
            ovel[k] += frac*vel[0][k];
        }
    }

    return 0;

}


int interpolateWGS84Orbit(cOrbit* orb, double tintp, double *opos, double *ovel)
{

    int i,j;

    double pos[4][3];
    double vel[4][3];
    double t[4];

    if(orb->nVectors < 4)
    {
//        printf("Need atleast 4 state vectors for Hermite polynomial orbit interpolation. \n");
        return 1;
    }

    // i=0;
    // while(orb->UTCtime[i] < tintp)
    //     i++;

    // i--;
    // if (i >= (orb->nVectors-3))
    //     i = orb->nVectors-4;

    // if (i < 2)
    //     i = 0;

//////////////////////////////////////////////////////////////////////////////

    for(i = 0; i < orb->nVectors; i++){
        if(orb->UTCtime[i] >= tintp)
            break;
    }
    i -= 2;
    if(i < 0)
        i = 0;
    if(i > orb->nVectors - 4)
        i = orb->nVectors - 4;


//////////////////////////////////////////////////////////////////////////////

    for(j=0; j<4; j++)
    {
        getStateVector(orb,i+j,t+j,pos[j],vel[j]);
    }

    orbitHermite(pos, vel, t, tintp, opos, ovel);

//////////////////////////////////////////////////////////////////////////////
    if((tintp < orb->UTCtime[0]) || (tintp > orb->UTCtime[orb->nVectors - 1]))
    {
        //even if this si the case, do the interpolation anyway, return 1 as indication
        //printf("Requested epoch outside the state vector span. \n");
        return 1;
    }
    else{
        return 0;
    }
//////////////////////////////////////////////////////////////////////////////
}

int interpolateLegendreOrbit(cOrbit* orb, double tintp, double *opos, double *ovel)
{

    int i,j;
    double pos[9][3];
    double vel[9][3];
    double t[9];    
    double trel, coeff;

    double noemer[] = { 40320.0, -5040.0, 1440.0, -720.0, 576.0, -720.0, 1440.0, -5040.0, 40320.0};
    double teller;

    //Init to 0.0
    opos[0] = 0.0; opos[1] = 0.0; opos[2] = 0.0;
    ovel[0] = 0.0; ovel[1] = 0.0; ovel[2] = 0.0;

    //Check for number of state vectors
    if(orb->nVectors < 9)
    {
//        printf("Need atleast 9 state vectors for Hermite polynomial orbit interpolation. \n");
        return 1;
    }

    //Search for appropriate interval
    for(i = 0; i < orb->nVectors; i++){
        if(orb->UTCtime[i] >= tintp)
            break;
    }
    i -= 5;
    if(i < 0)
        i = 0;
    if(i > orb->nVectors - 9)
        i = orb->nVectors - 9;

    for(j=0; j<9; j++)
    {
        getStateVector(orb,i+j,t+j,pos[j],vel[j]);
    }

    //Do the actual interpolation
    trel = 8.0*(tintp - t[0])/(t[8]-t[0]);

    //Checks if input time coincides with state vectors
    teller = 1.0;
    for(j=0;j<9;j++)
        teller *= (trel - j);

    if(teller == 0.0)
    {
        i = (int) trel;
        for(j=0; j<3; j++)
        {
            opos[j] = pos[i][j];
            ovel[j] = vel[i][j];
        }
    }
    else
    {
        for(i=0;i<9;i++)
        {
            coeff = teller/noemer[i]/(trel-i);
            for(j=0; j<3; j++)
            {
                opos[j] += coeff * pos[i][j];
                ovel[j] += coeff * vel[i][j];
            }
        }
    }

    if((tintp < orb->UTCtime[0]) || (tintp > orb->UTCtime[orb->nVectors - 1]))
    {
        //even if this si the case, do the interpolation anyway, return 1 as indication
        //printf("Requested epoch outside the state vector span. \n");
        return 1;
    }
    else{
        return 0;
    }
}

int computeAcceleration(cOrbit* orb, double tintp, double *acc)
{
    int i,j;
    int stat;

    double xbef[3];
    double vbef[3];
    double xaft[3];
    double vaft[3];
    double temp;

    for(i=0;i<3;i++)
    {
        acc[i] = 0.0;
    }

    temp = tintp - 0.01;
    stat = interpolateWGS84Orbit(orb, temp, xbef, vbef);

    if (stat != 0)
    {
        return 1;
    }

    temp = tintp + 0.01;
    stat = interpolateWGS84Orbit(orb, temp, xaft, vaft);

    if (stat != 0)
    {
        return 1;
    }

    for (i=0;i<3;i++)
    {
        acc[i] = (vaft[i] - vbef[i])/ 0.02;
    }

    return 0;
}

double quadInterpolate(double *x, double *y, double xintp)
{
    double x1[3], y1[3];
    double a,b,xin;
    double res;
    int i;

    xin = xintp - x[0];
    for(i=0; i<3; i++)
    {
        x1[i] = x[i] - x[0];
        y1[i] = y[i] - y[0];
    }

    a = (-y1[1]*x1[2]+y1[2]*x1[1]) / (-x1[2]*x1[1]*x1[1] + x1[1]*x1[2]*x1[2]);
    b = (y1[1] - a * x1[1]*x1[1])/x1[1];

    res = y[0] + a*xin*xin + b*xin;
    return res;
}


cOrbit* loadFromHDR(const char* filename, int basis)
{
    cOrbit* orb = (cOrbit*) malloc(sizeof(cOrbit));

    //First determine number of lines
    FILE *fp;
    char ch;
    char *line = NULL;
    int  count;
    size_t len;
    int nLines = 0;
    double t, pos[3], vel[3];

    if ((fp = fopen(filename, "r")) == NULL)
    {
        printf("Unable to open HDR file: %s \n", filename);
        exit(1);
    }

    while ((ch = getc(fp)) != EOF)
    {
        if (ch == '\n')
            nLines++;
    }

    rewind(fp);

    initOrbit(orb, nLines, basis); 
    
    count = 0;
    while(getline(&line, &len, fp) != -1)
    {
        sscanf(line, "%lf %lf %lf %lf %lf %lf %lf",&t,pos,pos+1,pos+2,vel,vel+1,vel+2);
        setStateVector(orb, count, t, pos, vel);
        count++;
    }
    printf("Read in %d State Vectors from %s \n", count, filename);
    fclose(fp);

    if (line != NULL)
        free(line);

    return orb;
}


void dumpToHDR(cOrbit* orb, const char* filename)
{

    FILE* fp;
    int i;
    double t, pos[3], vel[3];

    if ((fp = fopen(filename,"w")) == NULL)
    {
        printf("Unable to open HDR file: %s \n", filename);
        exit(1);
    }

    for(i=0; i< orb->nVectors; i++)
    {
        getStateVector(orb, i, &t, pos, vel);
        fprintf(fp, "%+g\t%+g\t%+g\t%+g\t%+g\t%+g\t%+g\n",t,pos[0],pos[1],pos[2],vel[0],vel[1],vel[2]);
    }

    printf("Writing %d vectors to %s \n", orb->nVectors, filename);
    fclose(fp);
}

void printOrbit(cOrbit *orb)
{
    int i;
    double t, pos[3], vel[3];

    for(i=0; i< orb->nVectors; i++)
    {
        getStateVector(orb, i, &t, pos, vel);
        printf("UTC = %lf \n", t);
        printf("Position = [ %lf, %lf, %lf] \n", pos[0], pos[1], pos[2]);
        printf("Velocity = [ %lf, %lf, %lf] \n", vel[0], vel[1], vel[2]);

    }
}

