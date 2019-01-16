//
// Author: Joshua Cohen
// Copyright 2016
//

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include "Constants.h"
#include "Orbit.h"

using std::getline;
using std::ifstream;
using std::ofstream;
using std::showpos;
using std::string;

// Default constructor
Orbit::Orbit() {
    position = NULL;
    velocity = NULL;
    UTCtime = NULL;
    nVectors = 0;
    basis = 0;
}

Orbit::Orbit(const Orbit &orb) {
    nVectors = orb.nVectors;
    basis = orb.basis;
    position = new double[3*nVectors];
    velocity = new double[3*nVectors];
    UTCtime = new double[nVectors];
    for (int i=0; i<nVectors; i++) {
        UTCtime[i] = orb.UTCtime[i];
        position[(3*i)] = orb.position[(3*i)];
        position[(3*i)+1] = orb.position[(3*i)+1];
        position[(3*i)+2] = orb.position[(3*i)+2];
        velocity[(3*i)] = orb.velocity[(3*i)];
        velocity[(3*i)+1] = orb.velocity[(3*i)+1];
        velocity[(3*i)+2] = orb.velocity[(3*i)+2];
    }
}

Orbit::~Orbit() {
    if (position) delete[] position;
    if (velocity) delete[] velocity;
    if (UTCtime) delete[] UTCtime;
}

// Unfortunately due to the way the algorithm works it will set up an Orbit and resize the internal
// vectors later
void Orbit::setOrbit(int nvec, int bs) { 
    if (position) delete[] position;
    if (velocity) delete[] velocity;
    if (UTCtime) delete[] UTCtime;
    position = new double[3*nvec];
    velocity = new double[3*nvec];
    UTCtime = new double[nvec];
    nVectors = nvec;
    basis = bs;
}

// Can set the Orbit by reading in a CSV HDR file (size=(nVec x 7))
void Orbit::setOrbit(const char *filename, int bs) {
    string line;
    double pos[3], vel[3];
    double t;
    int count = 0;

    nVectors = 0;
    basis = bs;
    ifstream fs(filename);
    if (!fs.is_open()) {
        printf("Error in Orbit::Orbit - Unable to open HDR file: %s\n", filename);
        exit(1);
    }

    // Rapid iterator to count number of lines safely
    while (getline(fs,line)) ++nVectors;
    if (position) delete[] position;
    if (velocity) delete[] velocity;
    if (UTCtime) delete[] UTCtime;
    position = new double[3*nVectors];
    velocity = new double[3*nVectors];
    UTCtime = new double[nVectors];

    // Reset filestream before reading lines
    fs.clear();
    fs.seekg(0);

    // Take advantage of filestream overridden >>/<< operators
    while (fs >> t >> pos[0] >> pos[1] >> pos[2] >> vel[0] >> vel[1] >> vel[2]) {
        setStateVector(count,t,pos,vel);
        count++;
    }
    fs.close();
    printf("Read in %d State Vectors from %s\n", nVectors, filename);
}

void Orbit::getPositionVelocity(double tintp, double pos[3], double vel[3]) {
    if (basis == WGS84_ORBIT) interpolateWGS84Orbit(tintp, pos, vel);
    else interpolateSCHOrbit(tintp, pos, vel);
}

void Orbit::setStateVector(int idx, double t, double pos[3], double vel[3]) {
    if ((idx >= nVectors) || (idx < 0)) {
        printf("Error in Orbit::setStateVector - Trying to set state vector %d out of %d\n", idx, nVectors);
        exit(1);
    }
    UTCtime[idx] = t;
    for (int i=0; i<3; i++) {
        position[(3*idx)+i] = pos[i];
        velocity[(3*idx)+i] = vel[i];
    }
}

void Orbit::getStateVector(int idx, double &t, double pos[3], double vel[3]) {
    if ((idx >= nVectors) || (idx < 0)) {
        printf("Error in Orbit::getStateVector - Trying to get state vector %d out of %d\n", idx, nVectors);
        exit(1);
    }
    t = UTCtime[idx];
    for (int i=0; i<3; i++) {
        pos[i] = position[(3*idx)+i];
        vel[i] = velocity[(3*idx)+i];
    }
}

// Common interface for orbit interpolation (avoid setting function pointers in main controller
int Orbit::interpolateOrbit(double tintp, double opos[3], double ovel[3], int method) {
    int ret;

    if (method == HERMITE_METHOD) ret = interpolateWGS84Orbit(tintp,opos,ovel);
    else if (method == SCH_METHOD) ret = interpolateSCHOrbit(tintp,opos,ovel);
    else if (method == LEGENDRE_METHOD) ret = interpolateLegendreOrbit(tintp,opos,ovel);
    else {
        printf("Error in Orbit::interpolateOrbit - Invalid orbit interpolation method.\n");
        exit(1);
    }
    return ret;
}

int Orbit::interpolateSCHOrbit(double tintp, double opos[3], double ovel[3]) {
    double pos[2][3], vel[2][3];
    double t[2];
    double frac,num,den;

    if (nVectors < 2) {
        printf("Error in Orbit::interpolateSCHOrbit - Need at least 2 state vectors for SCH orbit interpolation.\n");
        exit(1);
    }
    if ((tintp < UTCtime[0]) || (tintp > UTCtime[nVectors-1])) {
        printf("Error in Orbit::interpolateSCHOrbit - Requested epoch outside orbit state vector span.\n");
        exit(1);
    }
    for (int i=0; i<3; i++) {
        opos[i] = 0.0;
        ovel[i] = 0.0;
    }
    for (int i=0; i<nVectors; i++) {
        frac = 1.0;
        getStateVector(i,t[0],pos[0],vel[0]);
        for (int j=0; j<nVectors; j++) {
            if (i==j) continue;
            getStateVector(j,t[1],pos[1],vel[1]);
            num = t[1] - tintp;
            den = t[1] - t[0];
            frac = frac * (num / den);
        }
        for (int k=0; k<3; k++) {
            opos[k] = frac * pos[0][k];
            ovel[k] = frac * vel[0][k];
        }
    }
    return 0;
}

int Orbit::interpolateWGS84Orbit(double tintp, double opos[3], double ovel[3]) {
    double pos[4][3], vel[4][3];
    double t[4];
    int ii;

    if (nVectors < 4) return 1;
    for (int i=0; i<nVectors; i++) {
        ii = i;
        if (UTCtime[i] >= tintp) break;
    }
    ii = ii - 2;
    if (ii < 0) ii = 0;
    if (ii > (nVectors - 4)) ii = (nVectors - 4);
    
    for (int j=0; j<4; j++) getStateVector((ii+j),t[j],pos[j],vel[j]);
    orbitHermite(pos,vel,t,tintp,opos,ovel);

    if ((tintp < UTCtime[0]) || (tintp > UTCtime[(nVectors-1)])) return 1;
    else return 0;
}

int Orbit::interpolateLegendreOrbit(double tintp, double opos[3], double ovel[3]) {
    double pos[9][3], vel[9][3];
    double t[9];
    double noemer[] = {40320.0, -5040.0, 1440.0, -720.0, 576.0, -720.0, 1440.0, -5040.0, 40320.0};
    double trel, coeff, teller;
    int ii;

    for (int i=0; i<3; i++) {
        opos[i] = 0.0;
        ovel[i] = 0.0;
    }
    if (nVectors < 9) return 1;
    for (int i=0; i<nVectors; i++) {
        ii = i;
        if (UTCtime[i] >= tintp) break;
    }
    ii = ii - 5;
    if (ii < 0) ii = 0;
    if (ii > (nVectors - 9)) ii = (nVectors - 9);

    for (int j=0; j<9; j++) getStateVector((ii+j),t[j],pos[j],vel[j]);

    trel = (8.0 * (tintp - t[0])) / (t[8] - t[0]);
    teller = 1.0;
    for (int j=0; j<9; j++) teller = teller * (trel - j);

    if (teller == 0.0) {
        int i = int(trel);
        for (int j=0; j<3; j++) {
            opos[j] = pos[i][j];
            ovel[j] = vel[i][j];
        }
    } else {
        for (int i=0; i<9; i++) {
            coeff = (teller / noemer[i]) / (trel - i);
            for (int j=0; j<3; j++) {
                opos[j] = opos[j] + (coeff * pos[i][j]);
                ovel[j] = ovel[j] + (coeff * vel[i][j]);
            }
        }
    }
    if ((tintp < UTCtime[0]) || (tintp > UTCtime[(nVectors-1)])) return 1;
    else return 0;
}

int Orbit::computeAcceleration(double tintp, double acc[3]) {
    double xbef[3], vbef[3], xaft[3], vaft[3];
    double temp;
    int stat;

    for (int i=0; i<3; i++) acc[i] = 0.0;
    temp = tintp - 0.01;
    stat = interpolateWGS84Orbit(temp, xbef, vbef);
    if (stat != 0) return 1;
    temp = tintp + 0.01;
    stat = interpolateWGS84Orbit(temp, xaft, vaft);
    if (stat != 0) return 1;
    for (int i=0; i<3; i++) acc[i] = (vaft[i] - vbef[i]) / 0.02;
    return 0;
}

void Orbit::orbitHermite(double x[4][3], double v[4][3], double t[4], double time, double xx[3], double vv[3]) {
    double h[4], hdot[4], f0[4], f1[4], g0[4], g1[4];
    double sum, product;

    for (int i=0; i<4; i++) {
        h[i] = 0.;
        hdot[i] = 0.;
        f0[i] = 0.;
        f1[i] = 0.;
        g0[i] = 0.;
        g1[i] = 0.;
    }
    for (int i=0; i<4; i++) {
        f1[i] = time - t[i];
        sum = 0.0;
        for (int j=0; j<4; j++) {
            if (i != j) sum = sum + (1.0 / (t[i] - t[j]));
        }
        f0[i] = 1.0 - (2. * (time - t[i]) * sum);
    }
    for (int i=0; i<4; i++) {
        product = 1.0;
        for (int k=0; k<4; k++) {
            if (k != i) product = product * ((time - t[k]) / (t[i] - t[k]));
        }
        h[i] = product;
        sum = 0.0;
        for (int j=0; j<4; j++) {
            product = 1.0;
            for (int k=0; k<4; k++) {
                if ((k != i) && (k != j)) product = product * ((time - t[k]) / (t[i] - t[k]));
            }
            if (j != i) sum = sum + ((1.0 / (t[i] - t[j])) * product);
        }
        hdot[i] = sum;
    }
    for (int i=0; i<4; i++) {
        g1[i] = h[i] + (2.0 * (time - t[i]) * hdot[i]);
        sum = 0.0;
        for (int j=0; j<4; j++) {
            if (i != j) sum = sum + (1.0 / (t[i] - t[j]));
        }
        g0[i] = 2.0 * ((f0[i] * hdot[i]) - (h[i] * sum));
    }
    for (int k=0; k<3; k++) {
        sum = 0.0;
        for (int i=0; i<4; i++) sum = sum + (((x[i][k] * f0[i]) + (v[i][k] * f1[i])) * h[i] * h[i]);
        xx[k] = sum;
        sum = 0.0;
        for (int i=0; i<4; i++) sum = sum + (((x[i][k] * g0[i]) + (v[i][k] * g1[i])) * h[i]);
        vv[k] = sum;
    }
}

void Orbit::dumpToHDR(const char* filename) {
    ofstream fs(filename);
    if (!fs.is_open()) {
        printf("Error in Orbit::dumpToHDR - Unable to open HDR file: %s\n", filename);
        exit(1);
    }
    printf("Writing %d vectors to %s\n", nVectors, filename);
    fs << showpos;
    fs.precision(16);
    for (int i=0; i<nVectors; i++) {
        fs << UTCtime[i] << "\t" << position[(3*i)] << "\t" << position[(3*i)+1] << "\t" << position[(3*i)+2]
            << "\t" << velocity[(3*i)] << "\t" << velocity[(3*i)+1] << "\t" << velocity[(3*i)+2] << "\n";
    }
    fs.close();
}

void Orbit::printOrbit() {
    for (int i=0; i<nVectors; i++) {
        printf("UTC = %lf\n", UTCtime[i]);
        printf("Position = [ %lf, %lf, %lf ]\n", position[(3*i)], position[(3*i)+1], position[(3*i)+2]);
        printf("Velocity = [ %lf, %lf, %lf ]\n\n", velocity[(3*i)], velocity[(3*i)+1], velocity[(3*i)+2]);
    }
}

