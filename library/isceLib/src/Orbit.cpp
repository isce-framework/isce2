//
// Author: Joshua Cohen
// Copyright 2017
//

#include <fstream>
#include <stdio.h>
#include <string>
#include "isceLibConstants.h"
#include "Orbit.h"
using std::getline;
using std::ifstream;
using std::ofstream;
using std::showpos;
using std::string;
using isceLib::Orbit;
using isceLib::orbitHermite;
using isceLib::WGS84_ORBIT;
using isceLib::SCH_ORBIT;
using isceLib::HERMITE_METHOD;
using isceLib::SCH_METHOD;
using isceLib::LEGENDRE_METHOD;

Orbit::Orbit() {
    // Empty constructor

    basis = 0;
    nVectors = 0;
    position = NULL;
    velocity = NULL;
    UTCtime = NULL;
}

Orbit::Orbit(int bs, int nvec) {
    // Non-empty constructor

    basis = bs;
    nVectors = nvec;
    position = new double[3*nvec];
    velocity = new double[3*nvec];
    UTCtime = new double[nvec];
}

Orbit::Orbit(const Orbit &orb) {
    // Copy constructor

    basis = orb.basis;
    nVectors = orb.nVectors;
    position = new double[3*nVectors];
    velocity = new double[3*nVectors];
    UTCtime = new double[nVectors];
    for (int i=0; i<nVectors; i++) {
        position[3*i] = orb.position[3*i];
        position[3*i+1] = orb.position[3*i+1];
        position[3*i+2] = orb.position[3*i+2];
        velocity[3*i] = orb.velocity[3*i];
        velocity[3*i+1] = orb.velocity[3*i+1];
        velocity[3*i+2] = orb.velocity[3*i+2];
        UTCtime[i] = orb.UTCtime[i];
    }
}

Orbit::~Orbit() {
    // Destructor

    if (position) delete[] position;
    if (velocity) delete[] velocity;
    if (UTCtime) delete[] UTCtime;
}

int Orbit::isNull() {
    // Function used by Python module to determine if internal memory has been allocated (returns 1 if uninitialized)
    
    if (position || velocity || UTCtime) return 0;
    return 1;
}

void Orbit::resetStateVectors() {
    // Resets the internal memory mallocs to match stored nVectors

    if (nVectors >= 0) {
        if (position) delete[] position;
        if (velocity) delete[] velocity;
        if (UTCtime) delete[] UTCtime;
        position = new double[3*nVectors];
        velocity = new double[3*nVectors];
        UTCtime = new double[nVectors];
        for (int i=0; i<nVectors; i++) {
            position[3*i] = 0.;
            position[3*i+1] = 0.;
            position[3*i+2] = 0.;
            velocity[3*i] = 0.;
            velocity[3*i+1] = 0.;
            velocity[3*i+2] = 0.;
            UTCtime[i] = 0.;
        }
    } else {
        printf("Error: Invalid value set for nVectors (stored: %d).\n", nVectors);
    }
}

void Orbit::getPositionVelocity(double tintp, double pos[3], double vel[3]) {
    // Separately-named wrapper for interpolate based on stored basis. Does not check for interpolate success/fail

    if (basis == WGS84_ORBIT) interpolateWGS84Orbit(tintp, pos, vel);
    else if (basis == SCH_ORBIT) interpolateSCHOrbit(tintp, pos, vel);
    else return;
}

void Orbit::getStateVector(int idx, double &t, double pos[3], double vel[3]) {
    // Pull state vector values from the internal master list (0-indexed)
    
    if ((idx < 0) || (idx >= nVectors)) {
        printf("Error: Trying to get state vector %d out of %d\n", idx, nVectors);
        return;
    }
    t = UTCtime[idx];
    for (int i=0; i<3; i++) {
        pos[i] = position[3*idx+i];
        vel[i] = velocity[3*idx+i];
    }
}

void Orbit::setStateVector(int idx, double t, double pos[3], double vel[3]) {
    // Store state vector in the internal master list (0-indexed)

    if ((idx < 0) || (idx >= nVectors)) {
        printf("Error: Trying to set state vector %d out of %d\n", idx, nVectors);
        return;
    }
    UTCtime[idx] = t;
    for (int i=0; i<3; i++) {
        position[3*idx+i] = pos[i];
        velocity[3*idx+i] = vel[i];
    }
}

int Orbit::interpolate(double tintp, double opos[3], double ovel[3], int intp_type) {
    // Single-interface wrapper for orbit interpolation

    if (intp_type == HERMITE_METHOD) return interpolateWGS84Orbit(tintp, opos, ovel);
    else if (intp_type == SCH_METHOD) return interpolateSCHOrbit(tintp, opos, ovel);
    else if (intp_type == LEGENDRE_METHOD) return interpolateLegendreOrbit(tintp, opos, ovel);
    else {
        printf("Error: Unknown interpolation type (received %d)\n", intp_type);
        return 1;
    }
}

int Orbit::interpolateWGS84Orbit(double tintp, double opos[3], double ovel[3]) {
    // Interpolate WGS-84 orbit

    double pos[4][3], vel[4][3];
    double t[4];
    int idx;

    if (nVectors < 4) return 1;

    idx = -1;
    for (int i=0; i<nVectors; i++) {
        if (UTCtime[i] >= tintp) {
            idx = i;
            break;
        }
    }
    idx -= 2;
    if (idx < 0) idx = 0;
    if (idx > (nVectors-4)) idx = nVectors - 4;
    for (int i=0; i<4; i++) getStateVector(idx+i, t[i], pos[i], vel[i]);
    
    orbitHermite(pos, vel, t, tintp, opos, ovel);

    // Not sure why, but original code does the interpolation regardless if the time requested is
    // outside the epoch...
    if ((tintp < UTCtime[0]) || (tintp > UTCtime[nVectors-1])) return 1;
    else return 0;
}

void isceLib::orbitHermite(double x[4][3], double v[4][3], double t[4], double time, double xx[3], double vv[3]) {
    // Method used by interpolateWGS84Orbit but is not tied to an Orbit
    
    double h[4], hdot[4], f0[4], f1[4], g0[4], g1[4];
    double sum, product;
    
    for (int i=0; i<4; i++) {
        f1[i] = time - t[i];
        sum = 0.;
        for (int j=0; j<4; j++) {
            if (i != j) sum += 1. / (t[i] - t[j]);
        }
        f0[i] = 1. - (2. * (time - t[i]) * sum);
    }
    for (int i=0; i<4; i++) {
        product = 1.;
        for (int j=0; j<4; j++) {
            if (i != j) product *= (time - t[j]) / (t[i] - t[j]);
        }
        h[i] = product;
        sum = 0.;
        for (int j=0; i<4; j++) {
            product = 1.;
            for (int k=0; k<4; k++) {
                if ((i != k) && (j != k)) product *= (time - t[k]) / (t[i] - t[k]);
            }
            if (i != j) sum += (1. / (t[i] - t[j])) * product;
        }
        hdot[i] = sum;
    }
    for (int i=0; i<4; i++) {
        g1[i] = h[i] + (2. * (time - t[i]) * hdot[i]);
        sum = 0.;
        for (int j=0; j<4; j++) {
            if (i != j) sum += 1. / (t[i] - t[j]);
        }
        g0[i] = 2. * ((f0[i] * hdot[i]) - (h[i] * sum));
    }
    for (int j=0; j<3; j++) {
        sum = 0.;
        for (int i=0; i<4; i++) {
            sum += ((x[i][j] * f0[i]) + (v[i][j] * f1[i])) * h[i] * h[i];
        }
        xx[j] = sum;
        sum = 0.;
        for (int i=0; i<4; i++) {
            sum += ((x[i][j] * g0[i]) + (v[i][j] * g1[i])) * h[i];
        }
        vv[j] = sum;
    }
}

int Orbit::interpolateLegendreOrbit(double tintp, double opos[3], double ovel[3]) {
    // Interpolate Legendre orbit

    double pos[9][3], vel[9][3];
    double t[9];
    double noemer[] = {40320.0, -5040.0, 1440.0, -720.0, 576.0, -720.0, 1440.0, -5040.0, 40320.0};
    double trel, coeff, teller;
    int idx;

    if (nVectors < 9) return 1;

    for (int i=0; i<3; i++) {
        opos[i] = 0.;
        ovel[i] = 0.;
    }
    idx = -1;
    for (int i=0; i<nVectors; i++) {
        if (UTCtime[i] >= tintp) {
            idx = i;
            break;
        }
    }
    if (idx == -1) idx = nVectors;
    idx -= 5;
    if (idx < 0) idx = 0;
    if (idx > (nVectors-9)) idx = nVectors - 9;
    for (int i=0; i<9; i++) getStateVector(idx+i, t[i], pos[i], vel[i]);

    trel = (8. * (tintp - t[0])) / (t[8] - t[0]);
    teller = 1.;
    for (int i=0; i<9; i++) teller *= trel - i;
    
    if (teller == 0.) {
        for (int i=0; i<3; i++) {
            opos[i] = pos[int(trel)][i];
            ovel[i] = vel[int(trel)][i];
        }
    } else {
        for (int i=0; i<9; i++) {
            coeff = (teller / noemer[i]) / (trel - i);
            for (int j=0; j<3; j++) {
                opos[j] += coeff * pos[i][j];
                ovel[j] += coeff * vel[i][j];
            }
        }
    }

    if ((tintp < UTCtime[0]) || (tintp > UTCtime[nVectors-1])) return 1;
    else return 0;
}

int Orbit::interpolateSCHOrbit(double tintp, double opos[3], double ovel[3]) {
    // Interpolate SCH orbit

    double pos[2][3], vel[2][3];
    double t[2];
    double frac, num, den;

    if (nVectors < 2) {
        printf("Error: Need at least 2 state vectors for SCH orbit interpolation.\n");
        return 1;
    }
    if ((tintp < UTCtime[0]) || (tintp > UTCtime[nVectors-1])) {
        printf("Error: Requested epoch outside orbit state vector span.\n");
        return 1;
    }

    for (int i=0; i<3; i++) {
        opos[i] = 0.;
        ovel[i] = 0.;
    }

    for (int i=0; i<nVectors; i++) {
        frac = 1.;
        getStateVector(i, t[0], pos[0], vel[0]);
        for (int j=0; j<nVectors; j++) {
            if (i == j) continue;
            getStateVector(j, t[1], pos[1], vel[1]);
            num = t[1] - tintp;
            den = t[1] - t[0];
            frac *= num / den;
        }
        for (int j=0; j<3; j++) {
            opos[j] += frac * pos[0][j];
            ovel[j] += frac * vel[0][j];
        }
    }

    return 0;
}

int Orbit::computeAcceleration(double tintp, double acc[3]) {
    // Interpolate acceleration

    double vbef[3], vaft[3], dummy[3];
    double temp;
    int stat;

    for (int i=0; i<3; i++) acc[i] = 0.;
    temp = tintp - .01;
    stat = interpolateWGS84Orbit(temp, dummy, vbef);
    if (stat == 1) return stat;
    temp = tintp + .01;
    stat = interpolateWGS84Orbit(temp, dummy, vaft);
    if (stat == 1) return stat;
    for (int i=0; i<3; i++) acc[i] = (vaft[i] - vbef[i]) / .02;
    return 0;
}

void Orbit::printOrbit() {
    // Debug print the stored orbit

    double pos[3], vel[3];
    double t;

    for (int i=0; i<nVectors; i++) {
        getStateVector(i, t, pos, vel);
        printf("UTC = %f\n", t);
        printf("Position = [ %f , %f , %f ]\n", pos[0], pos[1], pos[2]);
        printf("Velocity = [ %f , %f , %f ]\n\n", vel[0], vel[1], vel[2]);
    }
}

void Orbit::loadFromHDR(const char *filename, int bs) {
    double pos[3], vel[3];
    double t;
    int count;
    string line;

    ifstream fs(filename);
    if (!fs.is_open()) {
        printf("Error: Unable to load orbit from HDR file: %s\n", filename);
        return;
    }

    count = 0;
    basis = bs;
    nVectors = 0;
    while (getline(fs, line)) nVectors++;

    if (position) delete[] position;
    if (velocity) delete[] velocity;
    if (UTCtime) delete[] UTCtime;
    position = new double[3*nVectors];
    velocity = new double[3*nVectors];
    UTCtime = new double[nVectors];

    fs.clear();
    fs.seekg(0);
    // Take advantage of the power of fstreams
    while (fs >> t >> pos[0] >> pos[1] >> pos[2] >> vel[0] >> vel[1] >> vel[2]) {
        setStateVector(count, t, pos, vel);
        count++;
    }
    fs.close();
    printf("Read in %d state vectors from %s\n", nVectors, filename);
}

void Orbit::dumpToHDR(const char* filename) {
    ofstream fs(filename);
    if (!fs.is_open()) {
        printf("Error: Unable to open HDR file '%s'\n", filename);
        fs.close();
        return;
    }
    printf("Writing %d vectors to '%s'\n", nVectors, filename);
    fs << showpos;
    fs.precision(16);
    for (int i=0; i<nVectors; i++) {
        fs << UTCtime[i] << "\t" << position[3*i] << "\t" << position[3*i+1] << "\t" << position[3*i+2]
            << "\t" << velocity[3*i] << "\t" << velocity[3*i+1] << "\t" << velocity[3*i+2] << "\n";
    }
    fs.close();
}

