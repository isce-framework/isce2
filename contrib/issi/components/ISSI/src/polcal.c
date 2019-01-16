#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include "polcal.h"

/**
 * Perform polarimetric calibration on ALOS PALSAR data.
 *
 * @param hhFile the name of the file containing the focused HH polarity SAR data
 * @param hvFile the name of the file containing the focused HV polarity SAR data
 * @param vhFile the name of the file containing the focused VH polarity SAR data
 * @param vvFile the name of the file containing the focused VV polarity SAR data
 * @param hhOutFile the name for the output file for the polarimetrically corrected HH polarity SAR data
 * @param hvOutFile the name for the output file for the polarimetrically corrected HV polarity SAR data
 * @param vhOutFile the name for the output file for the polarimetrically corrected VH polarity SAR data
 * @param vvOutFile the name for the output file for the polarimetrically corrected VV polarity SAR data
 * @param tcrossTalk1Real the real part of the first transmission cross-talk parameter
 * @param tcrossTalk2Real the real part of the second transmission cross-talk parameter
 * @param tchannelImbalanceReal the real part of the transmitted channel imbalance
 * @param tcrossTalk1Imag the imaginary part of the first transmission cross-talk parameter
 * @param tcrossTalk2Imag the imaginary part of the second transmission cross-talk parameter
 * @param tchannelImbalanceImag the imaginary part of the transmitted channel imbalance
 * @param rcrossTalk1Real the real part of the first reception cross-talk parameter
 * @param rcrossTalk2Real the real part of the second reception cross-talk parameter
 * @param rchannelImbalanceReal the real part of the received channel imbalance
 * @param rcrossTalk1Imag the imaginary part of the first reception cross-talk parameter
 * @param rcrossTalk2Imag the imaginary part of the second reception cross-talk parameter
 * @param rchannelImbalanceImag the imaginary part of the received channel imbalance
 * @param samples the number of samples in the range direction
 * @param lines the number of samples in the azimuth direction
 */
int
polcal(char *hhFile,   char *hvFile,   char *vhFile,   char *vvFile,
       char *hhOutFile,char *hvOutFile,char *vhOutFile,char *vvOutFile,
       float tcrossTalk1Real, float tcrossTalk2Real, float tchannelImbalanceReal,
       float tcrossTalk1Imag, float tcrossTalk2Imag, float tchannelImbalanceImag,
       float rcrossTalk1Real, float rcrossTalk2Real, float rchannelImbalanceReal,
       float rcrossTalk1Imag, float rcrossTalk2Imag, float rchannelImbalanceImag,
       int samples, int lines)
{
  struct distortion transmission, reception;

  transmission.crossTalk1 = tcrossTalk1Real + tcrossTalk1Imag*I;
  transmission.crossTalk2 = tcrossTalk2Real + tcrossTalk2Imag*I;
  transmission.channelImbalance = tchannelImbalanceReal + tchannelImbalanceImag*I;

  reception.crossTalk1 = rcrossTalk1Real + rcrossTalk1Imag*I;
  reception.crossTalk2 = rcrossTalk2Real + rcrossTalk2Imag*I;
  reception.channelImbalance = rchannelImbalanceReal + rchannelImbalanceImag*I;

  polarimetriccalibration_(hhFile,hvFile,vhFile,vvFile,
			   hhOutFile,hvOutFile,vhOutFile,vvOutFile,
			   &transmission,&reception,&samples,&lines);
  return 0;
}
