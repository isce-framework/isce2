struct distortion {
    float _Complex crossTalk1;
    float _Complex crossTalk2;
    float _Complex channelImbalance;
};

int
polcal(char *hhFile,   char *hvFile,   char *vhFile,   char *vvFile,
       char *hhOutFile,char *hvOutFile,char *vhOutFile,char *vvOutFile,
       float tcrossTalk1Real, float tcrossTalk2Real, float tchannelImbalanceReal,
       float tcrossTalk1Imag, float tcrossTalk2Imag, float tchannelImbalanceImag,
       float rcrossTalk1Real, float rcrossTalk2Real, float rchannelImbalanceReal,
       float rcrossTalk1Imag, float rcrossTalk2Imag, float rchannelImbalanceImag,
       int samples, int lines);
int
polarimetriccalibration_(char *hhFile, char *hvFile, char *vhFile, char *vvFile,
			 char *hhOutFile, char *hvOutFile, char *vhOutFile, char *vvOutFile,
			 struct distortion *transmission, struct distortion *reception,int *samples, int *lines);
