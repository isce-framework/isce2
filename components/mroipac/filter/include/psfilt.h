typedef struct{float re,im;} fcomplex;

void psfilt(char *int_filename, char *sm_filename, int width, int nlines, double alpha, int step, int xmin, int xmax, int ymin, int ymax);
void fourn(float *, unsigned int *, int ndim, int isign); 
void psd_wgt(fcomplex **cmp, fcomplex **seg_fft, double, int, int);

fcomplex **cmatrix(int nrl, int nrh, int ncl, int nch);
void free_cmatrix(fcomplex **m, int nrl, int nrh, int ncl, int nch);
void nrerror(char error_text[]);
