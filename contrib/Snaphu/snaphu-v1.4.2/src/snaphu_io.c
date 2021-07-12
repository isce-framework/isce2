/*************************************************************************

  snaphu input/output source file

  Written by Curtis W. Chen
  Copyright 2002 Board of Trustees, Leland Stanford Jr. University
  Please see the supporting documentation for terms of use.
  No warranty.

*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <signal.h>
#include <limits.h>
#include <float.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "snaphu.h"


/* function: SetDefaults()
 * -----------------------
 * Sets all parameters to their initial default values.
 */
void SetDefaults(infileT *infiles, outfileT *outfiles, paramT *params){


  /* input files */
  StrNCopy(infiles->weightfile,DEF_WEIGHTFILE,MAXSTRLEN);
  StrNCopy(infiles->corrfile,DEF_CORRFILE,MAXSTRLEN);
  StrNCopy(infiles->ampfile,DEF_AMPFILE,MAXSTRLEN);
  StrNCopy(infiles->ampfile2,DEF_AMPFILE2,MAXSTRLEN);
  StrNCopy(infiles->estfile,DEF_ESTFILE,MAXSTRLEN);  
  StrNCopy(infiles->magfile,DEF_MAGFILE,MAXSTRLEN);
  StrNCopy(infiles->costinfile,DEF_COSTINFILE,MAXSTRLEN);

  /* output and dump files */
  StrNCopy(outfiles->initfile,DEF_INITFILE,MAXSTRLEN);
  StrNCopy(outfiles->flowfile,DEF_FLOWFILE,MAXSTRLEN);
  StrNCopy(outfiles->eifile,DEF_EIFILE,MAXSTRLEN);
  StrNCopy(outfiles->rowcostfile,DEF_ROWCOSTFILE,MAXSTRLEN);
  StrNCopy(outfiles->colcostfile,DEF_COLCOSTFILE,MAXSTRLEN);
  StrNCopy(outfiles->mstrowcostfile,DEF_MSTROWCOSTFILE,MAXSTRLEN);
  StrNCopy(outfiles->mstcolcostfile,DEF_MSTCOLCOSTFILE,MAXSTRLEN);
  StrNCopy(outfiles->mstcostsfile,DEF_MSTCOSTSFILE,MAXSTRLEN);
  StrNCopy(outfiles->corrdumpfile,DEF_CORRDUMPFILE,MAXSTRLEN);
  StrNCopy(outfiles->rawcorrdumpfile,DEF_RAWCORRDUMPFILE,MAXSTRLEN);
  StrNCopy(outfiles->costoutfile,DEF_COSTOUTFILE,MAXSTRLEN);
  StrNCopy(outfiles->conncompfile,DEF_CONNCOMPFILE,MAXSTRLEN);
  StrNCopy(outfiles->outfile,DEF_OUTFILE,MAXSTRLEN);  
  StrNCopy(outfiles->logfile,DEF_LOGFILE,MAXSTRLEN);

  /* file formats */
  infiles->infileformat=DEF_INFILEFORMAT;
  infiles->unwrappedinfileformat=DEF_UNWRAPPEDINFILEFORMAT;
  infiles->magfileformat=DEF_MAGFILEFORMAT;
  infiles->corrfileformat=DEF_CORRFILEFORMAT;
  infiles->estfileformat=DEF_ESTFILEFORMAT;
  infiles->ampfileformat=DEF_AMPFILEFORMAT;
  outfiles->outfileformat=DEF_OUTFILEFORMAT;

  /* options and such */
  params->unwrapped=DEF_UNWRAPPED;
  params->regrowconncomps=DEF_REGROWCONNCOMPS;
  params->eval=DEF_EVAL;
  params->initonly=DEF_INITONLY;
  params->initmethod=DEF_INITMETHOD;
  params->costmode=DEF_COSTMODE;
  params->amplitude=DEF_AMPLITUDE;
  params->verbose=DEF_VERBOSE;

  /* SAR and geometry parameters */
  params->orbitradius=DEF_ORBITRADIUS;
  params->altitude=DEF_ALTITUDE;
  params->earthradius=DEF_EARTHRADIUS;
  params->bperp=DEF_BPERP; 
  params->transmitmode=DEF_TRANSMITMODE;
  params->baseline=DEF_BASELINE;
  params->baselineangle=DEF_BASELINEANGLE;
  params->nlooksrange=DEF_NLOOKSRANGE;
  params->nlooksaz=DEF_NLOOKSAZ;
  params->nlooksother=DEF_NLOOKSOTHER;
  params->ncorrlooks=DEF_NCORRLOOKS;           
  params->ncorrlooksrange=DEF_NCORRLOOKSRANGE;
  params->ncorrlooksaz=DEF_NCORRLOOKSAZ;
  params->nearrange=DEF_NEARRANGE;         
  params->dr=DEF_DR;               
  params->da=DEF_DA;               
  params->rangeres=DEF_RANGERES;         
  params->azres=DEF_AZRES;            
  params->lambda=DEF_LAMBDA;           

  /* scattering model parameters */
  params->kds=DEF_KDS;
  params->specularexp=DEF_SPECULAREXP;
  params->dzrcritfactor=DEF_DZRCRITFACTOR;
  params->shadow=DEF_SHADOW;
  params->dzeimin=DEF_DZEIMIN;
  params->laywidth=DEF_LAYWIDTH;
  params->layminei=DEF_LAYMINEI;
  params->sloperatiofactor=DEF_SLOPERATIOFACTOR;
  params->sigsqei=DEF_SIGSQEI;

  /* decorrelation model parameters */
  params->drho=DEF_DRHO;
  params->rhosconst1=DEF_RHOSCONST1;
  params->rhosconst2=DEF_RHOSCONST2;
  params->cstd1=DEF_CSTD1;
  params->cstd2=DEF_CSTD2;
  params->cstd3=DEF_CSTD3;
  params->defaultcorr=DEF_DEFAULTCORR;
  params->rhominfactor=DEF_RHOMINFACTOR;

  /* pdf model parameters */
  params->dzlaypeak=DEF_DZLAYPEAK;
  params->azdzfactor=DEF_AZDZFACTOR;
  params->dzeifactor=DEF_DZEIFACTOR;
  params->dzeiweight=DEF_DZEIWEIGHT;
  params->dzlayfactor=DEF_DZLAYFACTOR;
  params->layconst=DEF_LAYCONST;
  params->layfalloffconst=DEF_LAYFALLOFFCONST;
  params->sigsqshortmin=DEF_SIGSQSHORTMIN;
  params->sigsqlayfactor=DEF_SIGSQLAYFACTOR;
  
  /* deformation mode parameters */
  params->defoazdzfactor=DEF_DEFOAZDZFACTOR;
  params->defothreshfactor=DEF_DEFOTHRESHFACTOR;
  params->defomax=DEF_DEFOMAX;
  params->sigsqcorr=DEF_SIGSQCORR;
  params->defolayconst=DEF_DEFOLAYCONST;

  /* algorithm parameters */
  params->flipphasesign=DEF_FLIPPHASESIGN;
  params->initmaxflow=DEF_INITMAXFLOW;
  params->arcmaxflowconst=DEF_ARCMAXFLOWCONST;
  params->maxflow=DEF_MAXFLOW;
  params->krowei=DEF_KROWEI;
  params->kcolei=DEF_KCOLEI;   
  params->kperpdpsi=DEF_KPERPDPSI;
  params->kpardpsi=DEF_KPARDPSI;
  params->threshold=DEF_THRESHOLD;  
  params->initdzr=DEF_INITDZR;    
  params->initdzstep=DEF_INITDZSTEP;    
  params->maxcost=DEF_MAXCOST;
  params->costscale=DEF_COSTSCALE;      
  params->costscaleambight=DEF_COSTSCALEAMBIGHT;      
  params->dnomincangle=DEF_DNOMINCANGLE;
  params->srcrow=DEF_SRCROW;
  params->srccol=DEF_SRCCOL;
  params->p=DEF_P;
  params->nshortcycle=DEF_NSHORTCYCLE;
  params->maxnewnodeconst=DEF_MAXNEWNODECONST;
  params->maxcyclefraction=DEF_MAXCYCLEFRACTION;
  params->sourcemode=DEF_SOURCEMODE;
  params->maxnflowcycles=DEF_MAXNFLOWCYCLES;
  params->dumpall=DEF_DUMPALL;
  params->cs2scalefactor=DEF_CS2SCALEFACTOR;

  /* tile parameters */
  params->ntilerow=DEF_NTILEROW;
  params->ntilecol=DEF_NTILECOL;
  params->rowovrlp=DEF_ROWOVRLP;
  params->colovrlp=DEF_COLOVRLP;
  params->piecefirstrow=DEF_PIECEFIRSTROW;
  params->piecefirstcol=DEF_PIECEFIRSTCOL;
  params->piecenrow=DEF_PIECENROW;
  params->piecencol=DEF_PIECENCOL;
  params->tilecostthresh=DEF_TILECOSTTHRESH;
  params->minregionsize=DEF_MINREGIONSIZE;
  params->nthreads=DEF_NTHREADS;
  params->scndryarcflowmax=DEF_SCNDRYARCFLOWMAX;
  params->assembleonly=DEF_ASSEMBLEONLY;
  params->rmtmptile=DEF_RMTMPTILE;
  params->tileedgeweight=DEF_TILEEDGEWEIGHT;

  /* connected component parameters */
  params->minconncompfrac=DEF_MINCONNCOMPFRAC;
  params->conncompthresh=DEF_CONNCOMPTHRESH;
  params->maxncomps=DEF_MAXNCOMPS;

}


/* function: ProcessArgs()
 * -----------------------
 * Parses command line inputs passed to main().
 */
void ProcessArgs(int argc, char *argv[], infileT *infiles, outfileT *outfiles,
		 long *linelenptr, paramT *params){

  long i,j;
  signed char noarg_exit;

  /* required inputs */
  noarg_exit=FALSE;
  StrNCopy(infiles->infile,"",MAXSTRLEN);
  *linelenptr=0;

  /* loop over inputs */
  if(argc<2){                             /* catch zero arguments in */
    fprintf(sp1,OPTIONSHELPBRIEF);
    exit(ABNORMAL_EXIT);
  }
  for(i=1;i<argc;i++){                  
    /* if argument is an option */
    if(argv[i][0]=='-'){   
      if(strlen(argv[i])==1){
	fprintf(sp0,"invalid command line argument -\n");
	exit(ABNORMAL_EXIT);
      }else if(argv[i][1]!='-'){
	for(j=1;j<strlen(argv[i]);j++){
	  if(argv[i][j]=='h'){
	    fprintf(sp1,OPTIONSHELPFULL);
	    exit(ABNORMAL_EXIT);
	  }else if(argv[i][j]=='u'){
	    params->unwrapped=TRUE;
	  }else if(argv[i][j]=='t'){
	    params->costmode=TOPO;
	  }else if(argv[i][j]=='d'){
	    params->costmode=DEFO;
	  }else if(argv[i][j]=='s'){
	    params->costmode=SMOOTH;
	    params->defomax=0.0;
	  }else if(argv[i][j]=='q'){
	    params->eval=TRUE;
	    params->unwrapped=TRUE;
	  }else if(argv[i][j]=='f'){
	    if(++i<argc && j==strlen(argv[i-1])-1){
	      
	      /* read user-supplied configuration file */
	      ReadConfigFile(argv[i],infiles,outfiles,linelenptr,params);
	      break;
	    }else{
	      noarg_exit=TRUE;
	    }
	  }else if(argv[i][j]=='o'){
	    if(++i<argc && j==strlen(argv[i-1])-1){
	      StrNCopy(outfiles->outfile,argv[i],MAXSTRLEN);
	      break;
	    }else{
	      noarg_exit=TRUE;
	    }
	  }else if(argv[i][j]=='c'){
	    if(++i<argc && j==strlen(argv[i-1])-1){
	      StrNCopy(infiles->corrfile,argv[i],MAXSTRLEN);
	      break;
	    }else{
	      noarg_exit=TRUE;
	    }
	  }else if(argv[i][j]=='m'){
	    if(++i<argc && j==strlen(argv[i-1])-1){
	      StrNCopy(infiles->magfile,argv[i],MAXSTRLEN);
	      break;
	    }else{
	      noarg_exit=TRUE;
	    }
	  }else if(argv[i][j]=='a'){
	    if(++i<argc && j==strlen(argv[i-1])-1){
	      StrNCopy(infiles->ampfile,argv[i],MAXSTRLEN);
	      params->amplitude=TRUE;
	      break;
	    }else{
	      noarg_exit=TRUE;
	    }
	  }else if(argv[i][j]=='A'){
	    if(++i<argc && j==strlen(argv[i-1])-1){
	      StrNCopy(infiles->ampfile,argv[i],MAXSTRLEN);
	      params->amplitude=FALSE;
	      break;
	    }else{
	      noarg_exit=TRUE;
	    }
	  }else if(argv[i][j]=='e'){
	    if(++i<argc && j==strlen(argv[i-1])-1){
	      StrNCopy(infiles->estfile,argv[i],MAXSTRLEN);
	      break;
	    }else{
	      noarg_exit=TRUE;
	    }
	  }else if(argv[i][j]=='w'){
	    if(++i<argc && j==strlen(argv[i-1])-1){
	      StrNCopy(infiles->weightfile,argv[i],MAXSTRLEN);
	      break;
	    }else{
	      noarg_exit=TRUE;
	    }
	  }else if(argv[i][j]=='g'){
	    if(++i<argc && j==strlen(argv[i-1])-1){
	      StrNCopy(outfiles->conncompfile,argv[i],MAXSTRLEN);
	      break;
	    }else{
	      noarg_exit=TRUE;
	    }
	  }else if(argv[i][j]=='G'){
	    params->regrowconncomps=TRUE;
	    if(++i<argc && j==strlen(argv[i-1])-1){
	      StrNCopy(outfiles->conncompfile,argv[i],MAXSTRLEN);
	      break;
	    }else{
	      noarg_exit=TRUE;
	    }
	  }else if(argv[i][j]=='b'){
	    if(++i<argc && j==strlen(argv[i-1])-1){
	      if(StringToDouble(argv[i],&(params->bperp)) || !(params->bperp)){
		fprintf(sp0,"option -%c requires non-zero decimal argument\n",
			argv[i-1][j]);
		exit(ABNORMAL_EXIT);
	      }
	      break;
	    }else{
	      noarg_exit=TRUE;
	    }
	  }else if(argv[i][j]=='p'){
	    if(++i<argc && j==strlen(argv[i-1])-1){
	      if(StringToDouble(argv[i],&(params->p))){
		fprintf(sp0,"option -%c requires decimal argument\n",
			argv[i-1][j]);
		exit(ABNORMAL_EXIT);
	      }
	      break;
	    }else{
	      noarg_exit=TRUE;
	    }
	  }else if(argv[i][j]=='i'){
	    params->initonly=TRUE;
	  }else if(argv[i][j]=='n'){
	    params->costmode=NOSTATCOSTS;
	  }else if(argv[i][j]=='v'){
	    params->verbose=TRUE;
	  }else if(argv[i][j]=='l'){
	    if(++i<argc && j==strlen(argv[i-1])-1){
	      StrNCopy(outfiles->logfile,argv[i],MAXSTRLEN);
	      break;
	    }else{
	      noarg_exit=TRUE;
	    }
	  }else{
	    fprintf(sp0,"unrecognized option -%c\n",argv[i][j]);
	    exit(ABNORMAL_EXIT);
	  }
	  if(noarg_exit){
	    fprintf(sp0,"option -%c requires an argument\n",argv[i-1][j]);
	    exit(ABNORMAL_EXIT);
	  }
	}
      }else{
	/* argument is a "--" option */
	if(!strcmp(argv[i],"--costinfile")){
	  if(++i<argc){
	    StrNCopy(infiles->costinfile,argv[i],MAXSTRLEN);
	  }else{
	    noarg_exit=TRUE;
	  }
	}else if(!strcmp(argv[i],"--costoutfile")){
	  if(++i<argc){
	    StrNCopy(outfiles->costoutfile,argv[i],MAXSTRLEN);
	  }else{
	    noarg_exit=TRUE;
	  }
	}else if(!strcmp(argv[i],"--debug") || !strcmp(argv[i],"--dumpall")){
	  params->dumpall=TRUE;
	}else if(!strcmp(argv[i],"--mst")){
	  params->initmethod=MSTINIT;
	}else if(!strcmp(argv[i],"--mcf")){
	  params->initmethod=MCFINIT;
	}else if(!strcmp(argv[i],"--aa")){
	  if(i+2<argc){
	    StrNCopy(infiles->ampfile,argv[++i],MAXSTRLEN);
	    StrNCopy(infiles->ampfile2,argv[++i],MAXSTRLEN);
	    infiles->ampfileformat=FLOAT_DATA;
	    params->amplitude=TRUE;
	  }else{
	    noarg_exit=TRUE;
	  }
	}else if(!strcmp(argv[i],"--AA")){
	  if(++i+1<argc){
	    StrNCopy(infiles->ampfile,argv[i++],MAXSTRLEN);
	    StrNCopy(infiles->ampfile2,argv[i],MAXSTRLEN);
	    infiles->ampfileformat=FLOAT_DATA;
	    params->amplitude=FALSE;
	  }else{
	    noarg_exit=TRUE;
	  }
	}else if(!strcmp(argv[i],"--tile")){
	  if(++i+3<argc){
	    if(StringToLong(argv[i++],&(params->ntilerow))
	       || StringToLong(argv[i++],&(params->ntilecol))
	       || StringToLong(argv[i++],&(params->rowovrlp))
	       || StringToLong(argv[i],&(params->colovrlp))){
	      fprintf(sp0,"option %s requires four integer arguments\n",
		      argv[i-4]);
	      exit(ABNORMAL_EXIT);
	    }
	  }else{
	    noarg_exit=TRUE;
	  }
	}else if(!strcmp(argv[i],"--piece")){
	  if(++i+3<argc){
	    if(StringToLong(argv[i++],&(params->piecefirstrow))
	       || StringToLong(argv[i++],&(params->piecefirstcol))
	       || StringToLong(argv[i++],&(params->piecenrow))
	       || StringToLong(argv[i],&(params->piecencol))){
	      fprintf(sp0,"option %s requires four integer arguments\n",
		      argv[i-4]);
	      exit(ABNORMAL_EXIT);
	    }
	  }else{
	    noarg_exit=TRUE;
	  }
	}else if(!strcmp(argv[i],"--nproc")){
	  if(++i<argc){
	    if(StringToLong(argv[i],&(params->nthreads))){
	      fprintf(sp0,"option %s requires an integer arguemnt\n",
		      argv[i-1]);
	      exit(ABNORMAL_EXIT);
	    }
	  }else{
	    noarg_exit=TRUE;
	  }
	}else if(!strcmp(argv[i],"--assemble")){
	  params->assembleonly=TRUE;
	  if(++i<argc){
	    StrNCopy(params->tiledir,argv[i],MAXSTRLEN);
	  }else{
	    noarg_exit=TRUE;
	  }
	}else if(!strcmp(argv[i],"--copyright") || !strcmp(argv[i],"--info")){
	  fprintf(sp1,COPYRIGHT);
	  exit(ABNORMAL_EXIT);	  
	}else if(!strcmp(argv[i],"--help")){
	  fprintf(sp1,OPTIONSHELPFULL);
	  exit(ABNORMAL_EXIT);	  
	}else{
	  fprintf(sp0,"unrecognized option %s\n",argv[i]);
	  exit(ABNORMAL_EXIT);
	}
	if(noarg_exit){
	  fprintf(sp0,"incorrect number of arguments for option %s\n",
		  argv[i-1]);
	  exit(ABNORMAL_EXIT);
	}
      }
    }else{                                
      /* argument is not an option */
      if(!strlen(infiles->infile)){
        StrNCopy(infiles->infile,argv[i],MAXSTRLEN);
      }else if(*linelenptr==0){
	if(StringToLong(argv[i],linelenptr) || *linelenptr<=0){
	  fprintf(sp0,"line length must be positive integer\n");
	  exit(ABNORMAL_EXIT);
	}	  
      }else{
        fprintf(sp0,"multiple input files: %s and %s\n",
		infiles->infile,argv[i]);
	exit(ABNORMAL_EXIT);
      }
    }
  } /* end for loop over arguments */

  /* check to make sure we have required arguments */
  if(!strlen(infiles->infile) || !(*linelenptr)){
    fprintf(sp0,"not enough input arguments.  type %s -h for help\n",
	    PROGRAMNAME);
    exit(ABNORMAL_EXIT);
  }

} /* end of ProcessArgs */


/* function: CheckParams()
 * -----------------------
 * Checks all parameters to make sure they are valid.  This is just a boring
 * function with lots of checks in it.
 */
void CheckParams(infileT *infiles, outfileT *outfiles, 
		 long linelen, long nlines, paramT *params){

  long ni, nj, n;
  FILE *fp;

  /* make sure output file is writable (try opening in append mode) */
  /* file will be opened in write mode later, clobbering existing file */
  if((fp=fopen(outfiles->outfile,"a"))==NULL){
    fprintf(sp0,"file %s is not writable\n",outfiles->outfile);
    exit(ABNORMAL_EXIT);
  }else{
    if(ftell(fp)){
      fclose(fp);
    }else{
      fclose(fp);
      remove(outfiles->outfile);
    }
    if(!strcmp(outfiles->outfile,infiles->infile) 
       && !params->eval && !params->regrowconncomps){
      fprintf(sp0,"WARNING: output will overwrite input\n");
    }
  }

  /* make sure options aren't contradictory */
  if(params->initonly && params->unwrapped){
    fprintf(sp0,"cannot use initialize-only mode with unwrapped input\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->initonly && params->p>=0){
    fprintf(sp0,"cannot use initialize-only mode with Lp costs\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->costmode==NOSTATCOSTS && !(params->initonly || params->p>=0)){
    fprintf(sp0,"no-statistical-costs option can only be used in\n");
    fprintf(sp0,"  initialize-only or Lp-norm modes\n");
    exit(ABNORMAL_EXIT);
  }
  if(strlen(infiles->costinfile) && params->costmode==NOSTATCOSTS){
    fprintf(sp0,"no-statistical-costs option cannot be given\n");
    fprintf(sp0,"  if input cost file is specified\n");
    exit(ABNORMAL_EXIT);
  }
  if(strlen(outfiles->costoutfile) && params->costmode==NOSTATCOSTS){
    fprintf(sp0,"no-statistical-costs option cannot be given\n");
    fprintf(sp0,"  if output cost file is specified\n");
    exit(ABNORMAL_EXIT);
  }

  /* check geometry parameters */
  if(params->earthradius<=0){
    fprintf(sp0,"earth radius must be nonnegative\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->altitude){
    if(params->altitude>0){
      params->orbitradius=params->earthradius+params->altitude;
    }else{
      fprintf(sp0,"platform altitude must be positive\n");
      exit(ABNORMAL_EXIT);
    }
  }else if(params->orbitradius < params->earthradius){
    fprintf(sp0,"platform orbit radius must be greater than earth radius\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->costmode==TOPO && params->baseline<0){
    fprintf(sp0,"baseline length must be nonnegative\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->costmode==TOPO && params->baseline==0){
    fprintf(sp0,"WARNING: zero baseline may give unpredictable results\n");
  }
  if(params->ncorrlooks<=0){
    fprintf(sp0,"number of looks ncorrlooks must be positive\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->nearrange<=0){
    fprintf(sp0,"slant range parameter nearrange must be positive (meters)\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->dr<=0 || params->da<=0){
    fprintf(sp0,"pixel spacings dr and da must be positive (meters)\n");
    exit(ABNORMAL_EXIT);
  }
  /* dr and da after multilooking can be larger than rangeres, azres */
  /*
  if(params->rangeres<=(params->dr) 
     || params->azres<=(params->da)){
    fprintf(sp0,"resolutions parameters must be larger than pixel spacings\n");
    exit(ABNORMAL_EXIT);
  }
  */
  if(params->lambda<=0){
    fprintf(sp0,"wavelength lambda  must be positive (meters)\n");
    exit(ABNORMAL_EXIT);
  }

  /* check scattering model defaults */
  if(params->kds<=0){
    fprintf(sp0,"scattering model parameter kds must be positive\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->specularexp<=0){
    fprintf(sp0,"scattering model parameter SPECULAREXP must be positive\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->dzrcritfactor<0){
    fprintf(sp0,"dzrcritfactor must be nonnegative\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->laywidth<1){
    fprintf(sp0,"layover window width laywidth must be positive\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->layminei<0){
    fprintf(sp0,"layover minimum brightness must be nonnegative\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->sloperatiofactor<0){
    fprintf(sp0,"slope ratio fudge factor must be nonnegative\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->sigsqei<=0){
    fprintf(sp0,"intensity estimate variance must be positive\n");
    exit(ABNORMAL_EXIT);
  }

  /* check decorrelation model defaults */
  if(params->drho<=0){
    fprintf(sp0,"correlation step size drho must be positive\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->rhosconst1<=0 || params->rhosconst2<=0){
    fprintf(sp0,"parameters rhosconst1 and rhosconst2 must be positive\n");
    exit(ABNORMAL_EXIT);
  }
  if(!strlen(infiles->corrfile) 
     && (params->defaultcorr<0 || params->defaultcorr>1)){
    fprintf(sp0,"default correlation must be between 0 and 1\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->rhominfactor<0){
    fprintf(sp0,"parameter rhominfactor must be nonnegative\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->ncorrlooksaz<1 || params->ncorrlooksrange<1
     || params->nlooksaz<1 || params->nlooksrange<1
     || params->nlooksother<1){
    fprintf(sp0,"numbers of looks must be positive integer\n");
    exit(ABNORMAL_EXIT);
  }
  if(!strlen(infiles->corrfile)){
    if(params->ncorrlooksaz<params->nlooksaz){ 
      fprintf(sp0,"NCORRLOOKSAZ cannot be smaller than NLOOKSAZ\n");
      fprintf(sp0,"  setting NCORRLOOKSAZ to equal NLOOKSAZ\n");
      params->ncorrlooksaz=params->nlooksaz;
    }
    if(params->ncorrlooksrange<params->nlooksrange){ 
      fprintf(sp0,"NCORRLOOKSRANGE cannot be smaller than NLOOKSRANGE\n");
      fprintf(sp0,"  setting NCORRLOOKSRANGE to equal NLOOKSRANGE\n");
      params->ncorrlooksrange=params->nlooksrange;
    }
  }
    
  /* check pdf model parameters */
  if(params->azdzfactor<0){
    fprintf(sp0,"parameter azdzfactor must be nonnegative\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->dzeifactor<0){
    fprintf(sp0,"parameter dzeifactor must be nonnegative\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->dzeiweight<0 || params->dzeiweight>1.0){
    fprintf(sp0,"parameter dzeiweight must be between 0 and 1\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->dzlayfactor<0){
    fprintf(sp0,"parameter dzlayfactor must be nonnegative\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->layconst<=0){
    fprintf(sp0,"parameter layconst must be positive\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->layfalloffconst<0){
    fprintf(sp0,"parameter layfalloffconst must be nonnegative\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->sigsqshortmin<=0){
    fprintf(sp0,"parameter sigsqshortmin must be positive\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->sigsqlayfactor<0){
    fprintf(sp0,"parameter sigsqlayfactor must be nonnegative\n");
    exit(ABNORMAL_EXIT);
  }

  /* check deformation mode parameters */
  if(params->defoazdzfactor<0){
    fprintf(sp0,"parameter defoazdzfactor must be nonnegative\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->defothreshfactor<0){
    fprintf(sp0,"parameter defothreshfactor must be nonnegative\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->defomax<0){
    fprintf(sp0,"parameter defomax must be nonnegative\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->sigsqcorr<0){
    fprintf(sp0,"parameter sigsqcorr must be nonnegative\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->defolayconst<=0){
    fprintf(sp0,"parameter defolayconst must be positive\n");
    exit(ABNORMAL_EXIT);
  }
  
  /* check algorithm parameters */
  /* be sure to check for things that will cause type overflow */
  /* or floating point exception */
  if((params->initmaxflow)<1 && (params->initmaxflow)!=AUTOCALCSTATMAX){
    fprintf(sp0,"initialization maximum flow must be positive\n");
    exit(ABNORMAL_EXIT);
  }
  if((params->arcmaxflowconst)<1){
    fprintf(sp0,"arcmaxflowconst must be positive\n");
    exit(ABNORMAL_EXIT);
  }
  if((params->maxflow)<1){
    fprintf(sp0,"maxflow must be positive\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->krowei<=0 || params->kcolei<=0){
    fprintf(sp0,"averaging window sizes krowei and kcolei must be positive\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->kperpdpsi<=0 || params->kpardpsi<=0){
    fprintf(sp0,
	  "averaging window sizes kperpdpsi and kpardpsi must be positive\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->threshold<=0){
    fprintf(sp0,"numerical solver threshold must be positive\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->initdzr<=0){
    fprintf(sp0,"initdzr must be positive\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->initdzstep<=0){
    fprintf(sp0,"initdzstep must be positive\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->maxcost>POSSHORTRANGE || params->maxcost<=0){
    fprintf(sp0,"maxcost must be positive and within range or short int\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->costscale<=0){
    fprintf(sp0,"cost scale factor costscale must be positive\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->p<0 && params->p!=PROBCOSTP){
    fprintf(sp0,"Lp-norm parameter p should be nonnegative\n");
    exit(ABNORMAL_EXIT);
  }
  if((params->costmode==TOPO && params->maxflow*params->nshortcycle)
     >POSSHORTRANGE){
    fprintf(sp0,"maxflow exceeds range of short int for given nshortcycle\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->costmode==DEFO && ceil(params->defomax*params->nshortcycle)
     >POSSHORTRANGE){
    fprintf(sp0,"defomax exceeds range of short int for given nshortcycle\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->maxnewnodeconst<=0 || params->maxnewnodeconst>1){
    fprintf(sp0,"maxnewnodeconst must be between 0 and 1\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->sourcemode>1 || params->sourcemode<-1){
    fprintf(sp0,"sourcemode must be -1, 0, or 1\n");
    exit(ABNORMAL_EXIT);
  }
  if(infiles->infileformat!=FLOAT_DATA || strlen(infiles->magfile)){
    params->havemagnitude=TRUE;
  }else{
    params->havemagnitude=FALSE;
  }
  if(params->maxnflowcycles==USEMAXCYCLEFRACTION){
    params->maxnflowcycles=LRound(params->maxcyclefraction
				   *nlines/(double )params->ntilerow
				   *linelen/(double )params->ntilecol);
  }
  if(params->initmaxflow==AUTOCALCSTATMAX 
     && !(params->ntilerow==1 && params->ntilecol==1)){
    fprintf(sp0,"initial maximum flow cannot be calculated automatically in "
	    "tile mode\n");
    exit(ABNORMAL_EXIT);
  }
#ifdef NO_CS2
  if(params->initmethod==MCFINIT && !params->unwrapped){
    fprintf(sp0,"program not compiled with cs2 MCF solver module\n");
    exit(ABNORMAL_EXIT);
  }    
#endif  

  /* tile parameters */
  if(params->ntilerow<1 || params->ntilecol<1){
    fprintf(sp0,"numbers of tile rows and columns must be positive\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->rowovrlp<0 || params->colovrlp<0){
    fprintf(sp0,"tile overlaps must be nonnegative\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->ntilerow>1 || params->ntilecol>1){
    ni=ceil((nlines+(params->ntilerow-1)*params->rowovrlp)
	    /(double )params->ntilerow);
    nj=ceil((linelen+(params->ntilecol-1)*params->colovrlp)
	    /(double )params->ntilecol);
    if(params->p>=0){
      fprintf(sp0,"tile mode not enabled for Lp costs\n");
      exit(ABNORMAL_EXIT);
    }
    if(params->ntilerow+params->rowovrlp > nlines 
       || params->ntilecol+params->colovrlp > linelen
       || params->ntilerow*params->ntilerow > nlines
       || params->ntilecol*params->ntilecol > linelen){
      fprintf(sp0,"tiles too small or overlap too large for given input\n");
      exit(ABNORMAL_EXIT);
    }
    if(params->minregionsize 
       > ((nlines-(params->ntilerow-1)*(ni-params->rowovrlp))
	  *(linelen-(params->ntilecol-1)*(nj-params->colovrlp)))){
      fprintf(sp0,"minimum region size too large for given tile parameters\n");
      exit(ABNORMAL_EXIT);
    }
    if(TMPTILEOUTFORMAT!=ALT_LINE_DATA && TMPTILEOUTFORMAT!=FLOAT_DATA){
      fprintf(sp0,"unsupported TMPTILEOUTFORMAT value in complied binary\n");
      exit(ABNORMAL_EXIT);
    }
    if(TMPTILEOUTFORMAT==FLOAT_DATA && outfiles->outfileformat!=FLOAT_DATA){
      fprintf(sp0,"precompiled tile format precludes given output format\n");
      exit(ABNORMAL_EXIT);
    }
    if(params->scndryarcflowmax<1){
      fprintf(sp0,"parameter scndryarcflowmax too small\n");
      exit(ABNORMAL_EXIT);
    }
    if(params->initonly){
      fprintf(sp0,
	      "initialize-only mode and tile mode are mutually exclusive\n");
      exit(ABNORMAL_EXIT);
    }
    if(strlen(outfiles->conncompfile)){
      fprintf(sp0,
	      "connected components output not yet supported for tile mode\n");
      exit(ABNORMAL_EXIT);
    }
    if(params->assembleonly){
      n=strlen(params->tiledir);
      while(--n>0 && params->tiledir[n]=='/'){
	params->tiledir[n]='\0';
      }
      if(!strlen(params->tiledir)){
	fprintf(sp0,"tile directory name cannot have zero length\n");
	exit(ABNORMAL_EXIT);
      }
      if(!strcmp(params->tiledir,"/")){
	StrNCopy(params->tiledir,"",MAXSTRLEN);
      }
    }
    if(params->piecefirstrow!=DEF_PIECEFIRSTROW 
       || params->piecefirstcol!=DEF_PIECEFIRSTCOL
       || params->piecenrow!=DEF_PIECENROW
       || params->piecencol!=DEF_PIECENCOL){
      fprintf(sp0,"piece-only mode cannot be used with multiple tiles\n");
      exit(ABNORMAL_EXIT);
    }
  }else{
    if(params->assembleonly){
      fprintf(sp0,"assemble-only mode can only be used with multiple tiles\n");
      exit(ABNORMAL_EXIT);
    }
    if(params->nthreads>1){
      fprintf(sp0,"only one tile--disregarding multiprocessor option\n");
    }
    if(params->rowovrlp || params->colovrlp){
      fprintf(sp0,"only one tile--disregarding tile overlap values\n");
    }
    params->piecefirstrow--;                   /* index from 0 instead of 1 */
    params->piecefirstcol--;                   /* index from 0 instead of 1 */
    if(!params->piecenrow){
      params->piecenrow=nlines;
    }
    if(!params->piecencol){
      params->piecencol=linelen;
    }
    if(params->piecefirstrow<0 || params->piecefirstcol<0 
       || params->piecenrow<1 || params->piecencol<1
       || params->piecefirstrow+params->piecenrow>nlines
       || params->piecefirstcol+params->piecencol>linelen){
      fprintf(sp0,"illegal values for piece of interferogram to unwrap\n");
      exit(ABNORMAL_EXIT);
    }
  }
  if(params->nthreads<1){
    fprintf(sp0,"number of processors must be at least one\n");
    exit(ABNORMAL_EXIT);
  }else if(params->nthreads>MAXTHREADS){
    fprintf(sp0,"number of processors exceeds precomplied limit of %d\n",
	    MAXTHREADS);
    exit(ABNORMAL_EXIT);
  }

  /* connected component parameters */
  if(params->regrowconncomps){
    if(!strlen(outfiles->conncompfile)){
      fprintf(sp0,"no connected component output file specified\n");
      exit(ABNORMAL_EXIT);
    }      
    params->unwrapped=TRUE;
  }
  if(params->minconncompfrac<0 || params->minconncompfrac>1){
    fprintf(sp0,"illegal value for minimum connected component fraction\n");
    exit(ABNORMAL_EXIT);
  }
  if(params->maxncomps<=0){
    fprintf(sp0,"illegal value for maximum number of  connected components\n");
    exit(ABNORMAL_EXIT);
  }
  if(strlen(outfiles->conncompfile)){
    if(params->initonly){
      fprintf(sp0,"WARNING: connected component mask cannot be generated "
	      "in initialize-only mode\n         mask will not be output\n");
      StrNCopy(outfiles->conncompfile,"",MAXSTRLEN);
    }
    if(params->costmode==NOSTATCOSTS){
      fprintf(sp0,"WARNING: connected component mask cannot be generated "
	      "without statistical costs\n         mask will not be output\n");
      StrNCopy(outfiles->conncompfile,"",MAXSTRLEN);
    }
  }

  /* set global pointers to functions for calculating and evaluating costs */
  if(params->p<0){
    if(params->costmode==TOPO){
      CalcCost=CalcCostTopo;
      EvalCost=EvalCostTopo;
    }else if(params->costmode==DEFO){
      CalcCost=CalcCostDefo;
      EvalCost=EvalCostDefo;
    }else if(params->costmode==SMOOTH){
      CalcCost=CalcCostSmooth;
      EvalCost=EvalCostSmooth;
    }
  }else{
    if(params->p==0){
      CalcCost=CalcCostL0;
      EvalCost=EvalCostL0;
    }else if(params->p==1){
      CalcCost=CalcCostL1;
      EvalCost=EvalCostL1;
    }else if(params->p==2){
      CalcCost=CalcCostL2;
      EvalCost=EvalCostL2;
    }else{
      CalcCost=CalcCostLP;
      EvalCost=EvalCostLP;
    }
  }
}


/* function: ReadConfigFile()
 * --------------------------
 * Read in parameter values from a file, overriding existing parameters.
 */
void ReadConfigFile(char *conffile, infileT *infiles, outfileT *outfiles,
		    long *linelenptr, paramT *params){
  
  long nlines, nparams, nfields;
  FILE *fp;
  char buf[MAXLINELEN];
  char str1[MAXLINELEN], str2[MAXLINELEN];
  char *ptr;
  signed char badparam;

  /* open input config file */
  if(strlen(conffile)){
    if((fp=fopen(conffile,"r"))==NULL){

      /* abort if we were given a non-zero length name that is unreadable */
      fprintf(sp0,"unable to read configuration file %s\n",conffile);
      exit(ABNORMAL_EXIT);
    }
  }else{
    
    /* if we were given a zero-length name, just ignore it and go on */
    return;
  }

  /* read each line and convert the first two fields */
  nlines=0;
  nparams=0;
  badparam=FALSE;
  while(TRUE){

    /* read a line from the file and store it in buffer buf */
    buf[0]='\0';
    ptr=fgets(buf,MAXLINELEN,fp);

    /* break when we read EOF without reading any text */
    if(ptr==NULL && !strlen(buf)){
      break;
    }
    nlines++;

    /* make sure we got the whole line */
    if(strlen(buf)>=MAXLINELEN-1){
      fprintf(sp0,"line %ld in file %s exceeds maximum line length\n",
	      nlines,conffile);
      exit(ABNORMAL_EXIT);
    }
      
    /* read the first two fields */
    /* (str1, str2 same size as buf, so can't overflow them */
    nfields=sscanf(buf,"%s %s",str1,str2);

    /* if only one field is read, and it is not a comment, we have an error */
    if(nfields==1 && isalnum(str1[0])){
      fprintf(sp0,"unrecognized configuration parameter '%s' (%s:%ld)\n",
	      str1,conffile,nlines);
      exit(ABNORMAL_EXIT);
    }

    /* if we have (at least) two non-comment fields */
    if(nfields==2 && isalnum(str1[0])){

      /* do the conversions */
      nparams++;
      if(!strcmp(str1,"INFILE")){
	StrNCopy(infiles->infile,str2,MAXSTRLEN);
      }else if(!strcmp(str1,"OUTFILE")){
	StrNCopy(outfiles->outfile,str2,MAXSTRLEN);
      }else if(!strcmp(str1,"WEIGHTFILE")){
	StrNCopy(infiles->weightfile,str2,MAXSTRLEN);
      }else if(!strcmp(str1,"AMPFILE") || !strcmp(str1,"AMPFILE1")){
	if(strlen(infiles->ampfile2) && !params->amplitude){
	  fprintf(sp0,"cannot specify both amplitude and power\n");
	  exit(ABNORMAL_EXIT);
	}
	StrNCopy(infiles->ampfile,str2,MAXSTRLEN);
      }else if(!strcmp(str1,"AMPFILE2")){
	if(strlen(infiles->ampfile) && !params->amplitude){
	  fprintf(sp0,"cannot specify both amplitude and power\n");
	  exit(ABNORMAL_EXIT);
	}
	StrNCopy(infiles->ampfile2,str2,MAXSTRLEN);
	infiles->ampfileformat=FLOAT_DATA;
      }else if(!strcmp(str1,"PWRFILE") || !strcmp(str1,"PWRFILE1")){
	if(strlen(infiles->ampfile2) && params->amplitude){
	  fprintf(sp0,"cannot specify both amplitude and power\n");
	  exit(ABNORMAL_EXIT);
	}	
	StrNCopy(infiles->ampfile,str2,MAXSTRLEN);
	params->amplitude=FALSE;
      }else if(!strcmp(str1,"PWRFILE2")){
	if(strlen(infiles->ampfile) && params->amplitude){
	  fprintf(sp0,"cannot specify both amplitude and power\n");
	  exit(ABNORMAL_EXIT);
	}	
	StrNCopy(infiles->ampfile2,str2,MAXSTRLEN);
	params->amplitude=FALSE;
	infiles->ampfileformat=FLOAT_DATA;
      }else if(!strcmp(str1,"MAGFILE")){
	StrNCopy(infiles->magfile,str2,MAXSTRLEN);
      }else if(!strcmp(str1,"CORRFILE")){
	StrNCopy(infiles->corrfile,str2,MAXSTRLEN);
      }else if(!strcmp(str1,"ESTIMATEFILE")){
	StrNCopy(infiles->estfile,str2,MAXSTRLEN);
      }else if(!strcmp(str1,"LINELENGTH") || !strcmp(str1,"LINELEN")){
	badparam=StringToLong(str2,linelenptr);
      }else if(!strcmp(str1,"STATCOSTMODE")){
	if(!strcmp(str2,"TOPO")){
	  params->costmode=TOPO;
	}else if(!strcmp(str2,"DEFO")){
	  params->costmode=DEFO;
	}else if(!strcmp(str2,"SMOOTH")){
	  params->costmode=SMOOTH;
	}else if(!strcmp(str2,"NOSTATCOSTS")){
	  params->costmode=NOSTATCOSTS;
	}else{
	  badparam=TRUE;
	}
      }else if(!strcmp(str1,"INITONLY")){
	badparam=SetBooleanSignedChar(&(params->initonly),str2);
      }else if(!strcmp(str1,"UNWRAPPED_IN")){
	badparam=SetBooleanSignedChar(&(params->unwrapped),str2);
      }else if(!strcmp(str1,"DEBUG") || !strcmp(str1,"DUMPALL")){
	badparam=SetBooleanSignedChar(&(params->dumpall),str2);
      }else if(!strcmp(str1,"VERBOSE")){
	badparam=SetBooleanSignedChar(&(params->verbose),str2);
      }else if(!strcmp(str1,"INITMETHOD")){
	if(!strcmp(str2,"MST") || !strcmp(str2,"mst")){
	  params->initmethod=MSTINIT;
	}else if(!strcmp(str2,"MCF") || !strcmp(str2,"mcf") 
		 || !strcmp(str2,"CS2") || !strcmp(str2,"cs2")){
	  params->initmethod=MCFINIT;
	}else{
	  badparam=TRUE;
	}
      }else if(!strcmp(str1,"ORBITRADIUS")){
	if(!(badparam=StringToDouble(str2,&(params->orbitradius)))){
	  params->altitude=0;
	}
      }else if(!strcmp(str1,"ALTITUDE")){
	if(!(badparam=StringToDouble(str2,&(params->altitude)))){
	  params->orbitradius=0;
	}
      }else if(!strcmp(str1,"EARTHRADIUS")){
	badparam=StringToDouble(str2,&(params->earthradius));
      }else if(!strcmp(str1,"BPERP")){
	badparam=StringToDouble(str2,&(params->bperp));
      }else if(!strcmp(str1,"TRANSMITMODE")){
	if(!strcmp(str2,"PINGPONG") || !strcmp(str2,"REPEATPASS")){
	  params->transmitmode=PINGPONG;
	}else if(!strcmp(str2,"SINGLEANTENNATRANSMIT") || !strcmp(str2,"SAT")
		 || !strcmp(str2,"SINGLEANTTRANSMIT")){
	  params->transmitmode=SINGLEANTTRANSMIT;
	}else{
	  badparam=TRUE;
	}
      }else if(!strcmp(str1,"BASELINE")){
	if(!(badparam=StringToDouble(str2,&(params->baseline)))){
	  params->bperp=0;
	}
      }else if(!strcmp(str1,"BASELINEANGLE_RAD")){
	if(!(badparam=StringToDouble(str2,&(params->baselineangle)))){
	  params->bperp=0;
	}
      }else if(!strcmp(str1,"BASELINEANGLE_DEG")){
	if(!(badparam=StringToDouble(str2,&(params->baselineangle)))){
	  (params->baselineangle)*=(PI/180.0);
	  params->bperp=0;
	}
      }else if(!strcmp(str1,"NLOOKSRANGE")){
	badparam=StringToLong(str2,&(params->nlooksrange));
      }else if(!strcmp(str1,"NLOOKSAZ")){
	badparam=StringToLong(str2,&(params->nlooksaz));
      }else if(!strcmp(str1,"NLOOKSOTHER")){
	badparam=StringToLong(str2,&(params->nlooksother));
      }else if(!strcmp(str1,"NCORRLOOKS")){
	badparam=StringToDouble(str2,&(params->ncorrlooks));
      }else if(!strcmp(str1,"NCORRLOOKSRANGE")){
	badparam=StringToLong(str2,&(params->ncorrlooksrange));
      }else if(!strcmp(str1,"NCORRLOOKSAZ")){
	badparam=StringToLong(str2,&(params->ncorrlooksaz));
      }else if(!strcmp(str1,"NEARRANGE") || !strcmp(str1,"NOMRANGE")){
	badparam=StringToDouble(str2,&(params->nearrange));
      }else if(!strcmp(str1,"DR")){
	badparam=StringToDouble(str2,&(params->dr));
      }else if(!strcmp(str1,"DA")){
	badparam=StringToDouble(str2,&(params->da));
      }else if(!strcmp(str1,"RANGERES")){
	badparam=StringToDouble(str2,&(params->rangeres));
      }else if(!strcmp(str1,"AZRES")){
	badparam=StringToDouble(str2,&(params->azres));
      }else if(!strcmp(str1,"LAMBDA")){
	badparam=StringToDouble(str2,&(params->lambda));
      }else if(!strcmp(str1,"KDS") || !strcmp(str1,"KSD")){
	if(!strcmp(str1,"KSD")){
	  fprintf(sp0,"WARNING: parameter KSD interpreted as KDS (%s:%ld)\n",
		  conffile,nlines);
	}
	badparam=StringToDouble(str2,&(params->kds));
      }else if(!strcmp(str1,"SPECULAREXP") || !strcmp(str1,"N")){
	badparam=StringToDouble(str2,&(params->specularexp));
      }else if(!strcmp(str1,"DZRCRITFACTOR")){
	badparam=StringToDouble(str2,&(params->dzrcritfactor));
      }else if(!strcmp(str1,"SHADOW")){
	badparam=SetBooleanSignedChar(&(params->shadow),str2);
      }else if(!strcmp(str1,"DZEIMIN")){
	badparam=StringToDouble(str2,&(params->dzeimin));
      }else if(!strcmp(str1,"LAYWIDTH")){
	badparam=StringToLong(str2,&(params->laywidth));
      }else if(!strcmp(str1,"LAYMINEI")){
	badparam=StringToDouble(str2,&(params->layminei));
      }else if(!strcmp(str1,"SLOPERATIOFACTOR")){
	badparam=StringToDouble(str2,&(params->sloperatiofactor));
      }else if(!strcmp(str1,"SIGSQEI")){
	badparam=StringToDouble(str2,&(params->sigsqei));
      }else if(!strcmp(str1,"DRHO")){
	badparam=StringToDouble(str2,&(params->drho));
      }else if(!strcmp(str1,"RHOSCONST1")){
	badparam=StringToDouble(str2,&(params->rhosconst1));
      }else if(!strcmp(str1,"RHOSCONST2")){
	badparam=StringToDouble(str2,&(params->rhosconst2));
      }else if(!strcmp(str1,"CSTD1")){
	badparam=StringToDouble(str2,&(params->cstd1));
      }else if(!strcmp(str1,"CSTD2")){
	badparam=StringToDouble(str2,&(params->cstd2));
      }else if(!strcmp(str1,"CSTD3")){
	badparam=StringToDouble(str2,&(params->cstd3));
      }else if(!strcmp(str1,"DEFAULTCORR")){
	badparam=StringToDouble(str2,&(params->defaultcorr));
      }else if(!strcmp(str1,"RHOMINFACTOR")){
	badparam=StringToDouble(str2,&(params->rhominfactor));
      }else if(!strcmp(str1,"DZLAYPEAK")){
	badparam=StringToDouble(str2,&(params->dzlaypeak));
      }else if(!strcmp(str1,"AZDZFACTOR")){
	badparam=StringToDouble(str2,&(params->azdzfactor));
      }else if(!strcmp(str1,"DZEIFACTOR")){
	badparam=StringToDouble(str2,&(params->dzeifactor));
      }else if(!strcmp(str1,"DZEIWEIGHT")){
	badparam=StringToDouble(str2,&(params->dzeiweight));
      }else if(!strcmp(str1,"DZLAYFACTOR")){
	badparam=StringToDouble(str2,&(params->dzlayfactor));
      }else if(!strcmp(str1,"LAYCONST")){
	badparam=StringToDouble(str2,&(params->layconst));
      }else if(!strcmp(str1,"LAYFALLOFFCONST")){
	badparam=StringToDouble(str2,&(params->layfalloffconst));
      }else if(!strcmp(str1,"SIGSQSHORTMIN")){
	badparam=StringToLong(str2,&(params->sigsqshortmin));
      }else if(!strcmp(str1,"SIGSQLAYFACTOR")){
	badparam=StringToDouble(str2,&(params->sigsqlayfactor));
      }else if(!strcmp(str1,"DEFOAZDZFACTOR")){
	badparam=StringToDouble(str2,&(params->defoazdzfactor));
      }else if(!strcmp(str1,"DEFOTHRESHFACTOR")){
	badparam=StringToDouble(str2,&(params->defothreshfactor));
      }else if(!strcmp(str1,"DEFOMAX_CYCLE")){
	badparam=StringToDouble(str2,&(params->defomax));
      }else if(!strcmp(str1,"DEFOMAX_RAD")){
	if(!(badparam=StringToDouble(str2,&(params->defomax)))){
	  params->defomax/=TWOPI;
	}
      }else if(!strcmp(str1,"SIGSQCORR")){
	badparam=StringToDouble(str2,&(params->sigsqcorr));
      }else if(!strcmp(str1,"DEFOLAYCONST") || !strcmp(str1,"DEFOCONST")){
	badparam=StringToDouble(str2,&(params->defolayconst));
      }else if(!strcmp(str1,"INITMAXFLOW")){
	badparam=StringToLong(str2,&(params->initmaxflow));
      }else if(!strcmp(str1,"ARCMAXFLOWCONST")){
	badparam=StringToLong(str2,&(params->arcmaxflowconst));
      }else if(!strcmp(str1,"MAXFLOW")){
	badparam=StringToLong(str2,&(params->maxflow));
      }else if(!strcmp(str1,"KROWEI") || !strcmp(str1,"KROW")){
	badparam=StringToLong(str2,&(params->krowei));
      }else if(!strcmp(str1,"KCOLEI") || !strcmp(str1,"KCOL")){
	badparam=StringToLong(str2,&(params->kcolei));
      }else if(!strcmp(str1,"KPERPDPSI")){
	badparam=StringToLong(str2,&(params->kperpdpsi));
      }else if(!strcmp(str1,"KPARDPSI")){
	badparam=StringToLong(str2,&(params->kpardpsi));
      }else if(!strcmp(str1,"THRESHOLD")){
	badparam=StringToDouble(str2,&(params->threshold));
      }else if(!strcmp(str1,"INITDZR")){
	badparam=StringToDouble(str2,&(params->initdzr));
      }else if(!strcmp(str1,"INITDZSTEP")){
	badparam=StringToDouble(str2,&(params->initdzstep));
      }else if(!strcmp(str1,"MAXCOST")){
	badparam=StringToDouble(str2,&(params->maxcost));
      }else if(!strcmp(str1,"COSTSCALE")){
	badparam=StringToDouble(str2,&(params->costscale));
      }else if(!strcmp(str1,"COSTSCALEAMBIGHT")){
	badparam=StringToDouble(str2,&(params->costscaleambight));
      }else if(!strcmp(str1,"DNOMINCANGLE")){
	badparam=StringToDouble(str2,&(params->dnomincangle));
      }else if(!strcmp(str1,"CS2SCALEFACTOR")){
	badparam=StringToLong(str2,&(params->cs2scalefactor));
      }else if(!strcmp(str1,"PIECEFIRSTROW")){
	badparam=StringToLong(str2,&(params->piecefirstrow));
      }else if(!strcmp(str1,"PIECEFIRSTCOL")){
	badparam=StringToLong(str2,&(params->piecefirstcol));
      }else if(!strcmp(str1,"PIECENROW")){
	badparam=StringToLong(str2,&(params->piecenrow));
      }else if(!strcmp(str1,"PIECENCOL")){
	badparam=StringToLong(str2,&(params->piecencol));
      }else if(!strcmp(str1,"NTILEROW")){
	badparam=StringToLong(str2,&(params->ntilerow));
      }else if(!strcmp(str1,"NTILECOL")){
	badparam=StringToLong(str2,&(params->ntilecol));
      }else if(!strcmp(str1,"ROWOVRLP")){
	badparam=StringToLong(str2,&(params->rowovrlp));
      }else if(!strcmp(str1,"COLOVRLP")){
	badparam=StringToLong(str2,&(params->colovrlp));
      }else if(!strcmp(str1,"TILECOSTTHRESH")){
	badparam=StringToLong(str2,&(params->tilecostthresh));
      }else if(!strcmp(str1,"MINREGIONSIZE")){
	badparam=StringToLong(str2,&(params->minregionsize));
      }else if(!strcmp(str1,"TILEEDGEWEIGHT")){
	badparam=StringToDouble(str2,&(params->tileedgeweight));
      }else if(!strcmp(str1,"SCNDRYARCFLOWMAX")){
	badparam=StringToLong(str2,&(params->scndryarcflowmax));	
      }else if(!strcmp(str1,"ASSEMBLEONLY")){
	if(!strcmp(str2,"FALSE")){
	  params->assembleonly=FALSE;
	}else{
	  params->assembleonly=TRUE;
	  StrNCopy(params->tiledir,str2,MAXSTRLEN);
	}
      }else if(!strcmp(str1,"RMTMPTILE")){
	badparam=SetBooleanSignedChar(&(params->rmtmptile),str2);
      }else if(!strcmp(str1,"MINCONNCOMPFRAC")){
	badparam=StringToDouble(str2,&(params->minconncompfrac));
      }else if(!strcmp(str1,"CONNCOMPTHRESH")){
	badparam=StringToLong(str2,&(params->conncompthresh));
      }else if(!strcmp(str1,"MAXNCOMPS")){
	badparam=StringToLong(str2,&(params->maxncomps));
      }else if(!strcmp(str1,"NSHORTCYCLE")){
	badparam=StringToLong(str2,&(params->nshortcycle));
      }else if(!strcmp(str1,"MAXNEWNODECONST")){
	badparam=StringToDouble(str2,&(params->maxnewnodeconst));
      }else if(!strcmp(str1,"MAXNFLOWCYCLES")){
	badparam=StringToLong(str2,&(params->maxnflowcycles));
      }else if(!strcmp(str1,"MAXCYCLEFRACTION")){
	badparam=StringToDouble(str2,&(params->maxcyclefraction));
	params->maxnflowcycles=USEMAXCYCLEFRACTION;
      }else if(!strcmp(str1,"SOURCEMODE")){
	badparam=StringToLong(str2,&(params->sourcemode));
      }else if(!strcmp(str1,"NPROC") || !strcmp(str1,"NTHREADS")){
	badparam=StringToLong(str2,&(params->nthreads));
      }else if(!strcmp(str1,"COSTINFILE")){
	StrNCopy(infiles->costinfile,str2,MAXSTRLEN);
      }else if(!strcmp(str1,"COSTOUTFILE")){
	StrNCopy(outfiles->costoutfile,str2,MAXSTRLEN);
      }else if(!strcmp(str1,"LOGFILE")){
	StrNCopy(outfiles->logfile,str2,MAXSTRLEN);
      }else if(!strcmp(str1,"INFILEFORMAT")){
	if(!strcmp(str2,"COMPLEX_DATA")){
	  infiles->infileformat=COMPLEX_DATA;
	}else if(!strcmp(str2,"FLOAT_DATA")){
	  infiles->infileformat=FLOAT_DATA;
	}else if(!strcmp(str2,"ALT_LINE_DATA")){
	  infiles->infileformat=ALT_LINE_DATA;
	}else if(!strcmp(str2,"ALT_SAMPLE_DATA")){
	  infiles->infileformat=ALT_SAMPLE_DATA;
	}else{
	  badparam=TRUE;
	}
      }else if(!strcmp(str1,"UNWRAPPEDINFILEFORMAT")){
	if(!strcmp(str2,"ALT_LINE_DATA")){
	  infiles->unwrappedinfileformat=ALT_LINE_DATA;
	}else if(!strcmp(str2,"ALT_SAMPLE_DATA")){
	  infiles->unwrappedinfileformat=ALT_SAMPLE_DATA;
	}else if(!strcmp(str2,"FLOAT_DATA")){
	  infiles->unwrappedinfileformat=FLOAT_DATA;
	}else{
	  badparam=TRUE;
	}
      }else if(!strcmp(str1,"MAGFILEFORMAT")){
	if(!strcmp(str2,"ALT_LINE_DATA")){
	  infiles->magfileformat=ALT_LINE_DATA;
	}else if(!strcmp(str2,"ALT_SAMPLE_DATA")){
	  infiles->magfileformat=ALT_SAMPLE_DATA;
	}else if(!strcmp(str2,"FLOAT_DATA")){
	  infiles->magfileformat=FLOAT_DATA;
	}else if(!strcmp(str2,"COMPLEX_DATA")){
	  infiles->magfileformat=COMPLEX_DATA;
	}else{
	  badparam=TRUE;
	}
      }else if(!strcmp(str1,"OUTFILEFORMAT")){
	if(!strcmp(str2,"ALT_LINE_DATA")){
	  outfiles->outfileformat=ALT_LINE_DATA;
	}else if(!strcmp(str2,"ALT_SAMPLE_DATA")){
	  outfiles->outfileformat=ALT_SAMPLE_DATA;
	}else if(!strcmp(str2,"FLOAT_DATA")){
	  outfiles->outfileformat=FLOAT_DATA;
	}else{
	  badparam=TRUE;
	}
      }else if(!strcmp(str1,"CORRFILEFORMAT")){
	if(!strcmp(str2,"ALT_LINE_DATA")){
	  infiles->corrfileformat=ALT_LINE_DATA;
	}else if(!strcmp(str2,"ALT_SAMPLE_DATA")){
	  infiles->corrfileformat=ALT_SAMPLE_DATA;
	}else if(!strcmp(str2,"FLOAT_DATA")){
	  infiles->corrfileformat=FLOAT_DATA;
	}else{
	  badparam=TRUE;
	}
      }else if(!strcmp(str1,"AMPFILEFORMAT")){
	if(!strcmp(str2,"ALT_LINE_DATA")){
	  infiles->ampfileformat=ALT_LINE_DATA;
	}else if(!strcmp(str2,"ALT_SAMPLE_DATA")){
	  infiles->ampfileformat=ALT_SAMPLE_DATA;
	}else if(!strcmp(str2,"FLOAT_DATA")){
	  infiles->ampfileformat=FLOAT_DATA;
	}else{
	  badparam=TRUE;
	}
      }else if(!strcmp(str1,"ESTFILEFORMAT")){
	if(!strcmp(str2,"ALT_LINE_DATA")){
	  infiles->estfileformat=ALT_LINE_DATA;
	}else if(!strcmp(str2,"ALT_SAMPLE_DATA")){
	  infiles->estfileformat=ALT_SAMPLE_DATA;
	}else if(!strcmp(str2,"FLOAT_DATA")){
	  infiles->estfileformat=FLOAT_DATA;
	}else{
	  badparam=TRUE;
	}
      }else if(!strcmp(str1,"INITFILE")){
	StrNCopy(outfiles->initfile,str2,MAXSTRLEN);
      }else if(!strcmp(str1,"FLOWFILE")){
	StrNCopy(outfiles->flowfile,str2,MAXSTRLEN);
      }else if(!strcmp(str1,"EIFILE")){
	StrNCopy(outfiles->eifile,str2,MAXSTRLEN);
      }else if(!strcmp(str1,"ROWCOSTFILE")){
	StrNCopy(outfiles->rowcostfile,str2,MAXSTRLEN);
      }else if(!strcmp(str1,"COLCOSTFILE")){
	StrNCopy(outfiles->colcostfile,str2,MAXSTRLEN);
      }else if(!strcmp(str1,"MSTROWCOSTFILE")){
	StrNCopy(outfiles->mstrowcostfile,str2,MAXSTRLEN);
      }else if(!strcmp(str1,"MSTCOLCOSTFILE")){
	StrNCopy(outfiles->mstcolcostfile,str2,MAXSTRLEN);
      }else if(!strcmp(str1,"MSTCOSTSFILE")){
	StrNCopy(outfiles->mstcostsfile,str2,MAXSTRLEN);
      }else if(!strcmp(str1,"CORRDUMPFILE")){
	StrNCopy(outfiles->corrdumpfile,str2,MAXSTRLEN);
      }else if(!strcmp(str1,"RAWCORRDUMPFILE")){
	StrNCopy(outfiles->rawcorrdumpfile,str2,MAXSTRLEN);
      }else if(!strcmp(str1,"CONNCOMPFILE")){
	StrNCopy(outfiles->conncompfile,str2,MAXSTRLEN);
      }else if(!strcmp(str1,"REGROWCONNCOMPS")){
	badparam=SetBooleanSignedChar(&(params->regrowconncomps),str2);
      }else{
	fprintf(sp0,"unrecognized configuration parameter '%s' (%s:%ld)\n",
		str1,conffile,nlines);
	exit(ABNORMAL_EXIT);
      }

      /* give an error if we had trouble interpreting the line */
      if(badparam){
	fprintf(sp0,"illegal argument %s for parameter %s (%s:%ld)\n",
		str2,str1,conffile,nlines);
	exit(ABNORMAL_EXIT);
      }

    }
  }

  /* finish up */
  fclose(fp);
  if(nparams>1){
    fprintf(sp1,"%ld parameters input from file %s (%ld lines total)\n",
	    nparams,conffile,nlines);
  }else{
    if(nlines>1){
      fprintf(sp1,"%ld parameter input from file %s (%ld lines total)\n",
	      nparams,conffile,nlines);
    }else{
      fprintf(sp1,"%ld parameter input from file %s (%ld line total)\n",
	      nparams,conffile,nlines);
    }
  }

}


/* function: WriteConfigLogFile()
 * ------------------------------
 * Writes a text log file of configuration parameters and other
 * information.  The log file is in a format compatible to be used as
 * a configuration file.  
 */
void WriteConfigLogFile(int argc, char *argv[], infileT *infiles, 
			outfileT *outfiles, long linelen, paramT *params){

  FILE *fp;
  time_t t[1];
  long k;
  char buf[MAXSTRLEN], *ptr;

  /* see if we need to write a log file */
  if(strlen(outfiles->logfile)){

    /* open the log file */
    if((fp=fopen(outfiles->logfile,"w"))==NULL){
      fprintf(sp0,"unable to write to log file %s\n",outfiles->logfile);
      exit(ABNORMAL_EXIT);
    }
    fprintf(sp1,"Logging run-time parameters to file %s\n",outfiles->logfile);
    
    /* print some run-time environment information */
    fprintf(fp,"# %s v%s\n",PROGRAMNAME,VERSION);
    time(t);
    fprintf(fp,"# Log file generated %s",ctime(t));
    ptr=getcwd(buf,MAXSTRLEN);
    if(ptr!=NULL){
      fprintf(fp,"# Current working directory: %s\n",buf);
    }else{
      fprintf(fp,"# Could not determine current working directory\n");
    }
    fprintf(fp,"# Command line call:");
    for(k=0;k<argc;k++){
      fprintf(fp," %s",argv[k]);
    }
    fprintf(fp,"\n\n");

    /* print an entry for each run-time parameter */
    /* input and output files and main runtime options */
    fprintf(fp,"# File input and output and runtime options\n");
    LogStringParam(fp,"INFILE",infiles->infile);
    fprintf(fp,"LINELENGTH  %ld\n",linelen);
    LogStringParam(fp,"OUTFILE",outfiles->outfile);
    LogStringParam(fp,"WEIGHTFILE",infiles->weightfile);
    if(params->amplitude){
      if(strlen(infiles->ampfile2)){
	LogStringParam(fp,"AMPFILE1",infiles->ampfile);
	LogStringParam(fp,"AMPFILE2",infiles->ampfile2);
      }else{
	LogStringParam(fp,"AMPFILE",infiles->ampfile);
      }
    }else{
      if(strlen(infiles->ampfile2)){
	LogStringParam(fp,"PWRFILE1",infiles->ampfile);
	LogStringParam(fp,"PWRFILE2",infiles->ampfile2);
      }else{
	LogStringParam(fp,"PWRFILE",infiles->ampfile);
      }
    }
    LogStringParam(fp,"MAGFILE",infiles->magfile);
    LogStringParam(fp,"CORRFILE",infiles->corrfile);
    LogStringParam(fp,"ESTIMATEFILE",infiles->estfile);
    LogStringParam(fp,"COSTINFILE",infiles->costinfile);
    LogStringParam(fp,"COSTOUTFILE",outfiles->costoutfile);
    LogStringParam(fp,"LOGFILE",outfiles->logfile);
    if(params->costmode==TOPO){
      fprintf(fp,"STATCOSTMODE  TOPO\n");
    }else if(params->costmode==DEFO){
      fprintf(fp,"STATCOSTMODE  DEFO\n");
    }else if(params->costmode==SMOOTH){
      fprintf(fp,"STATCOSTMODE  SMOOTH\n");
    }else if(params->costmode==NOSTATCOSTS){
      fprintf(fp,"STATCOSTMODE  NOSTATCOSTS\n");
    }
    LogBoolParam(fp,"INITONLY",params->initonly);
    LogBoolParam(fp,"UNWRAPPED_IN",params->unwrapped);
    LogBoolParam(fp,"DEBUG",params->dumpall);
    if(params->initmethod==MSTINIT){
      fprintf(fp,"INITMETHOD  MST\n");
    }else if(params->initmethod==MCFINIT){
      fprintf(fp,"INITMETHOD  MCF\n");
    }
    LogBoolParam(fp,"VERBOSE",params->verbose);

    /* file formats */
    fprintf(fp,"\n# File Formats\n");
    LogFileFormat(fp,"INFILEFORMAT",infiles->infileformat);
    LogFileFormat(fp,"OUTFILEFORMAT",outfiles->outfileformat);
    LogFileFormat(fp,"AMPFILEFORMAT",infiles->ampfileformat);
    LogFileFormat(fp,"MAGFILEFORMAT",infiles->magfileformat);
    LogFileFormat(fp,"CORRFILEFORMAT",infiles->corrfileformat);
    LogFileFormat(fp,"ESTFILEFORMAT",infiles->estfileformat);
    LogFileFormat(fp,"UNWRAPPEDINFILEFORMAT",infiles->unwrappedinfileformat);

    /* SAR and geometry parameters */
    fprintf(fp,"\n# SAR and Geometry Parameters\n");
    fprintf(fp,"ALTITUDE  %.8f\n",
	    params->orbitradius-params->earthradius);
    fprintf(fp,"# ORBITRADIUS  %.8f\n",params->orbitradius);
    fprintf(fp,"EARTHRADIUS  %.8f\n",params->earthradius);
    if(params->bperp){
      fprintf(fp,"BPERP  %.8f\n",params->bperp);
    }else{
      fprintf(fp,"BASELINE %.8f\n",params->baseline);
      fprintf(fp,"BASELINEANGLE_DEG %.8f\n",
	      params->baselineangle*(180.0/PI));
    }
    if(params->transmitmode==PINGPONG){
      fprintf(fp,"TRANSMITMODE  REPEATPASS\n");
    }else if(params->transmitmode==SINGLEANTTRANSMIT){
      fprintf(fp,"TRANSMITMODE  SINGLEANTENNATRANSMIT\n");
    }
    fprintf(fp,"NEARRANGE  %.8f\n",params->nearrange);
    fprintf(fp,"DR  %.8f\n",params->dr);
    fprintf(fp,"DA  %.8f\n",params->da);
    fprintf(fp,"RANGERES  %.8f\n",params->rangeres);
    fprintf(fp,"AZRES  %.8f\n",params->azres);
    fprintf(fp,"LAMBDA  %.8f\n",params->lambda);
    fprintf(fp,"NLOOKSRANGE  %ld\n",params->nlooksrange);
    fprintf(fp,"NLOOKSAZ  %ld\n",params->nlooksaz);
    fprintf(fp,"NLOOKSOTHER  %ld\n",params->nlooksother);
    fprintf(fp,"NCORRLOOKS  %.8f\n",params->ncorrlooks);
    fprintf(fp,"NCORRLOOKSRANGE  %ld\n",params->ncorrlooksrange);
    fprintf(fp,"NCORRLOOKSAZ  %ld\n",params->ncorrlooksaz);
      
    /* scattering model parameters */
    fprintf(fp,"\n# Scattering model parameters\n");
    fprintf(fp,"KDS  %.8f\n",params->kds);
    fprintf(fp,"SPECULAREXP  %.8f\n",params->specularexp);
    fprintf(fp,"DZRCRITFACTOR  %.8f\n",params->dzrcritfactor);
    LogBoolParam(fp,"SHADOW",params->shadow);
    fprintf(fp,"DZEIMIN  %.8f\n",params->dzeimin);
    fprintf(fp,"LAYWIDTH  %ld\n",params->laywidth);
    fprintf(fp,"LAYMINEI  %.8f\n",params->layminei);
    fprintf(fp,"SLOPERATIOFACTOR  %.8f\n",params->sloperatiofactor);
    fprintf(fp,"SIGSQEI  %.8f\n",params->sigsqei);
    
    /* decorrelation model paramters */
    fprintf(fp,"\n# Decorrelation model parameters\n");
    fprintf(fp,"DRHO  %.8f\n",params->drho);
    fprintf(fp,"RHOSCONST1  %.8f\n",params->rhosconst1);
    fprintf(fp,"RHOSCONST2  %.8f\n",params->rhosconst2);
    fprintf(fp,"CSTD1  %.8f\n",params->cstd1);
    fprintf(fp,"CSTD2  %.8f\n",params->cstd2);
    fprintf(fp,"CSTD3  %.8f\n",params->cstd3);
    fprintf(fp,"DEFAULTCORR  %.8f\n",params->defaultcorr);
    fprintf(fp,"RHOMINFACTOR  %.8f\n",params->rhominfactor);
      
    /* PDF model paramters */
    fprintf(fp,"\n# PDF model parameters\n");
    fprintf(fp,"DZLAYPEAK  %.8f\n",params->dzlaypeak);
    fprintf(fp,"AZDZFACTOR  %.8f\n",params->azdzfactor);
    fprintf(fp,"DZEIFACTOR  %.8f\n",params->dzeifactor);
    fprintf(fp,"DZEIWEIGHT  %.8f\n",params->dzeiweight);
    fprintf(fp,"DZLAYFACTOR  %.8f\n",params->dzlayfactor);
    fprintf(fp,"LAYCONST  %.8f\n",params->layconst);
    fprintf(fp,"LAYFALLOFFCONST  %.8f\n",params->layfalloffconst);
    fprintf(fp,"SIGSQSHORTMIN  %ld\n",params->sigsqshortmin);
    fprintf(fp,"SIGSQLAYFACTOR  %.8f\n",params->sigsqlayfactor);

    /* deformation mode paramters */
    fprintf(fp,"\n# Deformation mode parameters\n");
    fprintf(fp,"DEFOAZDZFACTOR  %.8f\n",params->defoazdzfactor);
    fprintf(fp,"DEFOTHRESHFACTOR  %.8f\n",params->defothreshfactor);
    fprintf(fp,"DEFOMAX_CYCLE  %.8f\n",params->defomax);
    fprintf(fp,"SIGSQCORR  %.8f\n",params->sigsqcorr);
    fprintf(fp,"DEFOCONST  %.8f\n",params->defolayconst);

    /* algorithm parameters */
    fprintf(fp,"\n# Algorithm parameters\n");
    fprintf(fp,"INITMAXFLOW  %ld\n",params->initmaxflow);
    fprintf(fp,"ARCMAXFLOWCONST  %ld\n",params->arcmaxflowconst);
    fprintf(fp,"MAXFLOW  %ld\n",params->maxflow);
    fprintf(fp,"KROWEI  %ld\n",params->krowei);
    fprintf(fp,"KCOLEI  %ld\n",params->kcolei);
    fprintf(fp,"KPARDPSI  %ld\n",params->kpardpsi);
    fprintf(fp,"KPERPDPSI  %ld\n",params->kperpdpsi);
    fprintf(fp,"THRESHOLD  %.8f\n",params->threshold);
    fprintf(fp,"INITDZR  %.8f\n",params->initdzr);
    fprintf(fp,"INITDZSTEP  %.8f\n",params->initdzstep);
    fprintf(fp,"MAXCOST  %.8f\n",params->maxcost);
    fprintf(fp,"COSTSCALE  %.8f\n",params->costscale);
    fprintf(fp,"COSTSCALEAMBIGHT  %.8f\n",params->costscaleambight);
    fprintf(fp,"DNOMINCANGLE  %.8f\n",params->dnomincangle);
    fprintf(fp,"NSHORTCYCLE  %ld\n",params->nshortcycle);
    fprintf(fp,"MAXNEWNODECONST  %.8f\n",params->maxnewnodeconst);
    if(params->maxnflowcycles==USEMAXCYCLEFRACTION){
      fprintf(fp,"MAXCYCLEFRACTION  %.8f\n",params->maxcyclefraction);
    }else{
      fprintf(fp,"MAXNFLOWCYCLES  %ld\n",params->maxnflowcycles);
    }
    fprintf(fp,"SOURCEMODE  %ld\n",params->sourcemode);
    fprintf(fp,"CS2SCALEFACTOR  %ld\n",params->cs2scalefactor);
      
    /* file names for dumping intermediate arrays */
    fprintf(fp,"\n# File names for dumping intermediate arrays\n");
    LogStringParam(fp,"INITFILE",outfiles->initfile);
    LogStringParam(fp,"FLOWFILE",outfiles->flowfile);
    LogStringParam(fp,"EIFILE",outfiles->eifile);
    LogStringParam(fp,"ROWCOSTFILE",outfiles->rowcostfile);
    LogStringParam(fp,"COLCOSTFILE",outfiles->colcostfile);
    LogStringParam(fp,"MSTROWCOSTFILE",outfiles->mstrowcostfile);
    LogStringParam(fp,"MSTCOLCOSTFILE",outfiles->mstcolcostfile);
    LogStringParam(fp,"MSTCOSTSFILE",outfiles->mstcostsfile);
    LogStringParam(fp,"RAWCORRDUMPFILE",outfiles->rawcorrdumpfile);
    LogStringParam(fp,"CORRDUMPFILE",outfiles->corrdumpfile);

    /* piece extraction parameters */
    if(params->ntilerow==1 && params->ntilecol==1){
      fprintf(fp,"\n# Piece extraction parameters\n");
      fprintf(fp,"PIECEFIRSTROW  %ld\n",params->piecefirstrow+1);
      fprintf(fp,"PIECEFIRSTCOL  %ld\n",params->piecefirstcol+1);
      fprintf(fp,"PIECENROW  %ld\n",params->piecenrow);
      fprintf(fp,"PIECENCOL  %ld\n",params->piecencol);
    }else{
      fprintf(fp,"\n# Piece extraction parameters\n");
      fprintf(fp,"# Parameters ignored because of tile mode\n");
      fprintf(fp,"# PIECEFIRSTROW  %ld\n",params->piecefirstrow);
      fprintf(fp,"# PIECEFIRSTCOL  %ld\n",params->piecefirstcol);
      fprintf(fp,"# PIECENROW  %ld\n",params->piecenrow);
      fprintf(fp,"# PIECENCOL  %ld\n",params->piecencol);
    }


    /* tile control */
    fprintf(fp,"\n# Tile control\n");
    fprintf(fp,"NTILEROW  %ld\n",params->ntilerow);
    fprintf(fp,"NTILECOL  %ld\n",params->ntilecol);
    fprintf(fp,"ROWOVRLP  %ld\n",params->rowovrlp);
    fprintf(fp,"COLOVRLP  %ld\n",params->colovrlp);
    fprintf(fp,"NPROC  %ld\n",params->nthreads);
    fprintf(fp,"TILECOSTTHRESH  %ld\n",params->tilecostthresh);
    fprintf(fp,"MINREGIONSIZE  %ld\n",params->minregionsize);
    fprintf(fp,"TILEEDGEWEIGHT  %.8f\n",params->tileedgeweight);
    fprintf(fp,"SCNDRYARCFLOWMAX  %ld\n",params->scndryarcflowmax);
    LogBoolParam(fp,"RMTMPTILE",params->rmtmptile);
    if(params->assembleonly){
      LogStringParam(fp,"ASSEMBLEONLY",params->tiledir);
    }else{
      fprintf(fp,"ASSEMBLEONLY  FALSE\n");
    }

    /* connected component control */
    fprintf(fp,"\n# Connected component control\n");
    LogStringParam(fp,"CONNCOMPFILE",outfiles->conncompfile);
    LogBoolParam(fp,"REGROWCONNCOMPS",params->regrowconncomps);
    fprintf(fp,"MINCONNCOMPFRAC  %.8f\n",params->minconncompfrac);
    fprintf(fp,"CONNCOMPTHRESH  %ld\n",params->conncompthresh);
    fprintf(fp,"MAXNCOMPS  %ld\n",params->maxncomps);

    /* close the log file */
    fclose(fp);
  }
}


/* function: LogStringParam()
 * --------------------------
 * Writes a line to the log file stream for the given keyword/value 
 * pair.
 */
void LogStringParam(FILE *fp, char *key, char *value){

  /* see if we were passed a zero length value string */
  if(strlen(value)){
    fprintf(fp,"%s  %s\n",key,value);
    fflush(fp);
  }else{
    fprintf(fp,"# Empty value for parameter %s\n",key);
  }
}


/* LogBoolParam()
 * --------------
 * Writes a line to the log file stream for the given keyword/bool
 * pair.
 */
void LogBoolParam(FILE *fp, char *key, signed char boolvalue){

  if(boolvalue){
    fprintf(fp,"%s  TRUE\n",key);
  }else{
    fprintf(fp,"%s  FALSE\n",key);
  }
}

/* LogFileFormat()
 * ---------------
 * Writes a line to the log file stream for the given keyword/
 * file format pair.
 */
void LogFileFormat(FILE *fp, char *key, signed char fileformat){
  
  if(fileformat==COMPLEX_DATA){
    fprintf(fp,"%s  COMPLEX_DATA\n",key);
  }else if(fileformat==FLOAT_DATA){
    fprintf(fp,"%s  FLOAT_DATA\n",key);
  }else if(fileformat==ALT_LINE_DATA){
    fprintf(fp,"%s  ALT_LINE_DATA\n",key);
  }else if(fileformat==ALT_SAMPLE_DATA){
    fprintf(fp,"%s  ALT_SAMPLE_DATA\n",key);
  }
}


/* function: GetNLines() 
 * ---------------------
 * Gets the number of lines of data in the input file based on the file 
 * size.
 */
long GetNLines(infileT *infiles, long linelen){

  FILE *fp;
  long filesize, datasize;

  /* get size of input file in rows and columns */
  if((fp=fopen(infiles->infile,"r"))==NULL){
    fprintf(sp0,"can't open file %s\n",infiles->infile);
    exit(ABNORMAL_EXIT);
  }
  fseek(fp,0,SEEK_END);
  filesize=ftell(fp);
  fclose(fp);
  if(infiles->infileformat==FLOAT_DATA){
    datasize=sizeof(float);
  }else{
    datasize=2*sizeof(float);
  }
  if(filesize % (datasize*linelen)){
    fprintf(sp0,"extra data in file %s (bad linelength?)\n",
	    infiles->infile);
    exit(ABNORMAL_EXIT);
  }
  return(filesize/(datasize*linelen));               /* implicit floor */

}


/* function: WriteOutputFile()
 * ---------------------------
 * Writes the unwrapped phase to the output file specified, in the
 * format given in the parameter structure.
 */
void WriteOutputFile(float **mag, float **unwrappedphase, char *outfile,
		     outfileT *outfiles, long nrow, long ncol){

  if(outfiles->outfileformat==ALT_LINE_DATA){
    WriteAltLineFile(mag,unwrappedphase,outfile,nrow,ncol);
  }else if(outfiles->outfileformat==ALT_SAMPLE_DATA){
    WriteAltSampFile(mag,unwrappedphase,outfile,nrow,ncol);
  }else if(outfiles->outfileformat==FLOAT_DATA){
    Write2DArray((void **)unwrappedphase,outfile,
		 nrow,ncol,sizeof(float));
  }else{
    fprintf(sp0,"WARNING: Illegal format specified for output file\n");
    fprintf(sp0,"         using default floating-point format\n");
    Write2DArray((void **)unwrappedphase,outfile,
		 nrow,ncol,sizeof(float));
  }
}


/* function: OpenOutputFile()
 * --------------------------
 * Opens a file for writing.  If unable to open the file, tries to 
 * open a file in a dump path.  The name of the opened output file
 * is written into the string realoutfile, for which at least 
 * MAXSTRLEN bytes should already be allocated.
 */
FILE *OpenOutputFile(char *outfile, char *realoutfile){

  char path[MAXSTRLEN], basename[MAXSTRLEN], dumpfile[MAXSTRLEN];
  FILE *fp;

  if((fp=fopen(outfile,"w"))==NULL){

    /* if we can't write to the out file, get the file name from the path */
    /* and dump to the default path */
    ParseFilename(outfile,path,basename);
    StrNCopy(dumpfile,DUMP_PATH,MAXSTRLEN);
    strcat(dumpfile,basename);
    if((fp=fopen(dumpfile,"w"))!=NULL){
      fprintf(sp0,"WARNING: Can't write to file %s.  Dumping to file %s\n",
	     outfile,dumpfile);
      StrNCopy(realoutfile,dumpfile,MAXSTRLEN);
    }else{
      fprintf(sp0,"Unable to write to file %s or dump to file %s\nAbort\n",
	     outfile,dumpfile);
      exit(ABNORMAL_EXIT);
    }
  }else{
    StrNCopy(realoutfile,outfile,MAXSTRLEN);
  }
  return(fp);

}


/* function: WriteAltLineFile()
 * ----------------------------
 * Writes magnitude and phase data from separate arrays to file.
 * Data type is float.  For each line of data, a full line of magnitude data
 * is written, then a full line of phase data.  Dumps the file to a 
 * default directory if the file name/path passed in cannot be used.
 */
void WriteAltLineFile(float **mag, float **phase, char *outfile, 
		      long nrow, long ncol){

  int row;
  FILE *fp;
  char realoutfile[MAXSTRLEN];

  fp=OpenOutputFile(outfile,realoutfile);
  for(row=0; row<nrow; row++){
    if(fwrite(mag[row],sizeof(float),ncol,fp)!=ncol
       || fwrite(phase[row],sizeof(float),ncol,fp)!=ncol){
      fprintf(sp0,"Error while writing to file %s (device full?)\nAbort\n",
	      realoutfile);
      exit(ABNORMAL_EXIT);
    }
  }
  fclose(fp);
}


/* function: WriteAltSampFile()
 * ----------------------------
 * Writes data from separate arrays to file, alternating samples.
 * Data type is float.  nrow and ncol are the sizes of each input
 * array.  Dumps the file to a default directory if the file name/path 
 * passed in cannot be used.
 */
void WriteAltSampFile(float **arr1, float **arr2, char *outfile, 
		      long nrow, long ncol){

  long row, col;
  FILE *fp;
  float *outline;
  char realoutfile[MAXSTRLEN];

  outline=MAlloc(2*ncol*sizeof(float));
  fp=OpenOutputFile(outfile,realoutfile);
  for(row=0; row<nrow; row++){
    for(col=0;col<ncol;col++){
      outline[2*col]=arr1[row][col];
      outline[2*col+1]=arr2[row][col];
    }
    if(fwrite(outline,sizeof(float),2*ncol,fp)!=2*ncol){
      fprintf(sp0,"Error while writing to file %s (device full?)\nAbort\n",
	      realoutfile);
      exit(ABNORMAL_EXIT);
    }
  }
  fclose(fp);
}


/* function: Write2DArray()
 * ------------------------ 
 * Write data in a two dimensional array to a file.  Data elements are
 * have the number of bytes specified by size (use sizeof() when 
 * calling this function.  
 */
void Write2DArray(void **array, char *filename, long nrow, long ncol, 
		  size_t size){

  int row;
  FILE *fp;
  char realoutfile[MAXSTRLEN];

  fp=OpenOutputFile(filename,realoutfile);
  for(row=0; row<nrow; row++){
    if(fwrite(array[row],size,ncol,fp)!=ncol){
      fprintf(sp0,"Error while writing to file %s (device full?)\nAbort\n",
	      realoutfile);
      exit(ABNORMAL_EXIT);
    }
  }
  fclose(fp);
}


/* function: Write2DRowColArray()
 * ------------------------------ 
 * Write data in a 2-D row-and-column array to a file.  Data elements 
 * have the number of bytes specified by size (use sizeof() when 
 * calling this function.  The format of the array is nrow-1 rows
 * of ncol elements, followed by nrow rows of ncol-1 elements each.
 */
void Write2DRowColArray(void **array, char *filename, long nrow, 
			long ncol, size_t size){

  int row;
  FILE *fp;
  char realoutfile[MAXSTRLEN];

  fp=OpenOutputFile(filename,realoutfile);
  for(row=0; row<nrow-1; row++){
    if(fwrite(array[row],size,ncol,fp)!=ncol){
      fprintf(sp0,"Error while writing to file %s (device full?)\nAbort\n",
	      realoutfile);
      exit(ABNORMAL_EXIT);
    }
  }
  for(row=nrow-1; row<2*nrow-1; row++){
    if(fwrite(array[row],size,ncol-1,fp)!=ncol-1){
      fprintf(sp0,"Error while writing to file %s (device full?)\nAbort\n",
	      realoutfile);
      exit(ABNORMAL_EXIT);
    }
  }
  fclose(fp);
}


/* function: ReadInputFile()
 * -------------------------
 * Reads the input file specified on the command line.
 */
void ReadInputFile(infileT *infiles, float ***magptr, float ***wrappedphaseptr,
 		   short ***flowsptr, long linelen, long nlines, 
		   paramT *params, tileparamT *tileparams){

  long row, col, nrow, ncol;
  float **mag, **wrappedphase, **unwrappedphase;
  short **flows;

  /* initialize */
  mag=NULL;
  wrappedphase=NULL;
  unwrappedphase=NULL;
  flows=NULL;
  nrow=tileparams->nrow;
  ncol=tileparams->ncol;

  /* check data size */
  if(tileparams->ncol>LARGESHORT || tileparams->nrow>LARGESHORT){
    fprintf(sp0,"one or more interferogram dimensions too large\n");
    exit(ABNORMAL_EXIT);
  }
  if(tileparams->ncol<2 || tileparams->nrow<2){
    fprintf(sp0,"input interferogram must be at least 2x2\n");
    exit(ABNORMAL_EXIT);
  }

  /* is the input file already unwrapped? */
  if(!params->unwrapped){

    /* read wrapped phase and possibly interferogram magnitude data */
    fprintf(sp1,"Reading wrapped phase from file %s\n",infiles->infile);
    if(infiles->infileformat==COMPLEX_DATA){
      ReadComplexFile(&mag,&wrappedphase,infiles->infile,
		      linelen,nlines,tileparams);
    }else if(infiles->infileformat==ALT_LINE_DATA){
      ReadAltLineFile(&mag,&wrappedphase,infiles->infile,
		      linelen,nlines,tileparams);
    }else if(infiles->infileformat==ALT_SAMPLE_DATA){
      ReadAltSampFile(&mag,&wrappedphase,infiles->infile,
		      linelen,nlines,tileparams);
    }else if(infiles->infileformat==FLOAT_DATA){
      Read2DArray((void ***)&wrappedphase,infiles->infile,linelen,nlines,
		  tileparams,sizeof(float *),sizeof(float));
    }else{
      fprintf(sp0,"illegal input file format specification\n");
      exit(ABNORMAL_EXIT);
    }

    /* check to make sure the input data doesn't contain NaNs or infs */
    if(!ValidDataArray(wrappedphase,nrow,ncol) 
       || (mag!=NULL && !ValidDataArray(mag,nrow,ncol))){
      fprintf(sp0,"NaN or infinity found in input float data\nAbort\n");
      exit(ABNORMAL_EXIT);
    }

    /* flip the sign of the wrapped phase if flip flag is set */
    FlipPhaseArraySign(wrappedphase,params,nrow,ncol);

    /* make sure the wrapped phase is properly wrapped */
    WrapPhase(wrappedphase,nrow,ncol);

  }else{

    /* read unwrapped phase input */
    fprintf(sp1,"Reading unwrapped phase from file %s\n",infiles->infile);
    if(infiles->unwrappedinfileformat==ALT_LINE_DATA){
      ReadAltLineFile(&mag,&unwrappedphase,infiles->infile,
		      linelen,nlines,tileparams);
    }else if(infiles->unwrappedinfileformat==ALT_SAMPLE_DATA){
      ReadAltSampFile(&mag,&unwrappedphase,infiles->infile,
			   linelen,nlines,tileparams);
    }else if(infiles->unwrappedinfileformat==FLOAT_DATA){
      Read2DArray((void ***)&unwrappedphase,infiles->infile,linelen,nlines,
		  tileparams,sizeof(float *),sizeof(float));
    }else{
      fprintf(sp0,"Illegal input file format specification\nAbort\n");
      exit(ABNORMAL_EXIT);      
    }

    /* check to make sure the input data doesn't contain NaNs or infs */
    if(!ValidDataArray(unwrappedphase,nrow,ncol) 
       || (mag!=NULL && !ValidDataArray(mag,nrow,ncol))){
      fprintf(sp0,"NaN or infinity found in input float data\nAbort\n");
      exit(ABNORMAL_EXIT);
    }
    
    /* flip the sign of the input unwrapped phase if flip flag is set */
    FlipPhaseArraySign(unwrappedphase,params,nrow,ncol);

    /* parse flows of unwrapped phase */
    wrappedphase=ExtractFlow(unwrappedphase,&flows,nrow,ncol);

    /* free unwrapped phase array to save memory */
    Free2DArray((void **)unwrappedphase,nrow);

  }    

  /* get memory for mag (power) image and set to unity if not passed */
  if(mag==NULL){
    mag=(float **)Get2DMem(nrow,ncol,sizeof(float *),sizeof(float));
    for(row=0;row<nrow;row++){
      for(col=0;col<ncol;col++){
	mag[row][col]=1.0;
      }
    }
  }

  /* set passed pointers and return the number of rows in data */
  *wrappedphaseptr=wrappedphase;
  *magptr=mag;
  *flowsptr=flows;

}


/* function: ReadMagnitude()
 * -------------------------
 * Reads the interferogram magnitude in the specfied file if it exists.
 * Memory for the magnitude array should already have been allocated by
 * ReadInputFile().
 */
void ReadMagnitude(float **mag, infileT *infiles, long linelen, long nlines, 
		   tileparamT *tileparams){

  float **dummy;

  dummy=NULL;
  if(strlen(infiles->magfile)){
    fprintf(sp1,"Reading interferogram magnitude from file %s\n",
	    infiles->magfile);
    if(infiles->magfileformat==FLOAT_DATA){
      Read2DArray((void ***)&mag,infiles->magfile,linelen,nlines,tileparams,
		  sizeof(float *),sizeof(float));
    }else if(infiles->magfileformat==COMPLEX_DATA){
      ReadComplexFile(&mag,&dummy,infiles->magfile,linelen,nlines,
		      tileparams);
    }else if(infiles->magfileformat==ALT_LINE_DATA){
      ReadAltLineFile(&mag,&dummy,infiles->magfile,linelen,nlines,
		      tileparams);
    }else if(infiles->magfileformat==ALT_SAMPLE_DATA){
      ReadAltSampFile(&mag,&dummy,infiles->magfile,linelen,nlines,
		      tileparams);
    }
  }
  if(dummy!=NULL){
    Free2DArray((void **)dummy,tileparams->nrow);
  }
}


/* function: ReadUnwrappedEstimateFile()
 * -------------------------------------
 * Reads the unwrapped-phase estimate from a file (assumes file name exists).
 */
void ReadUnwrappedEstimateFile(float ***unwrappedestptr, infileT *infiles, 
			       long linelen, long nlines, 
			       paramT *params, tileparamT *tileparams){

  float **dummy;
  long nrow, ncol;


  /* initialize */
  dummy=NULL;
  nrow=tileparams->nrow;
  ncol=tileparams->ncol;

  /* read data */
  fprintf(sp1,"Reading coarse unwrapped estimate from file %s\n",
	  infiles->estfile);
  if(infiles->estfileformat==ALT_LINE_DATA){
    ReadAltLineFilePhase(unwrappedestptr,infiles->estfile,
			 linelen,nlines,tileparams);
  }else if(infiles->estfileformat==FLOAT_DATA){
    Read2DArray((void ***)unwrappedestptr,infiles->estfile,linelen,nlines,
		tileparams,sizeof(float *),sizeof(float));
  }else if(infiles->estfileformat==ALT_SAMPLE_DATA){
    ReadAltSampFile(&dummy,unwrappedestptr,infiles->estfile,
		    linelen,nlines,tileparams);
  }else{
    fprintf(sp0,"Illegal file format specification for file %s\nAbort\n",
	    infiles->estfile);
  }
  if(dummy!=NULL){
    Free2DArray((void **)dummy,nrow);
  }
  
  /* make sure data is valid */
  if(!ValidDataArray(*unwrappedestptr,nrow,ncol)){
    fprintf(sp0,"Infinity or NaN found in file %s\nAbort\n",infiles->estfile);
    exit(ABNORMAL_EXIT);
  }

  /* flip the sign of the unwrapped estimate if the flip flag is set */
  FlipPhaseArraySign(*unwrappedestptr,params,nrow,ncol);

}


/* function: ReadWeightsFile()
 * ---------------------------
 * Read in weights form rowcol format file of short ints.
 */
void ReadWeightsFile(short ***weightsptr,char *weightfile, 
		     long linelen, long nlines, tileparamT *tileparams){

  long row, col, nrow, ncol;
  short **rowweight, **colweight;
  signed char printwarning;


  /* set up and read data */
  nrow=tileparams->nrow;
  ncol=tileparams->ncol;
  if(strlen(weightfile)){
    fprintf(sp1,"Reading weights from file %s\n",weightfile);
    Read2DRowColFile((void ***)weightsptr,weightfile,linelen,nlines,
		     tileparams,sizeof(short));
    rowweight=*weightsptr;
    colweight=&(*weightsptr)[nrow-1];
    printwarning=FALSE;
    for(row=0;row<nrow-1;row++){
      for(col=0;col<ncol;col++){
	if(rowweight[row][col]<0){
	  rowweight[row][col]=0;
	  printwarning=TRUE;
	}
      }
    }
    for(row=0;row<nrow;row++){
      for(col=0;col<ncol-1;col++){
	if(colweight[row][col]<0){
	  colweight[row][col]=0;
	  printwarning=TRUE;
	}
      }
    }
    if(printwarning){
      fprintf(sp0,"WARNING: Weights cannot be negative.  Clipping to 0\n");
    }
  }else{
    fprintf(sp1,"No weight file specified.  Assuming uniform weights\n");
    *weightsptr=(short **)Get2DRowColMem(nrow,ncol,
					 sizeof(short *),sizeof(short));
    rowweight=*weightsptr;
    colweight=&(*weightsptr)[nrow-1];
    Set2DShortArray(rowweight,nrow-1,ncol,DEF_WEIGHT);
    Set2DShortArray(colweight,nrow,ncol-1,DEF_WEIGHT);
  }
}


/* function: ReadIntensity()
 * -------------------------
 * Reads the intensity information from specified file(s).  If possilbe,
 * sets arrays for average power and individual powers of single-pass
 * SAR images.  
 */
void ReadIntensity(float ***pwrptr, float ***pwr1ptr, float ***pwr2ptr, 
		   infileT *infiles, long linelen, long nlines, 
		   paramT *params, tileparamT *tileparams){
  
  float **pwr, **pwr1, **pwr2;
  long row, col, nrow, ncol;


  /* initialize */
  nrow=tileparams->nrow;
  ncol=tileparams->ncol;
  pwr=NULL;
  pwr1=NULL;
  pwr2=NULL;

  /* read the data */
  if(strlen(infiles->ampfile2)){

    /* data is given in two separate files */
    fprintf(sp1,"Reading brightness data from files %s and %s\n",
	    infiles->ampfile,infiles->ampfile2);
    if(infiles->ampfileformat==FLOAT_DATA){
      Read2DArray((void ***)&pwr1,infiles->ampfile,linelen,nlines,tileparams,
		  sizeof(float *),sizeof(float));
      Read2DArray((void ***)&pwr2,infiles->ampfile2,linelen,nlines,tileparams,
		  sizeof(float *),sizeof(float));
    }else{
      fprintf(sp0,"Illegal file formats specified for files %s, %s\nAbort\n",
	      infiles->ampfile,infiles->ampfile2);
      exit(ABNORMAL_EXIT);
    }

  }else{

    /* data is in single file */
    fprintf(sp1,"Reading brightness data from file %s\n",infiles->ampfile);
    if(infiles->ampfileformat==ALT_SAMPLE_DATA){
      ReadAltSampFile(&pwr1,&pwr2,infiles->ampfile,linelen,nlines,
		      tileparams);
    }else if(infiles->ampfileformat==ALT_LINE_DATA){
      ReadAltLineFile(&pwr1,&pwr2,infiles->ampfile,linelen,nlines,
		      tileparams);
    }else if(infiles->ampfileformat==FLOAT_DATA){
      Read2DArray((void ***)&pwr,infiles->ampfile,linelen,nlines,tileparams,
		  sizeof(float *),sizeof(float));
      pwr1=NULL;
      pwr2=NULL;
    }else{
      fprintf(sp0,"Illegal file format specified for file %s\nAbort\n",
	      infiles->ampfile);
      exit(ABNORMAL_EXIT);
    }
  }

  /* check data validity */
  if((pwr1!=NULL && !ValidDataArray(pwr1,nrow,ncol)) 
     || (pwr2!=NULL && !ValidDataArray(pwr2,nrow,ncol))
     || (pwr!=NULL && !ValidDataArray(pwr,nrow,ncol))){
    fprintf(sp0,"Infinity or NaN found in amplitude or power data\nAbort\n");
    exit(ABNORMAL_EXIT);
  }

  /* if data is amplitude, square to get power */
  if(params->amplitude){
    for(row=0;row<nrow;row++){
      for(col=0;col<ncol;col++){
	if(pwr1!=NULL && pwr2!=NULL){
	  pwr1[row][col]*=pwr1[row][col];
	  pwr2[row][col]*=pwr2[row][col];
	}else{
	  pwr[row][col]*=pwr[row][col];
	}
      }
    }
  }

  /* get the average power */
  if(pwr1!=NULL && pwr2!=NULL){
    if(pwr==NULL){
      pwr=(float **)Get2DMem(nrow,ncol,sizeof(float *),sizeof(float));
    }
    for(row=0;row<nrow;row++){
      for(col=0;col<ncol;col++){
	pwr[row][col]=(pwr1[row][col]+pwr2[row][col])/2.0;
      }
    }
  }
  
  /* set output pointers */
  *pwrptr=pwr;
  *pwr1ptr=pwr1;
  *pwr2ptr=pwr2;

}


/* function: ReadCorrelation()
 * ---------------------------
 * Reads the correlation information from specified file.
 */
void ReadCorrelation(float ***corrptr, infileT *infiles, 
		     long linelen, long nlines, tileparamT *tileparams){
  
  float **corr, **dummy;
  long nrow;


  /* initialize */
  nrow=tileparams->nrow;
  dummy=NULL;
  corr=NULL;

  /* read the data */
  fprintf(sp1,"Reading correlation data from file %s\n",infiles->corrfile);
  if(infiles->corrfileformat==ALT_SAMPLE_DATA){
    ReadAltSampFile(&dummy,&corr,infiles->corrfile,linelen,nlines,tileparams);
  }else if(infiles->corrfileformat==ALT_LINE_DATA){
    ReadAltLineFilePhase(&corr,infiles->corrfile,linelen,nlines,tileparams);
  }else if(infiles->corrfileformat==FLOAT_DATA){
    Read2DArray((void ***)&corr,infiles->corrfile,linelen,nlines,tileparams,
		sizeof(float *),sizeof(float));
  }else{
    fprintf(sp0,"Illegal file format specified for file %s\nAbort\n",
	    infiles->corrfile);
    exit(ABNORMAL_EXIT);
  }

  /* set output pointer and free memory */
  if(dummy!=NULL){
    Free2DArray((void **)dummy,nrow);
  }
  *corrptr=corr;

}


/* function: ReadAltLineFile()
 * ---------------------------
 * Read in the data from a file containing magnitude and phase
 * data.  File should have one line of magnitude data, one line
 * of phase data, another line of magnitude data, etc.  
 * ncol refers to the number of complex elements in one line of 
 * data.  
 */
void ReadAltLineFile(float ***mag, float ***phase, char *alfile, 
		     long linelen, long nlines, tileparamT *tileparams){

  FILE *fp;
  long filesize,row,nrow,ncol,padlen;

  /* open the file */
  if((fp=fopen(alfile,"r"))==NULL){
    fprintf(sp0,"Can't open file %s\nAbort\n",alfile);
    exit(ABNORMAL_EXIT);
  }

  /* get number of lines based on file size and line length */ 
  fseek(fp,0,SEEK_END);            
  filesize=ftell(fp);
  if(filesize!=(2*nlines*linelen*sizeof(float))){
    fprintf(sp0,"File %s wrong size (%ldx%ld array expected)\nAbort\n",
	    alfile,nlines,linelen);
    exit(ABNORMAL_EXIT);
  }
  fseek(fp,0,SEEK_SET);                 

  /* get memory */
  nrow=tileparams->nrow;
  ncol=tileparams->ncol;
  if(*mag==NULL){
    (*mag)=(float **)Get2DMem(nrow,ncol,sizeof(float *),sizeof(float));
  }
  if(*phase==NULL){
    (*phase)=(float **)Get2DMem(nrow,ncol,sizeof(float *),sizeof(float));
  }
  
  /* read the data */
  fseek(fp,(tileparams->firstrow*2*linelen+tileparams->firstcol)
	*sizeof(float),SEEK_CUR);
  padlen=(linelen-ncol)*sizeof(float);
  for(row=0; row<nrow; row++){
    if(fread((*mag)[row],sizeof(float),ncol,fp)!=ncol){
      fprintf(sp0,"Error while reading from file %s\nAbort\n",alfile);
      exit(ABNORMAL_EXIT);
    }
    fseek(fp,padlen,SEEK_CUR);
    if(fread((*phase)[row],sizeof(float),ncol,fp)!=ncol){
      fprintf(sp0,"Error while reading from file %s\nAbort\n",alfile);
      exit(ABNORMAL_EXIT);
    }
    fseek(fp,padlen,SEEK_CUR);
  }
  fclose(fp);

}


/* function: ReadAltLineFilePhase()
 * --------------------------------
 * Read only the phase data from a file containing magnitude and phase
 * data.  File should have one line of magnitude data, one line
 * of phase data, another line of magnitude data, etc.  
 * ncol refers to the number of complex elements in one line of 
 * data. 
 */
void ReadAltLineFilePhase(float ***phase, char *alfile, 
			  long linelen, long nlines, tileparamT *tileparams){

  FILE *fp;
  long filesize,row,nrow,ncol,padlen;

  /* open the file */
  if((fp=fopen(alfile,"r"))==NULL){
    fprintf(sp0,"Can't open file %s\nAbort\n",alfile);
    exit(ABNORMAL_EXIT);
  }

  /* get number of lines based on file size and line length */ 
  fseek(fp,0,SEEK_END);            
  filesize=ftell(fp);
  if(filesize!=(2*nlines*linelen*sizeof(float))){
    fprintf(sp0,"File %s wrong size (%ldx%ld array expected)\nAbort\n",
	    alfile,nlines,linelen);
    exit(ABNORMAL_EXIT);
  }
  fseek(fp,0,SEEK_SET);                 

  /* get memory */
  nrow=tileparams->nrow;
  ncol=tileparams->ncol;
  if(*phase==NULL){
    (*phase)=(float **)Get2DMem(nrow,ncol,sizeof(float *),sizeof(float));
  }
  
  /* read the phase data */
  fseek(fp,(tileparams->firstrow*2*linelen+linelen
	    +tileparams->firstcol)*sizeof(float),SEEK_CUR);
  padlen=(2*linelen-ncol)*sizeof(float);
  for(row=0; row<nrow; row++){
    if(fread((*phase)[row],sizeof(float),ncol,fp)!=ncol){
      fprintf(sp0,"Error while reading from file %s\nAbort\n",alfile);
      exit(ABNORMAL_EXIT);
    }
    fseek(fp,padlen,SEEK_CUR);
  }
  fclose(fp);

}


/* function: ReadComplexFile()
 * ---------------------------
 * Reads file of complex floats of the form real,imag,real,imag...
 * ncol is the number of complex samples (half the number of real
 * floats per line).  Ensures that phase values are in the range 
 * [0,2pi).
 */
void ReadComplexFile(float ***mag, float ***phase, char *rifile, 
		     long linelen, long nlines, tileparamT *tileparams){
         
  FILE *fp;
  long filesize,ncol,nrow,row,col,padlen;
  float *inpline;

  /* open the file */
  if((fp=fopen(rifile,"r"))==NULL){
    fprintf(sp0,"Can't open file %s\nAbort\n",rifile);
    exit(ABNORMAL_EXIT);
  }

  /* get number of lines based on file size and line length */ 
  fseek(fp,0,SEEK_END);
  filesize=ftell(fp);
  if(filesize!=(2*nlines*linelen*sizeof(float))){
    fprintf(sp0,"File %s wrong size (%ldx%ld array expected)\nAbort\n",
	    rifile,nlines,linelen);
    exit(ABNORMAL_EXIT);
  }
  fseek(fp,0,SEEK_SET);                 

  /* get memory */
  nrow=tileparams->nrow;
  ncol=tileparams->ncol;
  if(*mag==NULL){
    (*mag)=(float **)Get2DMem(nrow,ncol,sizeof(float *),sizeof(float));
  }
  if(*phase==NULL){
    (*phase)=(float **)Get2DMem(nrow,ncol,sizeof(float *),sizeof(float));
  }
  inpline=(float *)MAlloc(2*ncol*sizeof(float));

  /* read the data and convert to magnitude and phase */
  fseek(fp,(tileparams->firstrow*linelen+tileparams->firstcol)
	*2*sizeof(float),SEEK_CUR);
  padlen=(linelen-ncol)*2*sizeof(float);
  for(row=0; row<nrow; row++){
    if(fread(inpline,sizeof(float),2*ncol,fp)!=2*ncol){
      fprintf(sp0,"Error while reading from file %s\nAbort\n",rifile);
      exit(ABNORMAL_EXIT);
    }
    for(col=0; col<ncol; col++){
      (*mag)[row][col]=sqrt(inpline[2*col]*inpline[2*col]
			    +inpline[2*col+1]*inpline[2*col+1]);
      if(!IsFinite((*phase)[row][col]=atan2(inpline[2*col+1],inpline[2*col]))){
	(*phase)[row][col]=0;
      }else if((*phase)[row][col]<0){
        (*phase)[row][col]+=TWOPI;
      }else if((*phase)[row][col]>=TWOPI){
        (*phase)[row][col]-=TWOPI;
      }
    }
    fseek(fp,padlen,SEEK_CUR);
  }
  free(inpline);
  fclose(fp);

}


/* function: Read2DArray()
 * -------------------------
 * Reads file of real data of size elsize.  Assumes the native byte order 
 * of the platform. 
 */
void Read2DArray(void ***arr, char *infile, long linelen, long nlines, 
		 tileparamT *tileparams, size_t elptrsize, size_t elsize){
         
  FILE *fp;
  long filesize,row,nrow,ncol,padlen;

  /* open the file */
  if((fp=fopen(infile,"r"))==NULL){
    fprintf(sp0,"Can't open file %s\nAbort\n",infile);
    exit(ABNORMAL_EXIT);
  }

  /* get number of lines based on file size and line length */ 
  fseek(fp,0,SEEK_END);
  filesize=ftell(fp);
  if(filesize!=(nlines*linelen*elsize)){
    fprintf(sp0,"File %s wrong size (%ldx%ld array expected)\nAbort\n",
	    infile,nlines,linelen);
    exit(ABNORMAL_EXIT);
  }
  fseek(fp,0,SEEK_SET);                 

  /* get memory */
  nrow=tileparams->nrow;
  ncol=tileparams->ncol;
  if(*arr==NULL){
    (*arr)=(void **)Get2DMem(nrow,ncol,elptrsize,elsize);
  }

  /* read the data */
  fseek(fp,(linelen*tileparams->firstrow+tileparams->firstcol)
	*elsize,SEEK_CUR);
  padlen=(linelen-ncol)*elsize;
  for(row=0; row<nrow; row++){
    if(fread((*arr)[row],elsize,ncol,fp)!=ncol){
      fprintf(sp0,"Error while reading from file %s\nAbort\n",infile);
      exit(ABNORMAL_EXIT);
    }
    fseek(fp,padlen,SEEK_CUR);
  }
  fclose(fp);

}


/* function: ReadAltSampFile()
 * ---------------------------
 * Reads file of real alternating floats from separate images.  Format is
 * real0A, real0B, real1A, real1B, real2A, real2B,...
 * ncol is the number of samples in each image (note the number of
 * floats per line in the specified file).
 */
void ReadAltSampFile(float ***arr1, float ***arr2, char *infile, 
		     long linelen, long nlines, tileparamT *tileparams){
         
  FILE *fp;
  long filesize,row,col,nrow,ncol,padlen;
  float *inpline;

  /* open the file */
  if((fp=fopen(infile,"r"))==NULL){
    fprintf(sp0,"Can't open file %s\nAbort\n",infile);
    exit(ABNORMAL_EXIT);
  }

  /* get number of lines based on file size and line length */ 
  fseek(fp,0,SEEK_END);
  filesize=ftell(fp);
  if(filesize!=(2*nlines*linelen*sizeof(float))){
    fprintf(sp0,"File %s wrong size (%ldx%ld array expected)\nAbort\n",
	    infile,nlines,linelen);
    exit(ABNORMAL_EXIT);
  }
  fseek(fp,0,SEEK_SET);                 

  /* get memory */
  nrow=tileparams->nrow;
  ncol=tileparams->ncol;
  if(*arr1==NULL){
    (*arr1)=(float **)Get2DMem(nrow,ncol,sizeof(float *),sizeof(float));
  }
  if(*arr2==NULL){
    (*arr2)=(float **)Get2DMem(nrow,ncol,sizeof(float *),sizeof(float));
  }
  inpline=(float *)MAlloc(2*ncol*sizeof(float));

  /* read the data */
  fseek(fp,(tileparams->firstrow*linelen+tileparams->firstcol)
	*2*sizeof(float),SEEK_CUR);
  padlen=(linelen-ncol)*2*sizeof(float);
  for(row=0; row<nrow; row++){
    if(fread(inpline,sizeof(float),2*ncol,fp)!=2*ncol){
      fprintf(sp0,"Error while reading from file %s\nAbort\n",infile);
      exit(ABNORMAL_EXIT);
    }
    for(col=0; col<ncol; col++){
      (*arr1)[row][col]=inpline[2*col];
      (*arr2)[row][col]=inpline[2*col+1];
    }
    fseek(fp,padlen,SEEK_CUR);
  }
  free(inpline);
  fclose(fp);

}


/* function: Read2DRowColFile()
 * ----------------------------
 * Gets memory and reads single array from a file.  Array should be in the 
 * file line by line starting with the row array (size nrow-1 x ncol) and
 * followed by the column array (size nrow x ncol-1).  Both arrays 
 * are placed into the passed array as they were in the file.
 */
void Read2DRowColFile(void ***arr, char *filename, long linelen, long nlines, 
		      tileparamT *tileparams, size_t size){

  FILE *fp;
  long row, nel, nrow, ncol, padlen, filelen;
 
  /* open the file */
  if((fp=fopen(filename,"r"))==NULL){
    fprintf(sp0,"Can't open file %s\nAbort\n",filename);
    exit(ABNORMAL_EXIT);
  }

  /* get number of data elements in file */ 
  fseek(fp,0,SEEK_END);
  filelen=ftell(fp);
  fseek(fp,0,SEEK_SET);
  nel=(long )(filelen/size);

  /* check file size */
  if(2*linelen*nlines-nlines-linelen != nel || (filelen % size)){
    fprintf(sp0,"File %s wrong size (%ld elements expected)\nAbort\n",
	    filename,2*linelen*nlines-nlines-linelen);
    exit(ABNORMAL_EXIT);
  }

  /* get memory if passed pointer is NULL */
  nrow=tileparams->nrow;
  ncol=tileparams->ncol;
  if(*arr==NULL){
    (*arr)=Get2DRowColMem(nrow,ncol,sizeof(void *),size);
  }

  /* read arrays */
  fseek(fp,(linelen*tileparams->firstrow+tileparams->firstcol)
	*size,SEEK_SET);
  padlen=(linelen-ncol)*size;
  for(row=0; row<nrow-1; row++){
    if(fread((*arr)[row],size,ncol,fp)!=ncol){
      fprintf(sp0,"Error while reading from file %s\nAbort\n",filename);
      exit(ABNORMAL_EXIT);
    }
    fseek(fp,padlen,SEEK_CUR);
  }
  fseek(fp,(linelen*(nlines-1)+(linelen-1)*tileparams->firstrow
	    +tileparams->firstcol)*size,SEEK_SET);
  for(row=nrow-1; row<2*nrow-1; row++){
    if(fread((*arr)[row],size,ncol-1,fp)!=ncol-1){
      fprintf(sp0,"Error while reading from file %s\nAbort\n",filename);
      exit(ABNORMAL_EXIT);
    }
    fseek(fp,padlen,SEEK_CUR);
  }
  fclose(fp);
}


/* function: Read2DRowColFileRows()
 * --------------------------------
 * Similar to Read2DRowColFile(), except reads only row (horizontal) data
 * at specified locations.  tileparams->nrow is treated as the number of
 * rows of data to be read from the RowCol file, not the number of 
 * equivalent rows in the orginal pixel file (whose arcs are represented
 * in the RowCol file).
 */
void Read2DRowColFileRows(void ***arr, char *filename, long linelen, 
			  long nlines, tileparamT *tileparams, size_t size){

  FILE *fp;
  long row, nel, nrow, ncol, padlen, filelen;
 
  /* open the file */
  if((fp=fopen(filename,"r"))==NULL){
    fprintf(sp0,"Can't open file %s\nAbort\n",filename);
    exit(ABNORMAL_EXIT);
  }

  /* get number of data elements in file */ 
  fseek(fp,0,SEEK_END);
  filelen=ftell(fp);
  fseek(fp,0,SEEK_SET);
  nel=(long )(filelen/size);

  /* check file size */
  if(2*linelen*nlines-nlines-linelen != nel || (filelen % size)){
    fprintf(sp0,"File %s wrong size (%ld elements expected)\nAbort\n",
	    filename,2*linelen*nlines-nlines-linelen);
    exit(ABNORMAL_EXIT);
  }

  /* get memory if passed pointer is NULL */
  nrow=tileparams->nrow;
  ncol=tileparams->ncol;
  if(*arr==NULL){
    (*arr)=Get2DMem(nrow,ncol,sizeof(void *),size);
  }

  /* read arrays */
  fseek(fp,(linelen*tileparams->firstrow+tileparams->firstcol)
	*size,SEEK_SET);
  padlen=(linelen-ncol)*size;
  for(row=0; row<nrow; row++){
    if(fread((*arr)[row],size,ncol,fp)!=ncol){
      fprintf(sp0,"Error while reading from file %s\nAbort\n",filename);
      exit(ABNORMAL_EXIT);
    }
    fseek(fp,padlen,SEEK_CUR);
  }
  fclose(fp);
}


/* function: SetDumpAll()
 * ----------------------
 * Sets names of output files so that the program will dump intermediate
 * arrays.  Only sets names if they are not set already.
 */
void SetDumpAll(outfileT *outfiles, paramT *params){

  if(params->dumpall){
    if(!strlen(outfiles->initfile)){
      StrNCopy(outfiles->initfile,DUMP_INITFILE,MAXSTRLEN);
    }
    if(!strlen(outfiles->flowfile)){
      StrNCopy(outfiles->flowfile,DUMP_FLOWFILE,MAXSTRLEN);
    }
    if(!strlen(outfiles->eifile)){
      StrNCopy(outfiles->eifile,DUMP_EIFILE,MAXSTRLEN);
    }
    if(!strlen(outfiles->rowcostfile)){
      StrNCopy(outfiles->rowcostfile,DUMP_ROWCOSTFILE,MAXSTRLEN);
    }
    if(!strlen(outfiles->colcostfile)){
      StrNCopy(outfiles->colcostfile,DUMP_COLCOSTFILE,MAXSTRLEN);
    }
    if(!strlen(outfiles->mstrowcostfile)){
      StrNCopy(outfiles->mstrowcostfile,DUMP_MSTROWCOSTFILE,MAXSTRLEN);
    }
    if(!strlen(outfiles->mstcolcostfile)){
      StrNCopy(outfiles->mstcolcostfile,DUMP_MSTCOLCOSTFILE,MAXSTRLEN);
    }
    if(!strlen(outfiles->mstcostsfile)){
      StrNCopy(outfiles->mstcostsfile,DUMP_MSTCOSTSFILE,MAXSTRLEN);
    }
    if(!strlen(outfiles->corrdumpfile)){
      StrNCopy(outfiles->corrdumpfile,DUMP_CORRDUMPFILE,MAXSTRLEN);
    }
    if(!strlen(outfiles->rawcorrdumpfile)){
      StrNCopy(outfiles->rawcorrdumpfile,DUMP_RAWCORRDUMPFILE,MAXSTRLEN);
    }
  }
}


/* function: SetStreamPointers()
 * -----------------------------
 * Sets the default stream pointers (global variables).
 */
void SetStreamPointers(void){

  fflush(NULL);
  if((sp0=DEF_ERRORSTREAM)==NULL){
    if((sp0=fopen(NULLFILE,"w"))==NULL){
      fprintf(sp0,"unable to open null file %s\n",NULLFILE);
      exit(ABNORMAL_EXIT);
    }
  }
  if((sp1=DEF_OUTPUTSTREAM)==NULL){
    if((sp1=fopen(NULLFILE,"w"))==NULL){
      fprintf(sp0,"unable to open null file %s\n",NULLFILE);
      exit(ABNORMAL_EXIT);
    }
  }
  if((sp2=DEF_VERBOSESTREAM)==NULL){
    if((sp2=fopen(NULLFILE,"w"))==NULL){
      fprintf(sp0,"unable to open null file %s\n",NULLFILE);
      exit(ABNORMAL_EXIT);
    }
  }
  if((sp3=DEF_COUNTERSTREAM)==NULL){
    if((sp3=fopen(NULLFILE,"w"))==NULL){
      fprintf(sp0,"unable to open null file %s\n",NULLFILE);
      exit(ABNORMAL_EXIT);
    }
  }
}


/* function: SetVerboseOut()
 * -------------------------
 * Set the global stream pointer sp2 to be stdout if the verbose flag
 * is set in the parameter data type.
 */
void SetVerboseOut(paramT *params){

  fflush(NULL);
  if(params->verbose){
    if(sp2!=stdout && sp2!=stderr && sp2!=stdin && sp2!=NULL){
      fclose(sp2);
    }
    sp2=stdout;
    if(sp3!=stdout && sp3!=stderr && sp3!=stdin && sp3!=NULL){
      fclose(sp3);
    }
    sp3=stdout;
  }
}


/* function: ChildResetStreamPointers()
 * -----------------------------------
 * Reset the global stream pointers for a child.  Streams equal to stdout 
 * are directed to a log file, and errors are written to the screen.
 */
void ChildResetStreamPointers(pid_t pid, long tilerow, long tilecol, 
			      paramT *params){

  FILE *logfp;
  char logfile[MAXSTRLEN], cwd[MAXSTRLEN];

  fflush(NULL);
  sprintf(logfile,"%s/%s%ld_%ld",params->tiledir,LOGFILEROOT,tilerow,tilecol);
  if((logfp=fopen(logfile,"w"))==NULL){
    fprintf(sp0,"Unable to open log file %s\nAbort\n",logfile);
    exit(ABNORMAL_EXIT);
  }
  fprintf(logfp,"%s (pid %ld): unwrapping tile at row %ld, column %ld\n\n",
	  PROGRAMNAME,(long )pid,tilerow,tilecol);
  if(getcwd(cwd,MAXSTRLEN)!=NULL){
    fprintf(logfp,"Current working directory is %s\n",cwd);
  }
  if(sp2==stdout || sp2==stderr){
    sp2=logfp;
  }
  if(sp1==stdout || sp1==stderr){
    sp1=logfp;
  }
  if(sp0==stdout || sp0==stderr){
    sp0=logfp;
  }
  if(sp3!=stdout && sp3!=stderr && sp3!=stdin && sp3!=NULL){
    fclose(sp3);
  }
  if((sp3=fopen(NULLFILE,"w"))==NULL){
    fprintf(sp0,"Unable to open null file %s\n",NULLFILE);
    exit(ABNORMAL_EXIT);
  }
}


/* function: DumpIncrCostFiles()
 * -----------------------------
 * Dumps incremental cost arrays, creating file names for them.
 */
void DumpIncrCostFiles(incrcostT **incrcosts, long iincrcostfile, 
		       long nflow, long nrow, long ncol){

  long row, col, maxcol;
  char incrcostfile[MAXSTRLEN];
  char tempstr[MAXSTRLEN];
  short **tempcosts;

  /* get memory for tempcosts */
  tempcosts=(short **)Get2DRowColMem(nrow,ncol,sizeof(short *),sizeof(short));

  /* create the file names and dump the files */
  /* snprintf() is more elegant, but its unavailable on some machines */
  for(row=0;row<2*nrow-1;row++){
    if(row<nrow-1){
      maxcol=ncol;
    }else{
      maxcol=ncol-1;
    }
    for(col=0;col<maxcol;col++){
      tempcosts[row][col]=incrcosts[row][col].poscost;
    }
  }
  strncpy(incrcostfile,INCRCOSTFILEPOS,MAXSTRLEN-1);
  sprintf(tempstr,".%ld_%ld",iincrcostfile,nflow);
  strncat(incrcostfile,tempstr,MAXSTRLEN-strlen(incrcostfile)-1);
  Write2DRowColArray((void **)tempcosts,incrcostfile,
		     nrow,ncol,sizeof(short));
  for(row=0;row<2*nrow-1;row++){
    if(row<nrow-1){
      maxcol=ncol;
    }else{
      maxcol=ncol-1;
    }
    for(col=0;col<maxcol;col++){
      tempcosts[row][col]=incrcosts[row][col].negcost;
    }
  }
  strncpy(incrcostfile,INCRCOSTFILENEG,MAXSTRLEN-1);
  sprintf(tempstr,".%ld_%ld",iincrcostfile,nflow);
  strncat(incrcostfile,tempstr,MAXSTRLEN-strlen(incrcostfile)-1);
  Write2DRowColArray((void **)tempcosts,incrcostfile,
		     nrow,ncol,sizeof(short));

  /* free memory */
  Free2DArray((void **)tempcosts,2*nrow-1);

}


/* function: MakeTileDir()
 * ---------------------------
 * Create a temporary directory for tile files in directory of output file.  
 * Save directory name in buffer in paramT structure.
 */
void MakeTileDir(paramT *params, outfileT *outfiles){

  char path[MAXSTRLEN], basename[MAXSTRLEN];

  /* create name for tile directory (use pid to make unique) */
  ParseFilename(outfiles->outfile,path,basename);
  sprintf(params->tiledir,"%s%s%ld",path,TMPTILEDIRROOT,(long )getpid());

  /* create tile directory */
  fprintf(sp1,"Creating temporary directory %s\n",params->tiledir);
  if(mkdir(params->tiledir,TILEDIRMODE)){
    fprintf(sp0,"Error creating directory %s\nAbort\n",params->tiledir);
    exit(ABNORMAL_EXIT);
  }

}


/* function: ParseFilename()
 * -------------------------
 * Given a filename, separates it into path and base filename.  Output
 * buffers should be at least MAXSTRLEN characters, and filename buffer
 * should be no more than MAXSTRLEN characters.  The output path 
 * has a trailing "/" character.
 */
void ParseFilename(char *filename, char *path, char *basename){

  char tempstring[MAXSTRLEN];
  char *tempouttok;

  /* make sure we have a nonzero filename */
  if(!strlen(filename)){
    fprintf(sp0,"Zero-length filename passed to ParseFilename()\nAbort\n");
    exit(ABNORMAL_EXIT);
  }

  /* initialize path */
  if(filename[0]=='/'){
    StrNCopy(path,"/",MAXSTRLEN);
  }else{
    StrNCopy(path,"",MAXSTRLEN);
  }

  /* parse the filename */
  StrNCopy(tempstring,filename,MAXSTRLEN);
  tempouttok=strtok(tempstring,"/");
  while(TRUE){
    StrNCopy(basename,tempouttok,MAXSTRLEN);
    if((tempouttok=strtok(NULL,"/"))==NULL){
      break;
    }
    strcat(path,basename);
    strcat(path,"/");
  }

  /* make sure we have a nonzero base filename */
  if(!strlen(basename)){
    fprintf(sp0,"Zero-length base filename found in ParseFilename()\nAbort\n");
    exit(ABNORMAL_EXIT);
  }

}
