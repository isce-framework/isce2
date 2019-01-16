/*

Copyright [year created], by the California Institute of Technology.
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
Any commercial use must be negotiated with the Office of Technology
Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws and regulations.
By accepting this document, the user agrees to comply with all applicable
U.S. export laws and regulations.  User has the responsibility to obtain
export licenses, or other export authority as may be required before
exporting such information to foreign countries or providing access to
foreign persons.


Some code in this software may be derived of sample code obtained from programming manuals published by O'Reilly
& Associates, Inc.,  Copyright 1991.  Permission to use, copy, modify, distribute, and sell published O'Reilly &
Associates, Inc. code is explicitly granted in the programming manuals.  This notice and acknowledgement is provided
here insofar as such code may exist in this Software.


*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>

#include <X11/Xlib.h>
/*
#include <X11/Intrinsic.h>   Use these if no motif
#include <X11/StringDefs.h>
*/
#include <Xm/Xm.h>
#include <Xm/DrawingA.h>
#include <Xm/PanedW.h>
#include <Xm/Form.h>
#include <Xm/RowColumn.h>
#include <Xm/PushB.h>
#include <Xm/PushBG.h>
#include <Xm/DrawnB.h>
#include <Xm/ScrolledW.h>
#include <Xm/Label.h>
#include <X11/cursorfont.h>
#include <X11/keysym.h>    /* for popup window only */

/* for VAX
struct descriptor
  {
  unsigned short length;
  unsigned char data_type, dsc_class;
  char *string_ptr;
  };
*/

#define MAX_COLORS 256

#define SGImode

#define icon_width 29
#define icon_height 29
/* static char icon_bits[] = { */
const char icon_bits[] = {
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc6, 0x3e, 0x82, 0x00,
   0xee, 0x62, 0xc6, 0x00, 0xaa, 0x42, 0x6c, 0x00, 0xba, 0xc2, 0x38, 0x00,
   0x92, 0xc2, 0x38, 0x00, 0x82, 0x42, 0x62, 0x00, 0x82, 0x62, 0xc6, 0x00,
   0x82, 0x3e, 0x82, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

/* Global variables */
Display         *dgx;
GC              gc;
XtAppContext    app_context;
Window          root;
Window          top[6];
Window          wgx[321];
Window          fgx[321];
Window          lgx[321];
Widget          scrl[321];
Widget          labl[321];
Widget          draw[321];
Widget          form[321];
XmString        a_llll[321];
Widget          formy;
XEvent          event;
Pixmap          icon, disparity;
Colormap        cmap;
XVisualInfo *visualList;
char    b_bswap[4];
int     *i_bswap;
int     i_type[321];
int     i_dx[321];
int     i_wx[321];
int     i_gx[6][30];
int     i_tx[321];
int     i_dmax;
int     i_wmax;
int     i_gmax;
int     i_rmaxr, i_rmltr;
int     i_gmaxg, i_gmltg;
int     i_bmaxb, i_bmltb;
int     i_init = 0;
int     i_app = 0;
int     i_ctype = 0;
int     i_clrs = 0;
int             screen;
int             i_ctble;
int     allocated;
int     i_push = 0;
int     i_db;
int     i_mdsp;
int     i_mxxx;
int     i_myyy;
int     i_message = 0;
char    a_message[1600];
unsigned char   red[MAX_COLORS], green[MAX_COLORS], blue[MAX_COLORS];
unsigned char   rred[MAX_COLORS], ggreen[MAX_COLORS], bblue[MAX_COLORS];
Widget          gftop;
Widget          ewtop[10]={0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

/* Local functions */
static void  Button_quit();
static void  alo_colors();
static void  put_colors();
void         read_events();
int  myhandler();



/* For:  entry_window ***********************************************************************/
#include <Xm/LabelG.h>
#include <Xm/TextF.h>

/* For:  file window ***********************************************************************/
char ident[]="     Graphx v79.0      February 16, 2011                 ";

#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <Xm/MainW.h>
#include <Xm/CascadeB.h>
#include <Xm/Frame.h>
#include <Xm/FileSB.h>
#include <Xm/MessageB.h>

typedef struct
{
    Widget   widget;
} selectData;

void closeCallBack ();
void okCallBack ();

/*-------------------------------------------------------------
** Global Variables
*/

#define MAX_ARGS 20

static XmStringCharSet charset = (XmStringCharSet) XmSTRING_DEFAULT_CHARSET;


/*************************************************************************/
#ifdef SGI
int version_gx_()
#endif
#ifdef SUN
int version_gx_()
#endif
#ifdef M10
int version_gx__()
#endif
#if defined(HP) || defined(_linux)
int version_gx()
#endif

{   int i_version;

    i_version=79;
    return(i_version);

}

/*************************************************************************/
#ifdef SGI
void get_colorrgb_(a_color,i_color)
#endif
#ifdef SUN
void get_colorrgb_(a_color,i_color)
#endif
#ifdef M10
void get_colorrgb__(a_color,i_color)
#endif
#if defined(HP) || defined(_linux)
void get_colorrgb(a_color,i_color)
#endif
#ifdef VAX
void get_colorrgb__(dsc,i_color)
    struct descriptor *dsc;
#else
    char *a_color;
#endif

    int  i_color[];

{
    int i;
    int i_cnt;
    char a_lbl[80];
#ifdef VAX
    char *a_color;
    a_color=dsc->string_ptr;
#endif
    XColor rgbcolor[1], dspcolor[1];

    i_cnt = 0;
    for (i=0; i < 78; i++) {
          a_lbl[i] = a_color[i];
          if (a_lbl[i] != 0 && a_lbl[i] != 32 ) i_cnt = i+1;
    } /* enddo */
    a_lbl[i_cnt] = 0;
    a_lbl[79] = 0;
    for (i=i_cnt; i < 80; i++) {
/*          printf("i: %d\n",i); */
          a_lbl[i] = 0;
    } /* enddo */
/*    printf("number of letters in null color name: %d\n",i_cnt);
    printf("color: %s\n",a_lbl);
    printf("dgx: %d\n",dgx);
    printf("cmap: %d\n",cmap);
*/

    if(XLookupColor(dgx,cmap,a_lbl,rgbcolor,dspcolor) != 0) {
       /* printf("Found Color Name\n"); */
       i_color[0]=(int)rgbcolor[0].red/256;
       i_color[1]=(int)rgbcolor[0].green/256;
       i_color[2]=(int)rgbcolor[0].blue/256;
/*       printf("Colors found %d %d %d\n",i_color[0],i_color[1],i_color[2]); */
    }  /* end do */

/*    if(XParseColor(dgx,cmap,a_lbl,rgbcolor) != 0) {
       printf("made it to here\n");
       i_color[0]=rgbcolor[0].red/256;
       i_color[1]=rgbcolor[0].green/256;
       i_color[2]=rgbcolor[0].blue/256;
       printf("Colors found %d %d %d\n",i_color[0],i_color[1],i_color[2]);
    } */ /* end do */

/*    printf("Colors found are %d %d %d\n",i_color[0],i_color[1],i_color[2]); */

    return;
}

/*************************************************************************/
#ifdef SGI
void plot_data_(i_d,i_w,i_num,r_x,r_y)
#endif
#ifdef SUN
void plot_data_(i_d,i_w,i_num,r_x,r_y)
#endif
#ifdef M10
void plot_data__(i_d,i_w,i_num,r_x,r_y)
#endif
#if defined(HP) || defined(_linux)
void plot_data(i_d,i_w,i_num,r_x,r_y)
#endif
    int i_d[];
    int i_w[];
    int i_num[];
    float r_x[10000], r_y[10000];
{
    int           i;
    int           i_x1,i_y1;
    int           i_x2,i_y2;
    int           xr, yr;
    unsigned int  width, height, bwr, dr;


    /* Scale the values according to the size of the window */
    XGetGeometry(dgx, wgx[i_gx[i_d[0]][i_w[0]]], &root, &xr, &yr, &width, &height, &bwr, &dr);

/*    printf("plot_data1\n");
    printf("plot_data2 %d\n",i_num); */

    for (i = 0; i < i_num[0] - 1; i++) {

      i_x1 = (int) ((r_x[i]) * width );
      i_y1 = height - (int) ((r_y[i]) * height );
      i_x2 = (int) ((r_x[i+1]) * width );
      i_y2 = height - (int) ((r_y[i+1]) * height );

      if (i_x1 < 0)   i_x1 = 0;
      if (i_x1 > width) i_x1 = width;
      if (i_x2 < 0)   i_x2 = 0;
      if (i_x2 > width) i_x2 = width;

      if (i_y1 < 0)   i_y1 = 0;
      if (i_y1 > height) i_y1 = height;
      if (i_y2 < 0)   i_y2 = 0;
      if (i_y2 > height) i_y2 = height;

      /* printf("hi from plot_data3\n");
      printf("plot_data3 %d %d %d %d\n",i_x1, i_y1, i_x2, i_y2); */
      XDrawLine(dgx, wgx[i_gx[i_d[0]][i_w[0]]], gc, i_x1, i_y1, i_x2, i_y2);
     }

}


/*************************************************************************/
#ifdef SGI
void display_img_(i_d,i_w, i_x, i_y, i_width, i_height, i_bpl, r_rdat, r_gdat, r_bdat)
#endif
#ifdef SUN
void display_img_(i_d,i_w, i_x, i_y, i_width, i_height, i_bpl, r_rdat, r_gdat, r_bdat)
#endif
#ifdef M10
void display_img__(i_d,i_w, i_x, i_y, i_width, i_height, i_bpl, r_rdat, r_gdat, r_bdat)
#endif
#if defined(HP) || defined(_linux)
void display_img(i_d,i_w, i_x, i_y, i_width, i_height, i_bpl, r_rdat, r_gdat, r_bdat)
#endif

    int i_d[];
    int i_w[];
    int i_x[];
    int i_y[];
    int i_width[];
    int i_height[];
    int i_bpl[];
    float r_rdat[];
    float r_gdat[];
    float r_bdat[];

{
#ifdef M4
    unsigned char i_nbits[16000001];
#else
    unsigned char i_nbits[  400001];
#endif
        int i, j, k, l, m, i_bpp;

    union temp {
      unsigned char bbb[4];
      int           iii[2];
      long          lll[1];
    };

    union temp i_pix;

    XImage      xim;


        for (i = 0 ; i < i_height[0] ; i++) for(j = 0; j < i_width[0] ; j++) {
#ifdef M4
             if (i*i_width[0]+j > 4000000) {
#else
             if (i*i_width[0]+j >  100000) {
#endif
               printf("error - %d %d %d %d %d\n",i,j,i_width[0],i_height[0],i*i_width[0]+j);
               exit(0);
             } /* endif */
             k = (int)((float)(i_rmaxr)*r_rdat[i*i_bpl[0]+j]);
             l = (int)((float)(i_gmaxg)*r_gdat[i*i_bpl[0]+j]);
             m = (int)((float)(i_bmaxb)*r_bdat[i*i_bpl[0]+j]);
             if (k < 0) {
               printf("rdat < 0 %d \n",k);
               k = 0;
             } /* endif */
             if (k > i_rmaxr-1) {
/*               printf("rdat => i_rmaxr   %d %d \n",k,i_rmaxr); */
               k = i_rmaxr-1;
             } /* endif */

             if (l < 0) {
               printf("gdat < 0 %d \n",l);
               l = 0;
             } /* endif */
             if (l > i_gmaxg-1) {
/*               printf("gdat => i_gmaxg   %d %d \n",l,i_gmaxg); */
               l = i_gmaxg-1;
             } /* endif */

             if (m < 0) {
               printf("bdat < 0 %d \n",m);
               m = 0;
             } /* endif */
             if (m > i_bmaxb-1) {
/*               printf("bdat => i_bmaxb   %d %d \n",m,i_bmaxb); */
               m = i_bmaxb-1;
             } /* endif */

             i_pix.lll[0] = (long)(k*i_rmltr)+(long)(l*i_gmltg)+(long)(m*i_bmltb);

        if (*i_bswap == 1) {

             if (visualList[0].depth == 8) {
               i_bpp = 8;
               i_nbits[(i*i_width[0]+j)] = i_pix.bbb[3];
             } else if (visualList[0].depth == 15) {
               i_bpp = 16;
               i_nbits[(i*i_width[0]+j)*2+0] = i_pix.bbb[2];
               i_nbits[(i*i_width[0]+j)*2+1] = i_pix.bbb[3];
             } else if (visualList[0].depth == 16) {
               i_bpp = 16;
               i_nbits[(i*i_width[0]+j)*2+0] = i_pix.bbb[2];
               i_nbits[(i*i_width[0]+j)*2+1] = i_pix.bbb[3];
             } else if (visualList[0].depth == 24) {
               i_bpp = 32;
               i_nbits[(i*i_width[0]+j)*4+0] = i_pix.bbb[0];
               i_nbits[(i*i_width[0]+j)*4+1] = i_pix.bbb[1];
               i_nbits[(i*i_width[0]+j)*4+2] = i_pix.bbb[2];
               i_nbits[(i*i_width[0]+j)*4+3] = i_pix.bbb[3];
             } else {
               printf("depth not supported \n");
             } }
        else {
             if (visualList[0].depth == 8) {
               i_bpp = 8;
               i_nbits[(i*i_width[0]+j)] = i_pix.bbb[3];
             } else if (visualList[0].depth == 15) {
               i_bpp = 16;
               i_nbits[(i*i_width[0]+j)*2+1] = i_pix.bbb[2];
               i_nbits[(i*i_width[0]+j)*2+0] = i_pix.bbb[3];
             } else if (visualList[0].depth == 16) {
               i_bpp = 16;
               i_nbits[(i*i_width[0]+j)*2+1] = i_pix.bbb[2];
               i_nbits[(i*i_width[0]+j)*2+0] = i_pix.bbb[3];
             } else if (visualList[0].depth == 24) {
               i_bpp = 32;
               i_nbits[(i*i_width[0]+j)*4+3] = i_pix.bbb[0];
               i_nbits[(i*i_width[0]+j)*4+2] = i_pix.bbb[1];
               i_nbits[(i*i_width[0]+j)*4+1] = i_pix.bbb[2];
               i_nbits[(i*i_width[0]+j)*4+0] = i_pix.bbb[3];
             } else {
               printf("depth not supported \n");
             }
        }
             }

    xim.depth          = visualList[0].depth;
    xim.data           = (char *)i_nbits;
    xim.bitmap_pad     = 8;
    xim.width          = i_width[0];
    xim.height         = i_height[0];
    xim.format         = ZPixmap;
    xim.bits_per_pixel = i_bpp;
    xim.byte_order     = MSBFirst;
    xim.bytes_per_line = i_width[0]*i_bpp/8;
    xim.red_mask       = visualList[0].red_mask;
    xim.green_mask     = visualList[0].green_mask;
    xim.blue_mask      = visualList[0].blue_mask;

    XPutImage(dgx, wgx[i_gx[i_d[0]][i_w[0]]], gc, &xim, 0, 0, i_x[0], i_y[0], i_width[0], i_height[0]);
}


/*************************************************************************/
#ifdef SGI
void display_label_(i_d,i_w,a_string,i_center)
#endif
#ifdef SUN
void display_label_(i_d,i_w,a_string,i_center)
#endif
#ifdef M10
void display_label__(i_d,i_w,a_string,i_center)
#endif
#if defined(HP) || defined(_linux)
void display_label(i_d,i_w,a_string,i_center)
#endif

int i_d[];
int i_w[];
int i_center[];
char a_string[255];
{
    int i;
    int n;
    int i_cnt;
    char a_lbl[255];

    /* displays a string at the top of a window */
    XmString   motif_string;
    Arg        args[2];


    i_cnt = 0;
    for (i=0; i < 254; i++) {
          a_lbl[i] = a_string[i];
          if (a_lbl[i] != 0 && a_lbl[i] != 32 ) i_cnt = i+1;
        } /* enddo */
        a_lbl[i_cnt] = 0;

    motif_string = XmStringCreate((char *)a_lbl, XmSTRING_DEFAULT_CHARSET);

    if (i_db > 8-1) printf("i_center = %d\n",i_center[0]);

    n = 0;
    XtSetArg(args[n], XmNlabelString, motif_string);n++;
    if (i_center[0]==1) {XtSetArg(args[n],XmNalignment,XmALIGNMENT_BEGINNING);n++;}
      else {XtSetArg(args[n],XmNalignment,XmALIGNMENT_CENTER);n++; }
    /* endif */

    XtSetValues(labl[i_gx[i_d[0]][i_w[0]]], args, n);
    XFlush(dgx);

}

/*************************************************************************/
    char *a_lll;
    Widget main_windowew[10],rowcolew[10];

#ifdef SGI
void entry_window_(i_chn,a_label,a_data)
#endif
#ifdef SUN
void entry_window_(i_chn,a_label,a_data)
#endif
#ifdef M10
void entry_window__(i_chn,a_label,a_data)
#endif
#if defined(HP) || defined(_linux)
void entry_window(i_chn,a_label,a_data)
#endif

int i_chn[1];
char a_label[3360];
char a_data[3360];
{

    static char a_labels[21][160];
    static char a_datas[21][160];
    static char a_smenu[21][160];
    char a_title[160];
    Widget        textwin, formwin_e, labelwidgit_e, op_menu, pd_menu, menu_item[20];
    int           i;
    int j;
    int i_flag;
    int n;
    int num;
    int iii;
    int jjj;
    int nnn;
    Arg args[10];
    void          print_result();
    void          ewManager();
    void          ewdsp_reset();


  if (i_db > 8-1) printf("inside entry_window %d\n",i_chn[0]);
  if (ewtop[i_chn[0]] != 0) {
     if (i_db > 8-1) printf("raising window to forground %d\n",i_chn[0]);
     XRaiseWindow(dgx, XtWindow(ewtop[i_chn[0]]));

     if (i_db > 8-1) printf("destroying rowcol widget %d\n",i_chn[0]);
     XtDestroyWidget(rowcolew[i_chn[0]]); }

  else {
      /* printf("parsing title %d\n",i_chn[0]); */
      num=0;
      for (i=0; i<159; i++) {
         a_title[i] = a_label[i];
       if (a_title[i] != 0 && a_title[i] != 32 ) num = i+1;
      } /* enddo */
      /* printf("num= %d\n",num); */
      a_title[num] = 0;
      n = 0;
      XtSetArg(args[n], XmNtitle,     a_title); n++;
/*      XtSetArg(args[n], XmNdepth,   visualList[0].depth); n++;
      XtSetArg(args[n], XmNvisual, visualList[0].visual); n++;
      XtSetArg(args[n], XmNcolormap,         cmap); n++; */

      if (i_db > 7-1) printf("creating new entry window\n");

      ewtop[i_chn[0]] = XtAppCreateShell(NULL,"appClass",
                topLevelShellWidgetClass,dgx,
                args, n);

     /*
      *  Create main window.
      */

        n = 0;
        main_windowew[i_chn[0]] = XtCreateManagedWidget("main",
                            xmMainWindowWidgetClass, ewtop[i_chn[0]],
                            args, n);
        XtAddCallback(main_windowew[i_chn[0]], XmNdestroyCallback,
            ewdsp_reset, (XtPointer)i_chn[0]);

    } /* end if */

     if (i_db > 8-1) printf("Creating new rowcol widget %d\n",i_chn[0]);
    rowcolew[i_chn[0]] = XtVaCreateWidget("rowcol",
        xmRowColumnWidgetClass, main_windowew[i_chn[0]], NULL);

     if (i_db > 8-1) printf("Parsing lables\n");

/*    for (i = 0; i < XtNumber(a_labels); i++) { */
    for (i = 1; i < 20; i++) {
       num=0;
       iii=0;
       jjj=0;
       nnn=0;
       /* printf("i= %d\n",i); */

       for (j=0; j<159; j++) {
         a_datas[i][j] = a_data[(i*160+j)];
         if (a_datas[i][j] != 0 && a_datas[i][j] != 32 ) num = j+1;
         if (a_datas[i][j] != 124) {  /* looks for a "|" */
           a_smenu[iii][jjj]=a_datas[i][j];
           if (a_datas[i][j] != 0 && a_datas[i][j] != 32 ) nnn = jjj+1;
           jjj=jjj+1; }
         else {
           a_smenu[iii][nnn]=0;
           iii=iii+1;
           jjj=0;
           nnn=0;
         } /* endif */

       } /* enddo */
       a_datas[i][num] = 0;
       a_smenu[iii][nnn] = 0;
       jjj=a_smenu[0][0]-48;
       if (a_smenu[0][1] != 0) jjj=jjj*10+a_smenu[0][1]-48;
       if (i_db > 9-1) printf("a_datas[%d] = %s\n",i,a_datas[i]);
       num=0;
       for (j=0; j<159; j++) {
          a_labels[i][j] = a_label[160*i+j];
        if (a_labels[i][j] != 0 && a_labels[i][j] != 32 ) num = j+1;
       } /* enddo */
       a_labels[i][num] = 0;
       if (i_db > 9-1) printf("a_labels[%d] = %s\n",i,a_labels[i]);
/*       printf("num of %d = %d\n",i,num); */
       if (num != 0) {
         formwin_e = XtVaCreateWidget("form", xmFormWidgetClass, rowcolew[i_chn[0]],
             XmNfractionBase,  10,
             XmNheight,        35,  /* added to force correct size on mac and pc  !@#$%  SJS 2/5/03 */
             NULL);
         if (a_labels[i][0] != 124) {
           labelwidgit_e = XtVaCreateManagedWidget(a_labels[i],
               xmLabelGadgetClass, formwin_e,
               XmNtopAttachment,    XmATTACH_FORM,
               XmNbottomAttachment, XmATTACH_FORM,
               XmNleftAttachment,   XmATTACH_FORM,
               XmNrightAttachment,  XmATTACH_POSITION,
               XmNrightPosition,    3,
               XmNalignment,      XmALIGNMENT_END,
               NULL);

           if (iii==0) {

             textwin = XtVaCreateManagedWidget("textwin",
                 xmTextFieldWidgetClass, formwin_e,
                 XmNrightAttachment,  XmATTACH_FORM,
                 XmNleftAttachment,   XmATTACH_POSITION,
                 XmNleftPosition,     4,
                 XmNtraversalOn,      True,
                 XmNvalue,            a_datas[i],
                 NULL);


           /* When user hits return, print the label+value of textwin */
           if (i_db > 9-1) printf("a_labels[%d]  %d  %s\n",i,&a_labels[i],a_labels[i]);
           a_lll = (char *) &a_labels[i];
/*           XtAddCallback(textwin, XmNactivateCallback,print_result, (XtPointer)a_lll);   */
           i_flag=(i_chn[0]*10000)+((i)*100)+0;
           if (i_db > 9-1) printf("Adding Callback %d\n",i_flag);
           XtAddCallback(textwin, XmNlosingFocusCallback,ewManager, (XtPointer)i_flag);
           XtAddCallback(textwin, XmNactivateCallback,ewManager, (XtPointer)i_flag);

           if (i_db > 8-1) printf("a_lll = %d %s %d %d %d\n",a_lll,a_lll,*(a_lll+0),*(a_lll+1),*(a_lll+2));

           XtAddCallback(textwin, XmNactivateCallback,
              XmProcessTraversal, (XtPointer)XmTRAVERSE_NEXT_TAB_GROUP);

           }

         else {

           if (i_db > 8-1) printf("Creating option_menu widget.  Default=%d\n",jjj);

           n = 0;
           pd_menu = XmCreatePulldownMenu(formwin_e, "My_Pulldown_Menu",args, n);

           if (i_db > 8-1) printf("Creating submenus %d\n",iii);

           for (j=1;j<iii+1;j++) {
             i_flag=(i_chn[0]*10000)+((i)*100)+j;
             if (i_db > 8-1) printf("adding pulldown option %d: %s\n",j, a_smenu[j]);
/*             menu_item[j] = XtVaCreateManagedWidget(a_smenu[j],xmPushButtonGadgetClass,pd_menu,NULL); */
             n=0;
             menu_item[j] = XmCreatePushButton(pd_menu,a_smenu[j],args,n);
             XtManageChild(menu_item[j]);
             XtAddCallback(menu_item[j],XmNactivateCallback, ewManager, (XtPointer)i_flag);

           } /* End do */
           /* XtManageChild(pd_menu);  */

           n = 0;
           XtSetArg(args[n], XmNrightAttachment,      XmATTACH_FORM); n++;
           XtSetArg(args[n], XmNleftAttachment,   XmATTACH_POSITION); n++;
           XtSetArg(args[n], XmNleftPosition,                     4); n++;
           XtSetArg(args[n], XmNsubMenuId,                  pd_menu); n++;
           XtSetArg(args[n], XmNbuttonCount,                    iii); n++;
           XtSetArg(args[n], XmNmenuHistory,         menu_item[jjj]); n++;
           op_menu = XmCreateOptionMenu(formwin_e,(char *)XmStringCreateSimple("Option_Menu"),args,n);

           XtManageChild(op_menu);

         }
         } else {
           labelwidgit_e = XtVaCreateManagedWidget(NULL,
               xmLabelGadgetClass, formwin_e,
               XmNtopAttachment,    XmATTACH_FORM,
               XmNbottomAttachment, XmATTACH_FORM,
               XmNleftAttachment,   XmATTACH_FORM,
               XmNrightAttachment,  XmATTACH_POSITION,
               XmNrightPosition,    3,
               XmNalignment,        XmALIGNMENT_END,
               NULL);



           }


         XtManageChild(formwin_e);
      }
    }
    XtManageChild(rowcolew[i_chn[0]]);

    XtRealizeWidget(ewtop[i_chn[0]]);

}



void print_result(textwin, label)       /* To debug entry window  */
Widget textwin;
char  *label;
{
    char *value = XmTextFieldGetString(textwin);

    printf("%d %s %s\n", label, label, value);
    printf("%d %d %d %d\n", label, *(label+0), *(label+1), *(label+2));
    XtFree(value);
}

/*******************************************************************/
void ewManager(textwin, client_info, call_info)
Widget textwin;
XtPointer    client_info;
XtPointer    call_info;
{
/*    char *value = XmTextFieldGetString(textwin); */
    char *value;
    int i;
    int j;
    int i_data;
    int i_edsp;
    int i_exxx;
    int i_eyyy;

    XEvent client_event;

        i_data = (int) client_info;
        i_edsp = (int)(i_data/10000);
        i_exxx = (int)((i_data - (10000*i_edsp))/100);
        i_eyyy = (int)((i_data - (10000*i_edsp)) - (100*i_exxx));

        /* printf("eyyy=%d\n",i_eyyy); */
        value=" ";
        if (i_eyyy == 0) value = XmTextFieldGetString(textwin);
        if (i_db > 7-1) printf("Set: %d,   Entry: %d,  SubEntry: %d,  Value: %s\n",i_edsp,i_exxx,i_eyyy,value);

/*    printf("%d %s %s\n", label, label, value);
    printf("%d %d %d %d\n", label, *(label+0), *(label+1), *(label+2)); */

        i_message = i_message+1;
        if(i_message==10) i_message=0;

        client_event.xclient.type = ClientMessage;
        client_event.xclient.display = dgx;
        client_event.xclient.window = XtWindow(textwin);
        client_event.xclient.format = 32;
        client_event.xclient.data.l[0] = i_edsp;
        client_event.xclient.data.l[1] = i_exxx;
        client_event.xclient.data.l[2] = i_eyyy;
        client_event.xclient.data.l[3] = 10;
        client_event.xclient.data.l[4] = i_message;
/*        XSendEvent(dgx,wgx[i_gx[i_edsp][1]],False,ButtonReleaseMask,&client_event); */
/*        XSendEvent(dgx,XtWindow(ewtop[i_edsp]),False,ButtonReleaseMask,&client_event);  This should have worked arrrggg */
        for (i=1;i<5+1;i++) {
          if (top[i] != 0) j=i;
        }

        if (i_db > 7-1) printf("j= %d\n",j);
        XSendEvent(dgx,wgx[i_gx[j][1]],False,ButtonReleaseMask,&client_event);

        for (i=0;i<160;i++) {a_message[i_message*160+i] = *(value+i);}
        if (i_db > 7-1) printf("message= %s\n",&a_message[i_message*160]);

   /* XtFree(value); */
}

/*************************************************************************/
#ifdef SGI
void get_message_(i_msg,a_msg)
#endif
#ifdef SUN
void get_message_(i_msg,a_msg)
#endif
#ifdef M10
void get_message__(i_msg,a_msg)
#endif
#if defined(HP) || defined(_linux)
void get_message(i_msg,a_msg)
#endif

int  i_msg[1];
char a_msg[160];
{
    int i;
    int i_flag;

    i_flag = 0;
    for (i=0; i < 160; i++) {
      a_msg[i] = a_message[i_msg[0]*160+i];
      a_message[i_msg[0]*160+i]=0;
      if(a_msg[i]==0 | i_flag == 1) {
        a_msg[i]=32;
        i_flag = 1;
      }
    }
    if (i_db > 7-1) printf("msg=%s\n",a_msg);

}


void ewdsp_reset(w, client_info)
Widget w;
XtPointer    client_info;
{
    int idata;

    idata = (int) client_info;
    if (i_db > 11-1) printf("setting ewtop to 0 %d\n", idata);
    ewtop[idata]=0;

}



void closeCallBack (widgy, client_info, call_info)
Widget          widgy;          /*  widget id           */
XtPointer               client_info;    /*  data from application   */
XtPointer               call_info;      /*  data from widget class  */
{

/*      XDestroyWindow(dgx,XtWindow(gftop));  */
        XtDestroyWidget(gftop);
}

/*************************************************************************/
void MenuManager (widgy, client_info, call_info)
Widget          widgy;          /*  widget id           */
XtPointer               client_info;    /*  data from application   */
XtPointer               call_info;      /*  data from widget class  */
{
   int i_data;
   XEvent client_event;

        i_data = (int) client_info;
        i_mdsp = (int)(i_data/100);
        i_mxxx = (int)((i_data - (100*i_mdsp))/10);
        i_myyy = (int)((i_data - (100*i_mdsp)) - (10*i_mxxx));

        if (i_db > 8-1) printf("Pull Down Window Number = %d %d %d %d\n",i_data, i_mdsp, i_mxxx,i_myyy);

        client_event.xclient.type = ClientMessage;
        client_event.xclient.display = dgx;
        client_event.xclient.window = wgx[i_gx[i_mdsp][0]];
        client_event.xclient.format = 32;
        client_event.xclient.data.l[0] = i_mdsp;
        client_event.xclient.data.l[1] = i_mxxx;
        client_event.xclient.data.l[2] = i_myyy;
        client_event.xclient.data.l[3] = 0;
        client_event.xclient.data.l[4] = 0;
        XSendEvent(dgx,wgx[i_gx[i_mdsp][0]],False,ButtonReleaseMask,&client_event);
}

/*************************************************************************/
void ButtonManager (widgy, client_info, call_info)
Widget          widgy;          /*  widget id           */
XtPointer               client_info;    /*  data from application   */
XtPointer               call_info;      /*  data from widget class  */
{
   int i_g;
   /*    XFontStruct *font1; */
   XmFontList fontlist;
   int  xr, yr;
   unsigned int  width, height, bwr, dr;

        i_g = (int) client_info;

        /* printf("Button Number = %d %d %d\n",i_g, i_dx[i_g], i_wx[i_g]); */

        XtVaGetValues(draw[i_g],XmNfontList, &fontlist, NULL);

        /* XtVaGetValues(draw[i_gx[i_dx[i_g]][0]],XmNfontList, &fontlist, NULL); */

        /* font1==XLoadQueryFont(dgx, "-*-courier-*-r-*--12-*");
        XmFontListCreate(font1, XmSTRING_DEFAULT_CHARSET);  */

        XGetGeometry(dgx, wgx[i_g], &root, &xr, &yr, &width, &height, &bwr, &dr);
        /* printf("Button width/height = %d %d \n", width, height); */

        XmStringDraw(dgx,wgx[i_g],fontlist,a_llll[i_g],gc,width/2,height/2-9,0,XmALIGNMENT_CENTER,
           XmSTRING_DIRECTION_L_TO_R,NULL);
}



/*************************************************************************/
#ifdef SGI
void gx_getfile_(a_file,i_inflag)
#endif
#ifdef SUN
void gx_getfile_(a_file,i_inflag)
#endif
#ifdef M10
void gx_getfile__(a_file,i_inflag)
#endif
#if defined(HP) || defined(_linux)
void gx_getfile(dsc,i_inflag)
#endif

    char a_file[120];
    int i_inflag[0];
{
    int         n;
    Arg         args[10];
    Widget      main_window, menu_bar, menu_pane, button, fsbox;
    Widget              cascade, temp;
    XmString    str1;


    int i;
    int i_cnt;
    int i_flag;
    char a_lbl[120];

    i_flag=i_inflag[0];
/*    printf("i_flag in getfile = %d\n",i_flag); */
    i_cnt = 0;
    for (i=0; i < 118; i++) {
          a_lbl[i] = a_file[i];
          if (a_lbl[i] != 0 && a_lbl[i] != 32 ) i_cnt = i+1;
        } /* enddo */
        a_lbl[i_cnt] = 0;

   /*
    *  Initialize the toolkit.
    */

      n = 0;
/*      XtSetArg(args[n], XmNtitle,     "Filenames"); n++;
      XtSetArg(args[n], XmNdepth,   visualList[0].depth); n++;
      XtSetArg(args[n], XmNvisual, visualList[0].visual); n++;
      XtSetArg(args[n], XmNcolormap,         cmap); n++; */

      if (i_db > 3-1) printf("Creating shell\n");

      gftop = XtAppCreateShell(NULL,"appClass",
                topLevelShellWidgetClass,dgx,
                args, n);
/*      gftop = XtAppCreateShell(NULL,"appClass",
                overrideShellWidgetClass,dgx,
                args, n);  */
      if (i_db > 4-1) printf("Made top level shell\n");

        /*
     *  Create main window.
         */

        n = 0;
        main_window = XmCreateMainWindow (gftop, "main1", args, n);
        XtManageChild (main_window);

        /*
         *  Create menu bar in main window.
         */

        n = 0;
        menu_bar = XmCreateMenuBar (main_window, "menu_bar", args, n);
        XtManageChild (menu_bar);

        /*
         *  Create "Actions" pulldown menu.
         */

        n = 0;
        menu_pane = XmCreatePulldownMenu (menu_bar, "menu_pane", args, n);

        n = 0;
        button = XmCreatePushButton (menu_pane, "Close", args, n);
        XtManageChild (button);
        XtAddCallback (button, XmNactivateCallback, closeCallBack, NULL);

        n = 0;
        XtSetArg (args[n], XmNsubMenuId, menu_pane);  n++;
        cascade = XmCreateCascadeButton (menu_bar, "Window", args, n);
        XtManageChild (cascade);


        str1 = XmStringCreateLtoR ("Files", XmSTRING_DEFAULT_CHARSET);

    n = 0;
    XtSetArg (args[n], XmNlistLabelString, str1); n++;
    fsbox = XmCreateFileSelectionBox (main_window, "fileselect", args, n);
        XtManageChild (fsbox);
    XmStringFree (str1);

    XtAddCallback (fsbox, XmNokCallback, okCallBack, (XtPointer)i_flag);

        temp = XmFileSelectionBoxGetChild (fsbox, XmDIALOG_CANCEL_BUTTON);
        XtUnmanageChild (temp);
        temp = XmFileSelectionBoxGetChild (fsbox, XmDIALOG_HELP_BUTTON);
        XtUnmanageChild (temp);

    /*
     *  Set Main Window areas.
     */

        XmMainWindowSetAreas (main_window, menu_bar, NULL, NULL, NULL, fsbox);

    /*
     *  Realize the widget hierarchy and enter the main loop.
     */

    XtRealizeWidget (gftop);
}


void
okCallBack (widgy, client_info, call_info)
    Widget                widgy;
    XtPointer               client_info;
    XmFileSelectionBoxCallbackStruct  *call_info;
{
    Arg         args[10];
        XmString        pathstring = NULL;
        XmString        carraige_rtn = NULL;

    XEvent client_event;
    int i, j;
    int i_data;
    int i_edsp;

    char *value;

    int     n, stat_val;

    static XmStringCharSet   charset = (XmStringCharSet) XmSTRING_DEFAULT_CHARSET;
    char        *path;

        i_data = (int) client_info;
        i_edsp = 0;

        value=" ";
        value = XmTextFieldGetString(widgy);

    n = 0;

        carraige_rtn = XmStringCreateLtoR ("\012\012", charset);
        pathstring = XmStringConcat (call_info->value, carraige_rtn);
        XmStringGetLtoR (call_info->value, charset, &path);


        XtDestroyWidget(gftop);  /* this line closes file selection window */


        i_message = i_message+1;
        if(i_message==10) i_message=0;

        client_event.xclient.type = ClientMessage;
        client_event.xclient.display = dgx;
        client_event.xclient.format = 32;
        client_event.xclient.data.l[0] = 0;
        client_event.xclient.data.l[1] = 0;
        client_event.xclient.data.l[2] = i_data;
        client_event.xclient.data.l[3] = 12;
        client_event.xclient.data.l[4] = i_message;
        for (i=1;i<5+1;i++) {
          if (top[i] != 0) j=i;
        }

        if (i_db > 7-1) printf("okCallBack j= %d\n",j);
        XSendEvent(dgx,wgx[i_gx[j][1]],False,ButtonReleaseMask,&client_event);

        for (i=0;i<160;i++) {a_message[i_message*160+i] = *(path+i);}
        if (i_db > 7-1) printf("message= %s\n",&a_message[i_message*160]);


}


/*************************************************************************/
#ifdef SGI
void topwin_(i_w)
#endif
#ifdef SUN
void topwin_(i_w)
#endif
#ifdef M10
void topwin_(i_w)
#endif
#if defined(HP) || defined(_linux)
void topwin(i_w)
#endif

    int i_w[];

{

    if (top[i_w[0]] != 0) XRaiseWindow(dgx, top[i_w[0]]);

}

/*************************************************************************/
#ifdef SGI
void get_wininfo_(i_d, i_w, i_vx, i_vy, i_vw, i_vh, i_cw, i_ch,i_widget)
#endif
#ifdef SUN
void get_wininfo_(i_d, i_w, i_vx, i_vy, i_vw, i_vh, i_cw, i_ch,i_widget)
#endif
#ifdef M10
void get_wininfo__(i_d, i_w, i_vx, i_vy, i_vw, i_vh, i_cw, i_ch,i_widget)
#endif
#if defined(HP) || defined(_linux)
void get_wininfo(i_d, i_w, i_vx, i_vy, i_vw, i_vh, i_cw, i_ch,i_widget)
#endif

    int i_d[];
    int i_w[];
    int i_vx[], i_vy[];
    int i_vw[], i_vh[];
    int i_cw[], i_ch[];
    int i_widget[];

{
    int           xr, yr;
    unsigned int  width, height, bwr, dr;

    XGetGeometry(dgx, wgx[i_gx[i_d[0]][i_w[0]]], &root, &xr, &yr, &width, &height, &bwr, &dr);

    i_vx[0] = -xr;
    i_vy[0] = -yr;
    i_cw[0] = width;
    i_ch[0] = height;

    XGetGeometry(dgx, fgx[i_gx[i_d[0]][i_w[0]]], &root, &xr, &yr, &width, &height, &bwr, &dr);

    i_vw[0] = width;
    i_vh[0] = height;

    if (scrl[i_gx[i_d[0]][i_w[0]]] != 0) {  /* gets proper viewport size when scrollbars are present  Should
          probably correct for this elsewhere so the configure event always returns the correct size */

      XGetGeometry(dgx, XtWindow(scrl[i_gx[i_d[0]][i_w[0]]]), &root, &xr, &yr, &width, &height, &bwr, &dr);

      i_vw[0] = width+xr;
      i_vh[0] = height+yr;

      if (i_db == -21) printf("scroll bar size= %d %d %d %d\n",xr,yr,width,height);

    }

    i_widget[0] = i_push;

}

/*************************************************************************/
#ifdef SGI
void move_scroll_(i_d,i_w,i_x,i_y)
#endif
#ifdef SUN
void move_scroll_(i_d,i_w,i_x,i_y)
#endif
#ifdef M10
void move_scroll__(i_d,i_w,i_x,i_y)
#endif
#if defined(HP) || defined(_linux)
void move_scroll(i_d,i_w,i_x,i_y)
#endif

    int  i_d[];
    int  i_w[];
    int  i_x[];
    int  i_y[];

{
    Widget vsb;
    Widget hsb;

    /*     XWindowChanges    xwc; */

    int increment=0;
    int maximum=0;
    int minimum=0;
    int page_incr=0;
    int slider_size=0;
    int value=0;

    XtVaGetValues(scrl[i_gx[i_d[0]][i_w[0]]],XmNverticalScrollBar,  &vsb,NULL);
    XtVaGetValues(scrl[i_gx[i_d[0]][i_w[0]]],XmNhorizontalScrollBar,&hsb,NULL);

    XtVaGetValues(vsb,XmNincrement,     &increment,
                      XmNmaximum,       &maximum,
                      XmNminimum,       &minimum,
                      XmNpageIncrement, &page_incr,
                      XmNsliderSize,    &slider_size,
                      XmNvalue,         &value,
                      NULL);

/*
    printf("inc=%d, max=%d, min=%d, page=%d, slider=%d, value=%d\n",
         increment,maximum,minimum,page_incr,slider_size,value);
*/

    value=i_y[0];
    if (value < minimum) value = minimum;
    if (value > maximum-slider_size) value = maximum-slider_size;
    XmScrollBarSetValues(vsb,value,slider_size,increment,page_incr,True);

    XtVaGetValues(hsb,XmNincrement,     &increment,
                      XmNmaximum,       &maximum,
                      XmNminimum,       &minimum,
                      XmNpageIncrement, &page_incr,
                      XmNsliderSize,    &slider_size,
                      XmNvalue,         &value,
                      NULL);

    value=i_x[0];
    if (value < minimum) value = minimum;
    if (value > maximum-slider_size) value = maximum-slider_size;
    XmScrollBarSetValues(hsb,value,slider_size,increment,page_incr,True);

/*     The following code would change the slider positions, but not move the data properly
    n = 0;
    XtSetArg(args[n], XmNvalue, i_x[0]); n++;
    XtSetValues(hsb, args, n);

    n = 0;
    XtSetArg(args[n], XmNvalue, i_y[0]); n++;
    XtSetValues(vsb, args, n);

    xwc.x=-i_x[0];
    xwc.y=-i_y[0];

    XConfigureWindow(dgx, wgx[i_gx[i_d[0]][i_w[0]]], CWX | CWY, &xwc);
*/

}

/*************************************************************************/
#ifdef SGI
void resize_win_(i_d,i_w,i_x,i_y)
#endif
#ifdef SUN
void resize_win_(i_d,i_w,i_x,i_y)
#endif
#ifdef M10
void resize_win__(i_d,i_w,i_x,i_y)
#endif
#if defined(HP) || defined(_linux)
void resize_win(i_d,i_w,i_x,i_y)
#endif

    int  i_d[];
    int  i_w[];
    int  i_x[];
    int  i_y[];

{
    unsigned int   wide;
    unsigned int   high;

    int n;
    Arg   args[10];

    int maximum=30000;
    int minimum=100;

/*     XWindowChanges    xwc; */

    wide=i_x[0];
    if (wide < minimum) wide = minimum;
    if (wide > maximum) wide = maximum;

    high=i_y[0];
    if (high < minimum) high = minimum;
    if (high > maximum) high = maximum;

/*    xwc.width=wide;  ! for some reason, this code did not update the scoll bars properly
    xwc.height=high;
    XConfigureWindow(dgx, wgx[i_gx[i_d[0]][i_w[0]]], CWWidth | CWHeight, &xwc);  */

/*    XResizeWindow(dgx,wgx[i_gx[i_d[0]][i_w[0]]],wide,high); */

/*    XtResizeWidget(draw[i_gx[i_d[0]][i_w[0]]],wide,high);  */

    n = 0;
    XtSetArg(args[n], XmNwidth, wide); n++;
    XtSetArg(args[n], XmNheight, high); n++;
    XtSetValues(draw[i_gx[i_d[0]][i_w[0]]], args, n);
    XFlush(dgx);


}

/*************************************************************************/
#ifdef SGI
void set_button_shadow_(i_d,i_w,i_shadow,i_debug)
#endif
#ifdef SUN
void set_button_shadow_(i_d,i_w,i_shadow,i_debug)
#endif
#ifdef M10
void set_button_shadow__(i_d,i_w,i_shadow,i_debug)
#endif
#if defined(HP) || defined(_linux)
void set_button_shadow(i_d,i_w,i_shadow,i_debug)
#endif

    int  i_d[];
    int  i_w[];
    int  i_shadow[];
    int  i_debug[];

{
    int   n;
    Arg   args[10];

    n = 0;
    if (i_shadow[0] == 1) {
      XtSetArg(args[n], XmNshadowType, XmSHADOW_IN); n++;
      if (i_debug[0] > 7-1) printf("setting shadow in %d %d\n",i_d[0],i_w[0]);
      }
    else {
      XtSetArg(args[n], XmNshadowType, XmSHADOW_OUT); n++;
      if (i_debug[0] > 7-1) printf("setting shadow out %d %d\n",i_d[0],i_w[0]);
    }

    XtSetValues(draw[i_gx[i_d[0]][i_w[0]]], args, n);

/*      if (i_tx[i_g] == 0) {
          XtVaSetValues(draw[i_g],XmNshadowType, XmSHADOW_IN,NULL);
          }
        else {
          XtVaSetValues(draw[i_g],XmNshadowType, XmSHADOW_OUT,NULL);
          i_tx[i_g]=1;
        }
*/

}

/*************************************************************************/
#ifdef SGI
void move_win_(i_d,i_w,i_x,i_y)
#endif
#ifdef SUN
void move_win_(i_d,i_w,i_x,i_y)
#endif
#ifdef M10
void move_win__(i_d,i_w,i_x,i_y)
#endif
#if defined(HP) || defined(_linux)
void move_win(i_d,i_w,i_x,i_y)
#endif

    int  i_d[];
    int  i_w[];
    int  i_x[];
    int  i_y[];

{
    XWindowChanges    xwc;

    xwc.x=-i_x[0];
    xwc.y=-i_y[0];

    XConfigureWindow(dgx, wgx[i_gx[i_d[0]][i_w[0]]], CWX | CWY, &xwc);

}

/*************************************************************************/
#ifdef SGI
void destroy_display_(i_d)
#endif
#ifdef SUN
void destroy_display_(i_d)
#endif
#ifdef M10
void destroy_display__(i_d)
#endif
#if defined(HP) || defined(_linux)
void destroy_display(i_d)
#endif

    int  i_d[];

{
    XUnmapWindow(dgx,top[i_d[0]]);
    XDestroyWindow(dgx,top[i_d[0]]);
}

/*************************************************************************/
#ifdef SGI
void getevent_(i_flg,i_event)
#endif
#ifdef SUN
void getevent_(i_flg,i_event)
#endif
#ifdef M10
void getevent_(i_flg,i_event)
#endif
#if defined(HP) || defined(_linux)
void getevent(i_flg,i_event)
#endif

    int i_flg[];
    int i_event[10];
{
    XEvent        report;

    int           i;
    int      i_loop;

        char buffer[40];
        int bufsize = 40;
        KeySym keysym;
        XComposeStatus compose;

      i_event[0] = 0;
      i_event[1] = 0;
      i_event[2] = 0;
      i_event[3] = 0;
      i_event[4] = 0;
      i_event[5] = 0;
      i_event[6] = 0;
      i_event[7] = 0;
      if (i_flg[0] == 0 | XPending(dgx) ) {
        i_loop = 0;
        while(i_loop == 0) {
          XtAppNextEvent(app_context,&report);
          /* XNextEvent(dgx,&report); */
          /* printf("report.type = %d \n",report.type); */
          /* switch (report.type) {
            case Expose:
              printf("report=Expose %d\n",report.xexpose.window);
              break;
            case ConfigureNotify:
              printf("report=ConfigureNotify %d\n",report.xconfigure.window);
              break;
            case ButtonPress:
              printf("report=ButtonPress %d\n",report.xbutton.window);
              break;
            case ButtonRelease:
              printf("report=ButtonRelease %d\n",report.xbutton.window);
              break;
            case KeyPress:
              printf("report=KeyPress %d\n",report.xkey.window);
              break;
            case KeyRelease:
              printf("report=KeyRelease %d\n",report.xkey.window);
              break;
            case DestroyNotify:
              printf("report=DestroyNotify %d\n",report.xdestroywindow.window);
              break;
            default:
              break;  */ /* do nothing */
          /* } */ /* end case */
          switch (report.type) {
            case Expose:
              for(i=1; i<i_gmax+1; i++)
                if (report.xexpose.window == wgx[i])  {
                  i_event[0] = i_dx[i];
                  i_event[1] = i_wx[i];
                }
              if (i_event[0] == 0) break;
              i_event[2] = 1;
              i_event[3] = report.xexpose.x;
              i_event[4] = report.xexpose.y;
              i_event[5] = report.xexpose.width;
              i_event[6] = report.xexpose.height;
              i_event[7] = report.xexpose.count;
              i_loop = 1;
              break;
            case ConfigureNotify:
              for (i=1; i<i_gmax+1; i++)
                if (report.xconfigure.window == wgx[i]) {
                  i_event[0] = i_dx[i];
                  i_event[1] = i_wx[i];
                }
              for (i=1; i<i_gmax+1; i++)
                if (report.xconfigure.window == fgx[i]) {
                  i_event[0] =  i_dx[i];
                  i_event[1] = -i_wx[i];
                }
              if (i_event[0] == 0) break;
              if (i_event[1] > 0) {
                i_event[2] = 2;
                i_event[3] = -report.xconfigure.x;
                i_event[4] = -report.xconfigure.y;
                i_event[5] = report.xconfigure.width;
                i_event[6] = report.xconfigure.height;
                i_event[7] = 0;
                i_loop = 1; }
              else {
                i_event[1] = -i_event[1];
                i_event[2] = 3;
                i_event[3] = report.xconfigure.x;
                i_event[4] = report.xconfigure.y;
                i_event[5] = report.xconfigure.width;
                i_event[6] = report.xconfigure.height;
                i_event[7] = 0;
                i_loop = 1;
              } /* endif */
              break;
            case ButtonPress:
              for (i=1; i<i_gmax+1; i++) {
                if (report.xbutton.window == wgx[i]) {
                  i_event[0] = i_dx[i];
                  i_event[1] = i_wx[i];
                }
                if (report.xbutton.window == lgx[i]) {
                  i_event[0] =  i_dx[i];
                  i_event[1] = -i_wx[i];
                }
              }
              if (i_event[0] == 0) break;
              i_event[2] = 4;
              i_event[3] = report.xbutton.button;
              i_event[4] = report.xbutton.x;
              i_event[5] = report.xbutton.y;
              i_event[6] = 0;
              i_event[7] = 0;
              i_loop = 1;
              break;
            case ButtonRelease:
              for (i=1; i<i_gmax+1; i++) {
                if (report.xbutton.window == wgx[i]) {
                  i_event[0] = i_dx[i];
                  i_event[1] = i_wx[i];
                }
                if (report.xbutton.window == lgx[i]) {
                  i_event[0] =  i_dx[i];
                  i_event[1] = -i_wx[i];
                }
              }
              if (i_event[0] == 0) break;
              i_event[2] = 5;
              i_event[3] = report.xbutton.button;
              i_event[4] = report.xbutton.x;
              i_event[5] = report.xbutton.y;
              i_event[6] = 0;
              i_event[7] = 0;
/*              if (i_event[1] == 0) {
                i_event[4] = i_mxxx;
                i_event[5] = i_myyy;
                i_event[6] = i_mdsp;
                i_mxxx = 0;
                i_myyy = 0;
                i_mdsp = 0;
              }  */
              i_loop = 1;
              break;
            case KeyPress:
              for (i=1; i<i_gmax+1; i++) {
                if (report.xkey.window == wgx[i]) {
                  i_event[0] = i_dx[i];
                  i_event[1] = i_wx[i];
                }
                if (report.xkey.window == lgx[i]) {
                  i_event[0] =  i_dx[i];
                  i_event[1] = -i_wx[i];
                }
              }
              if (i_event[0] == 0) break;
              XLookupString(&report.xkey, buffer,bufsize,&keysym,&compose);
              i_event[2] = 6;
              i_event[3] = report.xkey.keycode;
              i_event[4] = report.xkey.x;
              i_event[5] = report.xkey.y;
              i_event[6] = keysym;
              i_event[7] = 0;
              i_loop = 1;
              break;
            case KeyRelease:
              for (i=1; i<i_gmax+1; i++) {
                if (report.xkey.window == wgx[i]) {
                  i_event[0] = i_dx[i];
                  i_event[1] = i_wx[i];
                }
                if (report.xkey.window == lgx[i]) {
                  i_event[0] =  i_dx[i];
                  i_event[1] = -i_wx[i];
                }
              }
              if (i_event[0] == 0) break;
              XLookupString(&report.xkey, buffer,bufsize,&keysym,&compose);
              i_event[2] = 7;
              i_event[3] = report.xkey.keycode;
              i_event[4] = report.xkey.x;
              i_event[5] = report.xkey.y;
              i_event[6] = keysym;
              i_event[7] = 0;
              i_loop = 1;
              break;
            case DestroyNotify:
              for (i=1; i<i_dmax+1; i++) {
                if (report.xdestroywindow.window == top[i]) top[i] = 0;
                /* printf("Setting top[%d] to zero\n",i); */
              }
              for (i=1; i<i_gmax+1; i++) {
                if (report.xdestroywindow.window == wgx[i]) {
                  i_event[0] = i_dx[i];
                  i_event[1] = i_wx[i];
                  i_dx[i] = 0;
                  i_wx[i] = 0;
                }
              }
              if (i_event[0] == 0) break;
              i_event[2] = 8;
              i_event[3] = 0;
              i_loop = 1;
              break;
            case MotionNotify:
              for (i=1; i<i_gmax+1; i++) {
                if (report.xmotion.window == wgx[i]) {
                  i_event[0] = i_dx[i];
                  i_event[1] = i_wx[i];
                }
                if (report.xmotion.window == lgx[i]) {
                  i_event[0] =  i_dx[i];
                  i_event[1] = -i_wx[i];
                }
              }
              if (i_event[0] == 0) break;
              i_event[2] = 9;
              i_event[3] = report.xmotion.state;
              i_event[4] = report.xmotion.x;
              i_event[5] = report.xmotion.y;
              i_event[6] = 0;
              i_event[7] = 0;
              i_loop = 1;
              break;
            case ClientMessage:
              if (i_db > 9-1) printf("report=ClientMessage %d %d %d\n",report.xclient.data.l[0],
                     report.xclient.data.l[1],report.xclient.data.l[2]);
              i_event[0] = report.xclient.data.l[0];
              i_event[1] = 0;
              i_event[2] = report.xclient.data.l[3];
              i_event[3] = 0;
              i_event[4] = report.xclient.data.l[1];
              i_event[5] = report.xclient.data.l[2];
              i_event[6] = report.xclient.data.l[4];
              i_event[7] = 0;
              i_loop = 1;
              break;
            default:
              break; /* do nothing */
          } /* end case */
          if (i_event[2] == 4 | i_event[2] == 5) {
            if (i_event[3] == 2 ) {
              if (i_event[1] > 0 ) {
                if (i_type[ i_event[1]] == 1 ) report.xbutton.button = 1; }
              else {
                if (i_type[-i_event[1]] == 4 ) report.xbutton.button = 1;
                if (i_type[-i_event[1]] == 3 ) report.xbutton.button = 1;
              } /* endif */
            } /* endif */
          } /* endif */
          XtDispatchEvent(&report);
          if (i_flg[0] == 1 && !XPending(dgx)) i_loop = 1;
        } /* end while */
      } /* end if */
}

/*************************************************************************/
#ifdef SGI
void clear_win_(i_d,i_w)
#endif
#ifdef SUN
void clear_win_(i_d,i_w)
#endif
#ifdef M10
void clear_win__(i_d,i_w)
#endif
#if defined(HP) || defined(_linux)
void clear_win(i_d,i_w)
#endif

    int i_d[];
    int i_w[];
{
    XClearWindow(dgx, wgx[i_gx[i_d[0]][i_w[0]]]);
}

/*************************************************************************/
#ifdef SGI
void get_dialog_(a_msg,a_rsp)
#endif
#ifdef SUN
void get_dialog_(a_msg,a_rsp)
#endif
#ifdef M10
void get_dialog__(a_msg,a_rsp)
#endif
#if defined(HP) || defined(_linux)
void get_dialog(a_msg,a_rsp)
#endif
    char a_msg[];
    char a_rsp[];
{
    XEvent report;

    int           j;
    int      i_loop;

    static Window pop_win;
        char buffer[40];
        int bufsize;
        int start_x,start_y;
        KeySym keysym;
        XComposeStatus compose;
        int count;
        unsigned int pop_width, pop_height;
        char a_lbl[40];
    int x,y;
        int length;
    int i_cnt;
    int i_event[10];
    GC                  def_gc;

      i_event[0] = 0;
      i_event[1] = 0;
      i_event[2] = 0;
      i_event[3] = 0;
      i_event[4] = 0;
      i_event[5] = 0;
      i_event[6] = 0;

        bufsize = 40;
        count = 0;
        x = 100;
        y = 100;


        i_cnt=0;
        for (j=0; j < 39; j++) {
          a_rsp[j] = 0;
          a_lbl[j] = a_msg[j];
          if (a_lbl[j] != 0 && a_lbl[j] != 32 ) i_cnt = j+1;
        } /* enddo */
        if (i_cnt == 40) i_cnt = 39;
        a_lbl[i_cnt] = 0;


        pop_width = 300;
        pop_height = 75;
        pop_win = XCreateSimpleWindow(dgx, root,x,y,pop_width,pop_height,
                     3,BlackPixel(dgx,screen),WhitePixel(dgx,screen));

        def_gc        = DefaultGC(dgx, screen);
        /* Calculate starting position of string in window */

        start_x = 5;
        start_y = 20;
        XSelectInput(dgx,pop_win,ExposureMask | KeyPressMask );

        XMapWindow(dgx,pop_win);

        i_loop = 0;
        while(i_loop == 0) {
          XNextEvent(dgx,&report);
          switch (report.type) {
            case Expose:
              if (report.xexpose.window == pop_win) {
                XDrawString(dgx,pop_win,def_gc,start_x,start_y   ,a_lbl,strlen(a_lbl));
                XDrawString(dgx,pop_win,def_gc,start_x,start_y+15,a_rsp,strlen(a_rsp));
              }
              break;
            case KeyPress:
              if (report.xkey.window == pop_win) {
                count = XLookupString(&report.xkey, buffer,bufsize,&keysym,&compose);
                if (count == 40) count=39;
                buffer[count]=0;
                if ((keysym == XK_Return) || (keysym == XK_KP_Enter) ||
                    (keysym== XK_Linefeed)) {
                  XUnmapWindow(dgx,pop_win);
                  XDestroyWindow(dgx,pop_win);
                  i_loop = 1;
                  break; }
                else if (((keysym >= XK_KP_Space) && (keysym <= XK_KP_9)) ||
                          ((keysym >= XK_space) && (keysym <= XK_asciitilde))) {
                  if ((strlen(a_rsp) + strlen(buffer)) >= 40 ) XBell(dgx,100);
                  else strcat(a_rsp,buffer); }
                else if ((keysym >= XK_Shift_L) && (keysym <= XK_Hyper_R));
                  /* Do Nothing because it's a modifier key */
                else if ((keysym >= XK_F1) && (keysym <= XK_F35)) {
                  if (buffer == NULL) printf("Unmapped function key\n");
                  else if ((strlen(a_rsp) + strlen(buffer)) >= 40) { XBell(dgx,100); }
                  else { strcat(a_rsp,buffer); } }
                else if ((keysym == XK_BackSpace) || (keysym == XK_Delete)) {
                  if ((length = strlen(a_rsp)) > 0) {
                    a_rsp[length - 1] = (char)NULL;
                    XClearWindow(dgx,pop_win);  }
                  else {
                    XBell(dgx,100); } }
                else {
                  printf("keysym %s is not handled\n",XKeysymToString(keysym));
                  XBell(dgx,100); }

                XDrawString(dgx,pop_win,def_gc,start_x,start_y   ,a_lbl,strlen(a_lbl));
                XDrawString(dgx,pop_win,def_gc,start_x,start_y+15,a_rsp,strlen(a_rsp));
                break;

              }
              break;
            default:
              break; /* do nothing */
          } /* end case */
          if (i_event[1] == 4 | i_event[1] == 5) {
            if (i_event[2] == 2 ) {
              if (i_event[0] > 0 ) {
                if (i_type[ i_event[0]] == 1 ) report.xbutton.button = 1; }
              else {
                if (i_type[-i_event[0]] == 4 ) report.xbutton.button = 1;
                if (i_type[-i_event[0]] == 3 ) report.xbutton.button = 1;
              } /* endif */
            } /* endif */
          } /* endif */
          XtDispatchEvent(&report);
        } /* end while */

        for (j=0; j < 38; j++) {
          if (a_rsp[j] == 0 ) a_rsp[j]=32;
        } /* enddo */
}

/*************************************************************************/
int myhandler (display, myerr)
Display *display;
XErrorEvent *myerr;
{
    char msg[80];
    char ttt[80];
    strcpy(ttt ,"BadDrawable (invalid Pixmap or Window parameter)");
    XGetErrorText(display, myerr->error_code,msg,80);
    if(strcmp(msg,ttt)!= 0 ) {
      fprintf(stderr, "error code %s\n", msg);
    }
    return(0);
}



/*************************************************************************/
#ifdef SGI
int init_gx_(i_wxi,i_typ,a_labl,i_wxs,i_wys,i_frx,i_fry,a_menu,a_lcolor,i_cin,i_din)
#endif
#ifdef SUN
int init_gx_(i_wxi,i_typ,a_labl,i_wxs,i_wys,i_frx,i_fry,a_menu,a_lcolor,i_cin,i_din)
#endif
#ifdef M10
int init_gx__(i_wxi,i_typ,a_labl,i_wxs,i_wys,i_frx,i_fry,a_menu,a_lcolor,i_cin,i_din)
#endif
#if defined(HP) || defined(_linux)
int init_gx(i_wxi,i_typ,a_labl,i_wxs,i_wys,i_frx,i_fry,a_menu,a_lcolor,i_cin,i_din)
#endif

    int  i_wxi[];
    int  i_typ[];
    int  i_wxs[];
    int  i_wys[];
    int  i_frx[];
    int  i_fry[];
    int  i_cin[];
    int  i_din[];

    char a_menu[];
    char a_labl[];
    char a_lcolor[];
{
    Widget          toplevel, maintoplevel;
    Widget          main_dx, form1, pane;
    Widget          temp;
    Widget          menu_bar, menu_pane, button, cascade;

    Arg             args[15];
    int             n = 1;
    char                        *ww1[2];
    char                        ww2[80];
    char            a_lbl[1000];
    char            a_clr[1000];
    char            a_title[20];
    int             i,j,k,l,num;
    int             ix;
    int             i_tttt;
    int             i_cnt[20];
    int             i_d;
    int             i_w;
    int             i_g;
    int             i_flag;
    XWindowAttributes     xwa;
    XSetWindowAttributes xswa;


    XVisualInfo vTemplate;
    int visualsMatched;
    int num_depths;
    int *depths;
    int default_depth;
    Visual *default_visual;
    Status rc;                  /* return status of various Xlib functions.  */
    XColor red, brown, blue, yellow, green, linec;
    static char *visual_class[] = {
          "StaticGray",
          "GrayScale",
          "StaticColor",
          "PseudoColor",
          "TrueColor",
          "DirectColor"
    };

    XErrorHandler defaulterr;
    Widget vsb;
    Widget hsb;

      ww1[0] = &ww2[0];
      strcpy (ww2,"Graphx");

      i_db = i_din[0];

      if (i_db > 3-1) printf("Start.\n");

      /* Initialize the intrinsics     */

      if (i_init == 0) {
        i_init = 1;
        i_bswap= (int *)&b_bswap;
        b_bswap[0]=0;
        b_bswap[1]=0;
        b_bswap[2]=0;
        b_bswap[3]=1;
        if (*i_bswap == 1) {
          if (i_db > 4-1) printf("This Machine is Big Endian\n"); }
        else {
          if (i_db > 4-1) printf("This Machine is Little Endian\n");
          if (i_db > 4-1) printf("i_bswap=%d\n",*i_bswap);
        }
        i_clrs = i_cin[0];
        if (i_clrs <   0) i_clrs = 0;
        if (i_clrs > 256) i_clrs = 256;
        if (i_db > 1-1) printf("Initializing X toolkit\n");
        i_cnt[0]=0;
        for (j=0; j < 78; j++) {
          a_lbl[j] = a_labl[(j)];
          if (a_lbl[j] != 0 && a_lbl[j] != 32 ) i_cnt[0] = j+1;
        } /* enddo */
        a_lbl[i_cnt[0]] = 0;
        if (i_cnt[0] == 0) strcpy(a_lbl,"GraphX");

        maintoplevel = XtAppInitialize(&app_context,
                                       "Graphx",
                                       NULL,0,
                                       &n,ww1,
                                       NULL,
                                       NULL,0);

        dgx       = XtDisplay(maintoplevel);

        screen    = DefaultScreen(dgx);
        gc        = DefaultGC(dgx, screen);
        root      = XDefaultRootWindow(dgx);


        depths = XListDepths(dgx,screen,&num_depths);
        if (i_db > 3-1) {
          printf(" \n");
          printf("Number of Depths avail = %d\n",num_depths);
          for (j=0; j<num_depths; j++) {
            printf("  Depth(%d)=%d\n",j,depths[j]);
          } /* enddo */
        } /* enddo */

        default_depth = DefaultDepth(dgx,screen);
        if (i_db > 3-1) printf("Default Depth = %d\n",default_depth);

        vTemplate.screen = screen;
        visualList = XGetVisualInfo(dgx, VisualScreenMask,
              &vTemplate, &visualsMatched);
        if (visualsMatched == 0) {
          printf("No visuals\n");
          exit(0);
        } /* endif */

        default_visual = DefaultVisual(dgx,screen);
        if (i_db > 3-1) {
          printf(" \n");
          printf("Number of visuals:  %d\n",visualsMatched);
          for (j=0; j<visualsMatched; j++) {
            printf("  %d) Visual ID = %d  size=%d  bpc=%d depth=%d type=%s\n",j,visualList[j].visualid,
                                                                           visualList[j].colormap_size,
                                                                           visualList[j].bits_per_rgb,
                                                                           visualList[j].depth,
                                                                           visual_class[visualList[j].class]);
          } /* enddo */
          printf("Default Visual  = %d\n",XVisualIDFromVisual(default_visual));
        } /* end if */

        vTemplate.screen = screen;
        vTemplate.depth = 24;
        vTemplate.class = TrueColor;
        visualList = XGetVisualInfo(dgx, VisualScreenMask | VisualDepthMask | VisualClassMask,
              &vTemplate, &visualsMatched);
        if (visualsMatched == 0) {
          vTemplate.class = TrueColor;
          visualList = XGetVisualInfo(dgx, VisualScreenMask | VisualClassMask,
              &vTemplate, &visualsMatched);
         if (visualsMatched == 0) {
          printf("No matching visuals\n");
          exit(0);
         } /* endif */
        } /* endif */

        if (i_db > 4-1) {
          printf(" \n");
          printf("Number of matching visuals:  %d\n",visualsMatched);
          for (j=0; j<visualsMatched; j++) {
            printf("%d)  Visual ID = %d  size=%d  bpc=%d depth=%d type=%s\n",j,visualList[j].visualid,
                                                                             visualList[j].colormap_size,
                                                                             visualList[j].bits_per_rgb,
                                                                             visualList[j].depth,
                                                                             visual_class[visualList[j].class]);
          } /* enddo */
        }

          printf(" \n");
          if (i_db > 2-1) printf("Using visual ID=%d  size=%d  bpc=%d depth=%d type=%s\n",
                                visualList[0].visualid,
                                visualList[0].colormap_size,
                                visualList[0].bits_per_rgb,
                                visualList[0].depth,
                                visual_class[visualList[0].class]);
          i_rmaxr = visualList[0].red_mask;
          i_rmltr = 1;
          i_gmaxg = visualList[0].green_mask;
          i_gmltg = 1;
          i_bmaxb = visualList[0].blue_mask;
          i_bmltb = 1;
          for (j=0;j<32; j++) {
/*            printf("i_rmaxr, 2*(int) (i_rmaxr/2) %d %d\n",i_rmaxr,2*(int) (i_rmaxr/2)); */
            if (i_rmaxr == 2*(int) (i_rmaxr/2)) {
              i_rmaxr = i_rmaxr/2;
              i_rmltr = i_rmltr*2;
            }
/*            printf("i_gmaxg, 2*(int) (i_gmaxr/2) %d %d\n",i_gmaxg,2*(int) (i_gmaxg/2)); */
            if (i_gmaxg == 2*(int) (i_gmaxg/2)) {
              i_gmaxg = i_gmaxg/2;
              i_gmltg = i_gmltg*2;
            }
/*            printf("i_bmaxb, 2*(int) (i_bmaxb/2) %d %d\n",i_bmaxb,2*(int) (i_bmaxb/2)); */
            if (i_bmaxb == 2*(int) (i_bmaxb/2)) {
              i_bmaxb = i_bmaxb/2;
              i_bmltb = i_bmltb*2;
            }
          }

          i_rmaxr=i_rmaxr+1;
          i_gmaxg=i_gmaxg+1;
          i_bmaxb=i_bmaxb+1;

          if (i_db > 4-1) {
            printf("  red_mask=%d\n",visualList[0].red_mask);
            printf("  grn_mask=%d\n",visualList[0].green_mask);
            printf("  blu_mask=%d\n",visualList[0].blue_mask);
            printf("red max,mult = %d  %d\n", i_rmaxr,i_rmltr);
            printf("grn max,mult = %d  %d\n", i_gmaxg,i_gmltg);
            printf("blu max,mult = %d  %d\n", i_bmaxb,i_bmltb);
          }

        defaulterr=XSetErrorHandler(myhandler);
        /*XSetErrorHandler(defaulterr);*/

          i_ctble = visualList[0].colormap_size;
/*        printf("hello = %d\n",1); */

        if (i_app == 0) {
/*          if (i_clrs == 0 && visualList[0].visualid == XVisualIDFromVisual(default_visual)) {
            if (i_db > 3-1) printf("using default color map\n");
            cmap  = XDefaultColormap(dgx, screen); }
          else {
*/
            if (i_db > 3-1) printf("creating private color map\n");
            cmap = XCreateColormap(dgx,RootWindow(dgx,screen),visualList[0].visual,AllocNone);
/*          }
*/
        }
      } /* endif */

      if (i_db > 5-1) printf("dgx = %d \n",dgx);
      if (i_db > 5-1) printf("root = %d\n",root);

      if (i_wxi[0] == 0) return(0);
      if (i_wxi[0] > 10) i_wxi[0] = 10;
      if (i_wxs[0] < 1 ) i_wxs[0] = 500;
      if (i_wys[0] < 1 ) i_wys[0] = 400;
      for (i_w=1; i_w < i_wxi[0]+1; i_w++) {
        if (i_db > 7-1) printf("i_w = %d\n",i_w);
        if (i_wxs[i_w] < 1) i_wxs[i_w] = i_wxs[0];
        if (i_wys[i_w] < 1) i_wys[i_w] = i_wys[0];
      }
/*      printf("hello = %d\n",2); */
      n = 0;
      XtSetArg(args[n], XmNtitle,         "SHELL"); n++;
      XtSetArg(args[n], XmNx,                   0); n++;
      XtSetArg(args[n], XmNy,                   0); n++;
      XtSetArg(args[n], XmNwidth,        i_wxs[0]); n++;
      XtSetArg(args[n], XmNheight,       i_wys[0]); n++;
      XtSetArg(args[n], XmNdepth,   visualList[0].depth); n++;
      XtSetArg(args[n], XmNvisual, visualList[0].visual); n++;
      XtSetArg(args[n], XmNcolormap,         cmap); n++;

      i_d=0;
      i_app = i_app+1;
      for (i=1; i < 5+1; i++) {
        if (i_db > 9-1) printf("i = %d\n",i);

        if (top[i] == 0) {
          i_d = i;
          break;
        }
      }
      if (i_d == 0) {
        printf("Too Many displays \n");
        return(0);
      }
      if (i_db > 3-1) printf("creating shell %d %d\n",i_app, i_d);

      toplevel = XtAppCreateShell(NULL,"appClass",
                topLevelShellWidgetClass,dgx,
                args, n);

      n = 0;
      XtSetArg(args[n], XmNsashWidth,    0); n++;
      XtSetArg(args[n], XmNsashHeight,   0); n++;
      XtSetArg(args[n], XmNdepth,   visualList[0].depth); n++;
      XtSetArg(args[n], XmNvisual, visualList[0].visual); n++;
      XtSetArg(args[n], XmNcolormap,  cmap); n++;
      main_dx = XtCreateManagedWidget("main",
                            xmMainWindowWidgetClass, toplevel,
                            args, n);

      n = 0;
      XtSetArg(args[n], XmNsashWidth,    0); n++;
      XtSetArg(args[n], XmNsashHeight,   0); n++;
      XtSetArg(args[n], XmNdepth,   visualList[0].depth); n++;
      XtSetArg(args[n], XmNvisual, visualList[0].visual); n++;
      XtSetArg(args[n], XmNcolormap,  cmap); n++;
      form1 = XtCreateManagedWidget("form",
                            xmFormWidgetClass, main_dx,
                            args, n);

        /*
         *  Create menu bar in main window.
         */

        if (i_db > 6-1) printf("Creating Menubar\n");
        n = 0;
/*        XtSetArg(args[n], XmNdepth,   visualList[0].depth); n++;
        XtSetArg(args[n], XmNvisual, visualList[0].visual); n++;
        XtSetArg(args[n], XmNcolormap,  cmap); n++; */
        menu_bar = XmCreateMenuBar (main_dx, "menu_bar", args, n);
        if (i_db > 6-1) printf("Managing Menubar\n");
        XtManageChild (menu_bar);

                i_g = 0;
                for (l=1; l<321;l++) {
                  if (i_dx[l]== 0) {
                    i_g = l;
                    break;
                  }
                }
                if (i_g == 0) {
                  printf("Too Many windows \n");
                  return(0);
                }
                i_w = 0;
                i_dx[i_g] = i_d;
                i_wx[i_g] = i_w;
                i_gx[i_dx[i_g]][i_wx[i_g]]=i_g;
                if (i_db > 5-1) printf("menu ** i_d,i_w,i_g = %d %d %d \n",i_d,i_w,i_g);

        for (i=0;i<9+1; i++) {
          num = 0;
          for (j=0; j < 19; j++) {
            a_lbl[j] = a_menu[(i*6*20+j)];
            /* if (a_lbl[j] == "|" ) num = j; */
            if (a_lbl[j] != 0 && a_lbl[j] != 32 ) num = j+1;
          } /* enddo */
          a_lbl[num] = 0;
          if (num != 0) {

            /*  Create pulldown menu. */

            i_w = 0;
/*            printf("hello 3\n"); */
            n = 0;
/*            XtSetArg(args[n], XmNdepth,   visualList[0].depth); n++;
            XtSetArg(args[n], XmNvisual, visualList[0].visual); n++;
            XtSetArg(args[n], XmNcolormap,  cmap); n++;  */
            menu_pane = XmCreatePulldownMenu (menu_bar, "", args, n);
/*            printf("hello 4\n"); */
            for (j=0; j<19; j++) a_title[j] = a_lbl[j];

            for (j=1; j<5+1; j++) {
              num=0;
              for (k=0; k<19; k++) {
                a_lbl[k] = a_menu[(i*20*6+j*20+k)];
                if (a_lbl[k] != 0 && a_lbl[k] != 32 ) num = k+1;
              } /* enddo */
              a_lbl[num] = 0;
              if (num != 0) {

                n = 0;
                if (i != 9) {
                  i_flag=(i_d*100)+((i+1)*10)+j; }
                else {
                  i_flag=(i_d*100)+j;
                }  /* end if */

/*                XtSetArg(args[n], XmNdepth,   visualList[0].depth); n++;
                XtSetArg(args[n], XmNvisual, visualList[0].visual); n++;
                XtSetArg(args[n], XmNcolormap,  cmap); n++;  */
                button = XmCreatePushButton (menu_pane, a_lbl, args, n);

                XtManageChild (button);
                XtAddCallback (button, XmNactivateCallback, MenuManager, (XtPointer)i_flag);

              }
            } /* enddo */

            n = 0;
/*            XtSetArg(args[n], XmNdepth,   visualList[0].depth); n++;
            XtSetArg(args[n], XmNvisual, visualList[0].visual); n++;
            XtSetArg(args[n], XmNcolormap,  cmap); n++;  */
            XtSetArg(args[n], XmNsubMenuId, menu_pane);  n++;
            cascade = XmCreateCascadeButton (menu_bar, a_title, args, n);
            XtManageChild (cascade);

            if (i == 9) {
               n = 0;
               XtSetArg (args[n], XmNmenuHelpWidget, cascade);  n++;
               XtSetValues (menu_bar, args, n);
            }


/*            printf("hello 5\n"); */


          }
        }

      if (i_db > 4-1) printf("creating pane\n");
      n = 0;
      XtSetArg(args[n], XmNsashWidth,    6); n++;
      XtSetArg(args[n], XmNsashHeight,   6); n++;
      XtSetArg(args[n], XmNdepth,   visualList[0].depth); n++;
      XtSetArg(args[n], XmNvisual, visualList[0].visual); n++;
      XtSetArg(args[n], XmNcolormap,  cmap); n++;
        XtSetArg(args[n], XmNtopAttachment,        XmATTACH_FORM); n++;
        XtSetArg(args[n], XmNbottomAttachment,     XmATTACH_FORM); n++;
        XtSetArg(args[n], XmNleftAttachment,       XmATTACH_FORM); n++;
        XtSetArg(args[n], XmNrightAttachment,      XmATTACH_FORM); n++;
        XtSetArg(args[n], XmNtopOffset,      10); n++;
      pane  = XtCreateManagedWidget("pane",
                        xmPanedWindowWidgetClass, form1,
                        args, n);
      ix = 0;
      if (i_db > 5-1) printf("i_d = %d \n",i_d);
      if (i_db > 5-1) printf("i_wxi = %d \n",i_wxi[0]);
      for (i_w=1; i_w < i_wxi[0]+1; i_w++) {
        if (i_db > 6-1) printf("loop. %d %d %d \n",i_w,ix,i_frx[i_w]);
        i_g = 0;
        for (j=1; j<321;j++) {
          if (i_dx[j]== 0) {
            i_g = j;
            break;
          }
        }
        if (i_g == 0) {
          printf("Too Many windows \n");
          return(0);
        }
        i_dx[i_g] = i_d;
        i_wx[i_g] = i_w;
        i_gx[i_d][i_w]=i_g;
        if (i_db > 5-1) printf("i_d,i_w,i_g = %d %d %d \n",i_d,i_w,i_g);
        if (ix == 0) {
          n = 0;
          XtSetArg(args[n], XmNborderWidth,         0); n++;
          XtSetArg(args[n], XmNfractionBase, i_frx[0]); n++;
          XtSetArg(args[n], XmNhorizontalSpacing,   0); n++;
          XtSetArg(args[n], XmNverticalSpacing,     0); n++;
          if (i_fry[i_w] < 0) {
            XtSetArg(args[n], XmNpaneMinimum, -i_fry[i_w]+10); n++;
            XtSetArg(args[n], XmNpaneMaximum, -i_fry[i_w]+10); n++;
          }
          else {
            if (i_fry[i_w] != 0) XtSetArg(args[n], XmNheight,       i_fry[i_w]); n++;
            XtSetArg(args[n], XmNpaneMinimum,        30); n++;
            XtSetArg(args[n], XmNpaneMaximum,      2000); n++;
          } /* endif */
/*          XtSetArg(args[n], XmNdepth,   visualList[0].depth); n++;
          XtSetArg(args[n], XmNvisual, visualList[0].visual); n++;
          XtSetArg(args[n], XmNcolormap,  cmap); n++; */
          formy = XtCreateManagedWidget("form",
                            xmFormWidgetClass, pane,
                            args, n);
        } /* endif */

        if (ix+i_frx[i_w] > i_frx[0]) i_frx[i_w] = i_frx[0]-ix;

        n = 0;
        XtSetArg(args[n], XmNborderWidth,                      0); n++;
        XtSetArg(args[n], XmNfractionBase,                     100); n++;
        XtSetArg(args[n], XmNbottomAttachment,     XmATTACH_FORM); n++;
        XtSetArg(args[n], XmNtopAttachment,        XmATTACH_FORM); n++;
        XtSetArg(args[n], XmNleftAttachment,   XmATTACH_POSITION); n++;
        XtSetArg(args[n], XmNleftPosition,                    ix); n++;
        XtSetArg(args[n], XmNrightAttachment,  XmATTACH_POSITION); n++;
        XtSetArg(args[n], XmNrightPosition,          ix+i_frx[i_w]); n++;
        XtSetArg(args[n], XmNhorizontalSpacing,                0); n++;
        XtSetArg(args[n], XmNverticalSpacing,                  0); n++;
/*        XtSetArg(args[n], XmNdepth,   visualList[0].depth); n++;
        XtSetArg(args[n], XmNvisual, visualList[0].visual); n++;
        XtSetArg(args[n], XmNcolormap,  cmap); n++; */
        form[i_g] = XtCreateWidget("subform",
                         xmFormWidgetClass, formy,
                         args, n);

        ix = ix+i_frx[i_w];
        if (ix == i_frx[0]) ix = 0;

        i_cnt[i_w] = 0;
        for (j=0; j < 78; j++) {
          a_lbl[j] = a_labl[(i_w*80+j)];
          if (a_lbl[j] != 0 && a_lbl[j] != 32 ) i_cnt[i_w] = j+1;
        } /* enddo */
        a_lbl[i_cnt[i_w]] = 0;

        if (i_db > 8-1) printf("i_cnt = %d %d \n",i_cnt[i_w],i_w);
        if (i_db > 8-1) printf("i_typ = %d \n",i_typ[i_w]);

        if (i_typ[i_w] == 5) i_typ[i_w] = -4;  /* to be backward compatible with graphx14 */
        i_tttt = i_typ[i_w];
        if (i_tttt < 0) i_tttt = -i_tttt;
        i_type[i_g]=i_tttt;
//          if (i_tttt == 6) i_tttt=1;
        scrl[i_g]=0;
        switch (i_tttt) {
          case 0:
            draw[i_g] = form[i_g];

          case 1:
            if (i_cnt[i_w] > 0) {
              n = 0;
              XtSetArg(args[n], XmNtopAttachment,        XmATTACH_FORM); n++;
              XtSetArg(args[n], XmNbottomAttachment,     XmATTACH_FORM); n++;
              XtSetArg(args[n], XmNleftAttachment,       XmATTACH_FORM); n++;
              XtSetArg(args[n], XmNrightAttachment,      XmATTACH_FORM); n++;
              XtSetArg(args[n], XmNwidth,                     i_wxs[i_w]); n++;
              XtSetArg(args[n], XmNheight,                    i_wys[i_w]); n++;
/*              XtSetArg(args[n], XmNdepth,   visualList[0].depth); n++;
              XtSetArg(args[n], XmNvisual, visualList[0].visual); n++;
              XtSetArg(args[n], XmNcolormap,  cmap); n++; */

              draw[i_g] = XtCreateManagedWidget(a_lbl,
                                 xmPushButtonWidgetClass, form[i_g],
                                 args, n); }

            else {
              draw[i_g] = form[i_g];
            } /* endif */

            XtManageChild(form[i_g]);
            XtManageChild(draw[i_g]);

            break;

          case 2:
            n = 0;
            XtSetArg(args[n], XmNborderWidth,                     0); n++;
            XtSetArg(args[n], XmNrightAttachment,     XmATTACH_FORM); n++;
            XtSetArg(args[n], XmNleftAttachment,      XmATTACH_FORM); n++;
            XtSetArg(args[n], XmNheight,                   i_wys[i_w]); n++;
/*            XtSetArg(args[n], XmNdepth, visualList[0].depth); n++;
            XtSetArg(args[n], XmNvisual, visualList[0].visual); n++;
            XtSetArg(args[n], XmNcolormap, cmap); n++; */
            XtSetArg(args[n], XmNlabelString, XmStringCreateSimple(a_lbl)); n++;
            draw[i_g] = XtCreateWidget("                                                  ",
                          xmLabelWidgetClass, form[i_g],
                          args, n);

            XtManageChild(form[i_g]);
            XtManageChild(draw[i_g]);

            break;

          case 3:
            n = 0;
            XtSetArg(args[n], XmNborderWidth,                     0); n++;
            XtSetArg(args[n], XmNrightAttachment,     XmATTACH_FORM); n++;
            XtSetArg(args[n], XmNleftAttachment,      XmATTACH_FORM); n++;
            labl[i_g] = XtCreateWidget("                                                  ",
                          xmLabelWidgetClass, form[i_g],
                          args, n);

            n = 0;
            temp = labl[i_g];
            XtSetArg(args[n], XmNtopAttachment,     XmATTACH_WIDGET); n++;
            XtSetArg(args[n], XmNtopWidget,                    temp); n++;
            XtSetArg(args[n], XmNbottomAttachment,    XmATTACH_FORM); n++;
            XtSetArg(args[n], XmNleftAttachment,      XmATTACH_FORM); n++;
            XtSetArg(args[n], XmNrightAttachment,     XmATTACH_FORM); n++;
            XtSetArg(args[n], XmNborderWidth,                     1); n++;
            XtSetArg(args[n], XmNwidth,                    i_wxs[i_w]); n++;
            XtSetArg(args[n], XmNheight,                   i_wys[i_w]); n++;
/*            XtSetArg(args[n], XmNdepth, visualList[0].depth); n++;
            XtSetArg(args[n], XmNvisual, visualList[0].visual); n++;
            XtSetArg(args[n], XmNcolormap, cmap); n++; */
            draw[i_g] = XtCreateWidget("draw",
                               xmDrawingAreaWidgetClass, form[i_g],
                               args, n);

            XtManageChild(form[i_g]);
            XtManageChild(labl[i_g]);
            XtManageChild(draw[i_g]);

            break;

          case 4:
            n = 0;
            XtSetArg(args[n], XmNborderWidth,                     0); n++;
            XtSetArg(args[n], XmNrightAttachment,     XmATTACH_FORM); n++;
            XtSetArg(args[n], XmNleftAttachment,      XmATTACH_FORM); n++;
            labl[i_g] = XtCreateWidget("                                                  ",
                          xmLabelWidgetClass, form[i_g],
                          args, n);
            if (i_db > 99-1) printf("labl = %d \n",labl[i_g]);

            n = 0;
            temp = labl[i_g];
            XtSetArg(args[n], XmNscrollingPolicy,              XmAUTOMATIC); n++;
            XtSetArg(args[n], XmNscrollBarDisplayPolicy,       XmAS_NEEDED); n++;
            XtSetArg(args[n], XmNbottomAttachment,           XmATTACH_FORM); n++;
            XtSetArg(args[n], XmNtopAttachment,            XmATTACH_WIDGET); n++;
            XtSetArg(args[n], XmNtopWidget,                           temp); n++;
            XtSetArg(args[n], XmNrightAttachment,        XmATTACH_POSITION); n++;
            XtSetArg(args[n], XmNrightPosition,                          100); n++;
            XtSetArg(args[n], XmNleftAttachment,             XmATTACH_FORM); n++;
            XtSetArg(args[n], XmNvisualPolicy,                  XmVARIABLE); n++;
/*            XtSetArg(args[n], XmNdepth, visualList[0].depth); n++;
            XtSetArg(args[n], XmNvisual, visualList[0].visual); n++;
            XtSetArg(args[n], XmNcolormap, cmap); n++; */
            scrl[i_g] = XtCreateWidget("scroll",
                           xmScrolledWindowWidgetClass, form[i_g],
                           args, n);
            if (i_db > 99-1) printf("scrl = %d \n",scrl[i_g]);

            n = 0;
            XtSetArg(args[n], XmNwidth,       i_wxs[i_w]); n++;
            XtSetArg(args[n], XmNheight,      i_wys[i_w]); n++;
            XtSetArg(args[n], XmNborderWidth,        1); n++;
/*            XtSetArg(args[n], XmNdepth, visualList[0].depth); n++;
            XtSetArg(args[n], XmNvisual, visualList[0].visual); n++;
            XtSetArg(args[n], XmNcolormap, cmap); n++; */
            draw[i_g] = XtCreateWidget("draw",
                               xmDrawingAreaWidgetClass, scrl[i_g],
                               args, n);
            if (i_db > 99-1) printf("draw = %d \n",draw[i_g]);

            XtVaGetValues(scrl[i_g],XmNhorizontalScrollBar,&hsb,NULL);
            XtVaGetValues(scrl[i_g],XmNverticalScrollBar,  &vsb,NULL);

            XmScrolledWindowSetAreas(scrl[i_g],hsb,vsb,draw[i_g]);
            if (i_db > 99-1) printf("Set scroll \n");

            XtManageChild(form[i_g]);
            if (i_db > 99-1) printf("Managing form \n");
            XtManageChild(labl[i_g]);
            if (i_db > 99-1) printf("Managing labl \n");
            XtManageChild(scrl[i_g]);
            if (i_db > 99-1) printf("Managing scrl \n");
            XtManageChild(draw[i_g]);
            if (i_db > 99-1) printf("Managing draw \n");

            break;

          case 5:

            break;
          case 6:
            if (i_cnt[i_w] > 0) {
              n = 0;

              XtSetArg(args[n], XmNtopAttachment,        XmATTACH_FORM); n++;
              XtSetArg(args[n], XmNbottomAttachment,     XmATTACH_FORM); n++;

/* For some reason, with both of the following statements active togeter, mdxs genrates error messages
    (BadDrawable) on the SGI and other machines, even though it appears to work properly.  To eliminate
    the error messages, I could have comment out next line.  Instead, I trap the error in myhandler and
    don't display it.  This could mask other errors, however, and should be better understood.   */

              XtSetArg(args[n], XmNrightAttachment,      XmATTACH_FORM); n++;
              XtSetArg(args[n], XmNleftAttachment,       XmATTACH_FORM); n++;

/*
              XtSetArg(args[n], XmNrightAttachment,      XmATTACH_POSITION); n++;
              XtSetArg(args[n], XmNleftAttachment,       XmATTACH_POSITION); n++;
              XtSetArg(args[n], XmNrightPosition,      90); n++;
              XtSetArg(args[n], XmNleftPosition,       10); n++;
*/

              XtSetArg(args[n], XmNrightOffset,                      3); n++;
              XtSetArg(args[n], XmNleftOffset,                       3); n++;
              XtSetArg(args[n], XmNshadowType,            XmSHADOW_OUT); n++;
              XtSetArg(args[n], XmNhighlightThickness,     0); n++;
              XtSetArg(args[n], XmNwidth,                   i_wxs[i_w]); n++;
              XtSetArg(args[n], XmNheight,                  i_wys[i_w]); n++;

              draw[i_g] = XtCreateManagedWidget(a_lbl,
                                 xmDrawnButtonWidgetClass, form[i_g],
                                 args, n);

              i_flag=i_g;   /* (i_d*100)+((i_w+1)*10)+j; */
              a_llll[i_g]=XmStringCreateSimple(a_lbl);
              XtAddCallback(draw[i_g],XmNactivateCallback,ButtonManager, (XtPointer)i_flag);
              XtAddCallback(draw[i_g],XmNexposeCallback,  ButtonManager, (XtPointer)i_flag);
              XtAddCallback(draw[i_g],XmNresizeCallback,  ButtonManager, (XtPointer)i_flag);

              n = 0;
              XtSetArg(args[n], XmNborderWidth,                     0); n++;
              XtSetArg(args[n], XmNrightAttachment,     XmATTACH_FORM); n++;
              XtSetArg(args[n], XmNleftAttachment,      XmATTACH_FORM); n++;
              XtSetArg(args[n], XmNheight,                   i_wys[i_w]); n++;
              XtSetArg(args[n], XmNlabelString, XmStringCreateSimple(a_lbl)); n++;
              labl[i_g] = XtCreateWidget("                                                  ",
                          xmLabelWidgetClass, draw[i_g],
                          args, n);

            }

            else {
              draw[i_g] = form[i_g];
            } /* endif */

            XtManageChild(form[i_g]);
            XtManageChild(draw[i_g]);

            break;

        } /* end case */

      } /* enddo */


    /*
     *  Set Main Window areas.
     */

/*    if (i_db > 5-1) printf("setting window areas\n");
        XmMainWindowSetAreas (main_dx, menu_bar, NULL, NULL, NULL, pane); */


    if (i_db > 5-1) printf("Realizing top level widget\n");
    XtRealizeWidget(toplevel);
    if (i_db > 5-1) printf("getting top window id\n");
    top[i_d] = XtWindow(toplevel);
    if (i_db > 5-1) printf("top= %d %d\n",top[i_d],i_d);
    XGetWindowAttributes(dgx,top[i_d],&xwa);
    if (i_db > 5-1) printf("got window attributes\n");
    XSetWindowColormap(dgx,top[i_d],cmap);

    i_g=i_gx[i_d][0];
    wgx[i_g] = XtWindow(menu_bar);
    if (i_db > 6-1) printf("menu wgx= %d %d\n",wgx[i_g],i_g);

    if (i_db > 6-1) printf("wgx= %d %d\n",wgx[i_g],i_g);
    XSelectInput(dgx,wgx[i_g],ExposureMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask |
           KeyPressMask | KeyReleaseMask | StructureNotifyMask);

    for (i_w=1; i_w < i_wxi[0]+1; i_w++) {
      i_g=i_gx[i_d][i_w];
      fgx[i_g] = XtWindow(form[i_g]);
      wgx[i_g] = XtWindow(draw[i_g]);   /* get the window id's for drawing */
      XSetWindowColormap(dgx,top[i_d],cmap);
      if (i_type[i_g] == 3) lgx[i_g] = XtWindow(labl[i_g]);     /* get the labels id's for drawing */
      if (i_type[i_g] == 4) lgx[i_g] = XtWindow(labl[i_g]);     /* get the labels id's for drawing */
      if (i_db > 6-1) printf("fgx= %d %d\n",fgx[i_g],i_g);
      if (i_db > 6-1) printf("wgx= %d %d\n",wgx[i_g],i_g);
      /* if (i_type[i_g] == 4) printf("lgx= %d %d\n",lgx[i_g],i_g);  */
      XSelectInput(dgx,wgx[i_g],ExposureMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask |
          KeyPressMask | KeyReleaseMask | StructureNotifyMask);
      XSelectInput(dgx,fgx[i_g],ExposureMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask |
          KeyPressMask | KeyReleaseMask | StructureNotifyMask);
      if (i_typ[i_w] < 0) {
        /* XGetWindowAttributes(dgx,wgx[i_g],xwa); */
        xswa.backing_store=Always;
        XChangeWindowAttributes(dgx,wgx[i_g],CWBackingStore,&xswa);
        if (i_db > 6-1) printf("BackingStore set to always %d, %d\n",i_d,i_w);
      } else {
        xswa.backing_store=NotUseful;
        XChangeWindowAttributes(dgx,wgx[i_g],CWBackingStore,&xswa);
        if (i_db > 6-1) printf("BackingStore set to NotUseful %d, %d\n",i_d,i_w);
      } /* Endif */
    } /* Enddo */


    /* Create the mdx icon */
    icon = XCreateBitmapFromData(dgx, top[i_d], icon_bits,
                               icon_width, icon_height);

    i_cnt[0]=0;
    for (j=0; j < 78; j++) {
      a_lbl[j] = a_labl[(j)];
      if (a_lbl[j] != 0 && a_lbl[j] != 32 ) i_cnt[0] = j+1;
    } /* enddo */
    a_lbl[i_cnt[0]] = 0;
    if (i_cnt[0] == 0) strcpy(a_lbl,"GraphX");
    XSetStandardProperties(dgx, top[i_d], a_lbl, a_lbl,
                         icon, ww1, 1, NULL);

/*    screen    = DefaultScreen(dgx);    */
    /*    gc            = DefaultGC(dgx, screen); */
/*    gc = XCreateGC(dgx,top[i_d],0,NULL); */

    i_cnt[0]=0;
    for (j=0; j < 78; j++) {
      a_clr[j] = a_lcolor[(j)];
      if (a_clr[j] != 0 && a_clr[j] != 32 ) i_cnt[0] = j+1;
    } /* enddo */
    a_clr[i_cnt[0]] = 0;

    if (i_cnt[0]=0) {
      rc = XAllocNamedColor(dgx, cmap, "white", &linec, &linec);}
    else {
      rc = XAllocNamedColor(dgx, cmap, a_clr, &linec, &linec);
    }
    if (rc == 0) {
      printf("XAllocNamedColor - failed to allocated forground color %s.  Using 'white' \n",a_clr);
      XSetForeground(dgx,gc,WhitePixel(dgx,screen)); }
    else {
      XSetForeground(dgx,gc,linec.pixel);
    }

    XSetBackground(dgx,gc,BlackPixel(dgx,screen));

    i_dmax=0;
    i_wmax=0;
    i_gmax=0;
    for (i_g=1; i_g<321; i_g++) {
      if (i_dx[i_g] != 0) i_gmax = i_g;
      if (i_dx[i_g]>i_dmax) i_dmax=i_dx[i_g];
      if (i_wx[i_g]>i_wmax) i_wmax=i_wx[i_g];
    }
    if (i_db > 2-1) printf("Graphx initialization complete\n");
    if (i_db > 6-1) printf("i_dmax= %d \n",i_dmax);
    if (i_db > 6-1) printf("i_wmax= %d \n",i_wmax);
    if (i_db > 6-1) printf("i_gmax= %d \n",i_gmax);
    if (i_db > 7-1) printf("i_wx= %d %d %d %d %d %d %d\n",i_wx[1],i_wx[2],i_wx[3],i_wx[4],i_wx[5],i_wx[6],i_wx[7]);
    if (i_db > 7-1) printf("wgx = %d %d %d %d %d %d %d\n",wgx[1],wgx[2],wgx[3],wgx[4],wgx[5],wgx[6],wgx[7]);
    return(i_d);

}

/*************************************************************************/
void free_graphics()
{
    read_events();


  XFlush(dgx);
  XCloseDisplay(dgx);
}


static void Button_quit(w, free, data)
    Widget w;
    Pixmap free;
    XmAnyCallbackStruct *data;
{
    /* Quit Graphsub */
    cmap  = XDefaultColormap(dgx, screen);
    if (top[1] != 0) XSetWindowColormap(dgx,top[1],cmap);
        XFlush(dgx);
    XFreeColormap(dgx,cmap);
    XFreePixmap(XtDisplay(w), free);
    free_graphics();
    exit(0);
}



void read_events()
{
    XFlush(dgx);
    while(XPending(dgx)) {
        XtNextEvent(&event);
        XtDispatchEvent(&event);
    }
}



/*        io routines - modified from Quyen's routines       */
#include <fcntl.h>

/*************************************************************************/
#ifdef SGI
int initdk_(i_flag, a_filename)
#endif
#ifdef SUN
int initdk_(i_flag, a_filename)
#endif
#ifdef M10
int initdk_(i_flag, a_filename)
#endif
#if defined(HP) || defined(_linux)
int initdk(i_flag, a_filename)
#endif

int *i_flag; char *a_filename;
{  int i;
   int i_stat;
   for(i=0; i < strlen(a_filename); i++)
     if( *(a_filename+i) == ' ') *(a_filename+i) = '\0' ;

   if (i_flag == 0) {
     if((i_stat=open(a_filename,O_RDWR)) < 0){
       if( (i_stat = open(a_filename,O_RDONLY)) < 0) {
         if( (i_stat = open(a_filename,O_CREAT|O_RDWR,0666)) < 0) {
           printf(" Cannot open the filename: %s\n",a_filename);
         }
       } else {
         printf(" Open filename %s as READ ONLY\n",a_filename);
       } /* end if */
     } else {
       printf(" Open filename %s as RDWR \n",a_filename);
     } /* endif */

     if( i_stat < 0 ) i_stat = open(a_filename,O_CREAT|O_RDWR,0666);
     if(i_stat == -1)printf(" Cannot open the filename: %s\n",a_filename);
   } else {
     if((i_stat=open(a_filename,O_RDONLY)) < 0){
       printf(" Cannot open the filename: %s\n",a_filename);
     }
   }
   return(i_stat);
}

/*************************************************************************/
#ifdef SGI
int iowrit_(i_chan, b_buff, bytes)
#endif
#ifdef SUN
int iowrit_(i_chan, b_buff, bytes)
#endif
#ifdef M10
int iowrit_(i_chan, b_buff, bytes)
#endif
#if defined(HP) || defined(_linux)
int iowrit(i_chan, b_buff, bytes)
#endif

int *i_chan, *bytes;
char *b_buff;
{
   int nbytes;
   nbytes = write(*i_chan, b_buff, *bytes);
   if(nbytes != *bytes) fprintf(stderr,
       " ** ERROR **: only %d bytes transfered out of %d bytes\n",
       nbytes, *bytes);
   return(nbytes);
}

/*************************************************************************/
#ifdef SGI
int ioread_(i_chan, b_buff, bytes)
#endif
#ifdef SUN
int ioread_(i_chan, b_buff, bytes)
#endif
#ifdef M10
int ioread_(i_chan, b_buff, bytes)
#endif
#if defined(HP) || defined(_linux)
int ioread(i_chan, b_buff, bytes)
#endif

int *i_chan, *bytes ;
char *b_buff;
{
   int nbytes;
   nbytes = read(*i_chan, b_buff, *bytes);
/*   if(nbytes != *bytes) fprintf(stderr,
     " ** ERROR **: only %d bytes are read out of %d requested\n",
     nbytes, *bytes); */
   return(nbytes);
}


/*************************************************************************/
#ifdef SGI
int ioseek_(i_chan, lbyte,i_flag)
#endif
#ifdef SUN
int ioseek_(i_chan, lbyte,i_flag)
#endif
#ifdef M10
int ioseek_(i_chan, lbyte,i_flag)
#endif
#if defined(HP) || defined(_linux)
int ioseek(i_chan, lbyte,i_flag)
#endif

int *i_chan, *i_flag, *lbyte;
{
   int nloc;
   off_t ibytes;
   ibytes = *lbyte ;
   if(*i_flag == 0) {
     nloc = lseek(*i_chan, ibytes, SEEK_SET); }
   else if (*i_flag == 1) {
     nloc = lseek(*i_chan, ibytes, SEEK_CUR); }
   else if (*i_flag == 2) {
     nloc = lseek(*i_chan, ibytes, SEEK_END); }
   else {
      nloc = lseek(*i_chan, ibytes, SEEK_CUR);
   } /* endif */
   return(nloc);
}

#ifdef IO64
/*************************************************************************/
#ifdef SGI
long long ioseek64_(i_chan, lbyte,i_flag)
#endif
#ifdef SUN
long long ioseek64_(i_chan, lbyte,i_flag)
#endif
#ifdef M10
long long ioseek64_(i_chan, lbyte,i_flag)
#endif
#if defined(HP) || defined(_linux)
long long ioseek64(i_chan, lbyte,i_flag)
#endif
int *i_chan, *i_flag;
long long *lbyte;
{
   long long nloc;
   off_t ibytes;
   ibytes = *lbyte;
   if(*i_flag == 0) {
     nloc = lseek(*i_chan, ibytes, SEEK_SET); }
   else if (*i_flag == 1) {
     nloc = lseek(*i_chan, ibytes, SEEK_CUR); }
   else if (*i_flag == 2) {
     nloc = lseek(*i_chan, ibytes, SEEK_END); }
   else {
      nloc = lseek(*i_chan, ibytes, SEEK_CUR);
   } /* endif */
   return(nloc);
}


#endif


/*************************************************************************/
#ifdef SGI
int closedk_(i_chan)
#endif
#ifdef SUN
int closedk_(i_chan)
#endif
#ifdef M10
int closedk_(i_chan)
#endif
#if defined(HP) || defined(_linux)
int closedk(i_chan)
#endif

int *i_chan;
{
   return(close(*i_chan));
}
