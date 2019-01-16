#ifndef alosglobals_h
#define alosglobals_h

struct GLOBALS
{
        int quad_pol;
        int ALOS_format;
        int dopp;
        int force_slope;

        double forced_slope;
        double tbias;

        char *imagefilename;
};

#endif //alosglobals_h
