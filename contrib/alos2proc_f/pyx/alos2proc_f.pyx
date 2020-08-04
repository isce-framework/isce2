#include "alos2proc_f.h"

cdef extern from "alos2proc_f.h":
    void c_fitoff(const char *, const char *, const double *, const double *, const int *);
    void c_rect(const char *, const char *, const int *, const int *, const int *, const int *,
                const double *, const double *, const double *, const double *, const double *, const double *,
                const char *, const char *);
    void c_rect_with_looks(const char *, const char *, const int *, const int *, const int *, const int *,
                const double *, const double *, const double *, const double *, const double *, const double *,
                const int *, const int *, const int *, const int *,
                const char *, const char *);


def fitoff(str infile, str outfile, double nsig, double maxrms, int minpoint):
    c_fitoff(infile.encode(), outfile.encode(), &nsig, &maxrms,  &minpoint)
    return

def rect(str infile, str outfile, int ndac, int nddn, int nrac, int nrdn,
        double a, double b, double c, double d, double e, double f,
        str filetype, str intstyle):
    c_rect(infile.encode(), outfile.encode(), &ndac, &nddn, &nrac, &nrdn,
            &a, &b, &c, &d, &e, &f, filetype.encode(), intstyle.encode())
    return

def rect_with_looks(str infile, str outfile, int ndac, int nddn, int nrac, int nrdn,
        double a, double b, double c, double d, double e, double f,
        int lac, int ldn, int lac0, int ldn0,
        str filetype, str intstyle):
    c_rect_with_looks(infile.encode(), outfile.encode(), &ndac, &nddn, &nrac, &nrdn,
            &a, &b, &c, &d, &e, &f, &lac, &ldn, &lac0, &ldn0,
            filetype.encode(), intstyle.encode())
    return
