#ifndef __ALOS2_F_H__
#define __ALOS2_F_H__

// define the C binding interfaces
#ifdef __cplusplus
extern "C" {
#endif

void c_fitoff(const char *, const char *, const double *, const double *, const int *);
void c_rect(const char *, const char *, const int *, const int *, const int *, const int *,
                const double *, const double *, const double *, const double *, const double *, const double *,
                const char *, const char *);
void c_rect_with_looks(const char *, const char *, const int *, const int *, const int *, const int *,
                const double *, const double *, const double *, const double *, const double *, const double *,
                const int *, const int *, const int *, const int *,
                const char *, const char *);
#ifdef __cplusplus
}
#endif

#endif // __ALOS2_F_H__
