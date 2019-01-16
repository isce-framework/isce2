//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                                  Giangi Sacco
//                        NASA Jet Propulsion Laboratory
//                      California Institute of Technology
//                        (C) 2009  All Rights Reserved
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#ifndef StdOEFFortTrans_h
#define StdOEFFortTrans_h

        #if defined(NEEDS_F77_TRANSLATION)

                #if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)
                        #define getStdErr_f getstderr_
                        #define getStdOut_f getstdout_
                        #define setStdErrFile_f setstderrfile_
                        #define setStdErr_f setstderr_
                        #define setStdOut_f setstdout_
                        #define setStdOutFileTag_f setstdoutfiletag_
                        #define setStdErrFileTag_f setstderrfiletag_
                        #define setStdLogFileTag_f setstdlogfiletag_
                        #define setStdOutFile_f setstdoutfile_
                        #define setStdLogFile_f setstdlogfile_
                        #define writeStdErr_f writestderr_
                        #define writeStdFile_f writestdfile_
                        #define writeStdOut_f writestdout_
                        #define writeStdLog_f writestdlog_
                        #define writeStd_f writestd_
                #else
                        #error Unknown traslation for FORTRAN external symbols
                #endif

        #endif

#endif //StdOEFFortTrans_h
