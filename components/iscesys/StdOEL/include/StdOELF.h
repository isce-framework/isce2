//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                                  Giangi Sacco
//                        NASA Jet Propulsion Laboratory
//                      California Institute of Technology
//                        (C) 2009  All Rights Reserved
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#ifndef StdOELF_h
#define StdOELF_h
#include "StdOELFFortTrans.h"
#include <stdint.h>
/**
  * @file
  * This is a C interface that allows fortran code to call public methods of a StdOE object.

  * The functions name in fortran will be the same except for the suffix "_f" that needs to be removed.
  * Moreover each function "func(args)" will be invoked from fortran using the syntax: call func(args).
  * The correspondence between C and fortran data types is:
  *     - char * <--> character*X (X integer number).
**/
extern "C"
{


        /**
            * Writes the string message on the standard output device. From fortran the function is called providing only the first two parameters. The last is implicit.
        @param stdOEL pointer of the StdOEL object.
            @param  message  character array containing the message to be output.

        **/
        void write_out_f(uint64_t * stdOEL,char * message, long int len);
        /**
            * Writes the string message on the standard error device. From fortran the function is called providing only the first two parameters. The last is implicit.
        @param stdOEL pointer of the StdOEL object.
            @param  message  character array containing the message to be output.

        **/
        void write_err_f(uint64_t * stdOEL,char * message, long int len);
        /**
            * Writes the string message on the standard log device. From fortran the function is called providing only the first two parameters. The last is implicit.
        @param stdOEL pointer of the StdOEL object.
            @param  message  character array containing the message to be output.

        **/
        void write_log_f(uint64_t * stdOEL,char * message, long int len);


}
#endif //StdOELF_h
