// -*- Makefile -*-
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                             Eric M Gurrola
//                        NASA Jet Propulsion Laboratory
//                      California Institute of Technology
//                      (C) 2004-2005  All Rights Reserved
//
// <LicenseText>
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

/**
  @file   FortChar.h
  @brief  A class to handle passing strings from Fortran to C++
  @author Eric M. Gurrola
*/

#ifndef _EMG_FORTCHAR_H_
#define _EMG_FORTCHAR_H_

#include <cstring>
#include <string>

struct FortCharError
{
  std::string message;
  FortCharError(std::string err){ message = err; }
};

class FortChar
{
  public:

    // Constructor: n = size of Fortran string buffer
    // The Fortran buffer is public so that it can be
    // passed directly to Fortran.  The user must ensure
    // that n is large enough to hold the string.

    FortChar(){ _flen = 0; _clen = 0; }

    FortChar(const int n) 
    { 
      _flen = 0;         // Necessary to set _flen=0 initially so allocate will not 
                         // try to delete previously allocated memory.
      _clen = 0;         // clen=0 until explicit call to set_cchar
      allocate(n);       // Allocate fchar and set _flen to the correct value.
      clear();           // Fill fchar with blanks
    }

    FortChar(char* fstr, const int n) 
    { 
      _flen = 0;                // Necessary to set _flen=0 initially so allocate will not 
                                // try to delete previously allocated memory.
      _clen = 0;                // clen=0 until explicit call to set_cchar
      allocate(n);              // Allocate fchar and set _flen to the correct value.
      clear();                  // Fill fchar with blanks
      for( int i=0; i<n; i++ )
      {
        fchar[i] = fstr[i];
      }
    }

    // Public member
    char* fchar;        // This is public so it can be given to Fortran directly

    // Public methods
    char* cchar()
    {
      if( _clen == 0 )
      {
        set_cchar();
      }
      return _cchar; 
    }
    int   flen() { return _flen; }
    int   clen() { return _clen; }
    std::string cstring()
    {
      if( _clen == 0 )
      {
        set_cchar();
      }
      _cstring = _cchar;      // Use std::string "=" to convert char* to string
      return _cstring; 
    }

    void set_cchar()
    {
      if( _flen > 0 )
      {
        int cl = _flen-1;
        while( fchar[cl] == ' ' )        // Locate the last non-blank character in the Fortran string
        {
            cl--;
        }
        cl += 2;                         // Length to last non-blank character plus one for null
        _clearc();                       // Clear the cchar buffer
        _cchar = new char[cl];           // Create appropriate amount of memory to hold cchar
        for( int i=0; i<cl-1; i++ )      // Carefully copy fchar to _cchar
        {
            _cchar[i] = fchar[i];
        }
        _cchar[cl-1] = '\0';             // Null terminate the C-string at the last blank
        _clen = cl + 1;
      }
      else
      {
        throw FortCharError("Error in FortChar.set_cchar: flen = 0");
      }
    }

    void allocate(int n)
    {
      if( _flen != 0 ) delete fchar;    // Delete fchar if previously allocated
      if( n > 0 ) 
      {
        fchar = new char[n];             // Allocate memory if not 0 length
      }
      else
      {
        throw FortCharError("Error in FortChar.allocate: n = 0");
      }
      _flen = n;                        // Set length
      clear();                          // Fill fchar with blanks, and delete cchar memory
    }

    void clear()
    {
      for( int i=0; i<_flen; i++ ) fchar[i] = ' ';  // Fill fchar with blanks
      _clearc();                                    // Delete the cchar memory
    }

    void release()
    {
      if( _flen != 0 ) delete fchar;                // Delete the fchar memory
      _flen = 0;                                    // Zero the flen
      _clearc();                                    // Delete the cchar memory
    }

    // Destructor
    ~FortChar() 
    { 
      if( _flen != 0 ) delete fchar; 
      if( _clen != 0 ) delete _cchar; 
    }

 private:

    char* _cchar;
    std::string _cstring;
    int   _clen;
    int   _flen;

    void _clearc()
    {
      if( _clen != 0 )
      {
        delete _cchar;
        _clen = 0;
      }
    }

};

#endif
