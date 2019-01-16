"""
Use "from py2to3 import *" to get the Python3 version of range, map, zip,
ascii, filter, and hex.  In Python3 range is equivalent to Python2.7 xrange,
an iterator rather than a list.  Similarly map, zip, and filter generate
iterators in Python 3 rather than lists.  The function ascii returns the
ascii version of a string and hex and oct return the hex, and oct 
representations of an integer

It is also necessary to use the following import from __future__ to get the
Python3 version of print (a function), import, unicode_literals (different in
Python3 from a byte string), and division (1/2 = 0.5, 1//2 = 0).

from __future__ import (print_function, absolute_import,
                        unicode_literals, division)
"""

try:
   range = xrange
   from future_builtins import *
except:
   pass



