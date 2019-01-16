from __future__ import print_function
from iscesys.Dumpers.XmlDumper import XmlDumper


## The factory can make these dumpers:
DUMPERS = {'xml' : XmlDumper}

def createFileDumper(type_):
    """dumper = createFileDumper(type_)
    
    str(type_) must be in DUMPERS = {'type_' : Dumper}

    dumper = Dumper() is the instance of the factory's class.
    """
    try:
         cls = DUMPERS[str(type_).lower()]
    except KeyError:
        raise TypeError(
            'Error. The type %s is an unrecognized dumper format.' %
            str(type_)
            )

    return cls()

