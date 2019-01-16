from iscesys.Parsers.XmlParser import XmlParser
from iscesys.Parsers.RscParser import RscParser    

## A table of recognozed parser types
PARSER = {'xml' : XmlParser,
          'rsc' : RscParser}

def createFileParser(type_):
    """get Parser class for 'xml' or 'rsc' input type."""
    try:
        cls = PARSER[str(type_).lower()]
    except KeyError:
        raise TypeError(
            'Error. The type %s is an unrecognized parser format.' % str(type_)
            )
    return cls()

