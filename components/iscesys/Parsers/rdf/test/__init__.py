#!/usr/bin/env python3
"""Test makes new.rdf, which shold be the same as old.rdf"""
## \namespace rdf.test A brief test suite
import rdf


SRC = "rdf.txt"
DST = "new.rdf"


## rdf.parse(SRC) >> DST
def main():
    """RDF...(SRC)>>DST"""
    data = rdf.parse(SRC)
    dst = data >> DST
    return None

if __name__ == '__main__':
    main()
