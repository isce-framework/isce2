#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2014 California Institute of Technology. ALL RIGHTS RESERVED.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# United States Government Sponsorship acknowledged. This software is subject to
# U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
# (No [Export] License Required except when exporting to an embargoed country,
# end user, or in support of a prohibited end use). By downloading this software,
# the user agrees to comply with all applicable U.S. export laws and regulations.
# The user has the responsibility to obtain export licenses, or other export
# authority as may be required before exporting this software to any 'EAR99'
# embargoed foreign country or citizen of those countries.
#
# Author: Eric Belz
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




"""data-->RDF is THE RDF OBJECT"""
##\namespace rdf.data.files Usable data object for files.
import sys
import collections    

#pylint:disable=E1101
try:
    DICT = collections.OrderedDict
except AttributeError:
    print >> sys.stderr, "Not 2.7: using (UnOrdered) dict for rdf mapping" 
    DICT = dict


## An RDF Mothership: A fully interpresed RDF File.
class RDF(object):
    """RDF object is made from the read_rdf helper function:

    >>>data = rdf_reader('rdf.txt')

    It is an associate array, so like a dict:

    >>>data[key]

    returns a value- as a float or string-or whatever "eval" returns.

    All the standard OrderDict methods can be used, and will return the
    full RDFField object that represent the value, units, dimension....
    comments.


    You may __setitem__:
    >>>rdf[key] = value #equivalent to
    >>>rdf[key] = RDFField(value)
    
    That is, it transforms assignee into an RDFField for you.
    """
    ## Make an instance from DICT argument
    ## \param dict_ is an rdf-enough dictionary
    ## \return RDF instance
    @classmethod
    def fromdict(cls, dict_):
        """instantiate from a DICT"""
        result = cls()
        for key, value in dict_.items():
            result[key.strip()] = value
        return result

    ## Make it from keyword arguments
    ## \param *args is an rdf-enough arguments (like dict)
    ## \param **dict_ is an rdf-enough dictionary
    ## \return RDF instance
    @classmethod
    def fromstar(cls, *args, **dict_):
        """instantiate from *args **dict_)"""
        # todo: (dict(*args) + dict_)
        rdf_ = cls()
        for pair in args:
            key, value = pair
            rdf_[str(key)] = value
            pass
        return rdf_ + cls.fromdict(dict_)

    ## Instaniate from a file
    ## \param src RDF file name
    ## \retval RDF an rdf instance
    @staticmethod
    def fromfile(src):
        """src -> RDF"""
        from iscesys.Parsers.rdf import rdfparse
        return rdfparse(src)
    
    ## \verb{ rdf << src } Sucks up an new rdf file.
    ## \param src RDF file name
    ## \returns src + RDF
    def __lshift__(self, src):
        return self + self.__class__.fromfile(src)

    def __ilshift__(self, src):
        return self<<src
    
    ## Don't use init-- it's internal.
    ## \param *args Internal constructor takes *args
    def __init__(self, *args):
        """RDF(*args)
        """
        ## The data are an associative array-or the rdf spec is worthless.
        self.data = DICT(args)

    ## Get attr from ::DICT If needed
    ## \param attr Attritubute
    ## \retval self.attr OR
    ## \retval self.data.attr If needed
    def __getattr__(self, attr):
        try:
            result = object.__getattribute__(self, attr)
        except AttributeError:
            result = getattr(self.data, attr)
        return result
    
    @property
    def rdf(self):
        return self

    ## Return value, vs "get" returns RDFField
    def __getitem__(self, key):
        return self.data[key].value
    
    ## Set the item to an RDFField
    ## \param key an RDF key
    ## \param value any kind of value
    ## \par Side Effects: 
    ##  assigns key in RDF.data with RDFField value
    ## \returns None
    def __setitem__(self, key, value):
        from rdf.data.entries import RDFField
        # Service If: convert to RDFField so you don't have to
        if not isinstance(value, RDFField):
            value = RDFField(value)
        self.data[key] = value

    ## Delete key, field
    ## \param key an RDF key
    ## \par Side Effects: 
    ## deletes key from RDF.data
    ## \returns None
    def __delitem__(self, key):
        self.data.__delitem__(key)

    ## iteritems()
    def __iter__(self):
        return iter(self.data.iteritems())

    ## Access as method or property (this is really for clients)
    def __call__(self):
        """self()->self so that x.rdf()()-->x.rdf()->x.rdf"""
        return self

    def record(self, key):
        """convery self[key] to an RDFRecord"""
        from iscesys.Parsers.rdf.data.entries import RDFRecord
        field = self.get(key)
        return RDFRecord(key, field)

    def records(self):
        """Get all records from record()"""
        return map(self.record, self.keys())

    ## Get maximum index (column) of OPERATOR in file's strings
    def _max_index(self):
        return max(map(int, self.records()))

    def __str__(self):
        from iscesys.Parsers.rdf.data.entries import RDFRecord
        max_index = self._max_index()
        ## now insert space...
        final_result = []
        for record in self.records():
            line = record.__str__(index=max_index)
            final_result.append(line + '\n')
        return "".join(final_result)


    ## rep the data attribute
    def __repr__(self):
        return repr(self.data)

    def __len__(self):
        return len(self.data)
    
    ## rdf >> dst \\n
    ## see tofile().
    def __rshift__(self, dst):
        return self.tofile(dst)

    ## Add is concatenation, and is not communative
    ## \param other An RDF instance
    ## \retval result Is another RDF instance
    def __add__(self, other):
        result = RDF()
        for key, field in self.items(): 
            result[key] = field
        for key, field in other.items():
            if result.has_key(key):   #Guard
                print ("WARNING: overwritting:", key)
            result[key] = field
        return result
    
    ## Incremental add an RDFRecord - very non polymorphic implementation...
    ## This is not pythonic and need fixing-the desibn flaw lies in
    ## either parse.py or the RDF spc itself
    ## \param other Could be RDF, tuple, list, other...
    def __iadd__(self, other):
        """You can increment with:

        Another RDF:
        OR RecursiveRecord or RDF

        this is in development
        """
        if other:
            if isinstance(other, self.__class__):
                self = self + other
            elif isinstance(other, tuple):
                self[other[0]] = other[1]
            elif isinstance(other, list):
                for item in other:
                    self += item
            else:
                try:
                    self = self + other.rdf()
                except AttributeError:
                    raise TypeError(
                        "Can't add type:" + other.__class__.__name__
                        )
        return self

    ## Write to file, with some formatting
    ## \param dst file name (writable)
    ## \par Side Effects: 
    ##  writes dst
    ## \returns
    def tofile(self, dst):
        """write data to a file"""
        with open(dst, 'w') as fdst:
            fdst.write(str(self))
        ## return dst to make idempotent
        return dst
    
    ## Convert to a standard (key, value) ::DICT
    def todict(self):
        """Convert to a normal dict()"""
        result = {}
        for key, field in self.iteritems():
            result.update( {key : field()} )
        return result
