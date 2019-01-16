#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2010 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Eric Gurrola
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




from contextlib import contextmanager



__all__ = ("createCoordinate", 
           "createImage", 
           "createRawImage",
           "createRawIQImage", 
           "createStreamImage", 
           "createSlcImage", 
           "createRgImage", 
           "createIntImage", 
           "createAmpImage", 
           "createOffsetImage", 
           "createDemImage",
           "contextIntImage",
           "contextOffsetImage",
           "contextRawImage",
           "contextStreamImage",
           "contextSlcImage",
           "contextRgImage",
           "contextAmpImage",
           "contextOffsetImage",
           "contextUnwImage",
           "contextAnyImage")



## Decorator to make image factroies into contextmanagers
def image_context(factory):
    @contextmanager
    def context_factory(filename=None, width=None, accessMode=None, create=True):
        """ %s image factory. Keywords arguments:

        kwarg            action
        -----------------------------------
        filename         setFilename
        width            setWidth
        accessMode       setFilename
        [create=True]    --call Image.createImage() -if all key
                           words are set
        """
        ## ONE: Build the context up:
        result = factory()
        if filename is not None:
            result.setFilename(filename)
        if width is not None:
            result.setWidth(width)
        if accessMode is not None:
            result.setAccessMode(accessMode)
        if width and filename and accessMode and create:
            result.createImage()

        yield result
        ## TWO: Tear it back down.
        result.finalizeImage()
        pass
    ## prepare context manager's docstring
    context_factory.__doc__ =  context_factory.__doc__  % (factory.__name__)
    return context_factory


def createCoordinate(name=''):
    from .Image import ImageCoordinate
    inst = ImageCoordinate(name=name)
    inst.configure()
    return inst

def createImage(name=''):
    from .Image import Image
    inst = Image(name=name)
    return inst

def createRawImage(name=''):
    from .RawImage import RawImage
    inst = RawImage(name=name)
    return inst

def createRawIQImage(name=''):
    from .RawIQImage import RawIQImage
    inst = RawIQImage(name=name)
    return inst
def createStreamImage(name=''):
    from .StreamImage import StreamImage
    inst = StreamImage(name=name)
    return inst

def createSlcImage(name=''):
    from .SlcImage import SlcImage
    inst = SlcImage(name=name)
    return inst

def createRgImage(name=''):
    from .RgImage import RgImage
    inst = RgImage(name=name)
    return inst

def createIntImage(name=''):
    from .IntImage import IntImage
    inst = IntImage(name=name)
    return inst

def createAmpImage(name=''):
    from .AmpImage import AmpImage
    inst = AmpImage(name=name)
    return inst

def createOffsetImage(name=''):
    from .OffsetImage import OffsetImage
    inst = OffsetImage(name=name)
    return inst

def createDemImage(name=''):
    from .DemImage import DemImage
    inst = DemImage(name=name)
    return inst

def createUnwImage(name=''):
    from .UnwImage import UnwImage
    inst = UnwImage(name=name)
    return inst

def getFactoriesInfo():
    return  {'ImageCoordinate':
                     {
                     'factory':'createCoordinate'                     
                     },
             'Image':
                     {
                     'factory':'createImage'                     
                     },
             'RawImage':
                     {
                     'factory':'createRawImage'                     
                     },
             'RawIQImage':
                     {
                     'factory':'createRawIQImage'                     
                     },
             'StreamImage':
                     {
                     'factory':'createStreamImage'                     
                     },
             'SlcImage':
                     {
                     'factory':'createSlcImage'                     
                     },
             'RgImage':
                     {
                     'factory':'createRgImage'                     
                     },
             'IntImage':
                     {
                     'factory':'createIntImage'                     
                     },
             'AmpImage':
                     {
                     'factory':'createAmpImage'                     
                     },
             'OffsetImage':
                     {
                     'factory':'createOffsetImage'                     
                     },
             'DemImage':
                     {
                     'factory':'createDemImage'                     
                     },
             'UnwImage':
                     {
                     'factory':'createUnwImage'                     
                     }
              }
## This is the IntImage factory's contect manager
contextIntImage = image_context(createIntImage)
contextRawImage = image_context(createRawImage)
contextStreamImage = image_context(createStreamImage)
contextSlcImage = image_context(createSlcImage)
contextRgImage = image_context(createRgImage)
contextAmpImage = image_context(createAmpImage)
contextOffsetImage = image_context(createOffsetImage)
contextDemImage = image_context(createDemImage)
contextUnwImage = image_context(createUnwImage)

## This manger takes a cls or instance, calls it factory in a context manager
@contextmanager
def contextAnyImage(cls,
               filename=None, width=None, accessMode=None, create=True):
    """imageFactory(cls,
                    filename=None, width=None, accessMode=None, create=True):

       cls:     as class OR instance of an Image subclass.

       returns a context manager that creates the class in a context.
       Keyword arguments are passed to the context manager, and are 
       use to build the class up.
       """
    if not isinstance(cls, type):
        cls = cls.__class__
        
    cls_name = cls.__name__

    hash_table = {
 'RawImage' : createRawImage,
 'StreamImage' : createStreamImage,
 'SlcImage' : createSlcImage,
 'RgImage' : createRgImage,
 'IntImage' : createIntImage,
 'AmpImage' : createAmpImage,
 'OffsetImage' : createOffsetImage,
 'DemImage' : createDemImage,
 'UnwImage' : createUnwImage
 }
    try:
        factory =  hash_table[cls_name]
    except KeyError:
        raise TypeError('Cannot find factory for: %s' % cls_name)

    ## ONE: Build the context up:
    result = factory()
    if filename is not None:
        result.setFilename(filename)
    if width is not None:
        result.setWidth(width)
    if accessMode is not None:
        result.setAccessMode(accessMode)
    if width and filename and accessMode and create:
        result.createImage()

    yield result
    try:
        result.finalizeImage()
    except TypeError:
        print("Image was not initialized, so finalizeImage failed")
    pass
