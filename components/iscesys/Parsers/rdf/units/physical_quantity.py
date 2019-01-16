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




## \namespace rdf.units.physical_quantity Classes for Physical Quantities
import abc
import operator
import sys

## use lower case IFF (prolly deprecated)
_LOWER = False
## Abbreviation case converter
_case = operator.methodcaller("lower") if _LOWER else lambda dum: dum

## This class is a class decorator factory that makes an instance
## of the class with a decorated prefix
class Prefix(object):
    """prefix = Prefix("symbol", exponent)

    INPUT:
           symbol   The prefix string symbol, e.g: "M" for mega
           exponent The exponent for the factor... 6   for 10**6

    OUTPUT:
            prefix   A class decorator that creates a new instance of
                     the decorated class (that must be a sub-class of Unit)
                     See Unit.__doc__ for why that works.

    @prefix
    class Dimension(Unit)
          si_unit = <what ever measure dimension>


    Note: Of course, you can stack them up.
    """

    ## Without a self.base, this class is no good
    __metaclass__ = abc.ABCMeta

    ## Sub's need a base to define what their prefixing means
    @abc.abstractproperty
    def base(self):
        pass

    @abc.abstractmethod
    def cast(self):
        return self.factor

    ## Construct with a symbol and in exponent
    ## \param symbol A string symbol that IS the abbreviation
    ## \param exponent sets the scale factor = base ** exponent
    def __init__(self, symbol, exponent):
        ## The prefix's official symbol
        self.symbol = str(symbol)
        ## \f$ f = B^x \f$
        self.factor = self.base ** exponent
        return None

    ## str(prefix) is the prefix's symbol.
    def __str__(self):
        return self.symbol

    ## Class decorator:
    ## \param cls A Unit sub-class
    ## \par Side Effects:
    ## instaniate deocrated intance and loh into Glossary
    ## \retval cls Class decorators return classes.
    def __call__(self, cls):
        """prefix(cls)-->cls'
        with SIDE EFFECTS"""
        # instansiate class with deocrated instance
        cls(str(self) + cls.si_unit, self.cast()(self))
        return cls

## <a href="http://en.wikipedia.org/wiki/Metric_prefix">Metric</a> Prefix.
class MetricPrefix(Prefix):
    """Prefix based on 10"""

    ## Metric is a Perfect 10
    base = 10

    def __float__(self):
        return float(self.factor)

    ## cast to float
    def cast(self):
        return float


## <a href="http://en.wikipedia.org/wiki/Binary_prefix">Binary</a> Prefix
## Note: limits/dIfferences of/between JEDEC and IEC
class BinaryPrefix(Prefix):
    """Prefix based on 1024"""

    ## \f$ 2^{10} \f$
    base = 1024

    ## cast to ling
    def __int__(self):
        return int(self.factor)

    def cast(self):
        return int


## <a href="http://en.wikipedia.org/wiki/Yotta-">\f$10^{24}\f$</a>
yotta = MetricPrefix('Z', 24)
## <a href="http://en.wikipedia.org/wiki/Zetta-">\f$10^{21}\f$</a>
zetta = MetricPrefix('Z', 21)
## <a href="http://en.wikipedia.org/wiki/Exa-">\f$10^{18}\f$</a>
exa = MetricPrefix('E', 18)
## <a href="http://en.wikipedia.org/wiki/Peta-">\f$10^{15}\f$</a>
peta = MetricPrefix('P', 15)
## <a href="http://en.wikipedia.org/wiki/Tera-">\f$10^{12}\f$</a>
tera = MetricPrefix('T', 12)
## <a href="http://en.wikipedia.org/wiki/Giga-">\f$10^9\f$</a>
giga = MetricPrefix('G', 9)
## <a href="http://en.wikipedia.org/wiki/Mega-">\f$10^6\f$</a>
mega = MetricPrefix('M', 6)
## <a href="http://en.wikipedia.org/wiki/Kilo-">\f$10^3\f$</a>
kilo = MetricPrefix('k', 3)
## <a href="http://en.wikipedia.org/wiki/Hecto-">\f$10^2\f$</a>
hecto = MetricPrefix('h', 2)
## <a href="http://en.wikipedia.org/wiki/Deca-">\f$10^1\f$</a>
deca = MetricPrefix('da', 1)
## Trival (but it does create an instance and put it in Unit.Glossary
base = MetricPrefix('', 0)
## <a href="http://en.wikipedia.org/wiki/Deci-">\f$10^{-1}\f$</a>
deci = MetricPrefix('d', -1)
## <a href="http://en.wikipedia.org/wiki/Centi-">\f$10^{-2}\f$</a>
centi = MetricPrefix('c', -2)
## <a href="http://en.wikipedia.org/wiki/Milli-">\f$10^{-3}\f$</a>
milli = MetricPrefix('m', -3)
## <a href="http://en.wikipedia.org/wiki/Micro-">\f$10^{-6}\f$</a>\n
## (NB: \f$"u"\f$ is used instead of \f$"\mu"\f$ for typographical reasons)
micro = MetricPrefix('u', -6)
## <a href="http://en.wikipedia.org/wiki/Nano-">\f$10^{-9}\f$</a>
nano = MetricPrefix('n', -9)
## <a href="http://en.wikipedia.org/wiki/Pico-">\f$10^{-12}\f$</a>
pico = MetricPrefix('p', -12)
## <a href="http://en.wikipedia.org/wiki/Femto-">\f$10^{-15}\f$</a>
femto = MetricPrefix('f', -15)
## <a href="http://en.wikipedia.org/wiki/Atto-">\f$10^{-18}\f$</a>
atto= MetricPrefix('a', -18)
## <a href="http://en.wikipedia.org/wiki/Zepto-">\f$10^{-21}\f$</a>
zepto = MetricPrefix('z', -21)
## <a href="http://en.wikipedia.org/wiki/Yocto-">\f$10^{-24}\f$</a>
yocto = MetricPrefix('y', -24)


## Trival (integer measurement)
base2 = BinaryPrefix('', 0)
## \f$ 2^{10} \f$, JEDEC
kilo2 = BinaryPrefix('k', 1)
## \f$ (2^{10})^2 \f$, JEDEC
mega2 = BinaryPrefix('M', 2)
## \f$ (2^{10})^3 \f$, JEDEC
giga2 = BinaryPrefix('G', 3)

## \f$ 2^{10} \f$, IEC
kibi = BinaryPrefix('Ki', 1)
## \f$ (2^{10})^2 \f$, IEC
mebi = BinaryPrefix('Mi', 2)
## \f$ (2^{10})^3 \f$, IEC
gibi = BinaryPrefix('Gi', 3)
## \f$ (2^{10})^4 \f$, IEC
tebi = BinaryPrefix('Ti', 4)
## \f$ (2^{10})^5 \f$, IEC
pebi = BinaryPrefix('Pi', 5)
## \f$ (2^{10})^6 \f$, IEC
exbi = BinaryPrefix('Ei', 6)
## \f$ (2^{10})^7 \f$, IEC
zebi = BinaryPrefix('Zi', 7)
## \f$ (2^{10})^8 \f$, IEC
yebi = BinaryPrefix('Yi', 8)


## The Unit class memoizes its instances
class Unit(str):
    """Unit(value, multiplier=1, adder=0 [,si_unit=None])

    On Units and Prefixes:

    Instances of the Prefix class deocrate Unit classes- and as such
    create instances of:

    Sym = <Prefix> + <Unit> when the Unit subclass is created (at import).

    That instance is, of course, also a <str> and is memoized in

    Unit.Glossary

    dictionary as:

    {Sym : Sym}

    At fist, that looks odd. The point is to do a hash-table search (not a list
    search) in the Glossary with "Sym" as a key-- here "Sym" is the ordinary
    string supplied by the RDF file's (unit) field.

    the resulting Value converts units to <Unit>'s si_unit with <unit>.factor
    as a scaling.

    Hence, of you're talking float(x) "km", you get:

    Glossary["km"](x) --> 1000*x, "m"
    """

    __metaclass__ = abc.ABCMeta

    ## When ever a unit is instantiated, it goes into here.
    Glossary = {}

    ## This is the target unit (SI or not) for all things in Unit subclass.
    @abc.abstractproperty
    def si_unit(self):
        pass


    ## The conversion function defined: \n
    ## \f$ y = m(x + b) \f$ \n
    # \param m is the multiplier for the conversion
    # \param b is the adder (applied 1st).
    # \par Side Effects:
    #     Instance is _memoized()'d.
    # \returns A string that can be looked up with a str in a hash-table
    #  and can then do unit conversion.
    def __new__(cls, string="", multiplier=1, adder=0, si_unit=None):
        """string="", multiplier=1, adder=0, si_unit=None):"""
        self = str.__new__(cls, _case(string) or si_unit or cls.si_unit)
        self._multiplier = multiplier
        self._adder = adder

        # Allow creation of a new unit that is not derivative of a module cnst.
        if si_unit is not None:  # Guard on keyword option
            self.si_unit = str(si_unit)

        ## All new instances get memoized
        self._memoize()

        return self

    ## Memoize into Unit.Glossary
    def _memoize(self, warn=True):
        """save self into Glossary, w/ overwite warning option"""
        # check key or not?
        if warn and self in self.Glossary: # Guard
            print >> sys.stderr, (
                'Warning: Overwriting Unit.Glossary["%s"]' % self
                )
        self.Glossary.update({self:self})

    ## The conversion function called: \n
    ## \f$ y = m(x + b) \f$ \n
    ## \param x is the value in non-base/SI units, and must support float()
    ## \retval y  is the value in self.__class__.si_unit
    def __call__(self, x):
        # todo: case x? who has case?
        return self._multiplier * float(x) + self._adder

    ## \param index Key to delete
    ## \par Side Effects:
    ## deletes key from rdf.units.GLOSSARY for ever.
    @classmethod
    def __delitem__(cls, index):
        del cls.Glossary[index]

    ## This is a TypeError: only Prefix.init can set Unit.Glossary
    @classmethod
    def __setitem__(cls, index, value):
        raise TypeError("Only Instaniation can set items for % class" %
                        cls.__name__)

## Length conversion to meters
@exa
@peta
@tera
@giga
@mega
@kilo
@base
@centi
@milli
@micro
@nano
@pico
@femto
@atto
class Length(Unit):
    si_unit = 'm'

## Conversion to kilograms
@base
class Mass(Unit):
    si_unit = 'kg'

@exa
@peta
@tera
@giga
@mega
@kilo
@base
@milli
@micro
@nano
@pico
@femto
@atto
## Time conversion to seconds
class Time(Unit):
    si_unit = 's'

@exa
@peta
@tera
@giga
@mega
@kilo
@milli
@micro
@nano
@pico
@femto
@atto
@base
class ElectricCurrent(Unit):
    si_unit = 'amp'

## Length conversion to square-meter
@base
class Area(Unit):
    si_unit = 'm*m'

## Speed conversion to meters per seconds
@base
@centi
@kilo
class Velocity(Unit):
    si_unit = 'm/s'


## Power conversion to Watts
@exa
@peta
@tera
@giga
@mega
@kilo
@milli
@micro
@nano
@pico
@femto
@atto
class Power(Unit):
    si_unit = 'W'


## decibel Power -is not power- it's just a number.
@base
class dBPower(Unit):
    si_unit = 'dbW'

## Blaise Pascal (19 June 1623 - 19 August 1662)
@base
class Pressure(Unit):
    """Pascal"""
    si_unit = 'Pa'

## Frequency conversion to Hz
@kilo
@mega
@giga
@tera
@base
class Frequency(Unit):
    si_unit = 'Hz'


## Temperature conversion to Celcius
@base
class Temperature(Unit):
    ## This just not right
    si_unit = 'degC'


@base
class AmountOfSubstance(Unit):
    si_unit = "mol"


@base
class LuminousIntensity(Unit):
    si_unit = "cd"


## Angle conversion to degrees
@base
@milli
class Angle(Unit):
    si_unit = 'rad'



## Data Volume conversion to bits
@base2
@kilo2
@mega2
@giga2
@kibi
@mebi
@gibi
@tebi
@pebi
@exbi
@zebi
@yebi
class Bit(Unit):
    si_unit = 'bits'


## Data rate conversion to bps
@base2
@kilo2
@mega2
@giga2
@mebi
@gibi
@tebi
@pebi
@exbi
@zebi
@yebi
class BitPerSecond(Unit):
    si_unit = 'bits/s'


## Data Volume conversion to bits
@base2
@kilo2
@mega2
@giga2
@kibi
@mebi
@gibi
@tebi
@pebi
@exbi
@zebi
@yebi
class Byte(Unit):
    si_unit = 'byte'

@base
class Pixel(Unit):
    si_unit = 'pixel'

## Data rate conversion to bytes per second
@base2
@kilo2
@mega2
@giga2
@mebi
@gibi
@tebi
@pebi
@exbi
@zebi
@yebi
class BytesPerSecond(Unit):
    si_unit = 'byte/s'


## TBD
class Ratio(Unit):
    pass


## Send these over to addendum.py
__all__ = ('Length', 'Mass', 'Area', 'Time', 'Velocity', 'Power',
           'dBPower', 'Frequency', 'Angle', 'Bit', 'BitPerSecond', 'Ratio',
           'BytesPerSecond' , 'Temperature', 'Byte', 'Pixel', 'Pressure')
