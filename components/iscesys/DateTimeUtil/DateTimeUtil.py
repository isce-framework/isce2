#Copyright 2010, by the California Institute of Technology.
#ALL RIGHTS RESERVED.
#United States Government Sponsorship acknowledged.
#Any commercial use must be negotiated with the Office of
#Technology Transfer at the California Institute of Technology.
#
#This software may be subject to U.S. export control laws. By
#accepting this software, the user agrees to comply with all applicable
#U.S. export laws and regulations. User has the responsibility to obtain
#export licenses, or other export authority as may be required before
#exporting such information to foreign countries or providing access
#to foreign persons.
import datetime
from isceobj.Planet import AstronomicalHandbook
from isceobj.Util.decorators import type_check



hour = AstronomicalHandbook.hour
day = AstronomicalHandbook.day
## Breaking PEP008 to conform to scipy.constants's convention.
micro = 1.e-6

## wrapped line for namespace greppage -may not be needed.
#__all__ = ('parseIsoDateTime', 'timedelta_to_seconds', 'seconds_since_midnight', 'date_time_to_decimal_year')


## Some format constants
_formats = ('%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S')


@type_check(datetime.timedelta)
def timedelta_to_seconds(td):
    """seconds = timedelta_to_seconds(td)

    td:        a timedelta-like object
    seconds    a float (s).
    """
    result = (
        td.microseconds * micro +
        td.seconds +
        td.days * day
        )
    return result

@type_check(datetime.datetime)
def seconds_since_midnight(dt):
    """s = seconds_since_midnight(dt)

    dt a datetime instance
    s  float, seconds since midnight
    """
    td = dt - dt.replace(hour=0,minute=0,second=0,microsecond=0)
    return timedelta_to_seconds(td)

@type_check(datetime.datetime)
def date_time_to_decimal_year(dt):
    """Given a datetime object, return the value of the year plus the
    percentage of the year"""
    decimalYear = dt.year + (dt.timetuple().tm_yday) / 365.25
    return decimalYear

def parseIsoDateTime(iso):
    for format in _formats:
        try:
            dt = datetime.datetime.strptime(iso, format)
        except ValueError:
            try:
                self.logger.error("Unable to parse date time %s" % (iso))
            except NameError:
                print("Can't log 'self' in function.")
                pass
            raise ValueError
            pass
        pass
    return dt

## To be Deprecated
class DateTimeUtil(object):


    @staticmethod
    def timeDeltaToSeconds(td):
        """
        Convert a datetime.timedelta object into an equivalent number of seconds.
        This function is a substitute for the timedelta.total_seconds() function available
        in Python 2.7
        """
        if (not isinstance(td,datetime.timedelta)):
            raise TypeError
        return (td.microseconds + (td.seconds + td.days * 24.0 * 3600.0) * 10**6) / 10**6

    @staticmethod
    def secondsSinceMidnight(dt):
        """
        Given a datetime object, return the number of seconds since midnight on that same day.
        """
        if (not isinstance(dt,datetime.datetime)):
            raise TypeError
        td = (dt - dt.replace(hour=0,minute=0,second=0,microsecond=0))
        numSeconds = DateTimeUtil.timeDeltaToSeconds(td)
        return numSeconds

    @staticmethod
    def dateTimeToDecimalYear(dt):
        """Given a datetime object, return the value of the year plus the percentage of the year"""
        if (not isinstance(dt,datetime.datetime)):
            raise TypeError
        decimalYear = dt.year + (dt.timetuple().tm_yday)/365.25
        return decimalYear

    @staticmethod
    def parseIsoDateTime(iso):

        dt = None
        formats = ('%Y-%m-%dT%H:%M:%S.%fZ',
                   '%Y-%m-%dT%H:%M:%S.%f',
                   '%Y-%m-%dT%H:%M:%SZ',
                   '%Y-%m-%dT%H:%M:%S')
        for format in formats:
            try:
                dt = datetime.datetime.strptime(iso,format)
            except ValueError:
                pass
        if (not dt):
            self.logger.error("Unable to parse date time %s" % (iso))
            raise ValueError

        return dt
