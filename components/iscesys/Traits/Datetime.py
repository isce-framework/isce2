#!/usr/bin/env python3

import datetime as DT

dtformat = "%Y-%m-%d %H:%M:%S.%f"
dtformat1 = "%Y-%m-%d %H:%M:%S"

class datetimeType(DT.datetime):
    '''
    Override inbuilt datetime.datetime to add functionality.
    '''

    def __new__(self, *args, **kwargs):
        '''
        Override the constructor.
        '''
        if len(args)==1 and isinstance(args[0], str):
            try:
                tag = DT.datetime.strptime(args[0], dtformat)
            except:
                tag = DT.datetime.strptime(args[0], dtformat1)

            return DT.datetime.__new__(self, tag.year, tag.month, tag.day,
                                       tag.hour, tag.minute, tag.second,
                                       tag.microsecond
                        )
        elif len(args)==1 and isinstance(args[0], DT.datetime):
            tag = args[0]
            return DT.datetime.__new__(self, tag.year, tag.month, tag.day,
                                       tag.hour, tag.minute, tag.second,
                                       tag.microsecond
            )
        else:
            return DT.datetime.__new__(self, *args, **kwargs)


    def __str__(self):
        return DT.datetime.strftime(self, dtformat)
