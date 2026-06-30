#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2013 California Institute of Technology. ALL RIGHTS RESERVED.
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




import collections

Tag = collections.namedtuple('Tag', 'version svn_revision yyyymmdd')

releases = (Tag('1.0.0',   '739', '20120814'),
            Tag('1.5.0',  '1180', '20131018'),
            Tag('1.5.01', '1191', '20131028'),
            Tag('2.0.0',  '1554', '20140724'),
            Tag('2.0.0_201409',  '1612', '20140918'),
            Tag('2.0.0_201410',  '1651', '20141103'),
            Tag('2.0.0_201505',  '1733', '20150504'),
            Tag('2.0.0_201506',  '1783', '20150619'),
            Tag('2.0.0_201511',  '1917', '20151123'),
            Tag('2.0.0_201512',  '1931', '20151221'),
            Tag('2.0.0_201604',  '2047', '20160426'),
            Tag('2.0.0_201604_dempatch', '2118:2047', '20160727'),
            Tag('2.0.0_201609',  '2143', '20160903'),
            Tag('2.0.0_20160906',  '2145', '20160906'),
            Tag('2.0.0_20160908',  '2150', '20160908'),
            Tag('2.0.0_20160912',  '2153', '20160912'),
            Tag('2.0.0_20170403',  '2256', '20170403'),
            Tag('2.1.0',  '2366', '20170806'),
            Tag('2.2.0',  '2497', '20180714'),
            Tag('2.2.1',  '2517', '20181221'),
            Tag('2.3',    '2531', '20190112'),
            # git migration
            Tag('2.3.1', '', '20190220'),
            Tag('2.3.2', '', '20190618'),
            Tag('2.3.3', '', '20200402'),
            Tag('2.4.0', '', '20200730'),
            Tag('2.4.1', '', '20200915'),
            Tag('2.4.2', '', '20201116'),
            Tag('2.5.0', '', '20210304'),
            Tag('2.5.1', '', '20210305'),
            Tag('2.5.2', '', '20210528'),
            Tag('2.5.3', '', '20210823'),
            Tag('2.6.0', '', '20220214'),
            Tag('2.6.1', '', '20220811'),
            Tag('2.6.2', '', '20230117'),
            Tag('2.6.3', '', '20230418'),
            Tag('2.6.4', '', '20250501'),
)


release_version = releases[-1].version
release_svn_revision = releases[-1].svn_revision
release_date = releases[-1].yyyymmdd
