from isceobj.Util.decorators import type_check

class Distortion(object):
    """A class to hold polarimetric distortion matrix information"""

    def __init__(self, crossTalk1=None, crossTalk2=None, channelImbalance=None):
        self._crossTalk1 = crossTalk1
        self._crossTalk2 = crossTalk2
        self._channelImbalance = channelImbalance
        return None

    def getCrossTalk1(self):
        return self._crossTalk1

    def getCrossTalk2(self):
        return self._crossTalk2

    def getChannelImbalance(self):
        return self._channelImbalance

    @type_check(complex)
    def setCrossTalk1(self, xtalk):
        self._crossTalk1 = xtalk
        return None

    @type_check(complex)
    def setCrossTalk2(self, xtalk):
        self._crossTalk2 = xtalk
        return None

    @type_check(complex)
    def setChannelImbalance(self, imb):
        self._channelImbalance = imb
        return None

    crossTalk1 = property(getCrossTalk1, setCrossTalk1)
    crossTalk2 = property(getCrossTalk1, setCrossTalk2)
    channelImbalance = property(getChannelImbalance, setChannelImbalance)

    pass

        
