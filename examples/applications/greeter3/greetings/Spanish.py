from iscesys.Component.Component import Component

class Spanish(Component):
    def __call__(self, name=None):
        if name:
            print("iHola {0}!".format(name))
        else:
            print("iHola!")
        return
