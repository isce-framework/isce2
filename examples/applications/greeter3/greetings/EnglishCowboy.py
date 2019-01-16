from iscesys.Component.Component import Component

class EnglishCowboy(Component):
    def __call__(self, name=None):
        if name:
            print("Howdy, {0}!".format(name))
        else:
            print("Howdy, Pardner")
        return
