from iscesys.Component.Component import Component

class EnglishStandard(Component):
    def __call__(self, name=None):
        if name:
            print("Hello, {0}!".format(name))
        else:
            print("Hello!")
        return
