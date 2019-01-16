from __future__ import print_function

def english_standard():
    from .EnglishStandard import EnglishStandard
    return EnglishStandard()

def english_cowboy():
    from .EnglishCowboy import EnglishCowboy
    return EnglishCowboy()


def spanish():
    from .Spanish import Spanish
    return Spanish()

facts={'english':english_standard,
       'cowboy':english_cowboy,
       'spanish':spanish
      }

def language(lang):
    try:
        return facts[lang.lower()]()
    except:
        return ValueError
