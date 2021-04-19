from copy import deepcopy
from pprint import pformat


class DDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo=None):
        return DDict(deepcopy(dict(self), memo=memo))

    def __str__(self):
        return pformat(self)
