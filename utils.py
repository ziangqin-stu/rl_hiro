"""
Project utils
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
from environments.create_maze_env import create_maze_env


class ParamDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self = d


def get_critic():
    pass


def get_actor():
    pass


def get_env(env_name):
    env = create_maze_env(env_name=env_name)
    return env
