"""
Project utils
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import wandb
import numpy as np
import gym
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


envnames_ant = ['AntBlock', 'AntBlockMaze', 'AntFall', 'AntMaze', 'AntPush']
envnames_mujoco = ['InvertedPendulum-v2', 'InvertedDoublePendulum-v2', 'Hopper-v2', 'Humanoid-v2', 'Ant-v2',
                   'Reacher-v2', 'HalfCheetah-v2', 'Walker2d-v1']


def get_env(env_name):
    global envnames_ant
    global envnames_mujoco
    env = None
    if env_name in envnames_ant:
        env = create_maze_env(env_name=env_name)
    elif env_name in envnames_mujoco:
        env = gym.make(env_name)
    return env


def log_video(env_name, actor):
    env = create_maze_env(env_name=env_name)
    print('\n    > Collecting current trajectory...')
    done = False
    step = 1
    state = env.reset()
    frame_buffer = []
    while not done:
        frame_buffer.append(env.render(mode='rgb_array'))
        action = actor(state)
        state, reward, done, info = env.step(action)
        step += 1
    print('    > Finished collection, saved video.\n')
    frame_buffer = np.array(frame_buffer).transpose(0, 3, 1, 2)
    wandb.log({"video": wandb.Video(frame_buffer, fps=30, format="mp4")})
    env.close()

