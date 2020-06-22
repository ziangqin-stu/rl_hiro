"""
Project utils
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import torch
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


class VideoLoggerTrigger:
    def __init__(self, start_ind=1):
        self.last_log = int(start_ind)
        self.first_log = False

    def good2log(self, t, interval):
        if int(t / interval) > int(self.last_log / interval):
            self.last_log = t
            return True
        elif not self.first_log and 0 < t - self.last_log < interval:
            self.first_log = True
            return True
        return False


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
    else:
        raise NotImplementedError("environment {} is not supported!".format(env_name))
    return env


def log_video(env_name, actor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if params.use_cuda else "cpu"
    # device = "cpu"
    if env_name in envnames_mujoco:
        env = gym.make(env_name)
    elif env_name in envnames_ant:
        env = create_maze_env(env_name=env_name)
    print('\n    > Collecting current trajectory...')
    done = False
    step = 1
    state = env.reset()
    frame_buffer = []
    while not done:
        frame_buffer.append(env.render(mode='rgb_array'))
        action = actor(torch.Tensor(state).to(device)).detach().cpu()
        state, reward, done, info = env.step(action)
        step += 1
    print('    > Finished collection, saved video.\n')
    frame_buffer = np.array(frame_buffer).transpose(0, 3, 1, 2)
    wandb.log({"video": wandb.Video(frame_buffer, fps=30, format="mp4")})
    env.close()


def log_video_hrl(env_name, actor_low, actor_high, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if params.use_cuda else "cpu"
    # device = "cpu"
    policy_params = params.policy_params
    if env_name in envnames_mujoco:
        env = gym.make(env_name)
    elif env_name in envnames_ant:
        env = create_maze_env(env_name=env_name)
    print('\n    > Collecting current trajectory...')
    done = False
    step = 1
    state = torch.Tensor(env.reset())
    goal = torch.Tensor(torch.randn_like(state))
    episode_reward, frame_buffer = 0, []
    while not done and step < 1000:
        frame_buffer.append(env.render(mode='rgb_array'))
        action = actor_low(torch.Tensor(state).to(device), torch.Tensor(goal).to(device)).detach().cpu()
        if (step + 1) % policy_params.c == 0 and step > 0: goal = actor_high(state)
        state, reward, done, info = env.step(action)
        episode_reward += reward
        step += 1
    print(f'    > Finished collection, saved video. Episode reward: {float(episode_reward):.3f}\n')
    frame_buffer = np.array(frame_buffer).transpose(0, 3, 1, 2)
    wandb.log({"video": wandb.Video(frame_buffer, fps=30, format="mp4")})
    env.close()


def print_cmd_hint(params, location):
    policy_params = params.policy_params
    if location == "start_train":
        print("\n\n==========================================================\nStart Train: {}".format(params.env_name))
        print("==========================================================")
    elif location == "end_train":
        print("\n==========================================================\nFinished Training! - {}".format(params.env_name))
        print("==========================================================\n\n")




