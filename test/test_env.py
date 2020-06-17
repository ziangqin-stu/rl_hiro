"""Test Ant Enviornments"""
import sys
import os
from utils import get_env

import torch
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

import numpy as np
import wandb
wandb.init(project="ziang-hiro")
import gym
from pyvirtualdisplay import Display
from environments.create_maze_env import create_maze_env

env_names = ['AntBlock', 'AntBlockMaze', 'AntFall', 'AntMaze', 'AntPush']


def show_env_property(env_name):
    env = create_maze_env.create_maze_env(env_name=env_name)
    print("\n\n-------------------------------------------------------------------------------------------------------")
    print(
        "{}: \n    MAZE_HEIGHT: {}\n    MAZE_SIZE_SCALING:{}\n    MAZE_STRUCTURE: {}".format(env_name, env.MAZE_HEIGHT,
                                                                                             env.MAZE_SIZE_SCALING,
                                                                                             env.MAZE_STRUCTURE))
    print("    MODEL_CLASS: {}".format(env.MODEL_CLASS))
    print(
        "    action_space: \n        dtype: {}\n        high: {}\n        low: {}\n        shape: {}\n        np_random: {}".format(
            env.action_space.dtype, env.action_space.high, env.action_space.low, env.action_space.shape,
            env.action_space.np_random))
    print(
        "    observation_space: \n        dtype: {}\n        high: {}\n        low: {}\n        shape: {}\n        np_random: {}".format(
            env.observation_space.dtype, env.observation_space.high[:14], env.observation_space.low[:14],
            env.observation_space.shape, env.observation_space.np_random))
    print("    reward_range: {}".format(env.reward_range))
    print("    movable_blocks: {}".format(env.movable_blocks))
    print("    wrapped_env: {}".format(env.wrapped_env))
    print("-------------------------------------------------------------------------------------------------------")
    return env


def show_envs():
    global env_names
    for env_name in env_names:
        show_env_property(env_name)


def interact_env(env_name, video=False):
    env = create_maze_env(env_name=env_name)
    print('\n    > Collecting random trajectory...')
    done = False
    step = 1
    obs = env.reset()
    frame_buffer = []
    while not (done or step > 100):
        if video:
            frame_buffer.append(env.render(mode='rgb_array'))
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        step += 1
        print(f"      > Reward: {reward:.3f}")
    print('    > Finished collection', end='')
    if video:
        frame_buffer = np.array(frame_buffer).transpose(0, 3, 1, 2)
        wandb.log({"video": wandb.Video(frame_buffer, fps=30, format="mp4")})
        print(', saved video.\n')
        env.close()
    else:
        print('.\n')
    return env


def interact_envs_display(video=False):
    global env_names
    if video:
        display = Display(backend='xvfb')
        display.start()
    for env_name in env_names:
        interact_env(env_name, video=video)
    if video:
        display.popen.kill()


def test_env(env_name):
    env = get_env(env_name)
    print(env.spec.id)


if __name__ == "__main__":
    gym.logger.set_level(40)
    # show_envs()
    # interact_envs_display(video=False)
    interact_env('AntBlock', video=False)
    # test_env("InvertedPendulum-v2")
