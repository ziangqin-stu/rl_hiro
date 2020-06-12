"""Test Ant Enviornments"""
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

import gym
from gym.wrappers import Monitor
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


def interact_env(env_name, display=False):
    env = create_maze_env(env_name=env_name)
    if display:
        abs_path = os.path.abspath(os.path.dirname(os.getcwd()))
        video_name = env_name + '_randtest.mp4'
        save_path_name = os.path.join(abs_path, 'save', 'video', video_name)
        env = Monitor(env, save_path_name, force=True)
    print('\n    > Collecting random trajectory...')
    done = False
    step = 1
    obs = env.reset()
    while not (done or step > 100):
        env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        step += 1
    if display:
        env.close()
    print('    > Finished collection, saved video in {}.\n'.format(save_path_name))
    return env


def interact_envs_display():
    global env_names
    display = Display(backend='xvfb')
    display.start()
    interact_env(env_names[0], display=True)
    display.popen.kill()


if __name__ == "__main__":
    gym.logger.set_level(40)
    # show_envs()
    interact_envs_display()
