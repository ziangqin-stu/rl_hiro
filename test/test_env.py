"""Test Ant Enviornments"""
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import torch
from utils import get_env
import numpy as np
import wandb
from utils import envnames_ant, log_video_hrl, log_video_hrl_dev, ParamDict
from network import ActorLow, ActorHigh

# wandb.init(project="ziang-hiro")
import gym
from pyvirtualdisplay import Display
from environments.create_maze_env import create_maze_env


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
    for env_name in envnames_ant:
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
    if video:
        display = Display(backend='xvfb')
        display.start()
    for env_name in envnames_ant:
        interact_env(env_name, video=video)
    if video:
        display.popen.kill()


def test_env(env_name):
    env = get_env(env_name)
    print(env.spec.id)


def test_log_video_hrl(use_cuda=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if use_cuda else "cpu"
    policy_params = ParamDict(c=10, )
    params = ParamDict(policy_params=policy_params, use_cuda=False)

    def rand_actor_low(x, y):
        return torch.Tensor(np.random.normal(size=action_dim)) * max_action

    def rand_actor_high(x):
        return torch.Tensor(np.random.normal(size=state_dim)) * 10

    for env_name in envnames_ant:
        env = create_maze_env(env_name=env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        actor_low = ActorLow(state_dim, action_dim, max_action).to(device)
        actor_high = ActorHigh(state_dim, max_action).to(device)
        log_video_hrl(env_name, rand_actor_low, rand_actor_high, params)


def probe_action_single(env_name, ind, use_cuda=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if use_cuda else "cpu"
    policy_params = ParamDict(c=10,)
    params = ParamDict(policy_params=policy_params, use_cuda=False)

    def rand_actor_low(x, y):
        action = torch.zeros(action_dim)
        # action[ind] = max_action / 500.
        # action = torch.zeros(action_dim)
        action = (torch.normal(mean=torch.zeros(action_dim), std=torch.ones(action_dim)) * max_action).clamp(-max_action, max_action)
        # action = (torch.ones(action_dim) * max_action / 1).clamp(-max_action, max_action)
        return action

    def rand_actor_high(x):
        action = torch.Tensor(np.random.normal(size=state_dim)) * 10
        return action

    env = create_maze_env(env_name=env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    log_video_hrl_dev(env_name, rand_actor_low, rand_actor_high, params)


def probe_action(env_name, use_cuda=False):
    env = create_maze_env(env_name=env_name)
    action_dim = env.action_space.shape[0]
    for i in range(action_dim):
        probe_action_single(env_name, i)


def test_mojoco(env_name):
    env = gym.make(env_name)
    obs_sequence, action_sequence, reward_sequence, episode_reward = [], [], [], 0
    done = False
    step = 0
    obs = env.reset()
    while not done and step < 20:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        step += 1
        obs_sequence.append(obs)
        action_sequence.append(action)
        reward_sequence.append(reward)
        episode_reward == reward

    print("        > observation:")
    for i in range(len(obs_sequence)):
        print("            {}".format(obs_sequence[i].tolist()))
    print("        > action:")
    for i in range(len(obs_sequence)):
        print("            {}".format(action_sequence[i].tolist()))
    print("        > reward:")
    for i in range(len(obs_sequence)):
        print("            {}".format(reward_sequence[i].tolist()))
    print("    Episode step: {}, reward: {}".format(step, episode_reward))


if __name__ == "__main__":
    gym.logger.set_level(40)
    # show_envs()
    # interact_envs_display(video=True)
    # interact_env('AntMaze', video=False)
    # test_env("InvertedPendulum-v2")
    # test_log_video_hrl()
    # probe_action("AntMaze")
    # probe_action_single("AntMaze", 0)
    test_mojoco("Ant-v2")
