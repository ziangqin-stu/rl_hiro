"""
Project utils
"""
import copy
import time
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


class LoggerTrigger:
    # trigger log behaviour intermittently
    def __init__(self, start_ind=1, first_log=True):
        self.last_log = int(start_ind)
        self.first_log = not first_log

    def good2log(self, t, interval):
        if int(t / interval) > int(self.last_log / interval):
            self.last_log = t
            return True
        elif not self.first_log and 0 < t - self.last_log < interval:
            self.first_log = True
            return True
        return False


class TimeLogger:
    def __init__(self):
        self.start_time = time.time()
        self.sps_start_time = time.time()
        self.sps_start_step = 0

    def sps(self, step):
        cur_time = time.time()
        time_span = cur_time - self.sps_start_time
        sps = (step - self.sps_start_step) / time_span
        self.sps_start_time = cur_time
        self.sps_start_step = step
        print("    >| state per second in past {}s: {}".format(time_span, sps))

    def time_spent(self):
        time_span = time.time() - self.start_time
        print("    >| training time: {} minutes".format(float('%.2f' % (time_span/60))))


envnames_ant = ['AntBlock', 'AntBlockMaze', 'AntFall', 'AntMaze', 'AntPush']
envnames_mujoco = ['InvertedPendulum-v2', 'InvertedDoublePendulum-v2', 'Hopper-v2', 'Humanoid-v2', 'Ant-v2',
                   'Reacher-v2', 'HalfCheetah-v2', 'Walker2d-v1']


def get_env(env_name):
    global envnames_ant
    global envnames_mujoco
    if env_name in envnames_ant:
        env = create_maze_env(env_name=env_name)
    elif env_name in envnames_mujoco:
        env = gym.make(env_name)
    else:
        raise NotImplementedError("environment {} is not supported!".format(env_name))
    return env


def get_target_position(env_name):
    if env_name == 'AntPush':
        target_pos = torch.Tensor([0, 19, 0.5])
    elif env_name == 'AntFall':
        target_pos = torch.Tensor([0, 27, 4.5])
    else:
        raise ValueError("{} is either wrong or not implemented!".format(env_name))
    return target_pos


def evaluate(actor_l, actor_h):
    success_number = 0
    for seed in range(10):
        for epi in range(5):
            for t in range(1000):
                if done:
                    success_number += 1
                    break
    success_rate = torch.Tensor([success_number / 50])
    return success_rate


def log_video(env_name, actor):
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
        action = actor(torch.Tensor(state)).detach().cpu()
        state, reward, done, info = env.step(action)
        step += 1
    print('    > Finished collection, saved video.\n')
    frame_buffer = np.array(frame_buffer).transpose(0, 3, 1, 2)
    wandb.log({"video": wandb.Video(frame_buffer, fps=30, format="mp4")})
    env.close()


def log_video_hrl(env_name, actor_low, actor_high, params):
    actor_low = copy.deepcopy(actor_low).cpu()
    actor_high = copy.deepcopy(actor_high).cpu()
    actor_high.max_goal = actor_high.max_goal.to('cpu')
    policy_params = params.policy_params
    goal_dim = params.goal_dim
    if env_name in envnames_mujoco:
        env = gym.make(env_name)
    elif env_name in envnames_ant:
        env = create_maze_env(env_name=env_name)
    print('\n    > Collecting current trajectory...')
    done = False
    step = 1
    state = torch.Tensor(env.reset())
    goal = torch.Tensor(torch.randn(goal_dim))
    episode_reward, frame_buffer = 0, []
    while not done and step < 600:
        frame_buffer.append(env.render(mode='rgb_array'))
        action = actor_low(torch.Tensor(state), torch.Tensor(goal)).detach()
        next_state, reward, done, info = env.step(action)
        if (step + 1) % policy_params.c == 0 and step > 0:
            goal = actor_high(state)
        else:
            goal = (torch.Tensor(state)[:goal_dim] + goal - torch.Tensor(next_state)[:goal_dim]).float()
        state = next_state
        episode_reward += reward
        step += 1
    print(f'    > Finished collection, saved video. Episode reward: {float(episode_reward):.3f}\n')
    frame_buffer = np.array(frame_buffer).transpose(0, 3, 1, 2)
    wandb.log({"video": wandb.Video(frame_buffer, fps=30, format="mp4")})
    env.close()


def log_video_hrl_dev(env_name, actor_low, actor_high, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if params.use_cuda else "cpu"
    # device = "cpu"
    policy_params = params.policy_params
    if env_name in envnames_mujoco:
        env = gym.make(env_name)
    elif env_name in envnames_ant:
        env = create_maze_env(env_name=env_name)
    state_dim = env.observation_space.shape[0]
    print('\n    > Collecting current trajectory...')
    done = False
    step = 1
    state = torch.Tensor(env.reset())
    goal = torch.Tensor(torch.randn_like(state))
    episode_reward, frame_buffer = 0, []
    while not done and step < 200:
        frame_buffer.append(env.render(mode='rgb_array'))
        action = actor_low(torch.Tensor(state).to(device), torch.Tensor(goal).to(device)).detach().cpu()
        next_state, reward, done, info = env.step(action)
        if (step + 1) % policy_params.c == 0 and step > 0:
            goal = actor_high(state)
        else:
            goal = state + goal - next_state
        state = next_state
        for i in range(state_dim):
            wandb.log({'state[{}]'.format(i): state[i]}, step=step)
        episode_reward += reward
        step += 1
    print(f'    > Finished collection, saved video. Episode reward: {float(episode_reward):.3f}\n')
    frame_buffer = np.array(frame_buffer).transpose(0, 3, 1, 2)
    wandb.log({"video": wandb.Video(frame_buffer, fps=30, format="mp4")})
    env.close()


def log_video_hrl_debug(env_name, actor_low, actor_high, params):
    actor_low = copy.deepcopy(actor_low).cpu()
    actor_high = copy.deepcopy(actor_high).cpu()
    actor_high.max_goal = actor_high.max_goal.to('cpu')
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
    while not done and step < 600:
        frame_buffer.append(env.render(mode='rgb_array'))
        action = actor_low(torch.Tensor(state), torch.Tensor(goal)).detach()
        next_state, reward, done, info = env.step(action)
        if (step + 1) % policy_params.c == 0 and step > 0:
            # goal = actor_high(state)
            if step < 200:
                goal = torch.Tensor([-10, 10, 0.5,
                                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,  # 3-13
                                     0., 0., 0., 0., 0., 0., 0.,  # 14-20
                                     0., 0., 0., 0., 0., 0., 0., 0.,  # 21-28
                                     0.]).cpu() - torch.Tensor(next_state)
            else:
                goal = torch.Tensor([0, 19, 0.5,
                                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,  # 3-13
                                     0., 0., 0., 0., 0., 0., 0.,  # 14-20
                                     0., 0., 0., 0., 0., 0., 0., 0.,  # 21-28
                                     0.]).cpu() - torch.Tensor(next_state)
        else:
            goal = (torch.Tensor(state) + goal - torch.Tensor(next_state)).float()
        state = next_state
        episode_reward += reward
        step += 1
    print(f'    > Finished collection, saved video. Episode reward: {float(episode_reward):.3f}\n')
    frame_buffer = np.array(frame_buffer).transpose(0, 3, 1, 2)
    wandb.log({"video": wandb.Video(frame_buffer, fps=30, format="mp4")})
    env.close()


def print_cmd_hint(params, location):
    if location == "start_train":
        print("\n\n==========================================================\nStart Train: {}".format(params.env_name))
        print("==========================================================")
    elif location == "end_train":
        print("\n==========================================================\nFinished Training! - {}".format(params.env_name))
        print("==========================================================\n\n")

    elif location == "training_state":
        state_sequence, goal_sequence, action_sequence, intri_reward_sequence, updated, goal_hat, reward_h_sequence = params[:]
        print("        > state:")
        for i in range(len(state_sequence)):
            print("            {}".format(["%.4f" % elem for elem in state_sequence[i].tolist()]))
        print("        > goal:")
        for i in range(len(goal_sequence)):
            print("            {}".format(["%.4f" % elem for elem in goal_sequence[i].tolist()]))
        print("        > action:")
        for i in range(len(action_sequence)):
            print("            {}, {}".format(["%.4f" % elem for elem in action_sequence[i].tolist()], float('%.4f' % intri_reward_sequence[i])))
        if updated:
            print("        > goal_hat: {}".format(goal_hat))
        else:
            print("        > goal_hat not updated")
        print("        > reward_h:")
        for i in range(len(reward_h_sequence)):
            print("            {}".format(["%.4f" % elem for elem in reward_h_sequence[i].tolist()]))
