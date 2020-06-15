"""
HIRO training process
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import copy
import torch
from torch.nn import functional
import numpy as np
import wandb
from utils import get_env, log_video, ParamDict
from network import Actor, Critic
from experience_buffer import ExperienceBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(params):
    # Initialization
    env = get_env(params.env_name)

    actor_h = Actor(params.state_dim, params.action_dim, params.policy_params.max_action).to(device)
    actor_target_h = copy.deepcopy(actor_h)
    actor_optimizer_h = torch.optim.Adam(actor_h.parameters(), lr=params.policy_params.lr)
    critic_h = Critic(params.state_dim, params.action_dim).to(device)
    critic_target_h = copy.deepcopy(critic_h)
    critic_optimizer_h = torch.optim.Adam(critic_h.parameters(), lr=params.policy_params.lr)
    experience_buffer_h = ExperienceBuffer(int(1e6), params.state_dim, params.action_dim)

    actor_l = Actor(params.state_dim, params.action_dim, params.policy_params.max_action).to(device)
    actor_target_l = copy.deepcopy(actor_l)
    actor_optimizer_l = torch.optim.Adam(actor_l.parameters(), lr=params.policy_params.lr)
    critic_l = Critic(params.state_dim, params.action_dim).to(device)
    critic_target_l = copy.deepcopy(critic_l)
    critic_optimizer_l = torch.optim.Adam(critic_l.parameters(), lr=params.policy_params.lr)
    experience_buffer_l = ExperienceBuffer(int(1e6), params.state_dim, params.action_dim)

    # Set Seed
    env.seed(params.policy_params.seed)
    torch.manual_seed(params.policy_params.seed)
    np.random.seed(params.policy_params.seed)

    # TD3 Algorithm
    total_it = [0]
    episode_reward, episode_timestep, episode_num = 0, 0, 0
    state, done = env.reset()
    goal = torch.randn_like(env.action_space.sample())
    for t in range(params.policy_params.max_timestep):
        # collect experience
        state_sequence, goal_sequence, action_sequence, reward_sequence, done_sequence = [], [], [], []
        # low-level collecion
        if t < params.policy_params.start_timestep:
            action = env.action_space.sample()
        else:
            max_action = params.policy_params.max_action
            expl_noise = params.policy_params.expl_noise
            action = (actor_l(state, goal).detach().cpu()
                      + np.random.normal(0, max_action * expl_noise, size=params.action_dim).astype(np.float32)).clamp(-max_action, max_action)
        next_state, reward, done, info = env.step(action)
        intri_reward = intrinsic_reward(state, goal, next_state)
        done =
        next_goal = h_function(state, goal, next_state)
        state = next_state
        experience_buffer_l.add(state, goal, action, next_state, next_goal, intri_reward, done)
        state_sequence.append(state)
        action_sequence.append(action)
        goal_sequence.append(goal)
        reward_sequence.append(reward)
        done_sequence.append(done)
        # high-level collecion
        if t % params.policy_params.c == 0 and t > 0:
            if t < params.policy_params.start_timestep:
                goal = torch.randn_like(action)
            else:
                goal_hat =
                max_goal = params.policy_params.max_goal
                expl_noise = params.policy_params.expl_noise
                goal = (actor_l(state, goal_hat).detach().cpu()
                          + np.random.normal(0, max_goal * expl_noise, size=params.action_dim).astype(np.float32)).clamp(-max_goal, max_goal)
            experience_buffer_h.add(state_sequence, goal_sequence, action_sequence, reward_sequence, done_sequence, state+goal)

        # update networks
        # low-level update
        if t >= params.policy_params.start_timestep:
            step_update_l(experience_buffer_l, params.policy_params.batch_size, total_it, actor_l, actor_target_l, critic_l,
                        critic_target_l, critic_optimizer_l, actor_optimizer_l, params)
        if done:
            print()
            if t >= params.policy_params.start_timestep:
                wandb.log()
            if params.save_video and (t % params.video_interval) == 0:
                log_video(params.env_name, actor_l)
            state, done = env.reset(), False
            episode_reward, episode_timestep = 0, 0
            episode_num += 1
        # high-level update
        if t >= params.policy_params.start_timestep and t % params.policy_params.c == 0 and t > 0:
            step_update_h(experience_buffer_h, params.policy_params.batch_size, total_it, actor_h, actor_target_h,
                            critic_h, critic_target_h, critic_optimizer_h, actor_optimizer_h, params)


if __name__ == "__main__":
    env_name = "AntBlockMaze"
    env = get_env(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    policy_params = ParamDict(
        seed=0,
        policy_noise=0.2,
        expl_noise=0.1,
        noise_clip=0.5,
        max_action=max_action,
        discount=0.99,
        policy_freq=2,
        tau=0.005,
        lr=3e-4,
        max_timestep=int(1e6),
        start_timestep=int(25e3),
        batch_size=256
    )
    params = ParamDict(
        policy_params=policy_params,
        env_name=env_name,
        state_dim=state_dim,
        action_dim=action_dim,
        save_video=True,
        video_interval=int(1e4)
    )
    wandb.init(project="ziang-hiro")
    train(params=params)









