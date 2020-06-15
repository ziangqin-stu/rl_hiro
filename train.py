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
    state, done = env.reset()
    for t in range(params.policy_params.max_timestep):
        # collect experience
        if t < params.policy_params.start_timestep:
            action = env.action_space.sample()
        else:
            max_action = params.policy_params.max_action
            action = (actor_l(state).detach().cpu()
                      + np.random.normal(0, params.policy_params.max_action * params.policy_params.expl_noise,
                                         size=params.action_dim).astype(np.float32)).clamp(-max_action, max_action)
        next_state, reward, done, info = env.step(action)








