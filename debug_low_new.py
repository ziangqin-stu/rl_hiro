import os
import datetime
from torch import Tensor
import copy
import torch
from torch.nn import functional
import numpy as np
import wandb
from utils import get_env, log_video_hrl, ParamDict, LoggerTrigger, TimeLogger
from network import ActorLow, ActorHigh, CriticLow, CriticHigh
from experience_buffer import ExperienceBufferLow, ExperienceBufferHigh

# train low-level policy to reach a static goal
def train(params):
    # Initialize
    # general setings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if params.use_cuda else "cpu"
    # utils
    max_step = params.max_step
    start_timestep = params.start_timestep
    expl_noise_std_l = params.expl_noise_std_l
    action_dim = params.action_dim
    state_dim = params.state_dim
    max_action = params.max_action
    # rl components
    env = get_env(params.env_name)
    actor_eval_l = ActorLow(state_dim, action_dim, policy_params.max_action).to(device)
    actor_target_l = copy.deepcopy(actor_eval_l).to(device)
    actor_optimizer_l = torch.optim.Adam(actor_eval_l.parameters(), lr=policy_params.actor_lr)
    critic_eval_l = CriticLow(state_dim, action_dim).to(device)
    critic_target_l = copy.deepcopy(critic_eval_l).to(device)
    critic_optimizer_l = torch.optim.Adam(critic_eval_l.parameters(), lr=policy_params.critic_lr)
    experience_buffer_l = ExperienceBufferLow(policy_params.max_timestep, state_dim, action_dim, params.use_cuda)

    # TD3
    # initialize
    # training loop
    state = Tensor(env.reset()).to(device)
    goal = torch.randn_like(state)
    for t in range(0, max_step):
        # generate action
        if t < start_timestep:
            action = env.action_space.sample()
        else:
            expl_noise_low = np.random.normal(mean=0, scale=expl_noise_std_l, size=action_dim).astype(np.float32)
            action = actor_eval_l(state, goal).detach().cpu()
            action = (action + expl_noise_low).clamp(-max_action, max_action)

