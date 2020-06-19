"""
Test Develop TD3 Algorithm
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import copy
import torch
from torch.nn import functional
import numpy as np
import wandb
from utils import get_env, log_video, ParamDict, VideoLoggerTrigger
from network import ActorTD3, CriticTD3
from experience_buffer import ExperienceBufferTD3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


def step_update(experience_buffer, batch_size, total_it, actor, actor_target, critic, critic_target, critic_optimizer,
                actor_optimizer, params):
    total_it[0] += 1
    # Sample Experience Batch
    state, action, next_state, reward, done = experience_buffer.sample(batch_size)
    with torch.no_grad():
        # select action according to policy and add clipped noise
        noise = (torch.randn_like(action) * params.policy_params.policy_noise).clamp(-params.policy_params.noise_clip,
                                                                                     params.policy_params.noise_clip)
        next_action = (actor_target(next_state) + noise).clamp(-params.policy_params.max_action,
                                                               params.policy_params.max_action)
        # compute target Q value
        target_q1, target_q2 = critic_target(next_state, next_action)
        target_q = torch.min(target_q1, target_q2)
        target_q = reward + (1 - done) * params.policy_params.discount * target_q
    # Get current Q estimates
    current_q1, current_q2 = critic(state, action)
    # Compute critic loss
    critic_loss = functional.mse_loss(current_q1, target_q) + functional.mse_loss(current_q2, target_q)
    # Optimize the critic
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    # delayed policy updates
    actor_loss = None
    if total_it[0] % params.policy_params.policy_freq == 0:
        # compute actor loss
        actor_loss = -critic.q1(state, actor(state)).mean()
        # optimize the actor
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        # Update the frozen target models
        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
            target_param.data.copy_(
                params.policy_params.tau * param.data + (1 - params.policy_params.tau) * target_param.data)
        for param, target_param in zip(actor.parameters(), actor_target.parameters()):
            target_param.data.copy_(
                params.policy_params.tau * param.data + (1 - params.policy_params.tau) * target_param.data)
    return target_q, critic_loss, actor_loss


def train(params):
    # Initialize
    policy_params = params.policy_params
    env = get_env(params.env_name)
    actor = ActorTD3(params.state_dim, params.action_dim, policy_params.max_action).to(device)
    actor_target = copy.deepcopy(actor)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=policy_params.lr)
    critic = CriticTD3(params.state_dim, params.action_dim).to(device)
    critic_target = copy.deepcopy(critic)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=policy_params.lr)
    experience_buffer = ExperienceBufferTD3(int(1e6), params.state_dim, params.action_dim)
    video_log_trigger = VideoLoggerTrigger(start_ind=policy_params.start_timestep)

    # Set Seed
    env.seed(policy_params.seed)
    torch.manual_seed(policy_params.seed)
    np.random.seed(policy_params.seed)

    # Training Loop
    print("=====================================================\nStart Train - {}".format(params.env_name))
    print("=====================================================")
    state, done = env.reset(), False
    episode_reward, episode_timestep, episode_num = 0, 0, 0
    total_it = [0]
    for t in range(policy_params.max_timestep):
        episode_timestep += 1
        # get action: epsilon-greedy variant
        action = None
        if t < policy_params.start_timestep:
            action = env.action_space.sample()
        else:
            max_action = policy_params.max_action
            action = (actor(torch.Tensor(state).to(device)).detach().cpu()
                      + np.random.normal(0, policy_params.max_action * policy_params.expl_noise,
                                         size=params.action_dim).astype(np.float32)).clamp(-max_action, max_action)
        # interact
        next_state, reward, done, info = env.step(action)
        # collect experience
        experience_buffer.add(state, action, next_state, reward, done)
        state = next_state
        episode_reward += reward
        # update networks
        if t >= policy_params.start_timestep:
            target_q, critic_loss, actor_loss = \
                step_update(experience_buffer, policy_params.batch_size, total_it, actor, actor_target, critic,
                            critic_target, critic_optimizer, actor_optimizer, params)
            wandb.log({'target_q': float(torch.mean(target_q).squeeze())}, step=t-policy_params.start_timestep)
            wandb.log({'critic_loss': float(torch.mean(critic_loss).squeeze())}, step=t-policy_params.start_timestep)
            if actor_loss is not None: wandb.log({'actor_loss': float(torch.mean(actor_loss).squeeze())}, step=t-policy_params.start_timestep)
        if done:
            print(
                f"> Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timestep} Reward: {episode_reward:.3f}")
            if t >= policy_params.start_timestep:
                wandb.log({'episode reward': episode_reward}, step=t-policy_params.start_timestep)
            if params.save_video and video_log_trigger.good2log(t, params.video_interval):
                log_video(params.env_name, actor)
            state, done = env.reset(), False
            episode_reward, episode_timestep = 0, 0
            episode_num += 1


if __name__ == "__main__":
    env_name = "InvtertedDoublePendulum-v2"
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
        video_interval=int(5e2)
    )
    wandb.init(project="ziang-hiro")
    train(params=params)
