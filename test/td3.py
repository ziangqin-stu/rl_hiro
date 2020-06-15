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
from utils import get_env, log_video, ParamDict
from network import Actor, Critic
from experience_buffer import ExperienceBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def train(params):
    # Initialize
    env = get_env(params.env_name)
    actor = Actor(params.state_dim, params.action_dim, params.policy_params.max_action).to(device)
    actor_target = copy.deepcopy(actor)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=params.policy_params.lr)
    critic = Critic(params.state_dim, params.action_dim).to(device)
    critic_target = copy.deepcopy(critic)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=params.policy_params.lr)
    experience_buffer = ExperienceBuffer(int(1e6), params.state_dim, params.action_dim)

    # Set Seed
    env.seed(params.policy_params.seed)
    torch.manual_seed(params.policy_params.seed)
    np.random.seed(params.policy_params.seed)

    # Training Loop
    print("=====================================================\nStart Train - {}".format(params.env_name))
    print("=====================================================")
    state, done = env.reset(), False
    episode_reward, episode_timestep, episode_num = 0, 0, 0
    total_it = [0]
    for t in range(params.policy_params.max_timestep):
        episode_timestep += 1
        # get action: epsilon-greedy variant
        action = None
        if t < params.policy_params.start_timestep:
            action = env.action_space.sample()
        else:
            max_action = params.policy_params.max_action
            action = (actor(state).detach().cpu()
                      + np.random.normal(0, params.policy_params.max_action * params.policy_params.expl_noise,
                                         size=params.action_dim).astype(np.float32)).clamp(-max_action, max_action)
        # interact
        next_state, reward, done, info = env.step(action)
        # collect experience
        experience_buffer.add(state, action, next_state, reward, done)
        state = next_state
        episode_reward += reward
        # update networks
        if t >= params.policy_params.start_timestep:
            step_update(experience_buffer, params.policy_params.batch_size, total_it, actor, actor_target, critic,
                        critic_target, critic_optimizer, actor_optimizer, params)
        if done:
            print(
                f"> Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timestep} Reward: {episode_reward:.3f}")
            if t >= params.policy_params.start_timestep:
                wandb.log({'episode reward': episode_reward}, step=t - params.policy_params.start_timestep)
            if params.save_video and (t % params.video_interval) == 0:
                log_video(params.env_name, actor)
            state, done = env.reset(), False
            episode_reward, episode_timestep = 0, 0
            episode_num += 1


if __name__ == "__main__":
    env_name = "Hopper-v2"
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
