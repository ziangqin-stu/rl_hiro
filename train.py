"""
HIRO training process
"""
import os
import sys

import gym

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import copy
import torch
from torch.nn import functional
import numpy as np
import wandb
from utils import get_env, log_video, ParamDict
from network import ActorLow, ActorHigh, CriticLow, CriticHigh
from experience_buffer import ExperienceBufferLow, ExperienceBufferHigh

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def h_function(state, goal, next_state):
    return state + goal - next_state


def intrinsic_reward(state, goal, next_state):
    state, goal, next_state = torch.Tensor(state), torch.Tensor(goal), torch.Tensor(next_state)
    return -torch.sum(torch.pow(state + goal - next_state, 2))


def done_judge_low(state, goal_state):
    # return torch.Tensor(state).equal(torch.Tensor(goal))
    return torch.sum(torch.pow((state - goal_state), 2)) < 1e12


def step_update_l(experience_buffer, batch_size, total_it, actor, actor_target, critic, critic_target, critic_optimizer,
                actor_optimizer, params):
    policy_params = params.policy_params
    total_it[0] += 1
    # Sample Experience Batch
    state, goal, action, next_state, next_goal, reward, done = experience_buffer.sample(batch_size)
    with torch.no_grad():
        # select action according to policy and add clipped noise
        noise = (torch.randn_like(action) * policy_params.policy_noise).clamp(-policy_params.noise_clip, policy_params.noise_clip)
        next_action = (actor_target(next_state, next_goal) + noise).clamp(-policy_params.max_action, policy_params.max_action)
        # compute target Q value
        target_q1, target_q2 = critic_target(next_state, next_goal, next_action)
        target_q = torch.min(target_q1, target_q2)
        target_q = reward + (1 - done) * policy_params.discount * target_q
    # Get current Q estimates
    current_q1, current_q2 = critic(state, goal, action)
    # Compute critic loss
    critic_loss = functional.mse_loss(current_q1, target_q) + functional.mse_loss(current_q2, target_q)
    # Optimize the critic
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    # delayed policy updates
    if total_it[0] % policy_params.policy_freq == 0:
        # compute actor loss
        actor_loss = -critic.q1(state, goal, actor(state, goal)).mean()
        # optimize the actor
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        # Update the frozen target models
        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
            target_param.data.copy_(policy_params.tau * param.data + (1 - policy_params.tau) * target_param.data)
        for param, target_param in zip(actor.parameters(), actor_target.parameters()):
            target_param.data.copy_(policy_params.tau * param.data + (1 - policy_params.tau) * target_param.data)
    return target_q.detach(), critic_loss.detach(), actor_loss.detach()


def step_update_h(experience_buffer, batch_size, total_it, actor, actor_target, critic, critic_target, critic_optimizer,
                actor_optimizer, params):
    policy_params = params.policy_params
    total_it[0] += 1
    # Sample Experience Batch
    state, goal, action, next_state, next_goal, reward, done = experience_buffer.sample(batch_size)
    with torch.no_grad():
        # select action according to policy and add clipped noise
        # noise = (torch.randn_like(goal) * policy_params.policy_noise).clamp(-policy_params.noise_clip, policy_params.noise_clip)
        noise = (torch.randn(goal.permute(0, 2, 1).shape[:2]) * policy_params.policy_noise).clamp(-policy_params.noise_clip, policy_params.noise_clip)
        new_goal = (actor_target(next_state, next_goal) + noise).clamp(-policy_params.max_goal, policy_params.max_goal)
        # compute target Q value
        target_q1, target_q2 = critic_target(next_state, next_goal, new_goal)
        target_q = torch.min(target_q1, target_q2)
        reward = torch.mean(reward, dim=1)
        done = torch.clamp(torch.sum(done, 1), 0., 1.)
        target_q = reward + (1 - done) * policy_params.discount * target_q
    # Get current Q estimates
    current_q1, current_q2 = critic(state, goal, new_goal)
    # Compute critic loss
    critic_loss = functional.mse_loss(current_q1, target_q) + functional.mse_loss(current_q2, target_q)
    # Optimize the critic
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    # delayed policy updates
    if total_it[0] % policy_params.policy_freq == 0:
        # compute actor loss
        actor_loss = -critic.q1(state, goal, actor(state, goal)).mean()
        # optimize the actor
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        # Update the frozen target models
        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
            target_param.data.copy_(policy_params.tau * param.data + (1 - policy_params.tau) * target_param.data)
        for param, target_param in zip(actor.parameters(), actor_target.parameters()):
            target_param.data.copy_(policy_params.tau * param.data + (1 - policy_params.tau) * target_param.data)
    return target_q.detach(), critic_loss.detach(), actor_loss.detach()


def train(params):
    # Initialization
    policy_params = params.policy_params
    env = get_env(params.env_name)

    actor_h = ActorHigh(policy_params.c, params.state_dim, policy_params.max_action).to(device)
    actor_target_h = copy.deepcopy(actor_h)
    actor_optimizer_h = torch.optim.Adam(actor_h.parameters(), lr=policy_params.lr)
    critic_h = CriticHigh(policy_params.c, params.state_dim).to(device)
    critic_target_h = copy.deepcopy(critic_h)
    critic_optimizer_h = torch.optim.Adam(critic_h.parameters(), lr=policy_params.lr)
    experience_buffer_h = ExperienceBufferHigh(policy_params.max_timestep, policy_params.c, params.state_dim, params.action_dim)

    actor_l = ActorLow(params.state_dim, params.action_dim, policy_params.max_action).to(device)
    actor_target_l = copy.deepcopy(actor_l)
    actor_optimizer_l = torch.optim.Adam(actor_l.parameters(), lr=policy_params.lr)
    critic_l = CriticLow(params.state_dim, params.action_dim).to(device)
    critic_target_l = copy.deepcopy(critic_l)
    critic_optimizer_l = torch.optim.Adam(critic_l.parameters(), lr=policy_params.lr)
    experience_buffer_l = ExperienceBufferLow(policy_params.max_timestep, params.state_dim, params.action_dim)

    # Set Seed
    env.seed(policy_params.seed)
    torch.manual_seed(policy_params.seed)
    np.random.seed(policy_params.seed)

    # TD3 Algorithm
    total_it = [0]
    episode_reward_l, episode_reward_h, episode_timestep, episode_num = 0, 0, 0, 0
    state, done_l = env.reset(), False
    goal = torch.randn_like(torch.Tensor(state))
    state_sequence, goal_sequence, action_sequence, next_state_sequence, next_goal_sequence, reward_sequence, done_h_sequence = [], [], [], [], [], [], []
    for t in range(policy_params.max_timestep):
        if t >= policy_params.start_timestep:
            print("x")
        episode_timestep += 1
        # > low-level collection
        # >> sample action
        max_action = policy_params.max_action
        if t < policy_params.start_timestep:
            action = env.action_space.sample()
        else:
            expl_noise = policy_params.expl_noise
            action = (actor_l(torch.Tensor(state).cuda(), torch.Tensor(goal).cuda()).detach().cpu()
                      + np.random.normal(0, max_action * expl_noise, size=params.action_dim).astype(np.float32)).clamp(-max_action, max_action).squeeze()
        # >> perform action
        next_state, reward, done_h, info = env.step(action)
        intri_reward = intrinsic_reward(state, goal, next_state)
        # >> collect step-low
        next_state, state, action, reward, intri_reward, done_h = \
            torch.Tensor(next_state), torch.Tensor(state), torch.Tensor(action), torch.Tensor([reward]), torch.Tensor([intri_reward]), torch.Tensor([done_h])
        done_l = torch.Tensor([done_judge_low(next_state, state + goal)])
        next_goal = h_function(state, goal, next_state)
        state = next_state
        episode_reward_l += intri_reward
        episode_reward_h += reward
        experience_buffer_l.add(state, goal, action, next_state, next_goal, intri_reward, done_l)
        state_sequence.append(state)
        action_sequence.append(action)
        next_state_sequence.append(next_state)
        next_goal_sequence.append(next_goal)
        goal_sequence.append(goal)
        reward_sequence.append(reward)
        done_h_sequence.append(done_h)
        # > high-level collection
        if (t + 1) % policy_params.c == 0 and t > 0:
            # >> sample goal
            max_goal = policy_params.max_goal
            if t < policy_params.start_timestep:
                new_goal = torch.randn_like(torch.Tensor(state))
            else:
                expl_noise = policy_params.expl_noise
                new_goal = (actor_h(torch.stack(state_sequence).cuda(), torch.stack(goal_sequence).cuda()).detach().cpu()
                            + np.random.normal(0, max_goal * expl_noise, size=params.state_dim).astype(np.float32)).clamp(
                    -max_goal, max_goal).squeeze()
                goal_hat = new_goal
                goal = goal_hat
            # >> collect step-high
            experience_buffer_h.add(state_sequence, goal_sequence, action_sequence, next_state_sequence, next_goal_sequence, reward_sequence, done_h_sequence)
            state_sequence, goal_sequence, action_sequence, next_state_sequence, next_goal_sequence, reward_sequence, done_h_sequence = [], [], [], [], [], [], []

        # > update networks
        # >> low-level update
        if t >= policy_params.start_timestep:
            target_q_l, critic_loss_l, actor_loss_l = step_update_l(experience_buffer_l, policy_params.batch_size, total_it, actor_l, actor_target_l,
                          critic_l, critic_target_l, critic_optimizer_l, actor_optimizer_l, params)
        if bool(done_l):
            print(f"> Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timestep} Reward_Low: {float(episode_reward_l):.3f} Reward_High: {float(episode_reward_h):.3f}")
            if t >= policy_params.start_timestep:
                wandb.log({'episode reward low': episode_reward_l}, step=t-params.policy_params.start_timestep)
                wandb.log({'episode reward high': episode_reward_h}, step=t-params.policy_params.start_timestep)
            if params.save_video and (t % params.video_interval) == 0:  # log_video(params.env_name, actor_l)
                pass
            state, done_l = env.reset(), False
            episode_reward_l, episode_reward_h, episode_timestep = 0, 0, 0
            episode_num += 1
        # >> high-level update
        if t >= policy_params.start_timestep and t % policy_params.c == 0 and t > 0:
            target_q_h, critic_loss_h, actor_loss_h = \
                step_update_h(experience_buffer_h, policy_params.batch_size, total_it, actor_h, actor_target_h, critic_h, critic_target_h, critic_optimizer_h, actor_optimizer_h, params)
            wandb.log({'target_q low': float(target_q_l)}, step=t - params.policy_params.start_timestep)
            wandb.log({'critic_loss low': float(critic_loss_l)}, step=t - params.policy_params.start_timestep)
            wandb.log({'actor_loss low': float(actor_loss_l)}, step=t - params.policy_params.start_timestep)
            wandb.log({'target_q high': float(target_q_h)}, step=t - params.policy_params.start_timestep)
            wandb.log({'critic_loss high': float(critic_loss_h)}, step=t - params.policy_params.start_timestep)
            wandb.log({'actor_loss high': float(actor_loss_h)}, step=t - params.policy_params.start_timestep)


if __name__ == "__main__":
    gym.logger.set_level(40)
    env_name = "AntMaze"
    env = get_env(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    policy_params = ParamDict(
        seed=0,
        c=10,
        policy_noise=0.2,
        expl_noise=0.1,
        noise_clip=0.5,
        max_action=max_action,
        max_goal=10,
        discount=0.99,
        policy_freq=1,
        tau=0.005,
        lr=3e-4,
        max_timestep=int(1e5),
        start_timestep=int(25e2),
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
