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
from utils import get_env, log_video_hrl, ParamDict, VideoLoggerTrigger
from network import ActorLow, ActorHigh, CriticLow, CriticHigh
from experience_buffer import ExperienceBufferLow, ExperienceBufferHigh


def h_function(state, goal, next_state):
    return state + goal - next_state


def intrinsic_reward(state, goal, next_state):
    state, goal, next_state = torch.Tensor(state), torch.Tensor(goal), torch.Tensor(next_state)
    return -torch.sum(torch.pow(state + goal - next_state, 2))


def done_judge_low(state, goal, next_state):
    # return torch.Tensor(state).equal(torch.Tensor(goal))
    done = torch.abs(intrinsic_reward(state, goal, next_state)) < 1.
    return torch.Tensor([done])


def off_policy_correction(actor, action_sequence, state_sequence, goal, params):
    # initialize
    policy_params = params.policy_params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if params.use_cuda else "cpu"
    action_sequence = torch.stack(action_sequence)
    state_sequence = torch.stack(state_sequence)
    max_goal = torch.Tensor(policy_params.max_goal)
    # prepare candidates
    mean = state_sequence[-1] - state_sequence[0]
    std = 0.5 * torch.Tensor(policy_params.max_goal)
    x = np.random.normal(loc=mean, scale=std, size=params.state_dim).astype(np.float32)
    candidates = [torch.min(torch.max(torch.Tensor(np.random.normal(loc=mean, scale=std, size=params.state_dim).astype(np.float32)), -max_goal), max_goal) for i in range(8)]
    candidates.append(state_sequence[-1] - state_sequence[0])
    candidates.append(goal)
    # select maximal
    candidates = torch.stack(candidates).to(device)
    action_sequence = action_sequence.to(device)
    state_sequence = state_sequence.to(device)
    actor(state_sequence, state_sequence[0] + candidates[0] - state_sequence)
    probability = [-functional.mse_loss(action_sequence, actor(state_sequence, state_sequence[0] + candidate - state_sequence)) for candidate in candidates]
    goal_hat = candidates[np.argmax(probability)]
    return torch.Tensor(goal_hat)


def step_update_l(experience_buffer, batch_size, total_it, actor_eval, actor_target, critic_eval, critic_target, critic_optimizer, actor_optimizer, params):
    policy_params = params.policy_params
    total_it[0] += 1
    # sample mini-batch transitions
    state, goal, action, reward, next_state, next_goal, done = experience_buffer.sample(batch_size)
    with torch.no_grad():
        # select action according to policy and add clipped noise
        policy_noise = torch.Tensor(np.random.normal(loc=0, scale=policy_params.sigma_l, size=params.action_dim).astype(np.float32) * policy_params.policy_noise)\
            .clamp(-policy_params.noise_clip, policy_params.noise_clip)
        next_action = (actor_target(next_state, next_goal) + policy_noise).clamp(-policy_params.max_action, policy_params.max_action)
        # clipped double Q-learning
        q_target_1, q_target_2 = critic_target(next_state, next_goal, next_action)
        q_target = torch.min(q_target_1, q_target_1)
        y = policy_params.reward_scal_h * reward + (1 - done) * policy_params.discount * q_target
    # update critic q_evaluate
    q_eval_1, q_eval_2 = critic_eval(state, goal, action)
    critic_loss = functional.mse_loss(q_eval_1, y) + functional.mse_loss(q_eval_2, y)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    # delayed policy update
    actor_loss = None
    if total_it[0] % policy_params.policy_freq == 0:
        # compute actor loss
        actor_loss = -critic_eval.q1(state, goal, actor_eval(state, goal)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        # soft update: critic q_target
        for param, target_param in zip(critic_eval.parameters(), critic_target.parameters()):
            target_param.data.copy_(policy_params.tau * param.data + (1 - policy_params.tau) * target_param.data)
        for param, target_param in zip(actor_eval.parameters(), actor_target.parameters()):
            target_param.data.copy_(policy_params.tau * param.data + (1 - policy_params.tau) * target_param.data)
        actor_loss = actor_loss.detach()
    return y.detach(), critic_loss.detach(), actor_loss


def step_update_h(experience_buffer, batch_size, total_it, actor_eval, actor_target, critic_eval, critic_target, critic_optimizer, actor_optimizer, params):
    policy_params = params.policy_params
    max_goal = torch.Tensor(policy_params.max_goal)
    # sample mini-batch transitions
    state_start, goal, reward, state_end, done = experience_buffer.sample(batch_size)
    with torch.no_grad():
        # select action according to policy and add clipped noise
        policy_noise = torch.Tensor(np.random.normal(loc=0, scale=policy_params.sigma_h, size=params.state_dim).astype(np.float32) * policy_params.policy_noise) \
            .clamp(-policy_params.noise_clip, policy_params.noise_clip)
        new_goal = torch.min(torch.max(actor_target(state_end) + policy_noise, -max_goal), max_goal)
        # clipped double Q-learning
        q_target_1, q_target_2 = critic_target(state_end, new_goal)
        target_q = torch.min(q_target_1, q_target_2)
        y = policy_params.reward_scal_h * reward + (1 - done) * policy_params.discount * target_q
    # update critic q_evaluate
    q_eval_1, q_eval_2 = critic_eval(state_start, goal)
    critic_loss = functional.mse_loss(q_eval_1, y) + functional.mse_loss(q_eval_2, y)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    # delayed policy updates
    actor_loss = None
    if int(total_it[0] / policy_params.c) % policy_params.policy_freq == 0:
        # compute actor loss
        actor_loss = -critic_eval.q1(state_start, actor_eval(state_start)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        # soft update: critic q_target
        for param, target_param in zip(critic_eval.parameters(), critic_target.parameters()):
            target_param.data.copy_(policy_params.tau * param.data + (1 - policy_params.tau) * target_param.data)
        for param, target_param in zip(actor_eval.parameters(), actor_target.parameters()):
            target_param.data.copy_(policy_params.tau * param.data + (1 - policy_params.tau) * target_param.data)
        actor_loss = actor_loss.detach()
    return y.detach(), critic_loss.detach(), actor_loss


def train(params):
    # Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if params.use_cuda else "cpu"
    policy_params = params.policy_params
    env = get_env(params.env_name)
    video_log_trigger = VideoLoggerTrigger(start_ind=policy_params.start_timestep)

    actor_eval_h = ActorHigh(params.state_dim, policy_params.max_action).to(device)
    actor_target_h = copy.deepcopy(actor_eval_h)
    actor_optimizer_h = torch.optim.Adam(actor_eval_h.parameters(), lr=policy_params.actor_lr)
    critic_eval_h = CriticHigh(params.state_dim).to(device)
    critic_target_h = copy.deepcopy(critic_eval_h)
    critic_optimizer_h = torch.optim.Adam(critic_eval_h.parameters(), lr=policy_params.critic_lr)
    experience_buffer_h = ExperienceBufferHigh(policy_params.max_timestep, params.state_dim, params.use_cuda)

    actor_eval_l = ActorLow(params.state_dim, params.action_dim, policy_params.max_action).to(device)
    actor_target_l = copy.deepcopy(actor_eval_l)
    actor_optimizer_l = torch.optim.Adam(actor_eval_l.parameters(), lr=policy_params.actor_lr)
    critic_eval_l = CriticLow(params.state_dim, params.action_dim).to(device)
    critic_target_l = copy.deepcopy(critic_eval_l)
    critic_optimizer_l = torch.optim.Adam(critic_eval_l.parameters(), lr=policy_params.critic_lr)
    experience_buffer_l = ExperienceBufferLow(policy_params.max_timestep, params.state_dim, params.action_dim, params.use_cuda)

    # Set Seed
    env.seed(policy_params.seed)
    torch.manual_seed(policy_params.seed)
    np.random.seed(policy_params.seed)

    # Training Algorithm (TD3)
    total_it = [0]
    episode_reward_l, episode_reward_h, episode_timestep_l, episode_timestep_h, episode_num_l, episode_num_h = 0, 0, 0, 0, 0, 0
    state, done_l = torch.Tensor(env.reset()), False
    goal = torch.Tensor(torch.randn_like(state))
    state_sequence, goal_sequence, action_sequence, next_state_sequence, next_goal_sequence, reward_sequence, done_h_sequence = [], [], [], [], [], [], []
    for t in range(policy_params.max_timestep):
        episode_timestep_l += 1
        episode_timestep_h += 1
        # > low-level collection
        # >> sample action: SARSA style
        max_action = policy_params.max_action
        if t < policy_params.start_timestep:
            action = env.action_space.sample()
        else:
            expl_noise_action = np.random.normal(loc=0, scale=max_action*policy_params.expl_noise, size=params.action_dim).astype(np.float32)
            action = (actor_eval_l(state.to(device), goal.to(device)).detach().cpu() + expl_noise_action).clamp(-max_action, max_action).squeeze()
        # >> perform action
        next_state, reward, done_h, info = env.step(action)
        intri_reward = intrinsic_reward(state, goal, next_state)
        # >> collect step_low
        next_state, state, action, reward, intri_reward, goal = \
            torch.Tensor(next_state), torch.Tensor(state), torch.Tensor(action), torch.Tensor([reward]), torch.Tensor([intri_reward]), torch.Tensor(goal)
        done_l = done_judge_low(state, goal, next_state)
        next_goal = h_function(state, goal, next_state)
        experience_buffer_l.add(state, goal, action, intri_reward, next_state, next_goal, done_l)
        state = next_state
        # >> update loggers
        episode_reward_l += intri_reward
        episode_reward_h += reward
        # >> collect low-level state sequence
        done_h = torch.Tensor([done_h])
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
            max_goal = torch.Tensor(policy_params.max_goal)
            if t < policy_params.start_timestep:
                new_goal = (torch.randn_like(state) * max_goal)
                new_goal = torch.min(torch.max(new_goal, -max_goal), max_goal)
            else:
                expl_noise_goal = np.random.normal(loc=0, scale=max_goal*policy_params.expl_noise, size=params.state_dim).astype(np.float32)
                new_goal = (actor_eval_h(state_sequence[0].to(device)).detach().cpu() + expl_noise_goal).squeeze()
                new_goal = torch.min(torch.max(new_goal, -max_goal), max_goal)
            goal_hat = off_policy_correction(actor_target_l, action_sequence, state_sequence, goal_sequence[0], params)
            # goal_hat = goal_sequence[0]
            # >> collect step-high
            experience_buffer_h.add(state_sequence[0], goal_hat, episode_reward_h, state, done_h)
            goal = new_goal
            state_sequence, goal_sequence, action_sequence, next_state_sequence, next_goal_sequence, reward_sequence, done_h_sequence = [], [], [], [], [], [], []

        # > update networks
        # >> low-level update
        if t >= policy_params.start_timestep:
            target_q_l, critic_loss_l, actor_loss_l = step_update_l(experience_buffer_l, policy_params.batch_size, total_it, actor_eval_l, actor_target_l,
                          critic_eval_l, critic_target_l, critic_optimizer_l, actor_optimizer_l, params)
        # >> high-level update
        if t >= policy_params.start_timestep and t % policy_params.c == 0 and t > 0:
            target_q_h, critic_loss_h, actor_loss_h = \
                step_update_h(experience_buffer_h, policy_params.batch_size, total_it, actor_eval_h, actor_target_h, critic_eval_h, critic_target_h, critic_optimizer_h, actor_optimizer_h, params)
        # >> record logger
        if t >= policy_params.start_timestep and t % params.log_interval == 0 and t > 0:
            wandb.log({'target_q low': torch.mean(target_q_l).squeeze()}, step=t-params.policy_params.start_timestep)
            wandb.log({'critic_loss low': torch.mean(critic_loss_l).squeeze()}, step=t-params.policy_params.start_timestep)
            if actor_loss_l is not None: wandb.log({'actor_loss low': torch.mean(actor_loss_l).squeeze()}, step=t-params.policy_params.start_timestep)
            wandb.log({'target_q high': torch.mean(target_q_h).squeeze()}, step=t-params.policy_params.start_timestep)
            wandb.log({'critic_loss high': torch.mean(critic_loss_h).squeeze()}, step=t-params.policy_params.start_timestep)
            if actor_loss_h is not None: wandb.log({'actor_loss high': torch.mean(actor_loss_h).squeeze()}, step=t-params.policy_params.start_timestep)
        # >> start new episode if done
        if bool(done_l) or episode_timestep_l >= policy_params.c:
            print(f"    > Total T: {t + 1} Episode_Low Num: {episode_num_l + 1} Episode_Low T: {episode_timestep_l} Reward_Low: {float(episode_reward_l):.3f} Reward_High: {float(episode_reward_h):.3f}")
            if t >= policy_params.start_timestep:
                wandb.log({'episode reward low': episode_reward_l}, step=t-params.policy_params.start_timestep)
                wandb.log({'episode reward high': episode_reward_h}, step=t-params.policy_params.start_timestep)
            if params.save_video and video_log_trigger.good2log(t, params.video_interval):  log_video_hrl(params.env_name, actor_target_l, actor_target_h, params)
            episode_reward_l, episode_reward_h, episode_timestep_l = 0, 0, 0
            episode_num_l += 1
        if bool(done_h):
            episode_num_h += 1
            print(f"    >>> Total T: {t + 1} Episode_High Num: {episode_num_h + 1} Episode_High T: {episode_timestep_h} Reward_High: {float(episode_reward_h):.3f}")
            state, done_h = torch.Tensor(env.reset()), False
            episode_timestep_h = 0


if __name__ == "__main__":
    # gym.logger.set_level(40)
    env_name = "AntMaze"
    env = get_env(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    max_goal = [
                10., 10., .5,                                   # 0-2
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,     # 3-13
                30, 30, 30, 30, 30, 30, 30,                     # 14-20
                30, 30, 30, 30, 30, 30, 30, 30,                 # 21-28
                1.]                                             # 29
    policy_params = ParamDict(
        seed=0,
        c=10,
        policy_noise=0.2,
        expl_noise=0.1,
        noise_clip=0.5,
        sigma_l=1.,
        sigma_h=1.,
        max_action=max_action,
        max_goal=max_goal,
        discount=0.99,
        policy_freq=2,
        tau=5e-3,
        actor_lr=1e-4,
        critic_lr=1e-3,
        reward_scal_l=1.,
        reward_scal_h=.1,
        sigma_g=1.,
        max_timestep=int(1e7),
        start_timestep=int(3e2),
        batch_size=100
    )
    params = ParamDict(
        policy_params=policy_params,
        env_name=env_name,
        state_dim=state_dim,
        action_dim=action_dim,
        video_interval=int(1e3),
        log_interval=1,
        save_video=True,
        use_cuda=False
    )
    wandb.init(project="ziang-hiro")
    train(params=params)
