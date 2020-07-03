import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import copy
import torch
from torch.nn import functional
import numpy as np
import wandb
from utils import get_env, log_video, print_cmd_hint, ParamDict, LoggerTrigger
from network import ActorTD3, CriticTD3
from experience_buffer import ExperienceBufferTD3


"""
TD3 Algorithm
    - DPG
    - A-C framework
    - experience replay
    - delayed policy update
    - target policy smoothing regularization 
"""

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


def step_update(experience_buffer, batch_size, total_it, actor_eval, actor_target, critic_eval, critic_target, critic_optimizer,
                actor_optimizer, params):
    total_it[0] += 1
    policy_params = params.policy_params
    # sample mini-batch transitions
    state, action, next_state, reward, done = experience_buffer.sample(batch_size)
    with torch.no_grad():
        # select action according to policy and add clipped noise
        noise = (torch.normal(size=action.size(), mean=0., std=policy_params.policy_noise_std) * policy_params.policy_noise_scale).clamp(-policy_params.policy_noise_clip, policy_params.policy_noise_clip)
        next_action = (actor_target(next_state) + noise).clamp(-policy_params.max_action_td3, policy_params.max_action_td3)
        # clipped double Q-learning
        q_target_1, q_target_2 = critic_target(next_state, next_action)
        q_target = torch.min(q_target_1, q_target_2)
        y = reward + (1 - done) * policy_params.discount * q_target
    # update critic q_evaluate
    q_eval_1, q_eval_2 = critic_eval(state, action)
    critic_loss = functional.mse_loss(q_eval_1, y) + functional.mse_loss(q_eval_2, y)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    # delayed policy update
    actor_loss = None
    if total_it[0] % policy_params.policy_freq == 0:
        # gradient ascent for actor policy DPG objective function J
        actor_loss = -critic_eval.q1(state, actor_eval(state)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        # soft update: critic q_target
        for param_eval, param_target in zip(critic_eval.parameters(), critic_target.parameters()):
            param_target.data.copy_(policy_params.tau * param_eval.data + (1 - policy_params.tau) * param_target.data)
        for param_eval, param_target in zip(actor_eval.parameters(), actor_target.parameters()):
            param_target.data.copy_(policy_params.tau * param_eval.data + (1 - policy_params.tau) * param_target.data)
    return y, critic_loss, actor_loss


def train(params):
    # Initialize
    policy_params = params.policy_params
    env = get_env(params.env_name)
    video_log_trigger = LoggerTrigger(start_ind=policy_params.start_timestep)
    experience_buffer = ExperienceBufferTD3(policy_params.max_timestep, params.state_dim_td3, params.action_dim_td3, params.use_cuda)
    actor_eval = ActorTD3(params.state_dim_td3, params.action_dim_td3, policy_params.max_action_td3).to(device)
    actor_target = copy.deepcopy(actor_eval)
    actor_optimizer = torch.optim.Adam(actor_eval.parameters(), lr=policy_params.lr_td3)
    critic_eval = CriticTD3(params.state_dim_td3, params.action_dim_td3).to(device)
    critic_target = copy.deepcopy(critic_eval)
    critic_optimizer = torch.optim.Adam(critic_eval.parameters(), lr=policy_params.lr_td3)

    # Set Seed
    env.seed(policy_params.seed)
    torch.manual_seed(policy_params.seed)
    np.random.seed(policy_params.seed)

    # Training Loop
    print_cmd_hint(params, "start_train")
    state, done = env.reset(), False
    episode_reward, episode_timestep, episode_num = 0, 0, 0
    total_it = [0]
    for t in range(policy_params.max_timestep):
        episode_timestep += 1
        # >>> select action: epsilon-greedy variant?
        if t < policy_params.start_timestep:
            action = env.action_space.sample()
        else:
            # target policy smoothing regularization
            max_action = policy_params.max_action_td3
            action = (actor_eval(torch.Tensor(state).to(device)).detach().cpu()
                      + np.random.normal(loc=0, scale=max_action*policy_params.expl_noise_std_scale,
                                         size=params.action_dim_td3).astype(np.float32)).clamp(-max_action, max_action)
        # observe
        next_state, reward, done, info = env.step(action)
        # store transition tuple
        experience_buffer.add(state, action, reward, next_state, done)
        # update episode logger
        state = next_state
        episode_reward = reward + episode_reward * policy_params.discount
        # TD step update
        if t >= policy_params.start_timestep:
            target_q, critic_loss, actor_loss = \
                step_update(experience_buffer, policy_params.batch_size, total_it, actor_eval, actor_target, critic_eval,
                            critic_target, critic_optimizer, actor_optimizer, params)
            wandb.log({'target_q': float(torch.mean(target_q).squeeze())}, step=t-policy_params.start_timestep)
            wandb.log({'critic_loss': float(torch.mean(critic_loss).squeeze())}, step=t-policy_params.start_timestep)
            if actor_loss is not None: wandb.log({'actor_loss': float(torch.mean(actor_loss).squeeze())}, step=t-policy_params.start_timestep)
        # start new episode
        if done:
            # record loggers
            episode_num += 1
            print(f"    > Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timestep} Reward: {episode_reward:.3f}")
            if t >= policy_params.start_timestep:
                wandb.log({'episode reward': episode_reward}, step=t-policy_params.start_timestep)
            # reset episode
            state, done = env.reset(), False
            episode_reward, episode_timestep = 0, 0
        # log video
        if params.save_video and video_log_trigger.good2log(t, params.video_interval): log_video(params.env_name, actor_target)
    print_cmd_hint(params, "end_train")
    for i in range(3):
        log_video(params.env_name, actor_target)


"""
Quick Test
"""
if __name__ == "__main__":
    env_name = "InvertedDoublePendulum-v2"
    env = get_env(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    policy_params = ParamDict(
        seed=0,
        policy_noise_std=0.1,
        policy_noise_scale=0.2,
        expl_noise_std_scale=0.1,
        policy_noise_clip=0.5,
        max_action_td3=max_action,
        discount=0.99,
        policy_freq=2,
        tau=5e-3,
        lr_td3=3e-4,
        max_timestep=int(5e4),
        start_timestep=int(25e3),
        batch_size=100
    )
    params = ParamDict(
        policy_params=policy_params,
        env_name=env_name,
        state_dim_td3=state_dim,
        action_dim_td3=action_dim,
        save_video=True,
        video_interval=int(5e3),
        use_cuda=False
    )
    wandb.init(project="ziang-hiro-new")
    train(params=params)
