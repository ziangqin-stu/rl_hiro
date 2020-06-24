"""
HIRO training process
"""
from torch import Tensor
import copy
import torch
from torch.nn import functional
import numpy as np
import wandb
from utils import get_env, log_video_hrl, ParamDict, LoggerTrigger, TimeLogger
from network import ActorLow, ActorHigh, CriticLow, CriticHigh
from experience_buffer import ExperienceBufferLow, ExperienceBufferHigh


def h_function(state, goal, next_state):
    return state + goal - next_state


def intrinsic_reward(state, goal, next_state):
    # low-level dense reward (L2 norm), provided by high-level policy
    return torch.pow(sum(torch.pow(state + goal - next_state, 2)), 1 / 2)


def done_judge_low(state, goal, next_state):
    # define low-level success: same as high-level success (L2 norm < 5, paper B.2.2)
    l2_norm = torch.pow(sum(torch.pow(state + goal - next_state, 2)), 1 / 2)
    done = (l2_norm < 5.)
    return Tensor([done])


def off_policy_correction(actor, action_sequence, state_sequence, state_dim, goal, max_goal, device):
    # initialize
    action_sequence = torch.stack(action_sequence).to(device)
    state_sequence = torch.stack(state_sequence).to(device)
    max_goal = max_goal.cpu()
    # prepare candidates
    mean = (state_sequence[-1] - state_sequence[0]).cpu()
    std = 0.5 * max_goal
    candidates = [torch.min(torch.max(Tensor(np.random.normal(loc=mean, scale=std, size=state_dim).astype(np.float32)), -max_goal), max_goal) for _ in range(8)]
    candidates.append(mean)
    candidates.append(goal.cpu())
    # select maximal
    candidates = torch.stack(candidates).to(device)
    surr_prob = [-functional.mse_loss(action_sequence, actor(state_sequence, state_sequence[0] + candidate - state_sequence)) for candidate in candidates]
    index = int(np.argmax(surr_prob))
    updated = (index != 9)
    goal_hat = candidates[index]
    return goal_hat.cpu(), updated


def step_update_l(experience_buffer, batch_size, total_it, actor_eval, actor_target, critic_eval, critic_target, critic_optimizer, actor_optimizer, params):
    policy_params = params.policy_params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if params.use_cuda else "cpu"
    total_it[0] += 1
    # sample mini-batch transitions
    state, goal, action, reward, next_state, next_goal, done = experience_buffer.sample(batch_size)
    with torch.no_grad():
        # select action according to policy and add clipped noise
        policy_noise = Tensor(np.random.normal(loc=0, scale=policy_params.sigma_l, size=params.action_dim).astype(np.float32) * policy_params.policy_noise)\
            .clamp(-policy_params.noise_clip, policy_params.noise_clip).to(device)
        next_action = (actor_target(next_state, next_goal) + policy_noise).clamp(-policy_params.max_action, policy_params.max_action)
        # clipped double Q-learning
        q_target_1, q_target_2 = critic_target(next_state, next_goal, next_action)
        q_target = torch.min(q_target_1, q_target_1)
        y = policy_params.reward_scal_l * reward + (1 - done) * policy_params.discount * q_target
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if params.use_cuda else "cpu"
    max_goal = Tensor(policy_params.max_goal).to(device)
    # sample mini-batch transitions
    state_start, goal, reward, state_end, done = experience_buffer.sample(batch_size)
    with torch.no_grad():
        # select action according to policy and add clipped noise
        policy_noise = Tensor(np.random.normal(loc=0, scale=policy_params.sigma_h, size=params.state_dim).astype(np.float32) * policy_params.policy_noise) \
            .clamp(-policy_params.noise_clip, policy_params.noise_clip).to(device)
        next_goal = torch.min(torch.max(actor_target(state_end) + policy_noise, -max_goal), max_goal)
        # clipped double Q-learning
        q_target_1, q_target_2 = critic_target(state_end, next_goal)
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
    # 1. Initialization
    # 1.1 utils
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if params.use_cuda else "cpu"
    policy_params = params.policy_params
    max_goal = Tensor(policy_params.max_goal).to(device)
    state_dim = params.state_dim
    action_dim = params.action_dim
    max_action = policy_params.max_action
    expl_noise = policy_params.expl_noise
    discount = policy_params.discount
    max_timestep = policy_params.max_timestep
    c = policy_params.c
    start_timestep = policy_params.start_timestep
    log_interval = params.log_interval
    env_name = params.env_name
    video_interval = params.video_interval
    save_video = params.save_video
    batch_size = policy_params.batch_size
    env = get_env(params.env_name)
    video_log_trigger = LoggerTrigger(start_ind=policy_params.start_timestep)
    state_print_trigger = LoggerTrigger(start_ind=policy_params.start_timestep)
    time_logger = TimeLogger()
    # 1.2 rl components
    # 1.2.1 high-level
    actor_eval_h = ActorHigh(state_dim, max_goal).to(device)
    actor_target_h = copy.deepcopy(actor_eval_h).to(device).to(device)
    actor_optimizer_h = torch.optim.Adam(actor_eval_h.parameters(), lr=policy_params.actor_lr)
    critic_eval_h = CriticHigh(state_dim).to(device)
    critic_target_h = copy.deepcopy(critic_eval_h).to(device).to(device)
    critic_optimizer_h = torch.optim.Adam(critic_eval_h.parameters(), lr=policy_params.critic_lr)
    experience_buffer_h = ExperienceBufferHigh(policy_params.max_timestep, state_dim, params.use_cuda)
    # 1.2.1 low-level
    actor_eval_l = ActorLow(state_dim, action_dim, policy_params.max_action).to(device)
    actor_target_l = copy.deepcopy(actor_eval_l).to(device)
    actor_optimizer_l = torch.optim.Adam(actor_eval_l.parameters(), lr=policy_params.actor_lr)
    critic_eval_l = CriticLow(state_dim, action_dim).to(device)
    critic_target_l = copy.deepcopy(critic_eval_l).to(device)
    critic_optimizer_l = torch.optim.Adam(critic_eval_l.parameters(), lr=policy_params.critic_lr)
    experience_buffer_l = ExperienceBufferLow(policy_params.max_timestep, state_dim, action_dim, params.use_cuda)
    # 1.3 set seeds
    env.seed(policy_params.seed)
    torch.manual_seed(policy_params.seed)
    np.random.seed(policy_params.seed)

    # 2. Training Algorithm (TD3)
    # 2.1 initialize
    time_logger.time_spent()
    total_it = [0]
    episode_reward_l, episode_reward_h, episode_num_l, episode_num_h, episode_timestep_l, episode_timestep_h, = 0, 0, 0, 0, 1, 1
    state, done_l = Tensor(env.reset()).to(device), False
    goal = torch.randn_like(state)
    state_sequence, goal_sequence, action_sequence = [], [], []
    # 2.2 training loop
    for t in range(max_timestep):
        # 2.2.1 sample low-level action
        if t < start_timestep:
            action = env.action_space.sample()
        else:
            expl_noise_action = np.random.normal(loc=0, scale=max_action*expl_noise, size=action_dim).astype(np.float32)
            action = (actor_eval_l(state, goal).detach().cpu() + expl_noise_action).clamp(-max_action, max_action).squeeze()
        # 2.2.2 interact environment
        next_state, reward, done_h, info = env.step(action)
        next_state, action, reward, done_h = Tensor(next_state).to(device), Tensor(action), Tensor([reward]), Tensor([done_h])
        # 2.2.3 collect low-level steps
        intri_reward = intrinsic_reward(state, goal, next_state)
        done_l = done_judge_low(state, goal, next_state)
        next_goal = h_function(state, goal, next_state)
        experience_buffer_l.add(state, goal, action, intri_reward, next_state, next_goal, done_l)
        # 2.2.4 update low-level loop
        state = next_state
        episode_reward_l = intri_reward + discount * episode_reward_l
        episode_reward_h = reward + discount * episode_reward_h
        # 2.2.5 collect low-level perform sequence
        state_sequence.append(state)
        action_sequence.append(action)
        goal_sequence.append(goal)

        if (t + 1) % c == 0 and t > 0:
            # 2.2.6 sample high-level goal
            if t < start_timestep:
                new_goal = (torch.randn_like(state) * max_goal)
                new_goal = torch.min(torch.max(new_goal, -max_goal), max_goal)
            else:
                expl_noise_goal = np.random.normal(loc=0, scale=max_goal.cpu()*expl_noise, size=state_dim).astype(np.float32)
                new_goal = (actor_eval_h(state_sequence[0].to(device)).detach().cpu() + expl_noise_goal).squeeze().to(device)
                new_goal = torch.min(torch.max(new_goal, -max_goal), max_goal)
            # > off-policy correction
            goal_hat, updated = off_policy_correction(actor_target_l, action_sequence, state_sequence, state_dim, goal_sequence[0], max_goal, device)
            # 2.2.7 collect high-level steps
            experience_buffer_h.add(state_sequence[0], goal_hat, episode_reward_h, state, done_h)
            if state_print_trigger.good2log(t, 200):
                print("\n        > state: {}".format(state_sequence[0]))
                print("        > goal-1: {}".format(goal_sequence[0]))
                if updated: print("        > goal_hat: {}".format(goal_hat))
                else: print("        > goal_hat not updated")
                print("        > action-1: {}\n".format(action_sequence[0]))
            # 2.2.8 update high-level loop
            goal = new_goal
            state_sequence, action_sequence, goal_sequence = [], [], []

        # 2.2.9 update networks
        # 2.2.9.1 low-level update
        if t >= start_timestep:
            target_q_l, critic_loss_l, actor_loss_l = \
                step_update_l(experience_buffer_l, batch_size, total_it, actor_eval_l, actor_target_l, critic_eval_l, critic_target_l, critic_optimizer_l, actor_optimizer_l, params)
        # 2.2.9.2 high-level update
        if t >= start_timestep and t % c == 0 and t > 0:
            target_q_h, critic_loss_h, actor_loss_h = \
                step_update_h(experience_buffer_h, batch_size, total_it, actor_eval_h, actor_target_h, critic_eval_h, critic_target_h, critic_optimizer_h, actor_optimizer_h, params)
        # 2.2.10 logger record
        if t >= start_timestep and t % log_interval == 0 and t > 0:
            wandb.log({'target_q low': torch.mean(target_q_l).squeeze()}, step=t-start_timestep)
            wandb.log({'critic_loss low': torch.mean(critic_loss_l).squeeze()}, step=t-start_timestep)
            if actor_loss_l is not None: wandb.log({'actor_loss low': torch.mean(actor_loss_l).squeeze()}, step=t-start_timestep)
            wandb.log({'target_q high': torch.mean(target_q_h).squeeze()}, step=t-start_timestep)
            wandb.log({'critic_loss high': torch.mean(critic_loss_h).squeeze()}, step=t-start_timestep)
            if actor_loss_h is not None: wandb.log({'actor_loss high': torch.mean(actor_loss_h).squeeze()}, step=t-start_timestep)
        # 2.2.11 start new episode
        if bool(done_l) or episode_timestep_l >= c:
            print(f"    > Total T: {t + 1} Episode_Low Num: {episode_num_l + 1} Episode_Low T: {episode_timestep_l} Reward_Low: {float(episode_reward_l):.3f} Reward_High: {float(episode_reward_h):.3f}")
            if t >= start_timestep:
                wandb.log({'episode reward low': episode_reward_l}, step=t-start_timestep)
                wandb.log({'episode reward high': episode_reward_h}, step=t-start_timestep)
            if save_video and video_log_trigger.good2log(t, video_interval):
                log_video_hrl(env_name, actor_target_l, actor_target_h, params)
                time_logger.sps(t)
                time_logger.time_spent()
            episode_reward_l, episode_reward_h, episode_timestep_l = 0, 0, 0
            episode_num_l += 1
        if bool(done_h):
            episode_num_h += 1
            print(f"    >>> Total T: {t + 1} Episode_High Num: {episode_num_h + 1} Episode_High T: {episode_timestep_h} Reward_High: {float(episode_reward_h):.3f}")
            state, done_h = Tensor(env.reset()), False
            episode_timestep_h = 0
        # 2.2.12 update training loop
        episode_timestep_l += 1
        episode_timestep_h += 1
    # 2.3 log training result
    for i in range(3):
        log_video_hrl(env_name, actor_target_l, actor_target_h, params)


if __name__ == "__main__":
    # gym.logger.set_level(40)
    env_name = "AntPush"
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
        seed=123,
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
        start_timestep=int(3e4),
        batch_size=100
    )
    params = ParamDict(
        policy_params=policy_params,
        env_name=env_name,
        state_dim=state_dim,
        action_dim=action_dim,
        video_interval=int(1e4),
        log_interval=1,
        save_video=True,
        use_cuda=True
    )

    wandb.init(project="ziang-hiro-new")
    train(params=params)
