"""
HIRO training process
"""
import os
import datetime
from torch import Tensor
import copy
import torch
from torch.nn import functional
import numpy as np
import wandb
from utils import get_env, log_video_hrl_debug, ParamDict, LoggerTrigger, TimeLogger
from network import ActorLow, ActorHigh, CriticLow, CriticHigh
from experience_buffer import ExperienceBufferLow, ExperienceBufferHigh


def initialize_params(params, device):
    policy_params = params.policy_params
    env_name = params.env_name
    state_dim = params.state_dim
    max_goal = Tensor(policy_params.max_goal).to(device)
    action_dim = params.action_dim
    max_action = policy_params.max_action
    expl_noise_std_l = policy_params.expl_noise_std_l
    # expl_noise_std_l = policy_params.expl_noise_l
    expl_noise_std_h = policy_params.expl_noise_std_h
    # expl_noise_std_h = policy_params.expl_noise_h
    c = policy_params.c
    max_timestep = policy_params.max_timestep
    start_timestep = policy_params.start_timestep
    discount = policy_params.discount
    batch_size = policy_params.batch_size
    log_interval = params.log_interval
    checkpoint_interval = params.checkpoint_interval
    save_video = params.save_video
    video_interval = params.video_interval
    env = get_env(params.env_name)
    video_log_trigger = LoggerTrigger(start_ind=policy_params.start_timestep)
    state_print_trigger = LoggerTrigger(start_ind=policy_params.start_timestep)
    checkpoint_logger = LoggerTrigger(start_ind=policy_params.start_timestep)
    time_logger = TimeLogger()
    return policy_params, env_name, state_dim, max_goal, action_dim, max_action, expl_noise_std_l, expl_noise_std_h,\
           c, max_timestep, start_timestep, discount, batch_size, \
           log_interval, checkpoint_interval, save_video, video_interval, env, video_log_trigger, state_print_trigger, checkpoint_logger, time_logger


def save_checkpoint(step, actor_l, critic_l, actor_optimizer_l, critic_optimizer_l, exp_l, actor_h, critic_h, actor_optimizer_h, critic_optimizer_h, exp_h, logger, params, file_path=None, file_name=None):
    if file_name is None:
        time = datetime.datetime.now()
        file_name = "hiro-{}_{}-it({})-[{}].tar".format(env_name.lower(), params.prefix, step, time)
    if file_path is None:
        file_path = os.path.join(".", "save", "model", file_name)
    print("\n    > saving training checkpoint...")
    torch.save({
        'step': step,
        'params': params,
        'logger': logger,
        'actor_l': actor_l.state_dict(),
        'critic_l': critic_l.state_dict(),
        'actor_optimizer_l': actor_optimizer_l.state_dict(),
        'critic_optimizer_l': critic_optimizer_l.state_dict(),
        'exp_l': exp_l,
        'actor_h': actor_h.state_dict(),
        'critic_h': critic_h.state_dict(),
        'actor_optimizer_h': actor_optimizer_h.state_dict(),
        'critic_optimizer_h': critic_optimizer_h.state_dict(),
        'exp_h': exp_h
    }, file_path)
    print("    > saved checkpoint to: {}\n".format(file_path))


def load_checkpoint(file_name):
    try:
        # load
        print("\n    > loading training checkpoint...")
        file_path = os.path.join(".", "save", "model", file_name)
        checkpoint = torch.load(file_path)
        print("\n    > checkpoint file loaded! parsing data...")
        params = checkpoint['params']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if params.use_cuda else "cpu"

        # initialize
        actor_eval_l = ActorLow(state_dim, action_dim, policy_params.max_action).to(device)
        actor_optimizer_l = torch.optim.Adam(actor_eval_l.parameters(), lr=policy_params.actor_lr)
        critic_eval_l = CriticLow(state_dim, action_dim).to(device)
        critic_optimizer_l = torch.optim.Adam(critic_eval_l.parameters(), lr=policy_params.critic_lr)

        actor_eval_h = ActorHigh(state_dim, max_goal, device).to(device)
        actor_optimizer_h = torch.optim.Adam(actor_eval_h.parameters(), lr=policy_params.actor_lr)
        critic_eval_h = CriticHigh(state_dim).to(device)
        critic_optimizer_h = torch.optim.Adam(critic_eval_h.parameters(), lr=policy_params.critic_lr)
        experience_buffer_h = ExperienceBufferHigh(policy_params.max_timestep, state_dim, params.use_cuda)

        # unpack
        step = checkpoint['step']
        logger = checkpoint['logger']

        actor_eval_l.load_state_dict(checkpoint['actor_l'])
        critic_eval_l.load_state_dict(checkpoint['critic_l'])
        actor_optimizer_l.load_state_dict((checkpoint['actor_optimizer_l']))
        critic_optimizer_l.load_state_dict(checkpoint['critic_optimizer_l'])
        experience_buffer_l = checkpoint['exp_l']

        actor_eval_h.load_state_dict(checkpoint['actor_h'])
        critic_eval_h.load_state_dict(checkpoint['critic_h'])
        actor_optimizer_h.load_state_dict((checkpoint['actor_optimizer_h']))
        critic_optimizer_h.load_state_dict(checkpoint['critic_optimizer_h'])
        experience_buffer_h = checkpoint['exp_h']

        actor_target_l = copy.deepcopy(actor_eval_l).to(device)
        critic_target_l = copy.deepcopy(critic_eval_l).to(device)
        actor_target_h = copy.deepcopy(actor_eval_h).to(device)
        critic_target_h = copy.deepcopy(critic_eval_h).to(device)

        actor_eval_l.train(), actor_target_l.train(), critic_eval_l.train(), critic_target_l.train()
        actor_eval_h.train(), actor_target_h.train(), critic_eval_h.train(), critic_target_h.train()
        print("    > checkpoint resume success!")
    except Exception as e:
        print(e)
    return step, params, device, logger, \
           actor_eval_l, actor_target_l, critic_eval_l, critic_target_l, actor_optimizer_l, critic_optimizer_l, experience_buffer_l, \
           actor_eval_h, actor_target_h, critic_eval_h, critic_target_h, actor_optimizer_h, critic_optimizer_h, experience_buffer_h


def create_rl_components(params, device):
    # low-level
    actor_eval_l = ActorLow(state_dim, action_dim, policy_params.max_action).to(device)
    actor_target_l = copy.deepcopy(actor_eval_l).to(device)
    actor_optimizer_l = torch.optim.Adam(actor_eval_l.parameters(), lr=policy_params.actor_lr)
    critic_eval_l = CriticLow(state_dim, action_dim).to(device)
    critic_target_l = copy.deepcopy(critic_eval_l).to(device)
    critic_optimizer_l = torch.optim.Adam(critic_eval_l.parameters(), lr=policy_params.critic_lr)
    experience_buffer_l = ExperienceBufferLow(policy_params.max_timestep, state_dim, action_dim, params.use_cuda)
    # high-level
    actor_eval_h = ActorHigh(state_dim, max_goal, device).to(device)
    actor_target_h = copy.deepcopy(actor_eval_h).to(device)
    actor_optimizer_h = torch.optim.Adam(actor_eval_h.parameters(), lr=policy_params.actor_lr)
    critic_eval_h = CriticHigh(state_dim).to(device)
    critic_target_h = copy.deepcopy(critic_eval_h).to(device)
    critic_optimizer_h = torch.optim.Adam(critic_eval_h.parameters(), lr=policy_params.critic_lr)
    experience_buffer_h = ExperienceBufferHigh(policy_params.max_timestep, state_dim, params.use_cuda)

    return actor_eval_l, actor_target_l, actor_optimizer_l, critic_eval_l, critic_target_l, critic_optimizer_l, experience_buffer_l, \
           actor_eval_h, actor_target_h, actor_optimizer_h, critic_eval_h, critic_target_h, critic_optimizer_h, experience_buffer_h


def h_function(state, goal, next_state):
    # return next goal
    return state + goal - next_state


def intrinsic_reward(state, goal, next_state):
    # low-level dense reward (L2 norm), provided by high-level policy
    # return -torch.pow(sum(torch.pow(state[:-1] + goal[:-1] - next_state[:-1], 2)), 1 / 2)
    return -torch.pow(sum(torch.pow(state + goal - next_state, 2)), 1 / 2)


def done_judge_low(state, goal, next_state):
    # define low-level success: same as high-level success (L2 norm < 5, paper B.2.2)
    l2_norm = torch.pow(sum(torch.pow(state + goal - next_state, 2)), 1 / 2)
    done = (l2_norm <= 2.5)
    return Tensor([done])


def success_judge(state):
    location = Tensor(state[:2])
    goal_state = Tensor([0, 19])
    l2_norm = torch.pow(sum(torch.pow(location + goal_state, 2)), 1 / 2)
    done = (l2_norm <= 5.)
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
    # initialize
    policy_params = params.policy_params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if params.use_cuda else "cpu"
    total_it[0] += 1
    # sample mini-batch transitions
    state, goal, action, reward, next_state, next_goal, done = experience_buffer.sample(batch_size)
    with torch.no_grad():
        # select action according to policy and add clipped noise
        policy_noise = Tensor(np.random.normal(loc=0, scale=policy_params.policy_noise_std, size=params.action_dim).astype(np.float32) * policy_params.policy_noise_scale) \
            .clamp(-policy_params.policy_noise_clip, policy_params.policy_noise_clip).to(device)
        next_action = (actor_target(next_state, next_goal) + policy_noise).clamp(-policy_params.max_action, policy_params.max_action)
        # clipped double Q-learning
        q_target_1, q_target_2 = critic_target(next_state, next_goal, next_action)
        q_target = torch.min(q_target_1, q_target_2)
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
        for param_eval, param_target in zip(critic_eval.parameters(), critic_target.parameters()):
            param_target.data.copy_(policy_params.tau * param_eval.data + (1 - policy_params.tau) * param_target.data)
        for param_eval, param_target in zip(actor_eval.parameters(), actor_target.parameters()):
            param_target.data.copy_(policy_params.tau * param_eval.data + (1 - policy_params.tau) * param_target.data)
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
        policy_noise = Tensor(np.random.normal(loc=0, scale=policy_params.policy_noise_std, size=params.state_dim).astype(np.float32) * policy_params.policy_noise_scale) \
            .clamp(-policy_params.policy_noise_clip, policy_params.policy_noise_clip).to(device)
        next_goal = torch.min(torch.max(actor_target(state_end) + policy_noise, -max_goal), max_goal)
        # clipped double Q-learning
        q_target_1, q_target_2 = critic_target(state_end, next_goal)
        q_target = torch.min(q_target_1, q_target_2)
        y = policy_params.reward_scal_h * reward + (1 - done) * policy_params.discount * q_target
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
        for param_eval, param_target in zip(critic_eval.parameters(), critic_target.parameters()):
            param_target.data.copy_(policy_params.tau * param_eval.data + (1 - policy_params.tau) * param_target.data)
        for param_eval, param_target in zip(actor_eval.parameters(), actor_target.parameters()):
            param_target.data.copy_(policy_params.tau * param_eval.data + (1 - policy_params.tau) * param_target.data)
        actor_loss = actor_loss.detach()
    return y.detach(), critic_loss.detach(), actor_loss


def train(params):
    # 1. Initialization
    # 1.1 rl components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if params.use_cuda else "cpu"
    if params.checkpoint is None:
        actor_eval_l, actor_target_l, actor_optimizer_l, critic_eval_l, critic_target_l, critic_optimizer_l, experience_buffer_l, \
        actor_eval_h, actor_target_h, actor_optimizer_h, critic_eval_h, critic_target_h, critic_optimizer_h, experience_buffer_h = create_rl_components(params, device)
        step = 0
    else:
        step, params, device, [time_logger, state_print_trigger, video_log_trigger, checkpoint_logger], \
        actor_eval_l, actor_target_l, critic_eval_l, critic_target_l, actor_optimizer_l, critic_optimizer_l, experience_buffer_l, \
        actor_eval_h, actor_target_h, critic_eval_h, critic_target_h, actor_optimizer_h, critic_optimizer_h, experience_buffer_h = load_checkpoint(params.checkpoint)
    # 1.2 utils
    policy_params, env_name, state_dim, max_goal, action_dim, max_action, expl_noise_std_l, expl_noise_std_h,\
    c, max_timestep, start_timestep, discount, batch_size, \
    log_interval, checkpoint_interval, save_video, video_interval, env, video_log_trigger, state_print_trigger, checkpoint_logger, time_logger = initialize_params(params, device)
    # 1.3 set seeds
    env.seed(policy_params.seed)
    torch.manual_seed(policy_params.seed)
    np.random.seed(policy_params.seed)

    # 2. Training Algorithm (TD3)
    # 2.1 initialize
    time_logger.time_spent()
    total_it = [0]
    episode_reward_l, episode_reward_h, episode_num_l, episode_num_h, episode_timestep_l, episode_timestep_h, = 0, 0, 0, 0, 1, 1
    state = Tensor(env.reset()).to(device)
    goal = torch.randn_like(state)
    state_sequence, goal_sequence, action_sequence, intri_reward_sequence = [], [], [], []
    # 2.2 training loop
    for t in range(step, max_timestep):
        # 2.2.1 sample low-level action
        if t < start_timestep:
            action = env.action_space.sample()
        else:
            # expl_noise_action = np.random.normal(loc=0, scale=max_action * expl_noise_std_l, size=action_dim).astype(np.float32)
            expl_noise_action = np.random.normal(loc=0, scale=expl_noise_std_l, size=action_dim).astype(np.float32)
            action = (actor_eval_l(state, goal).detach().cpu() + expl_noise_action).clamp(-max_action, max_action).squeeze()
        # 2.2.2 interact environment
        next_state, reward_h, _, info = env.step(action)
        done_h = success_judge(next_state)
        next_state, action, reward_h, done_h = Tensor(next_state).to(device), Tensor(action), Tensor([reward_h]), Tensor([done_h])
        # 2.2.3 collect low-level steps
        intri_reward = intrinsic_reward(state, goal, next_state)
        done_l = done_judge_low(state, goal, next_state)
        next_goal = h_function(state, goal, next_state)
        experience_buffer_l.add(state, goal, action, intri_reward, next_state, next_goal, done_l)
        # 2.2.4 update low-level loop
        state = next_state
        episode_reward_l += intri_reward
        episode_reward_h += reward_h
        # 2.2.5 collect low-level experience sequence
        state_sequence.append(state)
        action_sequence.append(action)
        intri_reward_sequence.append(intri_reward)
        goal_sequence.append(goal)

        if (t + 1) % c == 0 and t > 0:
            # 2.2.6 sample high-level goal
            if t < start_timestep:
                # next_goal = (torch.randn_like(state) * max_goal)
                # next_goal = torch.min(torch.max(next_goal, -max_goal), max_goal)
                next_goal = torch.Tensor([-10, 10, 0.5,
                                          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,  # 3-13
                                          0., 0., 0., 0., 0., 0., 0.,  # 14-20
                                          0., 0., 0., 0., 0., 0., 0., 0.,  # 21-28
                                          0.]).to(device) - next_state
            elif episode_timestep_h < 200:
                # expl_noise_goal = np.random.normal(loc=0, scale=max_goal.cpu() * expl_noise_std_h, size=state_dim).astype(np.float32)
                expl_noise_goal = np.random.normal(loc=0, scale=expl_noise_std_h, size=state_dim).astype(np.float32)
                # next_goal = (actor_eval_h(state_sequence[-1].to(device)).detach().cpu() + expl_noise_goal).squeeze().to(device)
                # next_goal = torch.min(torch.max(next_goal, -max_goal), max_goal)
                next_goal = torch.Tensor([-10, 10, 0.5,
                                          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,  # 3-13
                                          0., 0., 0., 0., 0., 0., 0.,  # 14-20
                                          0., 0., 0., 0., 0., 0., 0., 0.,  # 21-28
                                          0.]).to(device) - next_state
            else:
                next_goal = torch.Tensor([0, 19, 0.5,
                                          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,  # 3-13
                                          0., 0., 0., 0., 0., 0., 0.,  # 14-20
                                          0., 0., 0., 0., 0., 0., 0., 0.,  # 21-28
                                          0.]).to(device) - next_state
            # > off-policy correction
            # goal_hat, updated = off_policy_correction(actor_target_l, action_sequence, state_sequence, state_dim, goal_sequence[0], max_goal, device)
            # 2.2.7 collect high-level steps
            # experience_buffer_h.add(state_sequence[0], goal_hat, episode_reward_h, state, done_h)
            if state_print_trigger.good2log(t, 1000):
                print("        > state:")
                for i in range(len(state_sequence)):
                    print("            {}".format(["%.4f"%elem for elem in state_sequence[i].tolist()]))
                print("        > goal:")
                for i in range(len(goal_sequence)):
                    print("            {}".format(["%.4f" % elem for elem in goal_sequence[i].tolist()]))
                print("        > action:")
                for i in range(len(action_sequence)):
                    print("            {}, {}".format(["%.4f"%elem for elem in action_sequence[i].tolist()], float('%.4f'%intri_reward_sequence[i])))
                # if updated: print("        > goal_hat: {}".format(goal_hat))
                else: print("        > goal_hat not updated")
            # 2.2.8 update high-level loop
            state_sequence, action_sequence, intri_reward_sequence, goal_sequence = [], [], [], []
        goal = next_goal

        # 2.2.9 update networks
        # 2.2.9.1 low-level update
        if t >= start_timestep:
            target_q_l, critic_loss_l, actor_loss_l = \
                step_update_l(experience_buffer_l, batch_size, total_it, actor_eval_l, actor_target_l, critic_eval_l, critic_target_l, critic_optimizer_l, actor_optimizer_l, params)
        # 2.2.9.2 high-level update
        # target_q_h, critic_loss_h, actor_loss_h = None, None, None
        # if t >= start_timestep and t % c == 0 and t > 0:
        #     target_q_h, critic_loss_h, actor_loss_h = \
        #         step_update_h(experience_buffer_h, batch_size, total_it, actor_eval_h, actor_target_h, critic_eval_h, critic_target_h, critic_optimizer_h, actor_optimizer_h, params)
        # 2.2.10 logger record
        if t >= start_timestep and t % log_interval == 0 and t > 0:
            if target_q_l is not None: wandb.log({'target_q low': torch.mean(target_q_l).squeeze()}, step=t-start_timestep)
            if critic_loss_l is not None: wandb.log({'critic_loss low': torch.mean(critic_loss_l).squeeze()}, step=t-start_timestep)
            if actor_loss_l is not None: wandb.log({'actor_loss low': torch.mean(actor_loss_l).squeeze()}, step=t-start_timestep)
            # if target_q_h is not None: wandb.log({'target_q high': torch.mean(target_q_h).squeeze()}, step=t-start_timestep)
            # if critic_loss_h is not None: wandb.log({'critic_loss high': torch.mean(critic_loss_h).squeeze()}, step=t-start_timestep)
            # if actor_loss_h is not None: wandb.log({'actor_loss high': torch.mean(actor_loss_h).squeeze()}, step=t-start_timestep)
        # 2.2.11 start new episode
        if bool(done_l) or episode_timestep_l >= c:
            print(
                f"    > Total T: {t + 1} Episode_Low Num: {episode_num_l + 1} Episode_Low T: {episode_timestep_l} Reward_Low: {float(episode_reward_l):.3f} Reward_High: {float(episode_reward_h):.3f}")
            if t >= start_timestep:
                wandb.log({'episode reward low': episode_reward_l}, step=t-start_timestep)
                wandb.log({'episode reward high': episode_reward_h}, step=t-start_timestep)
            if save_video and video_log_trigger.good2log(t, video_interval):
                log_video_hrl_debug(env_name, actor_target_l, actor_target_h, params)
                time_logger.sps(t)
                time_logger.time_spent()
            state_sequence, action_sequence, intri_reward_sequence, goal_sequence = [], [], [], []
            episode_reward_l, episode_timestep_l = 0, 0
            episode_num_l += 1
        if bool(done_h) or episode_timestep_h > 400:
            episode_num_h += 1
            print(f"    >>> Episode End!: Total T: {t + 1} Episode_High Num: {episode_num_h + 1} Episode_High T: {episode_timestep_h} Reward_High: {float(episode_reward_h):.3f}\n")
            state, done_h = Tensor(env.reset()).to(device), Tensor([False])
            state_sequence, action_sequence, intri_reward_sequence, goal_sequence = [], [], [], []
            episode_reward_l, episode_timestep_l = 0, 0
            episode_reward_h, episode_timestep_h = 0, 0
        # 2.2.12 update training loop
        episode_timestep_l += 1
        episode_timestep_h += 1
        # if checkpoint_logger.good2log(t, checkpoint_interval):
        #     logger = [time_logger, state_print_trigger, video_log_trigger, checkpoint_logger]
        #     save_checkpoint(t,
        #                     actor_eval_l, critic_eval_l, actor_optimizer_l, critic_optimizer_l, experience_buffer_l,
        #                     actor_eval_h, critic_eval_h, actor_optimizer_h, critic_optimizer_h, experience_buffer_h,
        #                     logger, params)
    # 2.3 log training result
    for i in range(3):
        log_video_hrl_debug(env_name, actor_target_l, actor_target_h, params)


if __name__ == "__main__":
    # gym.logger.set_level(40)
    env_name = "AntPush"
    env = get_env(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    max_goal = [
        10., 10., .5,  # 0-2
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,  # 3-13
        30, 30, 30, 30, 30, 30, 30,  # 14-20
        30, 30, 30, 30, 30, 30, 30, 30,  # 21-28
        1.]  # 29
    policy_params = ParamDict(
        seed=12345,
        c=10,
        policy_noise_scale=0.2,
        policy_noise_std=1.,
        expl_noise_std_l=1.,
        expl_noise_std_h=1.,
        policy_noise_clip=0.5,
        max_action=max_action,
        max_goal=max_goal,
        discount=0.99,
        policy_freq=1,
        tau=5e-3,
        actor_lr=1e-4,
        critic_lr=1e-3,
        reward_scal_l=1.,
        reward_scal_h=.1,
        max_timestep=int(3e6),
        start_timestep=int(3e4),
        batch_size=100
    )
    params = ParamDict(
        policy_params=policy_params,
        env_name=env_name,
        state_dim=state_dim,
        action_dim=action_dim,
        video_interval=int(2e3),
        log_interval=1,
        checkpoint_interval=int(1e5),
        prefix="debuglow",
        save_video=True,
        use_cuda=True,
        # checkpoint="hiro-antpush_checked-it(1100000)-[2020-06-28 14:58:22.307268].tar"
        checkpoint=None
    )

    wandb.init(project="ziang-hiro-new")
    train(params=params)