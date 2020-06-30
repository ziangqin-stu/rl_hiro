"""
The default entrance of project
"""
import argparse
import csv

import copy

from train import train
from utils import ParamDict, get_env

# =============================
# Command-Line Argument Binding
# =============================
parser = argparse.ArgumentParser(description="Specific Hyper-Parameters for PPO training. ")
# >> experiment parameters
parser.add_argument('--param_id', default=0, type=int, help='index of parameter that will be loaded from local csv file for this run')
parser.add_argument('--algorithm', default="hiro", help='select experiment algorithm: hiro or td3')
parser.add_argument('--env_name', default="AntPush", help='environment name for this run, choose from AntPush, AntFall, AntMaze')
parser.add_argument('--state_dim', default=30, type=int, help='environment observation state dimension')
parser.add_argument('--action_dim', default=8, type=int, help='agent action dimension')
parser.add_argument('--save_video', help='whether sample and log episode video intermittently during training')
parser.add_argument('--video_interval', default=int(2e4), type=int, help='the interval of logging video')
parser.add_argument('--checkpoint_interval', default=int(1e5), type=int, help='the interval of log checkpoint')
parser.add_argument('--log_interval', default=1, type=int, help='the interval of print training state to interval')
parser.add_argument('--checkpoint', default="", help='the file name of checkpoint to be load, set to None if do not load data from local checkpoint')
parser.add_argument('--prefix', default="test", help='prefix of checkpoint files, used to distinguish different runs')
parser.add_argument('--use_cuda', help='whether use GPU')
# >> HIRO algorithm parameters
parser.add_argument('--seed', default=4321, type=int, help='manual seed')
parser.add_argument('--max_timestep', default=int(3e4), type=int, help='max training time step')
parser.add_argument('--start_timestep', default=int(2e4), type=int, help='amount of random filling experience')
parser.add_argument('--batch_size', default=int(1e2), type=int, help='batch sample size')
# parser.add_argument('--max_goal', default=None, help='goal boundary')
parser.add_argument('--max_action', default=30, type=int, help='action boundary')
parser.add_argument('--c', default=10, type=int, help='high-level policy update interval')
parser.add_argument('--policy_freq', default=2, type=int, help='delayed policy update interval')
parser.add_argument('--discount', default=.99, type=float, help='long-horizon reward discount')
parser.add_argument('--actor_lr', default=1e-4, type=float, help='actor policy learning rate')
parser.add_argument('--critic_lr', default=1e-3, type=float, help='critic policy learning rate')
parser.add_argument('--tau', default=5e-3, type=float, help='soft update parameter')
parser.add_argument('--reward_scal_l', default=1., type=float, help='low-level reward rescale parameter')
parser.add_argument('--reward_scal_h', default=.1, type=float, help='high-level reward rescale parameter')
parser.add_argument('--policy_noise_scale', default=.2, type=float, help='target policy smoothing regularization noise scale')
parser.add_argument('--policy_noise_std', default=1., type=float, help='target policy smoothing regularization noise standard deviation')
parser.add_argument('--policy_noise_clip', default=0.5, type=float, help='exploration noise boundary')
parser.add_argument('--expl_noise_std_l', default=1., type=float, help='low-level policy exploration noise standard deviation')
parser.add_argument('--expl_noise_std_h', default=1., type=float, help='low-level policy exploration noise standard deviation')
# >> TD3 algorithm special parameters
parser.add_argument('--policy_noise_std', default=0.1, type=float, help='policy noise standard derivation')
parser.add_argument('--expl_noise_std_scale', default=0.1, type=float, help='exploration noise standard derivation scale')
parser.add_argument('--lr_td3', default=0.1, type=float, help='td3 learning rate')
parser.add_argument('--max_action_td3', default=1, type=float, help='td3 action boundary')
parser.add_argument('--state_dim_td3', default=11, type=int, help='td3 learning rate')
parser.add_argument('--action_dim_td3', default=1, type=int, help='td3 learning rate')
# >> parse arguments
args = parser.parse_args()


# =========================
# Read Parameters from File
# =========================
bool_params_list = ['save_video', 'use_cuda', 'checkpoint']
true_strings = ['True', 'true', 'TRUE']
false_string = ['False', 'false', 'FALSE']
none_string = ['None', 'none', 'NONE']
max_goal = [
        10., 10., .5,  # 0-2
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,  # 3-13
        30, 30, 30, 30, 30, 30, 30,  # 14-20
        30, 30, 30, 30, 30, 30, 30, 30,  # 21-28
        1.]  # 29


def bool_args_preprocess(args):
    # Handle boolean cmd args
    global bool_params_list
    global true_strings
    global false_string
    for param_name in bool_params_list:
        param = getattr(args, param_name)
        if param is not None:
            if param in true_strings:
                setattr(args, param_name, True)
            elif param in false_string:
                setattr(args, param_name, False)
            elif param in none_string:
                setattr(args, param_name, None)
            else:
                raise ValueError('Command line boolean argument typo.')


def bool_params_preprocess(file_param):
    # Handle boolean cmd args
    global bool_params_list
    global true_strings
    global false_string
    for param_name in bool_params_list:
        param = file_param[param_name]
        if param is not None:
            if param in true_strings:
                file_param[param_name] = True
            elif param in false_string:
                file_param[param_name] = False
            else:
                raise ValueError('CSV boolean argument typo.')


def load_params(index):
    # load $ package training arguments
    if args.algorithm == 'td3':
        f = open('./train_param_td3.csv', 'r')
        with f:
            # read parameters from file
            reader = csv.DictReader(f)
            rows = [row for row in reader]
            file_param = rows[index]
            bool_args_preprocess(args)
            bool_params_preprocess(file_param)
            # load environment info
            env_name = args.env_name if args.env_name is not None else file_param['env_name']
            env = get_env(env_name)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            max_action = float(env.action_space.high[0])
            # build parameter container
            policy_params = ParamDict(
                seed=args.seed if args.seed is not None else int(file_param['seed']),
                policy_noise_std=args.policy_noise_std if args.policy_noise_std is not None else float(file_param['policy_noise_std']),
                policy_noise_scale=args.policy_noise_scale if args.policy_noise_scale is not None else float(file_param['policy_noise_scale']),
                expl_noise_std_scale=args.expl_noise_std_scale if args.expl_noise_std_scale is not None else float(file_param['expl_noise_std_scale']),
                policy_noise_clip=args.policy_noise_clip if args.policy_noise_clip is not None else float(file_param['policy_noise_clip']),
                max_action_td3=args.max_action_td3 if args.max_action_td3 is not None else float(file_param['max_action_td3']) if file_param['max_action_td3'] is not None else max_action,
                discount=args.discount if args.discount is not None else float(file_param['discount']),
                policy_freq=args.policy_freq if args.policy_freq is not None else int(file_param['policy_freq']),
                tau=args.tau if args.tau is not None else float(file_param['tau']),
                lr_td3=args.lr_td3 if args.lr_td3 is not None else float(file_param['lr_td3']),
                max_timestep=args.max_timestep if args.max_timestep is not None else int(file_param['max_timestep']),
                start_timestep=args.start_timestep if args.start_timestep is not None else int(file_param['start_timestep']),
                batch_size=args.batch_size if args.batch_size is not None else int(file_param['batch_size'])
            )
            params = ParamDict(
                policy_params=policy_params,
                env_name=env_name,
                state_dim_td3=args.state_dim_td3 if args.state_dim_td3 is not None else int(file_param['state_dim_td3']) if file_param['state_dim_td3'] is not None else state_dim,
                action_dim_td3=args.action_dim_td3 if args.action_dim_td3 is not None else int(file_param['action_dim_td3']) if file_param['action_dim_td3'] is not None else action_dim,
                save_video=args.save_video if args.save_video is not None else file_param['save_video'],
                video_interval=args.video_interval if args.video_interval is not None else int(file_param['video_interval']),
                use_cuda=args.use_cuda if args.use_cuda is not None else file_param['use_cuda']
            )
    elif args.algorithm == 'hiro':
        f = open('./train_param_hiro.csv', 'r')
        env_name = "AntPush"
        env = get_env(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        with f:
            # read parameters from file
            reader = csv.DictReader(f)
            rows = [row for row in reader]
            file_param = rows[index]
            bool_args_preprocess(args)
            bool_params_preprocess(file_param)
            # load environment info
            env_name = args.env_name if args.env_name is not None else file_param['env_name']
            env = get_env(env_name)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            # build parameter container
            policy_params = ParamDict(
                seed=args.seed if args.seed is not None else int(file_param['seed']),
                c=args.c if args.c is not None else int(file_param['c']),
                policy_noise_scale=args.policy_noise_scale if args.policy_noise_scale is not None else float(file_param['policy_noise_scale']),
                policy_noise_std=args.policy_noise_std if args.policy_noise_std is not None else float(file_param['policy_noise_std']),
                expl_noise_std_l=args.expl_noise_std_l if args.expl_noise_std_l is not None else float(file_param['expl_noise_std_l']),
                expl_noise_std_h=args.expl_noise_std_h if args.expl_noise_std_h is not None else float(file_param['expl_noise_std_h']),
                policy_noise_clip=args.policy_noise_clip if args.policy_noise_clip is not None else float(file_param['policy_noise_clip']),
                max_action=None,
                max_goal=None,
                discount=args.discount if args.discount is not None else float(file_param['discount']),
                policy_freq=args.policy_freq if args.policy_freq is not None else int(file_param['policy_freq']),
                tau=args.tau if args.tau is not None else float(file_param['tau']),
                actor_lr=args.actor_lr if args.actor_lr is not None else float(file_param['actor_lr']),
                critic_lr=args.critic_lr if args.critic_lr is not None else float(file_param['critic_lr']),
                reward_scal_l=args.reward_scal_l if args.reward_scal_l is not None else float(file_param['reward_scal_l']),
                reward_scal_h=args.reward_scal_h if args.reward_scal_h is not None else float(file_param['reward_scal_h']),
                episode_len=args.episode_len if args.episode_len is not None else int(file_param['episode_len']),
                max_timestep=args.max_timestep if args.max_timestep is not None else int(file_param['max_timestep']),
                start_timestep=args.start_timestep if args.start_timestep is not None else int(file_param['start_timestep']),
                batch_size=args.batch_size if args.batch_size is not None else int(file_param['batch_size'])
            )
            params = ParamDict(
                policy_params=policy_params,
                env_name=env_name,
                state_dim=args.state_dim if args.state_dim is not None else int(file_param['state_dim']) if file_param['state_dim'] is not None else state_dim,
                action_dim=args.action_dim if args.action_dim is not None else int(file_param['action_dim']) if file_param['action_dim'] is not None else action_dim,
                video_interval=args.video_interval if args.video_interval is not None else int(file_param['video_interval']),
                log_interval=args.log_interval if args.log_interval is not None else int(file_param['log_interval']),
                checkpoint_interval=args.checkpoint_interval if args.checkpoint_interval is not None else int(file_param['checkpoint_interval']),
                prefix=args.prefix if args.prefix is not None else file_param['prefix'],
                save_video=args.save_video if args.save_video is not None else file_param['save_video'],
                use_cuda=args.use_cuda if args.use_cuda is not None else file_param['use_cuda'],
                checkpoint=args.checkpoint if args.checkpoint is not None else file_param['checkpoint']
            )
    else:
        raise ValueError("value of algorithm argument is either 'hiro' or 'td3'.")
    return params, policy_params


# ==============
# Run Experiment
# ==============
def cmd_run(params):
    print("    >> function 'cmd_run' under developing...")
    training_param = copy.deepcopy(params)
    del training_param['policy_params']
    print("=========================================================")
    print("Start Training {}: env={}, #update={}".format(args.algorithm.upper(), params.env_name, params.iter_num))
    print("    -------------------------------------------------")
    print("    Training-Params: {}".format(training_param))
    print("    -------------------------------------------------")
    print("    Policy-Params: {}".format(params.policy_params))
    print("=========================================================")
    train(params)
    print(">=================================<")
    print("Training Finished!: alg=PPO, env={}, #update={}".format(params.env_name, params.iter_num))
    print(">=================================<")


# =============
# Main Entrance
# =============
if __name__ == "__main__":
    params, policy_params = load_params(args.param_id)
    cmd_run(params)
