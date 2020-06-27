"""
The default entrance of project
"""
import argparse
import csv
from train import train
from utils import ParamDict

# =============================
# Command-Line Argument Binding
# =============================
parser = argparse.ArgumentParser(description="Specific Hyper-Parameters for PPO training. ")
# >> select fundamental parameters from local file
parser.add_argument('--param_id', default=0, type=int, help='index of parameter that will be loaded from local csv file for this run')
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

parser.add_argument('--seed', default=4321, type=int, help='manual seed')
parser.add_argument('--max_timestep', default=int(3e4), type=int, help='max training time step')
parser.add_argument('--start_timestep', default=int(2e4), type=int, help='amount of random filling experience')
parser.add_argument('--batch_size', default=int(1e2), type=int, help='batch sample size')
parser.add_argument('--max_goal', default=None, help='goal boundary')
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
# >> parse arguments
args = parser.parse_args()

# =========================
# Read Parameters from File
# =========================
bool_params_list = ['save_video', 'use_cuda']
true_strings = ['True', 'true', 'TRUE']
false_string = ['False', 'false', 'FALSE']


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
    print("    >> function 'load_params' under developing...")
    # f = open('./train_param.csv', 'r')
    # with f:
    #     reader = csv.DictReader(f)
    #     rows = [row for row in reader]
    #     file_param = rows[index]
    #     bool_args_preprocess(args)
    #     bool_params_preprocess(file_param)
    #     policy_params = ParamDict(
    #         hidden_dim=args.hidden_dim if args.hidden_dim is not None else int(file_param['hidden_dim']),
    #         learning_rate=args.learning_rate if args.learning_rate is not None else float(file_param['learning_rate']),
    #         envs_num=args.envs_num if args.envs_num is not None else int(file_param['envs_num']),
    #         horizon=args.horizon if args.horizon is not None else int(file_param['horizon']),
    #         batch_size=args.batch_size if args.batch_size is not None else int(file_param['batch_size']),
    #         epochs_num=args.epochs_num if args.epochs_num is not None else int(file_param['epochs_num']),
    #         critic_coef=args.critic_coef if args.critic_coef is not None else float(file_param['critic_coef']),
    #         entropy_coef=args.entropy_coef if args.entropy_coef is not None else float(file_param['entropy_coef']),
    #         discount=args.discount if args.discount is not None else float(file_param['discount']),
    #         lambd=args.lambd if args.lambd is not None else float(file_param['lambd']),
    #         clip_param=args.clip_param if args.clip_param is not None else float(file_param['clip_param'])
    #     )
    #     params = ParamDict(
    #         policy_params=policy_params,
    #         env_name=args.env_name if args.env_name is not None else file_param['env_name'],
    #         prefix=args.prefix if args.prefix is not None else file_param['prefix'],
    #         save_path=args.save_path if args.save_path is not None else file_param['save_path'],
    #         use_pretrain=args.use_pretrain if args.use_pretrain is not None else file_param[
    #             'use_pretrain'],
    #         pretrain_file=args.pretrain_file if args.pretrain_file is not None else file_param['pretrain_file'],
    #         save_checkpoint=args.save_checkpoint if args.save_checkpoint is not None else file_param['save_checkpoint'],
    #         checkpoint_iter=args.checkpoint_iter if args.checkpoint_iter is not None else file_param['checkpoint_iter'],
    #         parallel=args.parallel if args.parallel is not None else file_param['parallel'],
    #         log_video=args.log_video if args.log_video is not None else file_param['log_video'],
    #         plotting_iters=args.plotting_iters if args.plotting_iters is not None else int(
    #             file_param['plotting_iters']),
    #         iter_num=args.iter_num if args.iter_num is not None else int(file_param['iter_num']),
    #         seed=args.seed if args.seed is not None else int(file_param['seed']),
    #         decay_entro_loss=args.decay_entro_loss if args.decay_entro_loss is not None else file_param[
    #             'decay_entro_loss']
    #     )
    # return params, policy_params
    return None, None


# ==============
# Run Experiment
# ==============
def cmd_run(params):
    print("    >> function 'cmd_run' under developing...")
    # training_param = copy.deepcopy(params)
    # del training_param['policy_params']
    # print("=========================================================")
    # print("Start PPO Training: env={}, #update={}".format(params.env_name, params.iter_num))
    # print("    -------------------------------------------------")
    # print("    Training-Params: {}".format(training_param))
    # print("    -------------------------------------------------")
    # print("    Policy-Params: {}".format(params.policy_params))
    # print("=========================================================")
    # display = Display(backend='xvfb')
    # display.start()
    # train(params)
    # display.popen.kill()
    # print(">=================================<")
    # print("Training Finished!: alg=PPO, env={}, #update={}".format(params.env_name, params.iter_num))
    # print(">=================================<")


# =============
# Main Entrance
# =============
if __name__ == "__main__":
    params, policy_params = load_params(args.param_id)
    cmd_run(params)
