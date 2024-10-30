import torch
import argparse


def command_parser():
    parser = argparse.ArgumentParser(description='Memory addressing dataset distillation')

    # Dataset setup
    parser.add_argument('--env', type=str, default='halfcheetah-medium-replay-v2', \
                        help='Which environment to use')
    parser.add_argument('--seed', type=int, default=42, \
                        help='Random seed of environment')
    parser.add_argument('--n_iters', type=int, default=50000, \
                        help='BPPT iterations')
    parser.add_argument('--eval_freq', type=int, default=1000, \
                        help='evaluation every x iterations')
    parser.add_argument('--normalize', type=bool, default=True, \
                        help='normalize observations') # ATTENTION: only True can be passed! use action="store_true" to pass False.
    

    # Synset setup
    parser.add_argument('--synset_size', type=int, default=256, \
                        help='the size of synthetic dataset')
    parser.add_argument('--synset_init', type=str, default='real', \
                        help='Ways to initialize synthetic data. \
                        real: use real offline data for initialization \
                        random: random initialize the synthetic data')
    parser.add_argument('--data_path', type=str, default=' ',  \
                        help='Path to load data')

    # Training setup
    parser.add_argument('--config', type=str, default='synset_bptt.yml',  \
                        help='Path to the config file')

    # Policy network setup
    parser.add_argument('--policy_type', type=str, default='gaussian',  \
                        help='Gaussian or deterministic policy network')

    # BPTT inner setup
    parser.add_argument('--bptt_inner_steps', type=int, default=100, \
                        help='backpropagation through time inner steps')
    parser.add_argument('--test_batch_num', type=int, default=10, \
                        help='num of batches to measure the generaliztion loss in BPTT') 
    parser.add_argument('--inner_optimizer', type=str, default='SGD', \
                        help='which optimizer to use for bptt inner loop')
    parser.add_argument('--inner_lr', type=float, default=0.1, \
                        help='learning rate for bptt inner loop')
    parser.add_argument('--inner_momentum', type=float, default=0., \
                        help='the momentum for bptt inner loop')
    
    # Synset optimizer setup
    parser.add_argument('--synset_optimizer', type=str, default='SGD', \
                        help='which optimizer to use for outer loop')
    parser.add_argument('--synset_lr', type=float, default=0.1, \
                        help='learning rate for outer loop')
    parser.add_argument('--outer_momentum', type=float, default=0.9, \
                        help='the momentum for bptt outer loop')


    # Match loss
    parser.add_argument('--match_objective', type=str, default='offline_policy', \
                        help='match offline data or offline policy to calculate the objective loss. \
                        offline_policy can only incoporate with Gaussian policy')
    parser.add_argument('--offline_policy_dir', type=str, default='./offline_policy_checkpoints', \
                        help='path of offline policy')
    parser.add_argument('--q_weight', action="store_true", \
                        help='use q value to weight the MSE loss')
    parser.add_argument('--beta_weight', type=float, default=0.02, \
                        help='beta * q_value')

    # Eval
    parser.add_argument('--eval_ensemble', action="store_true", \
                        help='use ensemble policy to evaluate synset')
    parser.add_argument('--ensemble_policy_num', type=int, default=10, \
                        help='number of ensemble committees')

    # Save
    parser.add_argument('--save_dir', type=str, default='', \
                        help='dir of saving synset')
    
    # wandb logging
    parser.add_argument('--project', type=str, default='Offline Behavior Distillation', \
                        help='wandb project')
    parser.add_argument('--group', type=str, default='OBD-BPTT', \
                        help='wandb group')
    parser.add_argument('--name', type=str, default=' ', \
                        help='wandb name')

    return parser


def merge_args(args, config):

    # set data
    setattr(config.dataset, 'name', args.env)
    setattr(config.dataset, 'data_path', args.data_path)
    config.synset_size = args.synset_size
    config.synset_init = args.synset_init
    config.save_dir = args.save_dir
    config.seed = args.seed
    config.normalize = args.normalize

    config.policy_type = args.policy_type

    # Eval
    config.eval_ensemble = args.eval_ensemble
    config.ensemble_policy_num = args.ensemble_policy_num

    setattr(config.training, 'n_iters', args.n_iters)
    setattr(config.training, 'eval_every', args.eval_freq)

    # BPTT
    setattr(config.bptt, 'inner_steps', args.bptt_inner_steps)
    setattr(config.bptt, 'test_batch_num', args.test_batch_num)

    setattr(config.bptt_optim, 'optimizer', args.inner_optimizer)
    setattr(config.bptt_optim, 'lr', args.inner_lr)
    setattr(config.bptt_optim, 'momentum', args.inner_momentum)

    setattr(config.synset_optim, 'optimizer', args.synset_optimizer)
    setattr(config.synset_optim, 'lr', args.synset_lr)
    setattr(config.synset_optim, 'momentum', args.outer_momentum)

    config.match_objective = args.match_objective
    config.offline_policy_dir = args.offline_policy_dir
    config.q_weight = args.q_weight
    config.beta_weight = args.beta_weight

    # wandb init
    config.project = args.project
    config.group = f"{args.group}-{config.match_objective}-q_weight-{str(config.q_weight)}"
    config.name = f"{args.env}-seed-{args.seed}"

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    config.device = device

    return config