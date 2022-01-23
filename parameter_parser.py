import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parameter_parser():
    parser = argparse.ArgumentParser()

    ######################### general parameters ################################
    parser.add_argument('--dataset_name', type=str, default='location',
                        choices=['adult', 'accident', 'location', 'mnist', 'cifar10', 'stl10'])
    parser.add_argument('--original_label', type=str, default='NY',
                        choices=['income', 'severity', 'LA', 'NY', 'default'])    
    parser.add_argument('--is_train_multiprocess', type=str2bool, default=False)
    parser.add_argument('--exp', type=str, default='mem_inf', 
                        choices=['model_train', 'mem_inf'], 
                        help="'mem_train' train the original-unlearning model pairs, 'mem_inf' launch the attack")
    parser.add_argument('--cuda', type=int, default=6, 
                        help="Choose the GPU device")
    ######################### target model related parameters ################################
    parser.add_argument('--original_model', type=str, default='RF',
                        choices=['DT', 'MLP', 'LR', 'RF', 'LRTorch', 'MLPTorch', 'scnn', 'resnet50', 'densenet'])
    parser.add_argument('--attack_model', type=str, default='DT', 
                        choices=['DT', 'MLP', 'LR', 'RF'])
    parser.add_argument('--unlearning_method', type=str, default='scratch', 
                        choices=['scratch', 'sisa'])
    parser.add_argument('--is_sample', type=str2bool, default=True)
    parser.add_argument('--is_obtain_posterior', type=str2bool, default=True)
    parser.add_argument('--num_process', type=int, default=1)
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--optim', type=str, default="Adam", 
                        choices=['Adam', 'SGD'])
    ######################### attack related parameters ################################
    parser.add_argument('--shadow_set_num', type=int, default=10, 
                        help="Number of shadow original model")
    parser.add_argument('--shadow_set_size', type=int, default=2000, 
                        help="Number of shadow model training samples")
    parser.add_argument('--shadow_unlearning_size', type=int, default=20, 
                        help="Number of unlearned model")
    parser.add_argument('--shadow_unlearning_num', type=int, default=1,
                        help="Number of deleted records to generate unlearned model")
    parser.add_argument('--shadow_num_shard', type=int, default=10, 
                        help="Number of shards")

    parser.add_argument('--target_set_num', type=int, default=10, 
                        help="Number of target original model")
    parser.add_argument('--target_set_size', type=int, default=2000, 
                        help="Number of target model training samples")
    parser.add_argument('--target_unlearning_size', type=int, default=20, 
                        help="Number of unlearned model")
    parser.add_argument('--target_unlearning_num', type=int, default=1,
                        help="Number of deleted records to generate unlearned model")
    parser.add_argument('--target_num_shard', type=int, default=10, 
                        help="Number of shards")

    parser.add_argument('--samples_to_evaluate', type=str, default="in_out", 
                        choices=['in_in', 'in_out', 'in_out_multi_version'],
                        help="Samples to evaluate")
    ######################### defense related parameters ################################
    parser.add_argument('--is_defense', type=str2bool, default=False)
    parser.add_argument('--top_k', type=int, default=4, choices=[0, 1, 2, 3, 4], 
                        help=" 0 (label), 4 (no defense)")
    parser.add_argument('--is_dp_defense', type=str2bool, default=False)
    parser.add_argument("--sigma", type=float, default=1.2, metavar="S", 
                        help="Noise multiplier (default 1.0)", )
    parser.add_argument("-c", "--max-per-sample-grad_norm", type=float, default=1.0, metavar="C", 
                        help="Clip per-sample gradients to this norm (default 1.0)", )
    parser.add_argument("--delta", type=float, default=1e-5, metavar="D", 
                        help="Target delta (default: 1e-5)", )
    parser.add_argument("-sr", "--sample-rate", type=float, default=0.001, metavar="SR", 
                        help="sample rate used for batch construction (default: 0.001)", )
    parser.add_argument("--secure_rng", action="store_true", default=False,
                        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost", )
    parser.add_argument("--lr", type=float, default=0.05, metavar="LR",
                        help="learning rate (default: .1)", )
    parser.add_argument('--n_accumulation_steps', type=float, default=1)

    args = vars(parser.parse_args())
    return args
