import argparse
import logging
import os

logging.getLogger("gym_minecraft").setLevel(logging.ERROR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--video_width', type=int, default=40)
    parser.add_argument('--video_height', type=int, default=30)
    parser.add_argument('--video_depth', action='store_true', default=True)
    parser.add_argument('--no_video_depth', action='store_false', dest='video_depth')
    parser.add_argument('--skip_steps', type=int, default=4)
    parser.add_argument('--start_minecraft', action='store_true', default=False)
    parser.add_argument('--num_buffer_frames', type=int, default=4)
    parser.add_argument('--action_space', choices=['continuous', 'discrete'], default='continuous')
    parser.add_argument('--allowed_actions', type=lambda s: [i for i in s.split(',')], default='move,turn')
    parser.add_argument('--use_allocator', action='store_true', default=False)
    parser.add_argument('--allocator_placement', choices=['public', 'private', 'pod'], default='pod')
    # model
    parser.add_argument('--policy', choices=['cnn', 'rnn', 'depth'], default='depth')
    parser.add_argument('--cnn_architecture', choices=['deepmind', 'openai', 'homogeneous'], default='openai')
    parser.add_argument('--cnn_layers', type=int, default=4)
    parser.add_argument('--cnn_filters', type=int, default=32)
    parser.add_argument('--cnn_kernel_size', type=int, default=3)
    parser.add_argument('--cnn_kernel_step', type=int, default=2)
    parser.add_argument('--rnn_layers', type=int, default=1)
    parser.add_argument('--rnn_size', type=int, default=128)
    parser.add_argument('--rnn_type', choices=['lstm', 'gru', 'simple'], default='lstm')
    parser.add_argument('--fc_layers', type=int, default=0)
    parser.add_argument('--fc_size', type=int, default=32)
    parser.add_argument('--cnn_init', choices=['orthogonal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'], default='glorot_normal')
    parser.add_argument('--fc_init', choices=['orthogonal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'], default='glorot_normal')
    parser.add_argument('--action_init', choices=['orthogonal', 'normal', 'uniform'], default='normal')
    parser.add_argument('--action_init_scale', type=float, default=0.01)
    parser.add_argument('--activation', choices=['tanh', 'relu', 'elu'], default='relu')
    parser.add_argument('--stochastic', '-s', action='store_true', default=True)
    parser.add_argument('--deterministic', '-d', action='store_false', dest='stochastic')
    # optimization
    parser.add_argument('--optimizer', choices=['adam', 'rmsprop', 'nadam'], default='adam')
    parser.add_argument('--optimizer_lr', type=float, default=0.0001)
    parser.add_argument('--clipnorm', type=float, default=40.0)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--lam', type=float, default=0.92)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--initial_std', type=float, default=0.5)
    parser.add_argument('--l2_reg', type=float, default=0)
    parser.add_argument('--policy_loss', choices=['pg', 'pposgd'], default='pposgd')
    parser.add_argument('--entropy_coef', type=float, default=0.001)
    parser.add_argument('--kld_coef', type=float, default=0.1)
    parser.add_argument('--value_coef', type=float, default=1.0)
    parser.add_argument('--depth_coef', type=float, default=1.0)
    parser.add_argument('--backward_coef', type=float, default=0.01)
    parser.add_argument('--normalize_observations', action='store_true', default=False)
    parser.add_argument('--no_normalize_observations', action='store_false', dest='normalize_observations')
    parser.add_argument('--normalize_advantage', action='store_true', default=True)
    parser.add_argument('--no_normalize_advantage', action='store_false', dest='normalize_advantage')
    parser.add_argument('--normalize_baseline', action='store_true', default=True)
    parser.add_argument('--no_normalize_baseline', action='store_false', dest='normalize_baseline')
    parser.add_argument('--repeat_updates', type=int, default=4)
    parser.add_argument('--num_local_steps', type=int, default=50)
    parser.add_argument('--adapt_kl', type=float, default=0.002)
    # parallelization
    parser.add_argument('--trainer', choices=['batched', 'batched_profiler', 'mix', 'dynamic'], default='batched')
    parser.add_argument('--num_runners', type=int, default=2)
    parser.add_argument('--queue_length', type=int, default=2)
    parser.add_argument('--queue_timeout', type=float, default=None)
    parser.add_argument('--runner_gpu', default=None)
    # how long
    parser.add_argument('--num_timesteps', type=int, default=100000)
    parser.add_argument('--stats_interval', type=int, default=10000)
    parser.add_argument('--save_interval', type=int, default=100000)
    parser.add_argument('--num_eval_steps', type=int, default=5000)
    # curriculum
    parser.add_argument('--curriculum_episodes', type=int, default=10)
    parser.add_argument('--curriculum_steps', type=int, default=10000)
    parser.add_argument('--curriculum_lr', type=float, default=0.1)
    parser.add_argument('--curriculum_epsilon', type=float, default=0.1)
    parser.add_argument('--curriculum_softmax_temperature', type=float, default=0.0004)
    parser.add_argument('--curriculum_policy', choices=['e_greedy', 'softmax'], default='e_greedy')
    parser.add_argument('--curriculum_teacher_gpu')
    parser.add_argument('--curriculum_abs', action='store_true', default=False)
    parser.add_argument('--reward_finish_threshold', type=int)
    # technical
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--no_display', dest='display', action='store_false')
    parser.add_argument('--monitor', action='store_true', default=False)
    parser.add_argument('--no_monitor', dest='monitor', action='store_false')
    parser.add_argument('--client_resize', action='store_true', default=False)
    parser.add_argument('--logdir', default="logs/minecraft")
    parser.add_argument('--csv_file')
    parser.add_argument('--load_weights')
    parser.add_argument('--weights_by_name', action='store_true', default=False)
    parser.add_argument('--load_mission', '-m', nargs='*', default=['missions/basic7x7.xml'])
    parser.add_argument('--host', default="127.0.0.1")
    parser.add_argument('--start_port', '-p', type=int, default=10001)
    parser.add_argument("--environment", "-e", default="MinecraftBasic-v0")

    subparsers = parser.add_subparsers(dest='command')
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('label')
    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('label')
    args = parser.parse_args()

    from common.envs import create_env
    from trainers.batched import BatchedTrainer, BatchedTrainerProfiler
    from trainers.batched_minecraft_mix import BatchedTrainer as BatchedTrainerMix
    from trainers.batched_minecraft_dynamic import BatchedTrainer as BatchedTrainerDynamic
    from policies.ppo import CNNPolicy, CNNRNNPolicy
    from policies.ppo_depth import CNNRNNPolicy as CNNRNNDepthPolicy

    if args.policy == 'cnn':
        policy_class = CNNPolicy
    elif args.policy == 'rnn':
        policy_class = CNNRNNPolicy
    elif args.policy == 'depth':
        policy_class = CNNRNNDepthPolicy
    else:
        assert False

    if args.trainer == 'batched':
        trainer_class = BatchedTrainer
    elif args.trainer == 'batched_profiler':
        trainer_class = BatchedTrainerProfiler
    elif args.trainer == 'mix':
        trainer_class = BatchedTrainerMix
    elif args.trainer == 'dynamic':
        trainer_class = BatchedTrainerDynamic
    else:
        assert False

    trainer = trainer_class(create_env, policy_class, args)

    if args.command == 'train':
        trainer.run(args.environment, args.num_timesteps, os.path.join(args.logdir, args.label))
    elif args.command == 'eval':
        trainer.eval(args.environment, args.num_timesteps, os.path.join(args.logdir, args.label))
    else:
        parser.print_usage()
