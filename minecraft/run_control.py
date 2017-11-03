import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--policy', choices=['mlp', 'rnn'], default='mlp')
    parser.add_argument('--rnn_layers', type=int, default=2)
    parser.add_argument('--rnn_size', type=int, default=100)
    parser.add_argument('--rnn_type', choices=['lstm', 'gru', 'simple'], default='lstm')
    parser.add_argument('--fc_layers', type=int, default=2)
    parser.add_argument('--fc_size', type=int, default=100)
    parser.add_argument('--cnn_init', choices=['orthogonal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'], default='glorot_normal')
    parser.add_argument('--fc_init', choices=['orthogonal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'], default='glorot_normal')
    parser.add_argument('--action_init', choices=['orthogonal', 'normal', 'uniform'], default='normal')
    parser.add_argument('--action_init_scale', type=float, default=0.01)
    parser.add_argument('--activation', choices=['tanh', 'relu', 'elu'], default='relu')
    parser.add_argument('--stochastic', action='store_true', default=True)
    parser.add_argument('--deterministic', action='store_false', dest='stochastic')
    # optimization
    parser.add_argument('--optimizer', choices=['adam', 'rmsprop', 'nadam'], default='adam')
    parser.add_argument('--optimizer_lr', type=float, default=0.001)
    parser.add_argument('--clipnorm', type=float, default=40.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--clip_param', type=float, default=0.3)
    parser.add_argument('--initial_std', type=float, default=1)
    parser.add_argument('--l2_reg', type=float, default=0)
    parser.add_argument('--policy_loss', choices=['pg', 'pposgd'], default='pposgd')
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--kld_coef', type=float, default=0.01)
    parser.add_argument('--value_coef', type=float, default=1.0)
    parser.add_argument('--normalize_observations', action='store_true', default=True)
    parser.add_argument('--no_normalize_observations', action='store_false', dest='normalize_observations')
    parser.add_argument('--normalize_advantage', action='store_true', default=True)
    parser.add_argument('--no_normalize_advantage', action='store_false', dest='normalize_advantage')
    parser.add_argument('--normalize_baseline', action='store_true', default=True)
    parser.add_argument('--no_normalize_baseline', action='store_false', dest='normalize_baseline')
    parser.add_argument('--repeat_updates', type=int, default=2)
    parser.add_argument('--num_local_steps', type=int, default=50)
    parser.add_argument('--adapt_kl', type=float)
    # parallelization
    parser.add_argument('--trainer', choices=['batched', 'batched_predictor'], default='batched')
    parser.add_argument('--num_runners', type=int, default=2)
    parser.add_argument('--queue_length', type=int, default=2)
    parser.add_argument('--queue_timeout', type=float, default=None)
    parser.add_argument('--runner_gpu', default=None)
    # how long
    parser.add_argument('--num_timesteps', type=int, default=100000)
    parser.add_argument('--stats_interval', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=100000)
    parser.add_argument('--num_eval_steps', type=int, default=1000)
    parser.add_argument('--max_episode_timesteps', type=int)
    # technical
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--no_display', dest='display', action='store_false')
    parser.add_argument('--monitor', action='store_true', default=False)
    parser.add_argument('--no_monitor', dest='monitor', action='store_false')
    parser.add_argument('--logdir', default="logs/control")
    parser.add_argument('--csv_file')
    parser.add_argument('--load_weights')
    parser.add_argument('--weights_by_name', action='store_true', default=False)
    parser.add_argument('--environment', '-e', default='CartPole-v0')

    subparsers = parser.add_subparsers(dest='command')
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('label')
    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('label')
    args = parser.parse_args()

    from common.envs import create_env
    from trainers.batched import BatchedTrainer
    from trainers.batched_predictor import BatchedPredictorTrainer
    from policies.ppo import MLPPolicy, RNNPolicy

    if args.policy == 'mlp':
        policy_class = MLPPolicy
    elif args.policy == 'rnn':
        policy_class = RNNPolicy
    else:
        assert False

    if args.trainer == 'batched':
        trainer_class = BatchedTrainer
    elif args.trainer == 'batched_predictor':
        trainer_class = BatchedPredictorTrainer
    else:
        assert False

    trainer = trainer_class(create_env, policy_class, args)

    if args.command == 'train':
        trainer.run(args.environment, args.num_timesteps, os.path.join(args.logdir, args.label))
    elif args.command == 'eval':
        trainer.eval(args.environment, args.num_timesteps, os.path.join(args.logdir, args.label))
    else:
        parser.print_usage()
