import argparse
import os

import numpy as np

from addition_rnn_model_2d import AdditionRNNModel, AdditionRNNEnvironment
from tensorboard_utils import create_summary_writer


def gen_curriculum_baseline(gen_digits):
    return [[1/gen_digits**2 for _ in range(gen_digits**2)]]


def gen_curriculum_naive(gen_digits):
    return [[1/(2*i + 1) if (j // gen_digits <= i and j % gen_digits == i) or (j // gen_digits == i and j % gen_digits <= i) else 0 for j in range(gen_digits**2)] for i in range(gen_digits)] + gen_curriculum_baseline(gen_digits)


def gen_curriculum_mixed(gen_digits):
    return [[1/(i+1)**2 if j // gen_digits <= i and j % gen_digits <= i else 0 for j in range(gen_digits**2)] for i in range(gen_digits)]


def gen_curriculum_combined(gen_digits):
    return [[1/(2*(i+1)**2) if (j // gen_digits < i and j % gen_digits < i) else 1/(2*(i+1)**2) + 1/(2*(2*i+1)) if (j // gen_digits <= i and j % gen_digits == i) or (j // gen_digits == i and j % gen_digits <= i) else 0 for j in range(gen_digits**2)] for i in range(gen_digits)] + gen_curriculum_baseline(gen_digits)


DIGITS_DIST_EXPERIMENTS = {
    'baseline': [[
        1/16, 1/16, 1/16, 1/16,
        1/16, 1/16, 1/16, 1/16,
        1/16, 1/16, 1/16, 1/16,
        1/16, 1/16, 1/16, 1/16
        ]],
    'naive': [[
        1, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
        ], [
        0, 1/3, 0, 0,
        1/3, 1/3, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
        ], [
        0, 0, 1/5, 0,
        0, 0, 1/5, 0,
        1/5, 1/5, 1/5, 0,
        0, 0, 0, 0
        ], [
        0, 0, 0, 1/7,
        0, 0, 0, 1/7,
        0, 0, 0, 1/7,
        1/7, 1/7, 1/7, 1/7
        ], [
        1/16, 1/16, 1/16, 1/16,
        1/16, 1/16, 1/16, 1/16,
        1/16, 1/16, 1/16, 1/16,
        1/16, 1/16, 1/16, 1/16
        ]],
    'mixed': [[
        1, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
        ], [
        1/4, 1/4, 0, 0,
        1/4, 1/4, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
        ], [
        1/9, 1/9, 1/9, 0,
        1/9, 1/9, 1/9, 0,
        1/9, 1/9, 1/9, 0,
        0, 0, 0, 0
        ], [
        1/16, 1/16, 1/16, 1/16,
        1/16, 1/16, 1/16, 1/16,
        1/16, 1/16, 1/16, 1/16,
        1/16, 1/16, 1/16, 1/16
        ]],
    'combined': [[
        1, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
        ], [
        1/8, 7/24, 0, 0,
        7/24, 7/24, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
        ], [
        1/18, 1/18, 7/45, 0,
        1/18, 1/18, 7/45, 0,
        7/45, 7/45, 7/45, 0,
        0, 0, 0, 0
        ], [
        1/32, 1/32, 1/32, 23/224,
        1/32, 1/32, 1/32, 23/224,
        1/32, 1/32, 1/32, 23/224,
        23/224, 23/224, 23/224, 23/224
        ], [
        1/16, 1/16, 1/16, 1/16,
        1/16, 1/16, 1/16, 1/16,
        1/16, 1/16, 1/16, 1/16,
        1/16, 1/16, 1/16, 1/16
        ]],
}
assert np.all([np.isclose(np.sum(a), 1) for a in DIGITS_DIST_EXPERIMENTS['baseline']])
assert np.all([np.isclose(np.sum(a), 1) for a in DIGITS_DIST_EXPERIMENTS['naive']])
assert np.all([np.isclose(np.sum(a), 1) for a in DIGITS_DIST_EXPERIMENTS['mixed']])
assert np.all([np.isclose(np.sum(a), 1) for a in DIGITS_DIST_EXPERIMENTS['combined']])
assert np.allclose(gen_curriculum_baseline(4), DIGITS_DIST_EXPERIMENTS['baseline'])
assert np.allclose(gen_curriculum_naive(4), DIGITS_DIST_EXPERIMENTS['naive'])
assert np.allclose(gen_curriculum_mixed(4), DIGITS_DIST_EXPERIMENTS['mixed'])
assert np.allclose(gen_curriculum_combined(4), DIGITS_DIST_EXPERIMENTS['combined'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('run_id')
    parser.add_argument('--curriculum', choices=['baseline', 'naive', 'mixed', 'combined'], default='baseline')
    parser.add_argument('--max_timesteps', type=int, default=2000)
    parser.add_argument('--max_digits', type=int, default=4)
    parser.add_argument('--invert', action='store_true', default=True)
    parser.add_argument('--no_invert', action='store_false', dest='invert')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--train_size', type=int, default=40960)
    parser.add_argument('--val_size', type=int, default=4096)
    parser.add_argument('--optimizer_lr', type=float, default=0.001)
    parser.add_argument('--clipnorm', type=float, default=2)
    parser.add_argument('--logdir', default='logs')
    args = parser.parse_args()

    if args.curriculum == 'baseline':
        curriculum_steps = gen_curriculum_baseline(args.max_digits)
    elif args.curriculum == 'naive':
        curriculum_steps = gen_curriculum_naive(args.max_digits)
    elif args.curriculum == 'mixed':
        curriculum_steps = gen_curriculum_mixed(args.max_digits)
    elif args.curriculum == 'combined':
        curriculum_steps = gen_curriculum_combined(args.max_digits)
    else:
        assert False

    logdir = os.path.join(args.logdir, "{0}digits-curriculum_{1}-{2}".format(args.max_digits, args.curriculum, args.run_id))
    writer = create_summary_writer(logdir)

    model = AdditionRNNModel(args.max_digits, args.hidden_size, args.batch_size, args.invert, args.optimizer_lr, args.clipnorm)

    val_dist = curriculum_steps[-1]
    env = AdditionRNNEnvironment(model, args.train_size, args.val_size, val_dist, writer)

    for train_dist in curriculum_steps:
        while model.epochs < args.max_timesteps:
            r, train_done, val_done = env.step(train_dist)
            if train_done:
                break

    print("Finished after", model.epochs, "epochs.")
    assert val_done
