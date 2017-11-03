import os
import logging
import glob
import re
import math
import time
import csv

import multiprocessing
from queue import Empty
from collections import defaultdict
import pickle
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BatchedTrainer(object):

    def __init__(self, create_environment, create_policy, args):
        self.create_environment = create_environment
        self.create_policy = create_policy
        self.args = args

    def runner(self, env_id, shared_buffer, fifo, num_timesteps, logdir, id):
        proc_name = multiprocessing.current_process().name
        logger.info("Runner %s started" % proc_name)

        # local environment for runner
        env = self.create_environment(env_id, id, os.path.join(logdir, 'gym'), **vars(self.args))

        # copy of policy
        policy = self.create_policy(env.observation_space, env.action_space, batch_size=1, stochastic=True, args=self.args)

        # record episode lengths and rewards for statistics
        episode_rewards = []
        episode_lengths = []
        episode_reward = 0
        episode_length = 0

        observation = env.reset()
        for i in range(math.ceil(float(num_timesteps) / self.args.num_local_steps)):
            # copy weights from main network at the beginning of iteration
            # the main network's weights are only read, never modified
            # but we create our own model instance, because Keras is not thread-safe
            policy.set_weights(pickle.loads(shared_buffer.raw))

            observations = []
            preds = []
            rewards = []
            terminals = []
            infos = defaultdict(list)

            for t in range(self.args.num_local_steps):
                if self.args.display:
                    env.render()

                # predict action probabilities (and state value)
                gym_action, pred = policy.predict([observation])
                # strip batch dimension
                pred = [p[0] for p in pred]

                # step environment and log data
                observations.append(observation)
                preds.append(pred)
                observation, reward, terminal, info = env.step(gym_action[0])
                rewards.append(reward)
                terminals.append(terminal)

                # record environment diagnostics from info
                for key, val in info.items():
                    try:
                        val = float(val)
                        infos[key].append(val)
                    except (TypeError, ValueError):
                        pass

                episode_reward += reward
                episode_length += 1

                # reset if terminal state
                if terminal:
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    episode_reward = 0
                    episode_length = 0
                    observation = env.reset()

            # predict value for the next observation
            # needed for calculating n-step returns
            _, pred = policy.predict([observation])
            # strip batch dimension
            pred = [p[0] for p in pred]
            preds.append(pred)

            # send observations, actions, rewards and returns
            # block if fifo is full
            fifo.put((
                observations,
                preds,
                rewards,
                terminals,
                episode_rewards,
                episode_lengths,
                {key: np.mean(val) for key, val in infos.items()}
            ))
            episode_rewards = []
            episode_lengths = []

        env.close()

        logger.info("Runner %s finished" % proc_name)

    def trainer(self, policy, fifos, shared_buffer, start_timestep, num_timesteps, logdir):
        proc_name = multiprocessing.current_process().name
        logger.info("Trainer %s started" % proc_name)

        # must import tensorflow here, otherwise sometimes it conflicts with multiprocessing
        from common.tensorboard_utils import create_summary_writer, add_summary
        writer = create_summary_writer(logdir)

        timestep = start_timestep
        total_episodes = 0
        total_timesteps = 0
        total_updates = 0
        total_rewards = []
        episode_rewards = []
        episode_lengths = []
        task_rewards = [[] for _ in range(len(self.args.load_mission))]
        stats_start = time.time()
        stats_timesteps = 0
        stats_updates = 0
        while timestep < num_timesteps:
            batch_observations = []
            batch_preds = []
            batch_rewards = []
            batch_terminals = []
            batch_timesteps = 0
            mean_infos = defaultdict(list)
            queue_sizes = []

            # loop over fifos from all runners
            for q, fifo in enumerate(fifos):
                try:
                    # Queue.qsize() is not implemented on Mac, ignore as it is used only for diagnostics
                    try:
                        queue_sizes.append(fifo.qsize())
                    except NotImplementedError:
                        pass

                    # wait for a new trajectory and statistics
                    observations, preds, rewards, terminals, episode_reward, episode_length, mean_info = \
                        fifo.get(timeout=self.args.queue_timeout)

                    # add to batch
                    batch_observations.append(observations)
                    batch_preds.append(preds)
                    batch_rewards.append(rewards)
                    batch_terminals.append(terminals)

                    # log statistics
                    total_rewards += episode_reward
                    episode_rewards += episode_reward
                    episode_lengths += episode_length
                    batch_timesteps += len(observations)
                    task_rewards[q % len(task_rewards)] += episode_reward

                    for key, val in mean_info.items():
                        mean_infos[key].append(val)

                except Empty:
                    # just ignore empty fifos, batch will be smaller
                    pass

            # if any of the runners produced trajectories
            if len(batch_observations) > 0:
                timestep += batch_timesteps

                # reorder dimensions for preds
                batch_preds = [list(zip(*p)) for p in batch_preds]
                batch_preds = list(zip(*batch_preds))

                # train model
                policy.train(batch_observations, batch_preds, batch_rewards, batch_terminals, timestep, writer)

                # share model parameters
                shared_buffer.raw = pickle.dumps(policy.get_weights(), pickle.HIGHEST_PROTOCOL)

                total_timesteps += batch_timesteps
                total_updates += self.args.repeat_updates
                stats_timesteps += batch_timesteps
                stats_updates += self.args.repeat_updates

                for key, val in mean_infos.items():
                    add_summary(writer, "diagnostics/"+key, np.mean(val), timestep)

                if timestep % self.args.stats_interval == 0:
                    total_episodes += len(episode_rewards)
                    stats_time = time.time() - stats_start
                    add_summary(writer, "game_stats/episodes", len(episode_rewards), timestep)
                    add_summary(writer, "game_stats/episode_reward_mean", np.mean(episode_rewards), timestep)
                    #add_summary(writer, "game_stats/episode_reward_stddev", np.std(episode_rewards), timestep)
                    add_summary(writer, "game_stats/episode_length_mean", np.mean(episode_lengths), timestep)
                    #add_summary(writer, "game_stats/episode_length_stddev", np.std(episode_lengths), timestep)

                    add_summary(writer, "game_stats/total_episodes", total_episodes, timestep)
                    add_summary(writer, "game_stats/total_timesteps", total_timesteps, timestep)
                    add_summary(writer, "game_stats/total_updates", total_updates, timestep)

                    add_summary(writer, "performance/updates_per_second", stats_updates / stats_time, timestep)
                    add_summary(writer, "performance/timesteps_per_second", stats_timesteps / stats_time, timestep)
                    add_summary(writer, "performance/estimated_runner_fps", stats_timesteps / self.args.num_runners / stats_time, timestep)
                    add_summary(writer, "performance/mean_queue_length", np.mean(queue_sizes), timestep)

                    for i, rewards in enumerate(task_rewards):
                        add_summary(writer, "curriculum_rewards/task%d_reward_mean" % i, np.mean(rewards), timestep)
                        add_summary(writer, "curriculum_episodes/task%d_episodes" % i, len(rewards), timestep)

                    logger.info("Step %d/%d: episodes %d, mean episode reward %.2f, mean episode length %.2f, timesteps/sec %.2f." %
                        (timestep, num_timesteps, len(episode_rewards), np.mean(episode_rewards), np.mean(episode_lengths),
                            stats_timesteps / stats_time))
                    episode_rewards = []
                    episode_lengths = []
                    task_rewards = [[] for _ in range(len(self.args.load_mission))]
                    stats_start = time.time()
                    stats_timesteps = 0
                    stats_updates = 0

                if timestep % self.args.save_interval == 0:
                    policy.save_weights(os.path.join(logdir, "weights_%d.hdf5" % timestep))

            #else:
                #logger.warn("Empty batch, runners are falling behind!")

        # save final weights
        policy.save_weights(os.path.join(logdir, "weights_%d.hdf5" % timestep))

        if self.args.csv_file:
            # save command-line parameters and most important performance metrics to file
            data = vars(self.args)
            data['episode_reward_mean'] = np.mean(total_rewards)
            data['total_episodes'] = total_episodes
            data['total_timesteps'] = total_timesteps
            data['total_updates'] = total_updates
            header = sorted(data.keys())

            # write the CSV file one directory above the experiment directory
            csv_file = os.path.join(os.path.dirname(logdir), self.args.csv_file)
            file_exists = os.path.isfile(csv_file)
            with open(csv_file, 'a') as file:
                writer = csv.DictWriter(file, delimiter=',', fieldnames=header)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data)

        # collect child processes
        while len(multiprocessing.active_children()) > 0:
            for fifo in fifos:
                # empty fifos just in case runners are waiting after them
                try:
                    fifo.get(timeout=1)
                except Empty:
                    pass

        logger.info("Trainer %s finished" % proc_name)

    def run(self, env_id, num_timesteps, logdir):
        # use spawn method for starting subprocesses
        ctx = multiprocessing.get_context('spawn')

        # create dummy environment to be able to create model
        env = self.create_environment(env_id, monitor_logdir=os.path.join(logdir, 'gym'), **vars(self.args))
        logger.info("Observation space: " + str(env.observation_space))
        logger.info("Action space: " + str(env.action_space))

        # use fixed batch size ONLY when queue timeout is None, i.e. blocks indefinitely until full batch is achieved
        # needed for stateful RNNs
        batch_size = self.args.num_runners if self.args.queue_timeout is None else None

        # create main model
        policy = self.create_policy(env.observation_space, env.action_space, batch_size, True, self.args)
        policy.summary()
        env.close()

        # check for commandline argument or previous weights file
        start_timestep = 0
        weights_file = None
        if self.args.load_weights:
            weights_file = self.args.load_weights
        else:
            files = glob.glob(os.path.join(logdir, "weights_*.hdf5"))
            if files:
                weights_file = max(files, key=lambda f: int(re.search(r'_(\d+).hdf5', f).group(1)))
                # set start timestep from file name when continuing previous session
                start_timestep = int(re.search(r'_(\d+).hdf5', weights_file).group(1))
                logger.info("Setting start timestep to %d" % start_timestep)

        # load saved weights
        if weights_file:
            logger.info("Loading weights: " + weights_file)
            policy.load_weights(weights_file)

        # create shared buffer for sharing weights
        blob = pickle.dumps(policy.get_weights(), pickle.HIGHEST_PROTOCOL)
        shared_buffer = ctx.Array('c', len(blob))
        shared_buffer.raw = blob

        # number of timesteps each runner has to make
        runner_timesteps = math.ceil((num_timesteps - start_timestep) / self.args.num_runners)

        # force runner processes to use cpu, child processes inherit environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # force child processes to use one thread only
        os.environ["OMP_NUM_THREADS"] = "1"

        # create fifos and threads for all runners
        fifos = []
        for i in range(self.args.num_runners):
            fifo = ctx.Queue(self.args.queue_length)
            fifos.append(fifo)
            process = ctx.Process(target=self.runner, args=(env_id, shared_buffer, fifo, runner_timesteps, logdir, i))
            process.start()

        # start trainer in main thread
        self.trainer(policy, fifos, shared_buffer, start_timestep, num_timesteps, logdir)

        logger.info("All done")

    def eval(self, env_id, num_timesteps, logdir):
        env = self.create_environment(env_id, monitor_logdir=os.path.join(logdir, 'gym'), **vars(self.args))
        logger.info("Observation space: " + str(env.observation_space))
        logger.info("Action space: " + str(env.action_space))

        # create main model
        batch_size = 1
        policy = self.create_policy(env.observation_space, env.action_space, batch_size, self.args.stochastic, self.args)
        policy.summary()

        weights_file = None
        if self.args.load_weights:
            weights_file = self.args.load_weights
        else:
            files = glob.glob(os.path.join(logdir, "weights_*.hdf5"))
            if files:
                weights_file = max(files, key=lambda f: int(re.search(r'_(\d+).hdf5', f).group(1)))

        # load saved weights
        if weights_file:
            logger.info("Loading weights: " + weights_file)
            policy.load_weights(weights_file)

        # record episode lengths and rewards for statistics
        episode_rewards = []
        episode_lengths = []
        episode_reward = 0
        episode_length = 0

        observation = env.reset()
        for i in range(num_timesteps):
            if self.args.display:
                env.render()

            # predict action probabilities (and state value)
            gym_action, _ = policy.predict([observation])

            # step environment and log data
            observation, reward, terminal, info = env.step(gym_action[0])

            episode_reward += reward
            episode_length += 1

            # reset if terminal state
            if terminal:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_reward = 0
                episode_length = 0
                observation = env.reset()

        logger.info("Episodes %d, mean episode reward %.2f, mean episode length %.2f." % (len(episode_rewards), np.mean(episode_rewards), np.mean(episode_lengths)))

        env.close()
        logger.info("All done")


class BatchedTrainerProfiler(BatchedTrainer):

    def runner(self, env_id, shared_buffer, fifo, num_timesteps, logdir, id):
        if id == 0:
            import cProfile
            command = """super(BatchedTrainerProfiler, self).runner(env_id, shared_buffer, fifo, num_timesteps, logdir, id)"""
            cProfile.runctx(command, globals(), locals(), filename=os.path.join(logdir, "runner.profile"))
        else:
            super(BatchedTrainerProfiler, self).runner(env_id, shared_buffer, fifo, num_timesteps, logdir, id)

    def trainer(self, policy, fifos, shared_buffer, start_timestep, num_timesteps, logdir):
        import cProfile
        command = """super(BatchedTrainerProfiler, self).trainer(policy, fifos, shared_buffer, start_timestep, num_timesteps, logdir)"""
        cProfile.runctx(command, globals(), locals(), filename=os.path.join(logdir, "trainer.profile"))
