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
from copy import copy
import pickle
import numpy as np

from trainers.batched import BatchedTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BatchedPredictorTrainer(BatchedTrainer):

    def runner(self, env_id, shared_buffer, fifo, num_timesteps, logdir):
        proc_name = multiprocessing.current_process().name
        logger.info("Runner %s started" % proc_name)

        # local environments for runner
        envs = []
        for i in range(self.args.num_runners):
            env = self.create_environment(env_id, i, os.path.join(logdir, 'gym'), **vars(self.args))
            envs.append(env)

        # copy of policy
        policy = self.create_policy(env.observation_space, env.action_space, batch_size=self.args.num_runners, stochastic=True, args=self.args)

        # record episode lengths and rewards for statistics
        episode_rewards = []
        episode_lengths = []
        episode_reward_n = np.zeros(len(envs))
        episode_length_n = np.zeros(len(envs))

        observation_n = [None] * len(envs)
        reward_n = [None] * len(envs)
        terminal_n = [None] * len(envs)
        info_n = [None] * len(envs)

        for i, env in enumerate(envs):
            observation_n[i] = env.reset()
        for k in range(math.ceil(float(num_timesteps) / (self.args.num_local_steps * self.args.num_runners))):
            # copy weights from main network at the beginning of iteration
            # the main network's weights are only read, never modified
            # but we create our own model instance, because Keras is not thread-safe
            policy.set_weights(pickle.loads(shared_buffer.raw))

            observations_tn = []
            preds_tn = []
            rewards_tn = []
            terminals_tn = []
            infos_kt = defaultdict(list)

            for t in range(self.args.num_local_steps):
                if self.args.display:
                    for env in envs:
                        env.render()

                # predict action probabilities (and state value)
                action_n, pred_n = policy.predict(observation_n)

                # step environment and log data
                observations_tn.append(copy(observation_n))
                preds_tn.append(copy(pred_n))
                for i, env in enumerate(envs):
                    #print(i, observation[i], action[i])
                    observation_n[i], reward_n[i], terminal_n[i], info_n[i] = env.step(action_n[i])
                    #print(i, reward[i], terminal[i])

                    episode_reward_n[i] += reward_n[i]
                    episode_length_n[i] += 1

                    # record environment diagnostics from info
                    for key, val in info_n[i].items():
                        try:
                            val = float(val)
                            infos_kt[key].append(val)
                        except (TypeError, ValueError):
                            pass

                    # reset if terminal state
                    if terminal_n[i]:
                        episode_rewards.append(episode_reward_n[i])
                        episode_lengths.append(episode_length_n[i])
                        episode_reward_n[i] = 0
                        episode_length_n[i] = 0
                        observation_n[i] = env.reset()

                rewards_tn.append(copy(reward_n))
                terminals_tn.append(copy(terminal_n))

            # otherwise calculate the value of the last state
            _, pred_n = policy.predict(observation_n)
            preds_tn.append(copy(pred_n))
            # flip dimensions for preds
            preds_tn = list(zip(*preds_tn))

            # swap batch and timestep axes
            observations_nt = np.swapaxes(observations_tn, 0, 1)
            preds_nt = [None] * len(preds_tn)
            for i in range(len(preds_tn)):
                preds_nt[i] = np.swapaxes(preds_tn[i], 0, 1)
            rewards_nt = np.swapaxes(rewards_tn, 0, 1)
            terminals_nt = np.swapaxes(terminals_tn, 0, 1)

            # send observations, actions, rewards and returns
            # block if fifo is full
            fifo.put((
                observations_nt,
                preds_nt,
                rewards_nt,
                terminals_nt,
                episode_rewards,
                episode_lengths,
                {key: np.mean(val) for key, val in infos_kt.items()}
            ))
            episode_rewards = []
            episode_lengths = []

        for env in envs:
            env.close()

        logger.info("Runner %s finished" % proc_name)

    def trainer(self, policy, fifo, shared_buffer, start_timestep, num_timesteps, logdir):
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
        stats_start = time.time()
        stats_timesteps = 0
        stats_updates = 0
        queue_sizes = []
        while timestep < num_timesteps:
            mean_infos = defaultdict(list)
            # Queue.qsize() is not implemented on Mac, ignore as it is used only for diagnostics
            try:
                queue_sizes.append(fifo.qsize())
            except NotImplementedError:
                pass

            # wait for a new trajectory and statistics
            batch_observations, batch_preds, batch_rewards, batch_terminals, episode_reward, episode_length, mean_info = fifo.get()

            # log statistics
            total_rewards += episode_reward
            episode_rewards += episode_reward
            episode_lengths += episode_length
            batch_timesteps = np.prod(batch_observations.shape[:2])

            for key, val in mean_info.items():
                mean_infos[key].append(val)

            timestep += batch_timesteps

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
                add_summary(writer, "game_stats/total_timesteps", int(total_timesteps), timestep)
                add_summary(writer, "game_stats/total_updates", total_updates, timestep)

                add_summary(writer, "performance/updates_per_second", stats_updates / stats_time, timestep)
                add_summary(writer, "performance/timesteps_per_second", stats_timesteps / stats_time, timestep)
                add_summary(writer, "performance/estimated_runner_fps", stats_timesteps / self.args.num_runners / stats_time, timestep)
                add_summary(writer, "performance/mean_queue_length", np.mean(queue_sizes), timestep)

                logger.info("Step %d/%d: episodes %d, mean episode reward %.2f, mean episode length %.2f, timesteps/sec %.2f." %
                    (timestep, num_timesteps, len(episode_rewards), np.mean(episode_rewards), np.mean(episode_lengths),
                        stats_timesteps / stats_time))
                episode_rewards = []
                episode_lengths = []
                stats_start = time.time()
                stats_timesteps = 0
                stats_updates = 0
                queue_sizes = []

            if timestep % self.args.save_interval == 0:
                policy.save_weights(os.path.join(logdir, "weights_%d.hdf5" % timestep))

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

        if self.args.runner_gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.runner_gpu

        # create fifo and process for runner
        fifo = ctx.Queue(self.args.queue_length)
        process = ctx.Process(target=self.runner, args=(env_id, shared_buffer, fifo, num_timesteps, logdir))
        process.start()

        # start trainer in main process
        self.trainer(policy, fifo, shared_buffer, start_timestep, num_timesteps, logdir)

        logger.info("All done")


class BatchedPredictorTrainerProfiler(BatchedPredictorTrainer):

    def runner(self, env_id, shared_buffer, fifo, num_timesteps, logdir):
        if id == 0:
            import cProfile
            command = """super(BatchedPredictorTrainerProfiler, self).runner(env_id, shared_buffer, fifo, num_timesteps, logdir)"""
            cProfile.runctx(command, globals(), locals(), filename=os.path.join(logdir, "runner.profile"))
        else:
            super(BatchedPredictorTrainerProfiler, self).runner(env_id, shared_buffer, fifo, num_timesteps, logdir)

    def trainer(self, policy, fifo, shared_buffer, start_timestep, num_timesteps, logdir):
        import cProfile
        command = """super(BatchedPredictorTrainerProfiler, self).trainer(policy, fifo, shared_buffer, start_timestep, num_timesteps, logdir)"""
        cProfile.runctx(command, globals(), locals(), filename=os.path.join(logdir, "trainer.profile"))
