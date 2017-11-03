import cv2
import numpy as np
import logging
import re

import gym
from gym.spaces import Discrete, Box, Tuple
from gym import Wrapper, ObservationWrapper, ActionWrapper
from gym.wrappers import Monitor

import time
import uuid
import getpass

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)
#universe.configure_logging()

#logging.getLogger("minecraft_py").setLevel(logging.DEBUG)
#logging.getLogger("gym_minecraft").setLevel(logging.DEBUG)

def create_env(env_id, client_id=0, monitor_logdir=None, n=1, **kwargs):

    if 'Minecraft' in env_id:
        return create_minecraft_env(env_id, client_id, monitor_logdir, **kwargs)
    elif 'NoFrameskip' in env_id:
        return create_atari_env(env_id, monitor_logdir, **kwargs)
    else:
        return create_other_env(env_id, monitor_logdir, **kwargs)


def create_minecraft_env(env_id, id=0, monitor_logdir=None, load_mission=None, allowed_actions=['move', 'turn'], action_space='discrete',
        client_resize=False, host='127.0.0.1', start_port=10001, start_minecraft=False,
        video_width=40, video_height=30, video_depth=False, skip_steps=0, policy='rnn', num_buffer_frames=4, **_):
    # import Minecraft locally to not introduce dependencies
    import gym_minecraft

    env = gym.make(env_id)

    port = start_port + id

    if load_mission:
        assert isinstance(load_mission, list)
        env.load_mission_file(load_mission[id % len(load_mission)])

    # call Minecraft-specific init function that sets up observation and action spaces
    env.init(allowContinuousMovement=allowed_actions,
             continuous_discrete=(action_space == 'discrete'),
             videoResolution=None if client_resize else (video_width, video_height),
             videoWithDepth=video_depth,
             client_pool=[(host, port)],
             start_minecraft=start_minecraft,
             skip_steps=skip_steps)

    # start monitor to record statistics and videos
    if monitor_logdir:
        env = Monitor(env, monitor_logdir, video_callable=False, resume=True)

    if client_resize:
        env = ResizeWrapper(env, video_width, video_height)

    if num_buffer_frames > 0 and policy == 'cnn':
        env = ObservationBuffer(env, num_buffer_frames)

    return env


# NB! Unvectorized!!!
class ResizeWrapper(ObservationWrapper):
    def __init__(self, env, width, height):
        assert isinstance(env.observation_space, Box)
        super(ResizeWrapper, self).__init__(env)
        self.width = width
        self.height = height
        self.observation_space = Box(0, 255, (height, width) + env.observation_space.low.shape[2:])

    def _observation(self, observation):
        return cv2.resize(observation, (self.width, self.height))


def create_atari_env(env_id, monitor_logdir=None, wrappers='deepmind', policy='rnn', num_buffer_frames=4, max_repeats=0, **_):
    env = gym.make(env_id)

    # start monitor to record statistics and videos
    if monitor_logdir:
        env = Monitor(env, monitor_logdir, video_callable=False, resume=True)

    if wrappers == 'deepmind':
        from common.atari_wrappers import wrap_deepmind
        env = wrap_deepmind(env)
    elif wrappers == 'universe':
        from universe.wrappers import Vectorize, Unvectorize
        from common.universe_wrappers import AtariRescale42x42, DiagnosticsInfo
        env = Vectorize(env)
        env = AtariRescale42x42(env)
        env = DiagnosticsInfo(env)
        env = Unvectorize(env)

    if policy == 'cnn' and num_buffer_frames > 0:
        env = ObservationBuffer(env, num_buffer_frames)

    if max_repeats > 0:
        env = FrameskipWrapper(env, max_repeats)

    return env


class ObservationBuffer(Wrapper):
    def __init__(self, env, buffer_size=4):
        super(ObservationBuffer, self).__init__(env)
        assert isinstance(self.env.observation_space, Box)
        assert len(self.env.observation_space.shape) == 3
        self.buffer_size = buffer_size
        low = np.tile(self.env.observation_space.low, (1, 1, self.buffer_size))
        high = np.tile(self.env.observation_space.high, (1, 1, self.buffer_size))
        shape = list(self.env.observation_space.shape)
        self.num_channels = shape[2]
        shape[2] *= self.buffer_size
        self.buffer = np.zeros(shape)
        self.observation_space = Box(low, high)

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.buffer = np.roll(self.buffer, shift=-self.num_channels, axis=2)
        self.buffer[:, :, -self.num_channels:] = observation
        return self.buffer, reward, done, info

    def _reset(self):
        observation = self.env.reset()
        self.buffer = np.tile(observation, (1, 1, self.buffer_size))
        return self.buffer
'''
    def _render(self, mode='human', close=False):
        if mode == 'human':
            import cv2

            for i in range(self.buffer_size):
                cv2.imshow("win%d" % i, self.buffer[:, :, i])
            cv2.waitKey()
        else:
            raise error.UnsupportedMode("Unsupported render mode: " + mode)
'''


class FrameskipWrapper(Wrapper):
    def __init__(self, env, max_repeats=30):
        super(FrameskipWrapper, self).__init__(env)
        self.max_repeats = max_repeats
        self.action_space = Tuple([self.action_space, Discrete(self.max_repeats)])

    def _step(self, action):
        assert self.action_space.contains(action)
        act, repeat = action
        rew = 0
        for i in range(1, repeat + 2):
            obs, r, done, info = self.env.step(act)
            rew += r
            if done:
                break
        return obs, rew, done, info


def create_other_env(env_id, monitor_logdir=None, **_):
    env = gym.make(env_id)

    # start monitor to record statistics and videos
    if monitor_logdir:
        env = Monitor(env, monitor_logdir, video_callable=False, resume=True)

    return env
