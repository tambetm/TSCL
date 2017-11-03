import time
import logging

import cv2
import numpy as np

from gym.spaces import Discrete, Box
from universe import pyprofile
from universe import vectorized
from universe import spaces as vnc_spaces
from universe.spaces.vnc_event import keycode, PointerEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)
#universe.configure_logging()


def _remove_outliers(reward, clip_value):
    if reward > clip_value or reward < -clip_value:
        return 0
    return reward


class RewardOutlierWrapper(vectorized.RewardWrapper):
    def __init__(self, env, outlier_limit):
        super(RewardOutlierWrapper, self).__init__(env)
        self.outlier_limit = outlier_limit

    def _reward(self, reward_n):
        return [_remove_outliers(reward, self.outlier_limit) for reward in reward_n]


class AddObservationNoise(vectorized.ObservationWrapper):
    def __init__(self, env, noise_level):
        super(AddObservationNoise, self).__init__(env)
        self.noise_level = noise_level

    def _observation(self, observation_n):
        return [observation + self.noise_level * np.random.randn(*observation.shape) for observation in observation_n]


class ObservationBuffer(vectorized.Wrapper):
    def __init__(self, env, buffer_size=4):
        super(ObservationBuffer, self).__init__(env)
        self.buffer_size = buffer_size
        assert len(self.env.observation_space.shape) == 3
        self._shape = list(self.env.observation_space.shape)
        self._num_channels = self._shape[2]
        self._shape[2] *= self.buffer_size
        self.observation_space = Box(-0.5, 0.5, self._shape)

    def _configure(self, **kwargs):
        super(ObservationBuffer, self)._configure(**kwargs)
        self.buffer_n = [np.empty(self._shape) for _ in range(self.n)]

    def _step(self, action_n):
        with pyprofile.push('env.obs_buffer.step'):
            with pyprofile.push('env.obs_buffer.after_step'):
                observation_n, reward_n, done_n, info = self.env.step(action_n)
            num_c = self._num_channels
            for i in range(self.n):
                buffer = self.buffer_n[i]
                buffer[..., 0:num_c * (self.buffer_size-1)] = buffer[..., num_c:num_c * self.buffer_size]
                buffer[..., -num_c:] = observation_n[0]
            return [buffer.copy() for buffer in self.buffer_n], reward_n, done_n, info

    def _reset(self):
        if self.metadata.get('semantics.async'):
            return self.env.reset()
        else:
            obs_n = self.env.reset()
            for i in range(self.n):
                self.buffer_n[i][:] = np.concatenate([obs_n[i]] * self.buffer_size, axis=2)
            return [buffer.copy() for buffer in self.buffer_n]

    def _render(self, mode='human', close=False):
        if mode == "rgb_array":
            return [buffer[..., -self._num_channels:] for buffer in self.buffer_n]
        return self.env.render(mode, close)


def DiagnosticsInfo(env, *args, **kwargs):
    return vectorized.VectorizeFilter(env, DiagnosticsInfoI, *args, **kwargs)


class DiagnosticsInfoI(vectorized.Filter):
    def __init__(self, log_interval=503):
        super(DiagnosticsInfoI, self).__init__()

        self._episode_time = time.time()
        self._last_time = time.time()
        self._local_t = 0
        self._log_interval = log_interval
        self._num_vnc_updates = 0
        self._last_episode_id = -1

        self._clear_state()

    def _clear_state(self):
        self._episode_reward = 0
        self._episode_length = 0
        self._all_rewards = []
        self._episode_time = time.time()

    def _after_reset(self, observation):
        self._clear_state()
        return observation

    def _after_step(self, observation, reward, done, info):
        tb_keys = {}  # which stats are to be displayed on tensorboard

        self._local_t += 1
        if info.get("stats.vnc.updates.n") is not None:
            self._num_vnc_updates += info.get("stats.vnc.updates.n")

        if self._local_t % self._log_interval == 0:
            cur_time = time.time()
            elapsed = cur_time - self._last_time
            fps = self._log_interval / elapsed
            self._last_time = cur_time
            cur_episode_id = info.get('vectorized.episode_id', 0)
            tb_keys["diagnostics/fps"] = fps
            if self._last_episode_id == cur_episode_id:
                tb_keys["diagnostics/fps_within_episode"] = fps
            self._last_episode_id = cur_episode_id
            if info.get("stats.gauges.diagnostics.lag.action") is not None:
                tb_keys["diagnostics/action_lag_lb"] = info["stats.gauges.diagnostics.lag.action"][0]
                tb_keys["diagnostics/action_lag_ub"] = info["stats.gauges.diagnostics.lag.action"][1]
            if info.get("reward.count") is not None:
                tb_keys["diagnostics/reward_count"] = info["reward.count"]
            if info.get("stats.gauges.diagnostics.clock_skew") is not None:
                tb_keys["diagnostics/clock_skew_lb"] = info["stats.gauges.diagnostics.clock_skew"][0]
                tb_keys["diagnostics/clock_skew_ub"] = info["stats.gauges.diagnostics.clock_skew"][1]
            if info.get("stats.gauges.diagnostics.lag.observation") is not None:
                tb_keys["diagnostics/observation_lag_lb"] = info["stats.gauges.diagnostics.lag.observation"][0]
                tb_keys["diagnostics/observation_lag_ub"] = info["stats.gauges.diagnostics.lag.observation"][1]

            if info.get("stats.vnc.updates.n") is not None:
                tb_keys["diagnostics/vnc_updates_n"] = info["stats.vnc.updates.n"]
                tb_keys["diagnostics/vnc_updates_n_ps"] = self._num_vnc_updates / elapsed
                self._num_vnc_updates = 0
            if info.get("stats.vnc.updates.bytes") is not None:
                tb_keys["diagnostics/vnc_updates_bytes"] = info["stats.vnc.updates.bytes"]
            if info.get("stats.vnc.updates.pixels") is not None:
                tb_keys["diagnostics/vnc_updates_pixels"] = info["stats.vnc.updates.pixels"]
            if info.get("stats.vnc.updates.rectangles") is not None:
                tb_keys["diagnostics/vnc_updates_rectangles"] = info["stats.vnc.updates.rectangles"]
            if info.get("env_status.state_id") is not None:
                tb_keys["diagnostics/env_state_id"] = info["env_status.state_id"]

        if reward is not None:
            self._episode_reward += reward
            if observation is not None:
                self._episode_length += 1
            self._all_rewards.append(reward)

        if done:
            if self._episode_length > 1:  # ignore 0 length episodes
                total_time_including_reset = time.time() - self._episode_time
                if 'reward_buffer.done_time' in info:
                    total_time = info['reward_buffer.done_time'] - self._episode_time
                else:
                    # in case this is being run on a non-VNC env (e.g. for sanity tests)
                    total_time = total_time_including_reset

                logger.info('Episode terminating: episode_reward=%s episode_length=%s total_time=%.2f total_time_including_reset=%.2f', self._episode_reward, self._episode_length, total_time, total_time_including_reset)

                tb_keys["global/episode_reward"] = self._episode_reward
                tb_keys["global/episode_length"] = self._episode_length
                tb_keys["global/episode_time"] = total_time
                tb_keys["global/reward_per_time"] = self._episode_reward / total_time
            else:
                logger.info('Episode terminating: episode_reward=%s episode_length=%s. WARNING: not adding to global stats since episode was empty', self._episode_reward, self._episode_length)
            # At this point, we've already completed the reset for the
            # previous episode.
            self._clear_state()

        info['tensorboard_keys'] = tb_keys
        return observation, reward, done, info


def _process_frame42(frame):
    frame = frame[34:34+160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [42, 42, 1])
    return frame


class AtariRescale42x42(vectorized.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [42, 42, 1])

    def _observation(self, observation_n):
        return [_process_frame42(observation) for observation in observation_n]


class FixedKeyState(object):
    def __init__(self, keys):
        self._keys = [keycode(key) for key in keys]
        self._down_keysyms = set()

    def apply_vnc_actions(self, vnc_actions):
        for event in vnc_actions:
            if isinstance(event, vnc_spaces.KeyEvent):
                if event.down:
                    self._down_keysyms.add(event.key)
                else:
                    self._down_keysyms.discard(event.key)

    def to_index(self):
        action_n = 0
        for key in self._down_keysyms:
            if key in self._keys:
                # If multiple keys are pressed, just use the first one
                action_n = self._keys.index(key) + 1
                break
        return action_n


class DiscreteToFixedKeysVNCActions(vectorized.ActionWrapper):
    """
    Define a fixed action space. Action 0 is all keys up. Each element of keys can be a single key or a space-separated list of keys

    For example,
       e=DiscreteToFixedKeysVNCActions(e, ['left', 'right'])
    will have 3 actions: [none, left, right]

    You can define a state with more than one key down by separating with spaces. For example,
       e=DiscreteToFixedKeysVNCActions(e, ['left', 'right', 'space', 'left space', 'right space'])
    will have 6 actions: [none, left, right, space, left space, right space]
    """
    def __init__(self, env, keys):
        super(DiscreteToFixedKeysVNCActions, self).__init__(env)

        self._keys = keys
        self._generate_actions()
        self.action_space = Discrete(len(self._actions))

    def _generate_actions(self):
        self._actions = []
        uniq_keys = set()
        for key in self._keys:
            for cur_key in key.split(' '):
                uniq_keys.add(cur_key)

        for key in [''] + self._keys:
            split_keys = key.split(' ')
            cur_action = []
            for cur_key in uniq_keys:
                cur_action.append(vnc_spaces.KeyEvent.by_name(cur_key, down=(cur_key in split_keys)))
            self._actions.append(cur_action)
        self.key_state = FixedKeyState(uniq_keys)

    def _action(self, action_n):
        # Each action might be a length-1 np.array. Cast to int to
        # avoid warnings.
        return [self._actions[int(action)] for action in action_n]


class CropScreen(vectorized.ObservationWrapper):
    """ crops out a [height]x[width] area starting from (top,left) """
    def __init__(self, env, height, width, top=0, left=0, normalize=False):
        super(CropScreen, self).__init__(env)
        self.height = height
        self.width = width
        self.top = top
        self.left = left
        self.normalize = normalize
        if normalize:
            self.observation_space = Box(0, 1.0, shape=(height, width, 3))
        else:
            self.observation_space = Box(0, 255, shape=(height, width, 3))

    def _observation(self, observation_n):
        with pyprofile.push('env.crop_screen.observation'):
            if self.normalize:
                return [ob[self.top:self.top+self.height, self.left:self.left+self.width, :].astype(np.float32) * (1.0/255.0) if ob is not None else None
                        for ob in observation_n]
            else:
                return [ob[self.top:self.top+self.height, self.left:self.left+self.width, :] if ob is not None else None
                        for ob in observation_n]


def _process_frame_flash(frame):
    frame = cv2.resize(frame, (200, 128))
    frame = frame.mean(2).astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [128, 200, 1])
    return frame


class FlashRescale(vectorized.ObservationWrapper):
    def __init__(self, env=None):
        super(FlashRescale, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [128, 200, 1])

    def _observation(self, observation_n):
        with pyprofile.push('env.flash_rescale.observation'):
            return [_process_frame_flash(observation) for observation in observation_n]


class VNCActionsMouseDirection(vectorized.ActionWrapper):
    """
    An action space for where only the direction from the center that you click in matters.
    Input is (sector, mousebuttons)

    """
    def __init__(self, env,
                 buttonmasks=(1,),
                 mouse_center=(251, 150),
                 mouse_radius=140,
                 mouse_sectors=16):
        super(VNCActionsMouseDirection, self).__init__(env)
        self._buttonmasks = buttonmasks
        self._mouse_center = mouse_center
        self._mouse_radius = mouse_radius
        self._mouse_sectors = mouse_sectors

        self._ra_x = None
        self._ra_y = None
        self._ra_buttonmask = None

        self._buttonmask_dim = (2 ** len(self._buttonmasks))

        self.action_space = Discrete(self._mouse_sectors * self._buttonmask_dim)
        logger.debug('VNCActionsMouseDirection action_space=%s' % (self.action_space))

    def _action(self, action_n):

        ret = []
        for action in action_n:
            assert self.action_space.contains(action)

            sector = action % self._mouse_sectors
            buttonmask = action // self._mouse_sectors

            angle = 2.0*np.pi*float(sector)/float(self._mouse_sectors)

            x = self._mouse_center[0] + self._mouse_radius*np.cos(angle)
            y = self._mouse_center[1] - self._mouse_radius*np.sin(angle)

            # translate these to VNC action events
            actions = []

            # emit PointerEvent. cast x,y to ints since they can be numpy arrays
            maskint = sum(self._buttonmasks[i] * ((buttonmask >> i) & 1) for i in range(len(self._buttonmasks)))
            actions.append(PointerEvent(x, y, maskint))

            if 0:
                logger.info('mouse action %s => %s' % (action, actions))

            ret.append(actions)

        logger.debug('_action(', action_n, ') = ', ret)
        return ret

    def _reverse_action(self, action_n):

        if self._ra_x is None:
            self._ra_x = np.zeros((self.n,))
            self._ra_y = np.zeros((self.n,))
            self._ra_buttonmask = np.zeros((self.n, len(self._buttonmasks)), dtype=np.int64)

        ret = []
        for i, action in enumerate(action_n):
            # action is a list of VNC events

            for event in action:
                """
                NOTE: it's tricky to handle multiple mouse events. Here we will
                take the x,y position of the LAST mouse event and we will do an
                OR operation across all button masks of all events, merging the
                pressed buttons together.
                """
                if isinstance(event, PointerEvent):
                    self._ra_x[i] = event.x
                    self._ra_y[i] = event.y
                    for j in range(len(self._buttonmasks)):
                        self._ra_buttonmask[i, j] = 1 if (event.buttonmask & self._buttonmasks[j]) else 0

            angle = np.atan2(self._ra_y[i] - self._mouse_center[1], self._ra_x[i] - self._mouse_center[0])   # returns 0 for (0,0)
            sector = np.round(angle/(2.0*np.pi) * self._mouse_sectors * angle)

            ret.append(sector, self._ra_buttonmasks[i, :])

        logger.debug('_reverse_action(', action_n, ') = ', ret)
        return ret
