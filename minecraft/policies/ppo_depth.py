import numpy as np
from gym.spaces import Tuple

from keras.models import Model
from keras.layers import Input, TimeDistributed, Dense, Convolution2D, Flatten, LSTM, GRU, SimpleRNN, Lambda, Reshape
from keras.objectives import mse
from keras.regularizers import l2
from keras.optimizers import Adam, RMSprop, Nadam
import keras.backend as K

from common.keras_utils import RunningMeanStd, NormalizeMeanStd, DenormalizeMeanStd, get_states, set_states
from common.actions import TupleAction
from common.math_utils import explained_variance

from policies import Policy


class MLPPolicy(Policy):
    def __init__(self, observation_space, action_space, batch_size, stochastic, args):
        self.observation_space = observation_space
        self.action_space = action_space
        self.batch_size = batch_size
        self.stochastic = stochastic
        self.args = args
        self.build()

    def build_hidden_layers(self, x):
        h = x

        # policy network
        for i in range(self.args.fc_layers):
            h = TimeDistributed(Dense(self.args.fc_size, activation=self.args.activation,
                    kernel_initializer=self.args.fc_init, kernel_regularizer=l2(self.args.l2_reg)), name="h%d" % (i + 1))(h)

        return h

    def build_action_layers(self, h):
        self.actions.build(h)

        # build policy layers
        if self.stochastic:
            u = self.actions.sample()
        else:
            u = self.actions.argmax()

        # wrap in Keras Lambda layer, otherwise it cannot be in graph
        logp = Lambda(lambda a: self.actions.logp(a)[0], output_shape=(None,), name="logp")(u)
        p = self.actions.params()

        return u, p, logp

    def build_value_layers(self, h):
        v = TimeDistributed(Dense(1, kernel_initializer=self.args.fc_init, kernel_regularizer=l2(self.args.l2_reg)), name="v")(h)
        return v

    def build(self):
        # wrap action space in Tuple to save ourselves from hierarchy generation
        self.actions = TupleAction(Tuple([self.action_space]), self.args)
        self.action_dim = self.actions.action_dim()   # dimensionality of action outputs
        #print("action_dim:", self.action_dim)

        x = Input(batch_shape=(self.batch_size, None,) + self.observation_space.shape[:2] + (3,), name="x")  # observations
        A = Input(batch_shape=(self.batch_size, None,), name="A")      # advantages
        a = Input(batch_shape=(self.batch_size, None, self.action_dim), name="a")    # taken actions
        R = Input(batch_shape=(self.batch_size, None, 1), name="R")    # discounted returns
        #print("a.ndim:", a.ndim)

        # apply running normalization
        if self.args.normalize_observations:
            # collect running mean and std
            rms = RunningMeanStd(name="xrms")
            xrms = rms(x)
            xz = NormalizeMeanStd(rms.mean(), rms.std(), -10, 10, name="xnorm")(xrms)
        else:
            xz = x

        # create intermediate layers
        h = self.build_hidden_layers(xz)

        # create action layers
        u, p, logp = self.build_action_layers(h)

        # value network
        vz = self.build_value_layers(h)

        d = TimeDistributed(Dense(np.prod(self.observation_space.shape[:2])))(h)
        d = TimeDistributed(Reshape(self.observation_space.shape[:2]), name="d")(d)

        # apply running normalization
        if self.args.normalize_baseline:
            # collect running mean and std
            self.rms = RunningMeanStd(name="Rrms")
            Rrms = self.rms(R)
            # apply normalization
            v = DenormalizeMeanStd(self.rms.mean(), self.rms.std(), name="vdenorm")(vz)
            Rz = NormalizeMeanStd(self.rms.mean(), self.rms.std(), -10, 10, name="Rnorm")(Rrms)
        else:
            Rz = R
            v = vz

        # normalize advantages
        if self.args.normalize_advantage:
            Az = (A - K.mean(A)) / K.std(A)
        else:
            Az = A

        # inputs to the model are observation and total reward,
        # outputs are action probabilities and state value
        #print(logp, u, p, v)
        #print(u.ndim)
        self.model = Model(inputs=[x, A, a, R], outputs=[logp, u, p, v, d])
        # HACK: to get rms updates and weights included in the model
        if self.args.normalize_baseline:
            self.model.layers.append(self.rms)

        # policy gradient loss
        def policy_gradient_loss(logp_old, logp_new):
            if self.args.policy_loss == 'pg':
                return K.mean(Az * -self.actions.logp(a)[0], axis=1)
            elif self.args.policy_loss == 'pposgd':
                ratio = K.exp(self.actions.logp(a)[0] - logp_old)
                surr1 = ratio * Az
                surr2 = K.clip(ratio, 1 - self.args.clip_param, 1 + self.args.clip_param) * Az
                return -K.mean(K.minimum(surr1, surr2), axis=1)
            else:
                assert False

        # entropy bonus
        def entropy_bonus_loss(a_old, a_new):
            # ignore parameters a_old, a_new, they are used in other loss functions or as output
            # instead use it to define entropy loss, which doesn't have external inputs anyway
            return self.args.entropy_coef * -K.mean(self.actions.entropy(), axis=1) \
                    + self.args.backward_coef * -K.sum(K.minimum(self.actions.argmax()[:, :, 0], 0))  # penalty for going backwards

        # KL penalty
        def kl_penalty_loss(p_old, p_new):
            self.kld_coef = K.variable(self.args.kld_coef, name="kld_coef")
            return self.kld_coef * K.mean(self.actions.kld(p_old)[0], axis=1)

        # value loss
        def value_loss(v_actual, v_predicted):
            # use normalized value instead of original
            return self.args.value_coef * K.mean(mse(vz, Rz), axis=1)

        # depth loss
        def depth_loss(d_actual, d_predicted):
            return self.args.depth_coef * K.mean(mse((d_actual / 255.0) ** 10, d_predicted), axis=1)

        if self.args.optimizer == 'adam':
            optimizer = Adam(lr=self.args.optimizer_lr, clipnorm=self.args.clipnorm)
        elif self.args.optimizer == 'rmsprop':
            optimizer = RMSprop(lr=self.args.optimizer_lr, clipnorm=self.args.clipnorm)
        elif self.args.optimizer == 'nadam':
            optimizer = Nadam(lr=self.args.optimizer_lr, clipnorm=self.args.clipnorm)
        else:
            assert False, "Unknown optimizer " + self.args.optimizer

        # losses are summed
        self.model.compile(optimizer=optimizer, loss=[policy_gradient_loss, entropy_bonus_loss, kl_penalty_loss, value_loss, depth_loss])

    def predict(self, observations):
        assert len(observations) == self.batch_size

        # create inputs for batch with one timestep
        x = np.array(observations)[:, np.newaxis, :, :, :3]  # add time axis
        A = np.zeros((self.batch_size, 1))  # dummy advantage
        a = np.zeros((self.batch_size, 1, self.action_dim))  # dummy action
        R = np.zeros((self.batch_size, 1, 1))  # dummy return

        # predict action probabilities (and state value)
        #print(x, A, a, R)
        #print("a:", a.shape, a)
        logp, u, p, v, _ = self.model.predict_on_batch([x, A, a, R])
        #print(logp.shape, u.shape, p.shape, v.shape)

        # convert raw actions into Gym actions
        # take the first element because we wrapped it into tuple
        actions = [self.actions.gym_action(flat)[0][0] for flat in u[:, 0]]

        # return auxiliary information along with actions
        return actions, (u[:, 0], v[:, 0, 0], logp[:, 0], p[:, 0])

    def discount(self, rewards, terminals, values):
        assert rewards.shape == terminals.shape
        assert rewards.shape[0] == values.shape[0]
        assert rewards.shape[1] == values.shape[1] - 1
        # switch batch and timestep axes
        rewards = np.swapaxes(rewards, 0, 1)
        terminals = np.swapaxes(terminals, 0, 1)
        values = np.swapaxes(values, 0, 1)
        # separate the value of the next state
        next_value = values[-1]
        values = values[:-1]

        # calculate discounted future rewards for this trajectory
        returns = []
        advantages = []
        advantage = 0
        # start with the predicted value of the last state
        for reward, terminal, value in zip(reversed(rewards), reversed(terminals), reversed(values)):
            nonterminal = 1 - terminal
            delta = reward + self.args.gamma * next_value * nonterminal - value
            advantage = delta + self.args.gamma * self.args.lam * nonterminal * advantage
            advantages.insert(0, advantage)
            returns.insert(0, advantage + value)
            next_value = value

        # swap timestep and batch axes back
        returns = np.swapaxes(returns, 0, 1)
        advantages = np.swapaxes(advantages, 0, 1)

        #print(returns.shape, advantages.shape)
        return returns, advantages

    def train(self, observations, preds, rewards, terminals, timestep=None, writer=None):
        assert len(observations) == self.batch_size
        assert len(rewards) == self.batch_size
        assert len(terminals) == self.batch_size

        # split preds into auxiliary information
        assert len(preds) == 4
        actions, values, logprobs, params = preds
        assert len(actions) == self.batch_size
        assert len(values) == self.batch_size
        assert len(logprobs) == self.batch_size
        assert len(params) == self.batch_size

        # convert to numpy arrays
        v = np.array(values)
        r = np.array(rewards)
        t = np.array(terminals)

        #print(v.shape, r.shape, t.shape)

        # calculate discounted returns
        returns, advantages = self.discount(r, t, v)

        # form training data from observations, actions and returns
        x = np.array(observations)
        a = np.array(actions)[:, :-1]
        R = np.array(returns)[:, :, np.newaxis]
        A = np.array(advantages)
        logp = np.array(logprobs)[:, :-1]
        p = np.array(params)[:, :-1]

        # decay learning rate
        #lr = max(self.args.optimizer_lr * (self.args.num_timesteps - timestep) / self.args.num_timesteps, 0)
        #K.set_value(self.model.optimizer.lr, lr)

        # extract depth
        assert x.shape[4] == 4
        d = x[:, :, :, :, 3]
        x = x[:, :, :, :, :3]

        #print(x.shape, a.shape, R.shape, A.shape, logp.shape, p.shape)

        # train the model
        states = get_states(self.model)
        for _ in range(self.args.repeat_updates):
            set_states(self.model, states)
            total_loss, policy_loss, entropy_bonus, kl_penalty, value_loss, depth_loss = self.model.train_on_batch([x, A, a, R], [logp, a, p, R, d])

        if self.args.adapt_kl:
            total_loss, policy_loss, entropy_bonus, kl_penalty, value_loss, depth_loss = self.model.test_on_batch([x, A, a, R], [logp, a, p, R, d])
            # check the KLD of last training batch
            if kl_penalty > self.args.adapt_kl * 2.0:
                newklcoeff = min(K.get_value(self.kld_coef) * 1.5, 1.0)
                K.set_value(self.kld_coef, newklcoeff)
            elif kl_penalty < self.args.adapt_kl / 2.0:
                newklcoeff = max(K.get_value(self.kld_coef) / 1.5, 1e-4)
                K.set_value(self.kld_coef, newklcoeff)

        if timestep and writer:
            # calculate statistics
            mean_expl_var = np.mean(explained_variance(v[:, :-1], R[:, :, 0]))
            mean_params = np.mean(p, axis=0)
            mean_values = np.mean(v)

            # must import tensorflow here, otherwise sometimes it conflicts with multiprocessing
            from common.tensorboard_utils import add_summary

            # log statistics
            #add_summary(writer, "training/learning_rate", lr, timestep)
            #add_summary(writer, "training/batch_timesteps", timesteps, timestep)
            add_summary(writer, "training/batch_size", len(observations), timestep)
            add_summary(writer, "training/loss_total", float(total_loss), timestep)
            add_summary(writer, "training/loss_policy", float(policy_loss), timestep)
            add_summary(writer, "training/loss_entropy_bonus", float(entropy_bonus), timestep)
            add_summary(writer, "training/loss_kl_penalty", float(kl_penalty), timestep)
            add_summary(writer, "training/loss_value", float(value_loss), timestep)
            add_summary(writer, "training/loss_depth", float(depth_loss), timestep)
            #add_summary(writer, "training/entropy_mean", float(np.mean(mean_entropies)), timestep)
            add_summary(writer, "training/explained_variance_mean", float(np.clip(mean_expl_var, -1, 1)), timestep)
            add_summary(writer, "training/value_mean", float(mean_values), timestep)
            add_summary(writer, "training/advantage_mean", float(np.mean(advantages)), timestep)
            add_summary(writer, "training/advantage_std", float(np.std(advantages)), timestep)

            if self.args.normalize_baseline:
                rms_mean = np.array(K.eval(self.rms.mean()))
                rms_std = np.array(K.eval(self.rms.std()))
                add_summary(writer, "training/rms_mean", float(rms_mean), timestep)
                add_summary(writer, "training/rms_std", float(rms_std), timestep)

            if self.args.adapt_kl:
                kld_coef = K.get_value(self.kld_coef)
                add_summary(writer, "training/kld_coef", float(kld_coef), timestep)

            for i in range(mean_params.shape[1]):
                add_summary(writer, "actions/param%d_mean" % i, float(np.mean(mean_params[:, i])), timestep)

        return total_loss, policy_loss, entropy_bonus, kl_penalty, value_loss, depth_loss

    def summary(self):
        return self.model.summary()

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        return self.model.set_weights(weights)

    def load_weights(self, filename):
        return self.model.load_weights(filename, by_name=self.args.weights_by_name)

    def save_weights(self, filename):
        return self.model.save_weights(filename)

    def get_states(self):
        return get_states(self.model)

    def set_states(self, values):
        return set_states(self.model, values)


class CNNPolicy(MLPPolicy):
    def build_cnn_layers(self, x):
        h = x

        if self.args.cnn_architecture == 'deepmind':
            h = TimeDistributed(Convolution2D(16, 8, strides=4, padding="valid", activation=self.args.activation,
                    data_format='channels_last', kernel_initializer=self.args.cnn_init, kernel_regularizer=l2(self.args.l2_reg)), name='c1')(h)
            h = TimeDistributed(Convolution2D(32, 4, strides=2, padding="valid", activation=self.args.activation,
                    data_format='channels_last', kernel_initializer=self.args.cnn_init, kernel_regularizer=l2(self.args.l2_reg)), name='c2')(h)
        elif self.args.cnn_architecture == 'alec':
            h = TimeDistributed(Convolution2D(32, 8, strides=4, padding="valid", activation=self.args.activation,
                    data_format='channels_last', kernel_initializer=self.args.cnn_init, kernel_regularizer=l2(self.args.l2_reg)), name='c1')(h)
            h = TimeDistributed(Convolution2D(64, 4, strides=2, padding="valid", activation=self.args.activation,
                    data_format='channels_last', kernel_initializer=self.args.cnn_init, kernel_regularizer=l2(self.args.l2_reg)), name='c2')(h)
            h = TimeDistributed(Convolution2D(64, 3, strides=1, padding="valid", activation=self.args.activation,
                    data_format='channels_last', kernel_initializer=self.args.cnn_init, kernel_regularizer=l2(self.args.l2_reg)), name='c3')(h)
        elif self.args.cnn_architecture == 'openai':
            h = TimeDistributed(Convolution2D(32, 3, strides=2, padding="same", activation='elu',
                    data_format='channels_last', kernel_initializer=self.args.cnn_init, kernel_regularizer=l2(self.args.l2_reg)), name='c1')(h)
            h = TimeDistributed(Convolution2D(32, 3, strides=2, padding="same", activation='elu',
                    data_format='channels_last', kernel_initializer=self.args.cnn_init, kernel_regularizer=l2(self.args.l2_reg)), name='c2')(h)
            h = TimeDistributed(Convolution2D(32, 3, strides=2, padding="same", activation='elu',
                    data_format='channels_last', kernel_initializer=self.args.cnn_init, kernel_regularizer=l2(self.args.l2_reg)), name='c3')(h)
            h = TimeDistributed(Convolution2D(64, 3, strides=2, padding="same", activation='elu',
                    data_format='channels_last', kernel_initializer=self.args.cnn_init, kernel_regularizer=l2(self.args.l2_reg)), name='c4')(h)
        elif self.args.cnn_architecture == 'homogeneous':
            for i in range(self.args.cnn_layers):
                h = TimeDistributed(Convolution2D(self.args.cnn_filters, self.args.cnn_kernel_size, strides=self.args.cnn_kernel_step, 
                        activation=self.args.activation, padding="valid", data_format='channels_last', 
                        kernel_initializer=self.args.cnn_init, kernel_regularizer=l2(self.args.l2_reg)
                    ), name='c%d' % (i + 1))(h)
        else:
            assert False
        h = TimeDistributed(Flatten(), name="fl")(h)
        return h

    def build_hidden_layers(self, x):
        # create convolutional layers
        h = self.build_cnn_layers(x)
        # let the main code generate fully connected layers
        return super(CNNPolicy, self).build_hidden_layers(h)


# TODO: reset RNN at episode boundaries?
class RNNPolicy(MLPPolicy):
    def build_rnn_layers(self, x):
        h = x
        if self.args.rnn_type == 'lstm':
            RNN = LSTM
        elif self.args.rnn_type == 'gru':
            RNN = GRU
        elif self.args.rnn_type == 'simple':
            RNN = SimpleRNN
        else:
            assert False
        # create recurrent layers
        for i in range(self.args.rnn_layers):
            h = RNN(self.args.rnn_size, return_sequences=True, stateful=True, name="r%d" % (i + 1))(h)
        # let the main code generate fully connected layers
        return h

    def build_hidden_layers(self, x):
        # create recurrent layers
        h = self.build_rnn_layers(x)
        # let the main code generate fully connected layers
        return super(RNNPolicy, self).build_hidden_layers(h)


class CNNRNNPolicy(CNNPolicy, RNNPolicy):
    def build_hidden_layers(self, x):
        # create convolutional layers
        h = CNNPolicy.build_cnn_layers(self, x)
        # create recurrent layers
        h = RNNPolicy.build_rnn_layers(self, h)
        # let the main code generate fully connected layers
        return MLPPolicy.build_hidden_layers(self, h)
