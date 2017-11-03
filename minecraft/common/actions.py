from gym.spaces import Box, Discrete, MultiDiscrete, Tuple

from keras.layers import TimeDistributed, Dense, Lambda
from keras.objectives import categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy, kullback_leibler_divergence
from keras.regularizers import l2
import keras.backend as K
import numpy as np

from .keras_utils import SampleMultinomial, SampleGaussian, SampleBinomial, maybe_merge


class Action(object):
    def __init__(self, action_space, args, nr=""):
        self.action_space = action_space
        self.args = args
        self.nr = nr

    def build(self, h):
        raise NotImplementedError

    def argmax(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def logp(self, a):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def params(self):
        raise NotImplementedError

    def kld(self, params):
        raise NotImplementedError

    def gym_action(self, flat):
        raise NotImplementedError

    def action_dim(self):
        raise NotImplementedError


class DiscreteAction(Action):
    def build(self, h):
        self.p = TimeDistributed(Dense(self.action_space.n, activation="softmax",
                    kernel_initializer=self.args.action_init, kernel_regularizer=l2(self.args.l2_reg)), name="p%s" % self.nr)(h)

    def argmax(self):
        return Lambda(lambda x: K.expand_dims(K.argmax(x)), output_shape=(None, 1), name="u%s" % self.nr)(self.p)

    def sample(self):
        s = SampleMultinomial(name="s%s" % self.nr)(self.p)
        return Lambda(lambda x: K.expand_dims(x), output_shape=(None, 1), name="u%s" % self.nr)(s)

    def logp(self, a):
        return -sparse_categorical_crossentropy(a[:, :, :1], self.p), a[:, :, 1:]

    def entropy(self):
        return categorical_crossentropy(self.p, self.p)

    def params(self):
        return self.p

    def kld(self, params):
        return kullback_leibler_divergence(self.p, params[:, :, :self.action_space.n]), params[:, :, self.action_space.n:]

    def gym_action(self, flat):
        return int(flat[0]), flat[1:]

    def action_dim(self):
        return 1


class GaussianAction(Action):
    def __init__(self, action_space, args, nr):
        super(GaussianAction, self).__init__(action_space, args, nr)
        self.n = np.prod(self.action_space.shape)

    def build(self, h):
        self.mu = TimeDistributed(Dense(self.n, kernel_initializer=self.args.action_init,
                        kernel_regularizer=l2(self.args.l2_reg)), name="mu%s" % self.nr)(h)
        self.gaussian = SampleGaussian(self.args.initial_std, name="u%s" % self.nr)
        # have to build layer here, otherwise std will not be available
        self.samp = self.gaussian(self.mu)
        # broadcast std to the same dimensions as mu
        self.std = K.zeros_like(self.mu) + self.gaussian.std()

    def argmax(self):
        return self.mu

    def sample(self):
        return self.samp

    def logp(self, a):
        x = a[:, :, :self.n]
        a = a[:, :, self.n:]
        return K.sum(-0.5 * K.square(x - self.mu) / K.square(self.std) - 0.5 * K.log(2 * K.square(self.std) * np.pi), axis=2), a

    def entropy(self):
        return K.sum(0.5 * K.log(2 * np.pi * np.e * K.square(self.std)), axis=2)

    def params(self):
        # this didn't work
        #return merge([self.mu, self.std[np.newaxis, np.newaxis, :]], mode="concat")
        return Lambda(lambda mu: K.concatenate([mu, self.std], axis=2), name="p%s" % self.nr,
            output_shape=(None, 2 * self.n))(self.mu)

    def kld(self, params):
        other_mu = params[:, :, :K.shape(self.mu)[2]]
        params = params[:, :, K.shape(self.mu)[2]:]
        other_std = params[:, :, :K.shape(self.std)[2]]
        params = params[:, :, K.shape(self.std)[2]:]
        return K.sum(K.log(other_std / self.std) + 0.5 * (K.square(self.std) + K.square(self.mu - other_mu)) / K.square(other_std) - 0.5, 2), params

    def gym_action(self, flat):
        a = flat[:self.n]
        a = np.reshape(a, self.action_space.shape)
        a = np.clip(a, self.action_space.low, self.action_space.high)
        return a, flat[self.n:]

    def action_dim(self):
        return self.n


class MultiDiscreteAction(Action):
    def build(self, h):
        self.ps = []
        for i, (low, high) in enumerate(zip(self.action_space.low, self.action_space.high)):
            # if action is binary use sigmoid
            if low + 1 == high:
                p = TimeDistributed(Dense(1, activation="sigmoid",
                    kernel_initializer=self.args.action_init, kernel_regularizer=l2(self.args.l2_reg)), name="p%s_%d" % (self.nr, i))(h)
            else:
                n = high - low + 1
                p = TimeDistributed(Dense(n, activation='softmax',
                    kernel_initializer=self.args.action_init, kernel_regularizer=l2(self.args.l2_reg)), name="p%s_%d" % (self.nr, i))(h)
            self.ps.append(p)

    def argmax(self):
        us = []
        for i, (low, high) in enumerate(zip(self.action_space.low, self.action_space.high)):
            if low + 1 == high:
                u = Lambda(lambda x: K.cast(x > 0.5, K.floatx()), name="u%s_%d" % (self.nr, i))(self.ps[i])
            else:
                u = Lambda(lambda x: K.expand_dims(K.argmax(x)), output_shape=(None, 1), name="u%s_%d" % (self.nr, i))(self.ps[i])
            us.append(u)
        return maybe_merge(us, mode="concat", name="u%s" % self.nr)

    def sample(self):
        us = []
        for i, (low, high) in enumerate(zip(self.action_space.low, self.action_space.high)):
            if low + 1 == high:
                u = SampleBinomial(name="u%s_%d" % (self.nr, i))(self.ps[i])
            else:
                s = SampleMultinomial(name="s%s_%d" % (self.nr, i))(self.ps[i])
                u = Lambda(lambda x: K.expand_dims(x), output_shape=(None, 1), name="u%s_%d" % (self.nr, i))(s)
            us.append(u)
        return maybe_merge(us, mode="concat", name="u%s" % self.nr)

    def logp(self, a):
        logps = []
        for i, (low, high) in enumerate(zip(self.action_space.low, self.action_space.high)):
            if low + 1 == high:
                logp = (a[:, :, :1] * K.log(self.ps[i]) + (1 - a[:, :, :1]) * K.log(1 - self.ps[i]))[:, :, 0]
                a = a[:, :, 1:]
            else:
                logp = -sparse_categorical_crossentropy(a[:, :, :1], self.ps[i])
                a = a[:, :, 1:]
            logps.append(logp)
        return maybe_merge(logps, mode="sum"), a

    def entropy(self):
        es = []
        for i, (low, high) in enumerate(zip(self.action_space.low, self.action_space.high)):
            if low + 1 == high:
                e = binary_crossentropy(self.ps[i], self.ps[i])
            else:
                e = categorical_crossentropy(self.ps[i], self.ps[i])
            es.append(e)
        return maybe_merge(es, mode="sum")

    def params(self):
        ps = []
        for i, (low, high) in enumerate(zip(self.action_space.low, self.action_space.high)):
            if low + 1 == high:
                p = self.ps[i]
            else:
                p = self.ps[i]
            ps.append(p)
        return maybe_merge(ps, mode="concat", name="p%s" % self.nr)

    def kld(self, params):
        ks = []
        for i, (low, high) in enumerate(zip(self.action_space.low, self.action_space.high)):
            if low + 1 == high:
                k = kullback_leibler_divergence(self.ps[i], params[:, :, :1]) + kullback_leibler_divergence(1 - self.ps[i], 1 - params[:, :, :1])
                params = params[:, :, 1:]
            else:
                n = high - low + 1
                k = kullback_leibler_divergence(self.ps[i], params[:, :, :n])
                params = params[:, :, n:]
            ks.append(k)
        return maybe_merge(ks, mode="sum"), params

    def gym_action(self, flat):
        return (flat[:self.action_space.num_discrete_space] + self.action_space.low).astype(np.int32), flat[self.action_space.num_discrete_space:]

    def action_dim(self):
        return self.action_space.num_discrete_space


class TupleAction(Action):
    def __init__(self, action_space, args, nr=""):
        super(TupleAction, self).__init__(action_space, args, nr)

        self.spaces = []
        for i, space in enumerate(self.action_space.spaces):
            if len(self.action_space.spaces) > 1:
                suffix = "%s_%d" % (self.nr, i)
            else:
                suffix = nr

            if isinstance(space, Discrete):
                self.spaces.append(DiscreteAction(space, self.args, suffix))
            elif isinstance(space, Box):
                self.spaces.append(GaussianAction(space, self.args, suffix))
            elif isinstance(space, MultiDiscrete):
                self.spaces.append(MultiDiscreteAction(space, self.args, suffix))
            elif isinstance(space, Tuple):
                self.spaces.append(TupleAction(space, self.args, suffix))
            else:
                raise NotImplementedError

    def build(self, h):
        for space in self.spaces:
            space.build(h)

    def argmax(self):
        us = []
        for space in self.spaces:
            us.append(space.argmax())
        return maybe_merge(us, mode="concat", name="u%s" % self.nr)

    def sample(self):
        us = []
        for space in self.spaces:
            us.append(space.sample())
        return maybe_merge(us, mode="concat", name="u%s" % self.nr)

    def logp(self, a):
        logps = []
        for space in self.spaces:
            logp, a = space.logp(a)
            logps.append(logp)
        return maybe_merge(logps, mode="sum"), a

    def entropy(self):
        es = []
        for space in self.spaces:
            es.append(space.entropy())
        return maybe_merge(es, mode="sum")

    def params(self):
        ps = []
        for space in self.spaces:
            p = space.params()
            ps.append(p)
        return maybe_merge(ps, mode="concat", name="p%s" % self.nr)

    def kld(self, params):
        ks = []
        for space in self.spaces:
            k, params = space.kld(params)
            ks.append(k)
        return maybe_merge(ks, mode="sum"), params

    def gym_action(self, flat):
        us = []
        for space in self.spaces:
            u, flat = space.gym_action(flat)
            us.append(u)
        return us, flat

    def action_dim(self):
        n = 0
        for space in self.spaces:
            n += space.action_dim()
        return n
