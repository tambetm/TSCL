from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import merge
from keras import initializers

import numpy as np


class RunningMeanStd(Layer):
    def __init__(self, **kwargs):
        super(RunningMeanStd, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = input_shape[2:]

        # accumulators
        self._mean = self.add_weight('mean', shape, initializer=initializers.Zeros(), trainable=False)
        self._var = self.add_weight('var', shape, initializer=initializers.Ones(), trainable=False)
        self._count = self.add_weight('count', (1,), initializer=initializers.Zeros(), trainable=False)
        self._std = K.sqrt(self._var)
        
        super(RunningMeanStd, self).build(input_shape)

    def call(self, x, mask=None):
        batch_count = K.cast(K.prod(K.shape(x)[:2]), K.floatx())
        batch_mean = K.mean(x, axis=(0, 1))
        batch_var = K.var(x, axis=(0, 1))
        total_count = self._count + batch_count
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        delta = batch_mean - self._mean
        m_a = self._var * self._count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + K.square(delta) * self._count * batch_count / total_count
        # add updates to the graph
        self.add_update([
            K.update(self._mean, self._mean + delta * batch_count / total_count),
            K.update(self._var, M2 / total_count),
            K.update(self._count, total_count)
        ])
        # dummy addition to suppress Keras warning
        return x + 0

    def mean(self):
        return self._mean

    def std(self):
        return self._std


class NormalizeMeanStd(Layer):
    def __init__(self, mean, std, minval=-10, maxval=10, **kwargs):
        self._mean = mean
        self._std = std
        self._minval = minval
        self._maxval = maxval
        super(NormalizeMeanStd, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return K.clip((x - self._mean) / K.maximum(self._std, K.epsilon()), self._minval, self._maxval)


class DenormalizeMeanStd(Layer):
    def __init__(self, mean, std, **kwargs):
        self._mean = mean
        self._std = std
        super(DenormalizeMeanStd, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x * self._std + self._mean


class SampleMultinomial(Layer):
    def __init__(self, from_logits=False, **kwargs):
        super(SampleMultinomial, self).__init__(**kwargs)
        self.from_logits = from_logits
        if K.backend() == 'theano':
            from theano.tensor.shared_randomstreams import RandomStreams
            self.random = RandomStreams()
        elif K.backend() == 'tensorflow':
            import tensorflow as tf
        else:
            raise NotImplementedError

    def call(self, x, mask=None):
        if K.backend() == 'theano':
            if self.from_logits:
                # TODO: there is a more direct way from logits
                return K.argmax(self.random.multinomial(pvals=K.softmax(x)))
            else:
                return K.argmax(self.random.multinomial(pvals=x))
        elif K.backend() == 'tensorflow':
            import tensorflow as tf
            shape = K.shape(x)
            if not self.from_logits:
                x = tf.clip_by_value(x, K.epsilon(), 1 - K.epsilon())
                x = tf.log(x)
            return K.reshape(tf.multinomial(K.reshape(x, [-1, shape[-1]]), 1), shape[:-1])
        else:
            raise NotImplementedError

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class SampleGaussianFixedVariance(Layer):
    def __init__(self, std=1.0, **kwargs):
        super(SampleGaussianFixedVariance, self).__init__(**kwargs)
        self._std = std
        if K.backend() == 'theano':
            from theano.tensor.shared_randomstreams import RandomStreams
            self.random = RandomStreams()
        elif K.backend() == 'tensorflow':
            import tensorflow as tf
        else:
            raise NotImplementedError

    def build(self, input_shape):
        self.stdshape = input_shape[-1:]
        super(SampleGaussianFixedVariance, self).build(input_shape)

    def call(self, x, mask=None):
        if K.backend() == 'theano':
            return self.random.normal(x.shape, x, self._std)
        elif K.backend() == 'tensorflow':
            import tensorflow as tf
            return tf.random_normal(tf.shape(x), x, self._std)
        else:
            raise NotImplementedError

    def std(self):
        return np.tile(self._std, self.stdshape)


class SampleGaussian(Layer):
    def __init__(self, initial_std=1.0, **kwargs):
        super(SampleGaussian, self).__init__(**kwargs)
        self.initial_std = initial_std
        if K.backend() == 'theano':
            from theano.tensor.shared_randomstreams import RandomStreams
            self.random = RandomStreams()
        elif K.backend() == 'tensorflow':
            import tensorflow as tf
        else:
            raise NotImplementedError

    def build(self, input_shape):
        shape = input_shape[-1:]
        def my_init(shape, dtype=None):
            return K.zeros(shape, dtype=dtype) + K.log(self.initial_std)
        self._logstd = self.add_weight('logstd', shape, initializer=my_init, trainable=True)
        super(SampleGaussian, self).build(input_shape)

    def call(self, x, mask=None):
        if K.backend() == 'theano':
            return self.random.normal(x.shape, x, self.std())
        elif K.backend() == 'tensorflow':
            import tensorflow as tf
            return tf.random_normal(tf.shape(x), x, self.std())
        else:
            raise NotImplementedError

    def std(self):
        return K.exp(self._logstd)


class SampleGaussianLearned(Layer):
    def __init__(self, **kwargs):
        super(SampleGaussian, self).__init__(**kwargs)
        if K.backend() == 'theano':
            from theano.tensor.shared_randomstreams import RandomStreams
            self.random = RandomStreams()
        elif K.backend() == 'tensorflow':
            import tensorflow as tf
        else:
            raise NotImplementedError

    def call(self, x, mask=None):
        if K.backend() == 'theano':
            return self.random.normal(x.shape, x[..., :(x.shape[-1] // 2)], x[..., (x.shape[-1] // 2):])
        elif K.backend() == 'tensorflow':
            import tensorflow as tf
            return tf.random_normal(tf.shape(x), x[..., :(tf.shape(x)[-1] // 2)], x[..., (tf.shape(x) // 2):])
        else:
            raise NotImplementedError

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (input_shape[-1] // 2,)


class SampleBinomial(Layer):
    def __init__(self, from_logits=False, **kwargs):
        super(SampleBinomial, self).__init__(**kwargs)
        self.from_logits = from_logits
        if K.backend() == 'theano':
            from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
            self.random = RandomStreams()
        elif K.backend() == 'tensorflow':
            import tensorflow as tf
        else:
            raise NotImplementedError

    def call(self, x, mask=None):
        if K.backend() == 'theano':
            s = self.random.uniform(x.shape, low=0, high=1)
        elif K.backend() == 'tensorflow':
            import tensorflow as tf
            s = tf.random_uniform(tf.shape(x), minval=0, maxval=1)
        else:
            raise NotImplementedError

        if self.from_logits:
            # TODO: there might be more direct way from logits
            return K.cast(K.sigmoid(x) > s, K.floatx())
        else:
            return K.cast(x > s, K.floatx())

def maybe_merge(layers, **kwargs):
    if len(layers) == 1:
        return layers[0]
    else:
        return merge(layers, **kwargs)


def get_states(model):
    """Returns state of recurrent layers as list of numpy arrays."""
    states = []
    for layer in model.layers:
        if hasattr(layer, 'reset_states') and getattr(layer, 'stateful', False):
            states += layer.states
    return K.batch_get_value(states)


def set_states(model, values):
    """Sets state of recurrent layers from list of numpy arrays."""
    states = []
    for layer in model.layers:
        if hasattr(layer, 'reset_states') and getattr(layer, 'stateful', False):
            states += layer.states
    return K.batch_set_value(zip(states, values))


def normc_init(std=1.0):

    def _init(shape, name=None, dim_ordering='tf'):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return K.variable(out, name=name)

    return _init


def test_runningmeanstd():
    from keras.models import Model
    from keras.layers import Input

    x = Input(shape=(None, 3))
    rms = RunningMeanStd()
    y = rms(x)
    m = Model(x, y)
    m.compile('adam', 'mse')

    xs = [np.random.randn(10, 200, 3) for i in range(1, 10000)]
    for xx in xs:
        m.train_on_batch(xx, xx)

    xs = np.concatenate(xs, axis=0)
    mean = np.mean(xs, axis=(0, 1))
    std = np.std(xs, axis=(0, 1))

    rms_mean = K.eval(rms.mean())
    rms_std = K.eval(rms.std())

    print(mean, rms_mean)
    print(std, rms_std)
    assert np.allclose(mean, rms_mean)
    assert np.allclose(std, rms_std)
