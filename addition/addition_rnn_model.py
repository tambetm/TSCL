import random

from keras.layers import TimeDistributed, Dense, RepeatVector, recurrent, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import keras.backend as K

import numpy as np

from tensorboard_utils import add_summary


class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    '''

    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)


class AdditionRNNModel(object):
    def __init__(self, max_digits=15, hidden_size=128, batch_size=4096, invert=True, optimizer_lr=0.001, clipnorm=None, logdir=None):
        self.max_digits = max_digits
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.invert = invert
        self.optimizer_lr = optimizer_lr
        self.clipnorm = clipnorm
        self.logdir = logdir
        self.maxlen = max_digits + 1 + max_digits

        self.chars = '0123456789+ '
        self.num_chars = len(self.chars)
        self.ctable = CharacterTable(self.chars, self.maxlen)

        self.epochs = 0
        self.make_model()
        if logdir:
            self.callbacks = [TensorBoard(log_dir=self.logdir)]
        else:
            self.callbacks = []

    def make_model(self):
        input = Input(shape=(self.maxlen, self.num_chars))

        # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
        # note: in a situation where your input sequences have a variable length,
        # use input_shape=(None, nb_feature).
        x = recurrent.LSTM(self.hidden_size)(input)
        # For the decoder's input, we repeat the encoded input for each time step
        x = RepeatVector(self.max_digits + 1)(x)
        # The decoder RNN could be multiple layers stacked or a single layer
        x = recurrent.LSTM(self.hidden_size, return_sequences=True)(x)
        # For each of step of the output sequence, decide which character should be chosen
        x = TimeDistributed(Dense(self.num_chars, activation='softmax'))(x)

        def full_number_accuracy(y_true, y_pred):
            y_true_argmax = K.argmax(y_true)
            y_pred_argmax = K.argmax(y_pred)
            tfd = K.equal(y_true_argmax, y_pred_argmax)
            tfn = K.all(tfd, axis=1)
            tfc = K.cast(tfn, dtype='float32')
            tfm = K.mean(tfc)
            return tfm

        self.model = Model(inputs=input, outputs=x)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(lr=self.optimizer_lr, clipnorm=self.clipnorm),
            metrics=['accuracy', full_number_accuracy])

    def generate_data(self, dist, size):
        questions = []
        expected = []
        lengths = []
        #print('Generating data...')
        while len(questions) < size:
            gen_digits = 1 + np.random.choice(len(dist), p=dist)
            f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(gen_digits)))
            a, b = f(), f()

            # Pad the data with spaces such that it is always MAXLEN
            q = '{}+{}'.format(a, b)
            query = q + ' ' * (self.maxlen - len(q))
            ans = str(a + b)
            # Answers can be of maximum size DIGITS + 1
            ans += ' ' * (self.max_digits + 1 - len(ans))
            if self.invert:
                query = query[::-1]
            questions.append(query)
            expected.append(ans)
            lengths.append(gen_digits)
        #print('Total addition questions:', len(questions))

        #print('Vectorization...')
        X = np.zeros((len(questions), self.maxlen, self.num_chars), dtype=np.bool)
        y = np.zeros((len(questions), self.max_digits + 1, self.num_chars), dtype=np.bool)
        for i, sentence in enumerate(questions):
            X[i] = self.ctable.encode(sentence, maxlen=self.maxlen)
        for i, sentence in enumerate(expected):
            y[i] = self.ctable.encode(sentence, maxlen=self.max_digits + 1)

        return X, y, np.array(lengths)

    def accuracy_per_length(self, X, y, lens):
        # TODO: get rid of this, should be part of train FF pass
        p = self.model.predict(X, batch_size=self.batch_size)

        y = np.argmax(y, axis=-1)
        p = np.argmax(p, axis=-1)

        accs = []
        for i in range(self.max_digits):
            yl = y[lens == i+1]
            pl = p[lens == i+1]
            tf = np.all(yl == pl, axis=1)
            accs.append(np.mean(tf))

        return np.array(accs)

    def train_epoch(self, train_data, val_data=None):
        train_X, train_y, train_lens = train_data
        if val_data is not None:
            val_X, val_y, val_lens = val_data

        history = self.model.fit(
            train_X, train_y,
            batch_size=self.batch_size,
            epochs=self.epochs + 1,
            validation_data=(val_X, val_y) if val_data else None,
            initial_epoch=self.epochs,
            callbacks=self.callbacks
        )
        self.epochs += 1

        return history.history


class AdditionRNNEnvironment:
    def __init__(self, model, train_size, val_size, val_dist, writer=None):
        self.model = model
        self.num_actions = model.max_digits
        self.train_size = train_size
        self.val_data = model.generate_data(val_dist, val_size)
        self.writer = writer

    def step(self, train_dist):
        print("Training on", train_dist)
        train_data = self.model.generate_data(train_dist, self.train_size)
        history = self.model.train_epoch(train_data, self.val_data)
        #train_accs = self.model.accuracy_per_length(*train_data)
        val_accs = self.model.accuracy_per_length(*self.val_data)

        train_done = history['full_number_accuracy'][-1] > 0.99
        val_done = history['val_full_number_accuracy'][-1] > 0.99

        if self.writer:
            for k, v in history.items():
                add_summary(self.writer, "model/" + k, v[-1], self.model.epochs)
            for i in range(self.num_actions):
                #add_summary(self.writer, "train_accuracies/task_%d" % (i + 1), train_accs[i], self.model.epochs)
                add_summary(self.writer, "valid_accuracies/task_%d" % (i + 1), val_accs[i], self.model.epochs)

        return val_accs, train_done, val_done
