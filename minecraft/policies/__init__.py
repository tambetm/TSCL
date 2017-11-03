class Policy(object):

    def build(self):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError

    def train(self, observations, preds, rewards, terminals, timestep, writer):
        raise NotImplementedError

    def predict(self, observation):
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError

    def set_weights(self, weights):
        raise NotImplementedError

    def load_weights(self, filename):
        raise NotImplementedError

    def save_weights(self, filename):
        raise NotImplementedError

    # for recurrent policies
    def get_states(self):
        return None

    def set_states(self, values):
        pass
