import numpy as np

np.random.seed(1234)


class Initializer(object):
    def __init__(self, n_input_dims):
        self.n_input_dims = n_input_dims

    def initial_value(self):
        return np.random.normal(size=self.n_input_dims)
