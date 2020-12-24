import numpy as np


class Perceptron:
    def __init__(self, num_of_features, learning_rate, iterations, with_bias):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.with_bias = with_bias

        weights_num = num_of_features + 1 if with_bias else num_of_features
        self.weights = np.zeros(weights_num)

    def fit(self, training_data):

        for _ in range(self.iterations):
            updated = False
            for sample, target in training_data:
                output = self.predict(sample)

                if target != output:
                    update = self.learning_rate * (target - output)

                    if self.with_bias:
                        self.weights[0] += update
                        self.weights[1:] += sample * update
                    else:
                        self.weights += sample * update
                    updated = True

            if not updated:
                print("In iteration", _, "the network stopped updating")
                break;
            else:
                print("Iteration {} weights: {}".format(_, self.weights))

    def predict(self, x):
        if self.with_bias:
            output = np.dot(x, self.weights[1:]) + self.weights[0]
        else:
            output = np.dot(x, self.weights)
        prediction = 1 if output >= 0 else -1
        return prediction
