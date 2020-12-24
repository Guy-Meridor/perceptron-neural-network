import random as rnd
import numpy as np
from Perceptron.Perceptron import Perceptron

TRAINING_DATA_SIZE = 300
TESTING_DATA_SIZE = 100
NUMBER_LENGTH = 21
LEARNING_RATE = 0.1
ITERATIONS = 15
BIAS = False


def random_binary_rep():
    if BIAS:
        return np.array([np.random.choice([0, 1]) for i in range(NUMBER_LENGTH)])
    else:
        return np.array([np.random.choice([-1, 1]) for i in range(NUMBER_LENGTH)])



def get_target(binary_rep):
    if BIAS:
        return 1 if sum(binary_rep) > NUMBER_LENGTH / 2 else -1
    else:
        return 1 if sum(binary_rep) >= 0 else -1



def create_training_data(SIZE):
    data = []

    for _ in range(SIZE):
        binary_rep = random_binary_rep()

        target = get_target(binary_rep)

        data.append((binary_rep, target))
    return data


if __name__ == "__main__":
    net = Perceptron(NUMBER_LENGTH, LEARNING_RATE, ITERATIONS, BIAS)
    training_data = create_training_data(TRAINING_DATA_SIZE)

    net.fit(training_data)

    checks = [random_binary_rep() for _ in range(TESTING_DATA_SIZE)]
    errors = 0
    for check in checks:
        prediction = net.predict(check)
        target = get_target(check)
        if target != prediction:
            errors += 1

    prediction_percent = 100 * (TESTING_DATA_SIZE - errors) / TESTING_DATA_SIZE
    print("Prediction percentage is: {}%".format(prediction_percent))
    print("Final Weights: ", net.weights)
