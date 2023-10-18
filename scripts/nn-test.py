import numpy as np
import argparse

from machine_learning_intuition.neural_network import NeuralNetwork
from machine_learning_intuition.utils import generate_data, generate_apartment_data, minmax_scale


def convert_samples_to_np_array(samples, dim=1):
    samples_np = np.array(samples)
    return samples_np.reshape(-1, dim)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", default="simple",
                        help="simple for our basic * by 100 experiment. complex for our mock-housing example. "
                             "load to load the model (assumes complex)")
    parser.add_argument("--seed", default=12345, help="random seed")
    args = parser.parse_args()
    np.random.seed(args.seed)
    if args.type == "simple":
        x, y = generate_data()

        global_min = min(0, np.min(x))
        global_max = max(1, np.max(y))
        x_scaled = minmax_scale(x, global_min, global_max)
        y_scaled = minmax_scale(y, global_min, global_max)

        x = convert_samples_to_np_array(x_scaled)
        y = convert_samples_to_np_array(y_scaled)


        nn = NeuralNetwork([1, 5, 5, 1])

        nn.train(x, y, 1000)

        print("me:", nn.predict(np.array([5])))
        print("me:", nn.predict(np.array([3])))
    else:
        # Generate data
        X, y = generate_apartment_data()
        x = convert_samples_to_np_array(X, dim=5)
        y = convert_samples_to_np_array(y, dim=1)
        nn = NeuralNetwork([5, 10, 25, 10, 1])
        nn.train(x, y, epochs=100000, patience_limit=500)

        X_test, y_test = generate_apartment_data(num_samples=10)
        for i in range(0, 10):
            print("Trained: we generated an apartment of:  ", X_test[i], ". We expect price:", y_test[i],
                  " but we predicted:",
                  nn.predict(X_test[i])[0])