from machine_learning_intuition.utils import (generate_apartment_data, generate_data, scale_data,
                                              convert_samples_to_np_array)
from machine_learning_intuition.neural_network import NeuralNetwork, ReLU, Linear
import argparse
import os

save_point = "./nn-apartments.json"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", default="simple",
                        help="simple for our basic * by 100 experiment. complex for our mock-housing example. "
                             "load to load the model (assumes complex)")

    args = parser.parse_args()

    if args.type == "simple":
        X, y = generate_data()

        X, y = scale_data(X, y)

        X = convert_samples_to_np_array(X)
        y = convert_samples_to_np_array(y)

        # Initialize and train the neural network
        nn = NeuralNetwork([1, 4, 3, 2, 1],
                           activation_functions=[ReLU(), ReLU(), ReLU(), Linear()])
        X_test, y_test = generate_data(num_samples=1)
        X_test = convert_samples_to_np_array(X_test)
        y_test = convert_samples_to_np_array(y_test)
        print("Untrained: we generated:  ", X_test, ". We expect:", y_test, " but we predicted:",
              nn.predict(X_test)[0])

        print("Let's train! ")

        nn.train(X, y)

        print("We're trained, let's predict again!")
        # Test the trained network
        for i in range(0, 10):
            X_test, y_test = generate_data(num_samples=1)
            print("Trained: we generated: ", X_test, ". We expect:", y_test, " and we predicted:",
                  nn.predict(X_test)[0])

    elif args.type == "complex":
        # 5 inputs, 1 output

        if os.path.exists(save_point):
            print("Loading the network.")
            nn = NeuralNetwork.load(save_point)
        else:
            print("New network")
            nn = NeuralNetwork([5, 10, 25, 10, 1])

        # Generate data
        X, y = generate_apartment_data(num_samples=1)

        print("Untrained: we generated an apartment of:  ", X, ". We expect price:", y, " but we predicted:",
              nn.predict(X)[0])

        print("Let's train! ")
        X, y = generate_apartment_data(num_samples=2000)

        nn.train(X, y, epochs=100000, patience_limit=500)
        nn.save(save_point)
        print("We're trained, let's predict again!")
        X_test, y_test = generate_apartment_data(num_samples=10)
        for i in range(0, 10):
            print("Trained: we generated an apartment of:  ", X_test[i], ". We expect price:", y_test[i],
                  " but we predicted:",
                  nn.predict(X_test[i])[0])
    elif args.type == "load":
        mlp = NeuralNetwork.load(save_point)
        X_test, y_test = generate_apartment_data(num_samples=10)
        for i in range(0, 10):
            print("Loaded! we generated an apartment of:  ", X_test[i], ". We expect price:", y_test[i],
                  " but we predicted:",
                  mlp.predict(X_test[i])[0])

    else:
        print(f"I'm not sure what to do with type: {args.type}. Values are: simple or complex")
