import argparse
import os
from machine_learning_intuition import MultiLevelPerceptron, mlp_utils

generate_apartment_data = mlp_utils.generate_apartment_data
generate_data = mlp_utils.generate_data
scale_data = mlp_utils.scale_data

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", default="simple",
                        help="simple for our basic * by 100 experiment. complex for our mock-housing example. "
                             "load to load the model (assumes complex)")

    args = parser.parse_args()

    if args.type == "simple":
        X, y = generate_data()

        X, y = scale_data(X, y)

        # Initialize and train the neural network
        mlp = MultiLevelPerceptron([1, 4, 3, 2, 1])
        mlp.activation_function = mlp.linear
        mlp.activation_derivative = mlp.linear_derivative
        X_test, y_test = generate_data(num_samples=1)
        print("Untrained: we generated:  ", X_test, ". We expect:", y_test, " but we predicted:",
              mlp.predict(X_test)[0])

        print("Let's train! ")
        mlp.train(X, y, epochs=1000, learning_rate=0.01)

        print("We're trained, let's predict again!")
        # Test the trained network
        for i in range(0, 10):
            X_test, y_test = generate_data(num_samples=1)
            print("Trained: we generated: ", X_test, ". We expect:", y_test, " and we predicted:",
                  mlp.predict(X_test)[0])

    elif args.type == "complex":
        # 5 inputs, 1 output
        save_point = "./apartments.json"
        if os.path.exists(save_point):
            print("Loading the network.")
            mlp = MultiLevelPerceptron.load(save_point)
        else:
            print("New network")
            mlp = MultiLevelPerceptron([5, 10, 25, 10, 1])

        # Generate data
        X, y = generate_apartment_data(num_samples=1)

        print("Untrained: we generated an apartment of:  ", X, ". We expect price:", y, " but we predicted:",
              mlp.predict(X)[0])

        print("Let's train! ")
        X, y = generate_apartment_data(num_samples=2000)

        mlp.train(X, y, epochs=100000, patience_limit=500)
        mlp.save("./apartments.json")
        print("We're trained, let's predict again!")
        X_test, y_test = generate_apartment_data(num_samples=10)
        for i in range(0, 10):
            print("Trained: we generated an apartment of:  ", X_test[i], ". We expect price:", y_test[i],
                  " but we predicted:",
                  mlp.predict(X_test[i])[0])
    elif args.type == "load":
        mlp = MultiLevelPerceptron.load("./apartments.json")
        X_test, y_test = generate_apartment_data(num_samples=10)
        for i in range(0, 10):
            print("Loaded! we generated an apartment of:  ", X_test[i], ". We expect price:", y_test[i],
                  " but we predicted:",
                  mlp.predict(X_test[i])[0])

    else:
        print(f"I'm not sure what to do with type: {args.type}. Values are: simple or complex")
