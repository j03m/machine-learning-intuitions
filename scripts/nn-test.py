import numpy as np
import argparse
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

        nn.train(x, y, 1000, patience_limit=1000)

        print("me:", nn.predict(np.array([5])))
        print("me:", nn.predict(np.array([3])))
    elif args.type == "fake-apartments":
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
    elif args.type == "iris":
        # Load the Iris dataset
        iris = datasets.load_iris()
        scaler = StandardScaler()
        X, y = iris.data, iris.target

        # split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # scale the data after you have performed the train-test split.
        # Fit the scaler only on the training data and transform both the training and test data.
        # This prevents data leakage.
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        nn = NeuralNetwork([4, 8, 1, 8, 1])
        nn.train(X_train, y_train, epochs=1000, patience_limit=10000)

        y_pred = [round(float(nn.predict(sample_x).item())) for sample_x in X_test]

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

        # Precision, Recall, F1-Score
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion Matrix
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
