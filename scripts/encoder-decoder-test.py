import argparse
import os
from machine_learning_intuition import EncoderDecoder, MultiLevelPerceptron, mlp_utils

import numpy as np


def generate_data(num_samples=1000, seq_len=5):
    X = np.random.randint(1, 100, size=(num_samples, seq_len))
    y = np.array([sorted(x) for x in X])
    return X, y


def predict_one(model):
    X, y = generate_data(num_samples=1)
    X, y = mlp_utils.scale_data(X, y)
    (_,
     output_seq,
     _,
     _,
     _,
     _) = model.predict(X)
    return X, y, output_seq


def generate_and_predict(model):
    X, y, output_seq = predict_one(model)
    print(f"I am un trained. Given {X} I predicted: {output_seq}. We expected {y}")

    X, y = generate_data(num_samples=2000)
    X, y = mlp_utils.scale_data(X, y)
    print("Let's train")
    model.train(X, y, learning_rate=0.01, epochs=5000, patience_limit=500, warm_up_epochs=500)

    X, y, output_seq = predict_one(model)
    print(f"I am un trained. Given {X} I predicted: {output_seq}. We expected {y}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", default="simple",
                        help="simple for our basic sort.")

    args = parser.parse_args()

    if args.type == "create-train":
        model = EncoderDecoder(encoder_layers=[5, 8, 8, 3], decoder_layers=[3, 8, 8, 5])
        generate_and_predict(model)
        model.save("sample")
    elif args.type == "load-train":
        model = EncoderDecoder.load("sample")
        generate_and_predict(model)
        model.save("sample")


if __name__ == "__main__":
    main()
