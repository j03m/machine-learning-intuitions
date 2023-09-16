import argparse
from machine_learning_intuition import EncoderDecoder, MultiLevelPerceptron, utils


def predict_one(model):
    X, y = utils.generate_arithmetic_sequence_data(num_samples=1)
    prediction, _, _ = model.predict(X)
    return X, y, prediction


def generate_and_predict(model):
    X, y, output_seq = predict_one(model)
    print(f"I am un trained. Given {X} I predicted: {output_seq}. We expected {y}")

    X, y = utils.generate_arithmetic_sequence_data(num_samples=1000)
    print("Let's train")
    model.train(X, y, learning_rate=0.01, epochs=2000, patience_limit=100, warm_up_epochs=500)

    X, y, output_seq = predict_one(model)
    print(f"I am trained. Given {X} I predicted: {output_seq}. We expected {y}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", default="simple",
                        help="simple for our basic sort.")

    args = parser.parse_args()

    if args.type == "create-train":
        model = EncoderDecoder(encoder_layers=[7, 25, 3], decoder_layers=[3, 25, 1])
        generate_and_predict(model)
        model.save("sample")
    elif args.type == "load-train":
        model = EncoderDecoder.load("sample")
        generate_and_predict(model)
        model.save("sample")
    else:
        model = MultiLevelPerceptron([7, 25, 3, 3, 25, 1])
        X, y = utils.generate_arithmetic_sequence_data()
        print("Let's train")
        model.train(X, y, learning_rate=0.01, epochs=5000, patience_limit=500, warm_up_epochs=500)
        X, y = utils.generate_arithmetic_sequence_data(num_samples=1)
        print(f"Perceptron, received {X}, predicted: {model.predict(X)[0]}, expected {y}")
        model.save("sort.json")


if __name__ == "__main__":
    main()
