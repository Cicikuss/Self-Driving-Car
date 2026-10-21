import argparse
import os

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils import (
    balance_steering_data,
    batch_generator,
    load_driving_log,
    load_img_steering,
    nvidia_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the behavioral cloning model.")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "Data", "Tracks"),
        help="Directory containing driving_log.csv and the camera images.",
    )
    parser.add_argument("--output", default="model.keras", help="Output model path.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=300)
    parser.add_argument("--validation-steps", type=int, default=200)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not display matplotlib figures.",
    )
    return parser.parse_args()


def plot_distribution(data, title):
    figure, axis = plt.subplots()
    axis.hist(data["steering"], bins=25)
    axis.set_title(title)
    axis.set_xlabel("Steering angle")
    axis.set_ylabel("Samples")
    return figure


def plot_loss(history):
    figure, axis = plt.subplots()
    axis.plot(history.history["loss"], label="Training")
    axis.plot(history.history["val_loss"], label="Validation")
    axis.legend()
    axis.set_title("Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE")
    return figure


def main():
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    data = load_driving_log(data_dir)
    data, _, _ = balance_steering_data(data)

    image_paths, steerings = load_img_steering(data_dir + "/IMG", data)
    x_train, x_valid, y_train, y_valid = train_test_split(
        image_paths, steerings, test_size=0.2, random_state=6
    )
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_valid)}")

    model = nvidia_model()
    model.summary()
    history = model.fit(
        batch_generator(x_train, y_train, args.batch_size, True),
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        validation_data=batch_generator(x_valid, y_valid, args.batch_size, False),
        validation_steps=args.validation_steps,
        verbose=1,
        shuffle=True,
    )
    model.save(args.output)
    print(f"Model saved to {os.path.abspath(args.output)}")

    if not args.no_plot:
        plot_distribution(data, "Steering distribution")
        plot_loss(history)
        plt.show()


if __name__ == "__main__":
    main()
