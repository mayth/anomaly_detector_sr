from enum import Enum
from typing import TextIO
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import matplotlib.dates as mdates
import pandas as pd
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Plot the results of the anomaly detection"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to the CSV file containing the results (\"-\" to read from stdin)"
    )
    parser.add_argument(
        "-n", "--name", type=str, help="title of the value plotting", default="value"
    )
    parser.add_argument(
        "-t", "--threshold", type=float, help="Threshold for the anomaly score"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to the output image",
        default="output.png",
    )
    parser.add_argument(
        "-s", "--show", action="store_true", help="Show the plot", default=False
    )
    args = parser.parse_args()

    src: TextIO | str
    if args.path == "-":
        src = sys.stdin
    else:
        src = args.path
    threshold = args.threshold

    data =  pd.read_csv(src)
    anomaly_points = data[data["output"] == 1]
    t_anomalies = pd.to_datetime(anomaly_points["Time"])

    fig = plt.figure(figsize=[12.8, 9.6])
    t = pd.to_datetime(data["Time"])

    axs: list[axes.Axes] = []

    ax = fig.add_subplot(3, 1, 1)
    ax.set_title(args.name)
    ax.plot(t, data["value"] / 1000)
    ax.plot(
        t_anomalies,
        anomaly_points["value"] / 1000,
        marker="o",
        linestyle="None",
        color="red",
    )
    axs.append(ax)

    ax = fig.add_subplot(3, 1, 2)
    ax.set_title("Saliency Map")
    ax.plot(t, data["saliency"])
    ax.plot(
        t_anomalies,
        anomaly_points["saliency"],
        marker="o",
        linestyle="None",
        color="red",
    )
    axs.append(ax)

    ax = fig.add_subplot(3, 1, 3)
    ax.set_title("Score")
    ax.plot(t, data["score"])
    ax.plot(
        t_anomalies, anomaly_points["score"], marker="o", linestyle="None", color="red"
    )
    if threshold:
        ax.axhline(y=threshold, color="red", linestyle="--")
    axs.append(ax)

    for ax in axs:
        ax.xaxis.set_major_locator(mdates.HourLocator([0, 6, 12, 18]))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))

    fig.autofmt_xdate()

    plt.savefig(args.output)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
