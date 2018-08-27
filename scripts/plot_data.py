import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MAX_LENGTH = 99999


def get_plot_data_from_single_experiment(exp_dir, file_name, column_name):
    data = pd.read_csv(os.path.join(exp_dir, file_name))
    return data[column_name][:options.x_max]


def add_curve_for_experiment(data_dir, label=None):
    if os.path.exists(os.path.join(data_dir, options.file_name)):
        plot_data = [
            get_plot_data_from_single_experiment(
                data_dir, options.file_name, options.column_to_plot)
        ]
    else:
        plot_data = []
        shortest_length = None
        for sub_name in os.listdir(data_dir):
            sub_dir = os.path.join(data_dir, sub_name)
            if os.path.isdir(sub_dir) and "seed" in sub_name and "viz" not in sub_name and \
                    os.path.exists(os.path.join(sub_dir, options.file_name)):
                print("=> Obtaining data from subdirectory {}".format(sub_dir))
                plot_data_single_exp = get_plot_data_from_single_experiment(
                    sub_dir, options.file_name, options.column_to_plot)
                plot_data.append(plot_data_single_exp)

                if shortest_length is None:
                    shortest_length = len(plot_data_single_exp)
                elif len(plot_data_single_exp) < shortest_length:
                    shortest_length = len(plot_data_single_exp)
        for i in range(len(plot_data)):
            plot_data[i] = plot_data[i][:shortest_length]

    add_curve(plot_data, label)


def create_plot():
    plt.figure()
    plt.title(options.title)


def save_plot():
    plt.xlabel("Training Iteration")
    plt.ylabel(options.column_to_plot)
    plt.savefig(os.path.join(options.save_fig_dir, options.save_fig_filename))
    plt.close()


def add_curve(data, label):
    mean_data = np.mean(data, axis=0)
    std_data = np.std(data, axis=0)
    x = range(1, len(mean_data) + 1)
    if label is None:
        plt.plot(x, mean_data)
    else:
        plt.plot(x, mean_data, label=label)
    plt.fill_between(x, mean_data + std_data, mean_data - std_data, alpha=0.3)


def add_legend():
    plt.legend()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training curve with progress data")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--multi_curve", type=bool, default=False)
    parser.add_argument("--file_name", type=str, default="progress.csv")
    parser.add_argument("--column_to_plot", type=str, default="AverageReturn")
    parser.add_argument("--title", type=str, default="Training Average Reward Curve")
    parser.add_argument("--save_fig_filename", type=str, default="plot.png")
    parser.add_argument("--save_fig_dir", type=str, default=os.path.curdir)
    parser.add_argument("--x_max", type=int, default=MAX_LENGTH)
    parser.add_argument("--extra_dir1", type=str, default=None)
    parser.add_argument("--extra_dir2", type=str, default=None)

    options = parser.parse_args()

    create_plot()
    if options.multi_curve:
        for sub_name in os.listdir(options.data_dir):
            sub_dir = os.path.join(options.data_dir, sub_name)
            if os.path.isdir(sub_dir) and "viz" not in sub_dir:
                label = input("Enter label for data in directory '{}': ".format(sub_dir))
                add_curve_for_experiment(sub_dir, label=label)
        add_legend()
    else:
        if options.extra_dir1:
            label = input("Enter label for data in directory '{}': ".format(options.extra_dir1))
            add_curve_for_experiment(options.extra_dir1, label=label)
        if options.extra_dir2:
            label = input("Enter label for data in directory '{}': ".format(options.extra_dir2))
            add_curve_for_experiment(options.extra_dir2, label=label)
        add_curve_for_experiment(options.data_dir, label="default-curve")
        add_legend()
    save_plot()
