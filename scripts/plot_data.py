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
        for sub_name in os.listdir(data_dir):
            sub_dir = os.path.join(data_dir, sub_name)
            if os.path.isdir(sub_dir) and "seed" in sub_name and "viz" not in sub_name and \
                    os.path.exists(os.path.join(sub_dir, options.file_name)):
                print("=> Obtaining data from subdirectory {}".format(sub_dir))
                plot_data_single_exp = get_plot_data_from_single_experiment(
                    sub_dir, options.file_name, options.column_to_plot)
                plot_data.append(plot_data_single_exp)

    add_curve(plot_data, label)


def add_curve_for_experiment_with_dir_substr(cur_dir, dir_substr, label=None):
    print("=> Obtaining data in directory {} containing sub string {}".format(cur_dir, dir_substr))
    plot_data = []
    for sub_name in os.listdir(cur_dir):
        sub_dir = os.path.join(cur_dir, sub_name)
        if dir_substr in sub_dir:
            plot_data_single_exp = get_plot_data_from_single_experiment(
                sub_dir, options.file_name, options.column_to_plot)
            plot_data.append(plot_data_single_exp)

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
    longest_length = len(data[0])
    for i in range(1, len(data)):
        if len(data[i]) > longest_length:
            longest_length = len(data[i])

    mean_data = []
    std_data = []
    for itr in range(longest_length):
        itr_values = []
        for curve_i in range(len(data)):
            if itr < len(data[curve_i]):
                itr_values.append(data[curve_i][itr])
        mean_data.append(np.mean(itr_values))
        std_data.append(np.std(itr_values))

    mean_data = np.array(mean_data)
    std_data = np.array(std_data)
    x = range(1, len(mean_data) + 1)
    if label is None:
        plt.plot(x, mean_data)
    else:
        plt.plot(x, mean_data, label=label)
    plt.fill_between(x, mean_data + std_data, mean_data - std_data, alpha=0.3)


def add_legend():
    plt.legend()


def use_option_or_input_label(option_argument, data_dir):
    if option_argument is None:
        return input("Enter label for data in directory '{}': ".format(data_dir))
    return option_argument


def plot_in_dir_with_substr(sub_str):
    if sub_str is not None:
        label = input("Enter label for data in directory with sub string {}: ".format(sub_str))
        add_curve_for_experiment_with_dir_substr(options.data_dir, sub_str, label=label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training curve with progress data")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--multi_curve", type=bool, default=False)
    parser.add_argument("--use_dir_substr", type=bool, default=False)
    parser.add_argument("--dir1_substr", type=str, default=None)
    parser.add_argument("--dir2_substr", type=str, default=None)
    parser.add_argument("--dir3_substr", type=str, default=None)
    parser.add_argument("--dir4_substr", type=str, default=None)
    parser.add_argument("--dir5_substr", type=str, default=None)
    parser.add_argument("--dir6_substr", type=str, default=None)
    parser.add_argument("--file_name", type=str, default="progress.csv")
    parser.add_argument("--column_to_plot", type=str, default="AverageReturn")
    parser.add_argument("--title", type=str, default="Training Average Reward Curve")
    parser.add_argument("--save_fig_filename", type=str, default="plot.png")
    parser.add_argument("--save_fig_dir", type=str, default=os.path.curdir)
    parser.add_argument("--x_max", type=int, default=MAX_LENGTH)
    parser.add_argument("--extra_dir1", type=str, default=None)
    parser.add_argument("--extra_dir1_curve_name", type=str, default=None)
    parser.add_argument("--extra_dir2", type=str, default=None)
    parser.add_argument("--extra_dir2_curve_name", type=str, default=None)
    parser.add_argument("--extra_dir3", type=str, default=None)
    parser.add_argument("--extra_dir3_curve_name", type=str, default=None)
    parser.add_argument("--extra_dir4", type=str, default=None)
    parser.add_argument("--extra_dir4_curve_name", type=str, default=None)
    parser.add_argument("--default_curve_name", type=str, default='default-curve')

    options = parser.parse_args()

    create_plot()
    if options.multi_curve:
        if options.use_dir_substr:
            plot_in_dir_with_substr(options.dir1_substr)
            plot_in_dir_with_substr(options.dir2_substr)
            plot_in_dir_with_substr(options.dir3_substr)
            plot_in_dir_with_substr(options.dir4_substr)
            plot_in_dir_with_substr(options.dir5_substr)
            plot_in_dir_with_substr(options.dir6_substr)
        else:
            for sub_name in os.listdir(options.data_dir):
                sub_dir = os.path.join(options.data_dir, sub_name)
                if os.path.isdir(sub_dir) and "viz" not in sub_dir:
                    label = input("Enter label for data in directory '{}': ".format(sub_dir))
                    add_curve_for_experiment(sub_dir, label=label)
        add_legend()
    else:
        if options.extra_dir1:
            label = use_option_or_input_label(options.extra_dir1_curve_name, options.extra_dir1)
            add_curve_for_experiment(options.extra_dir1, label=label)
        if options.extra_dir2:
            label = use_option_or_input_label(options.extra_dir2_curve_name, options.extra_dir2)
            add_curve_for_experiment(options.extra_dir2, label=label)
        if options.extra_dir3:
            label = use_option_or_input_label(options.extra_dir3_curve_name, options.extra_dir3)
            add_curve_for_experiment(options.extra_dir3, label=label)
        if options.extra_dir4:
            label = use_option_or_input_label(options.extra_dir4_curve_name, options.extra_dir4)
            add_curve_for_experiment(options.extra_dir4, label=label)
        add_curve_for_experiment(options.data_dir, label=options.default_curve_name)
        add_legend()
    save_plot()
