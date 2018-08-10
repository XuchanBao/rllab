import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_plot_data_from_single_experiment(exp_dir, file_name, column_name):
    data = pd.read_csv(os.path.join(exp_dir, file_name))
    return data[column_name]


def generate_plot(data):
    plt.figure()
    mean_data = np.mean(data, axis=0)
    std_data = np.std(data, axis=0)
    x = range(1, len(mean_data) + 1)
    plt.plot(x, mean_data)
    plt.fill_between(x, mean_data + std_data, mean_data - std_data, alpha=0.3)

    plt.title(options.title)
    plt.savefig(os.path.join(options.save_fig_dir, options.save_fig_filename))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training curve with progress data")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--file_name", type=str, default="progress.csv")
    parser.add_argument("--column_to_plot", type=str, default="AverageReturn")
    parser.add_argument("--title", type=str, default="Training Average Return Curve")
    parser.add_argument("--save_fig_filename", type=str, default="plot.png")
    parser.add_argument("--save_fig_dir", type=str, default=os.path.curdir)

    options = parser.parse_args()

    if os.path.exists(os.path.join(options.data_dir, options.file_name)):
        plot_data = [
            get_plot_data_from_single_experiment(
                options.data_dir, options.file_name, options.column_to_plot)
        ]
    else:
        plot_data = []
        shortest_length = None
        for sub_name in os.listdir(options.data_dir):
            sub_dir = os.path.join(options.data_dir, sub_name)
            if os.path.isdir(sub_dir) and "seed" in sub_name and \
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

    generate_plot(plot_data)
