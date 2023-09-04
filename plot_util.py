import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import os
import glob
from matplotlib.ticker import FuncFormatter


def thousands(x, pos):
    return '%1.0fk' % (x * 1e-3)


COLOR = ["blue", "green", "red", "cyan", "magenta", "yellow", "black", "white", "purple", "pink", "brown", "orange"]


class plot_entity:
    def __init__(self, file_name, ignore_first=False):
        self.file_path = os.path.join(os.path.abspath(''), file_name)
        self.dataset_name = None
        self.strategy_name = None
        self.acquisition_step = []
        self.accuracy = None
        self.std = None
        self.__init_data(ignore_first)
        self.folder_name = os.path.basename(os.path.dirname(self.file_path))

    def __init_data(self, ignore_first):
        df = pd.read_csv(self.file_path, sep='\t', header=None)
        # The first line of the txt file is dataset_name
        self.dataset_name = df[0][0]
        # The second line of the txt file is strategy_name
        self.strategy_name = df[0][1]
        # The rest of the line before the last is the accuracy and acquisition_step and std
        self.accuracy = []
        self.std = []
        it_start = 3 if ignore_first else 2
        for i in range(it_start, len(df[0])):
            line = df[0][i].split(',')
            self.acquisition_step.append(int(line[0]))
            self.accuracy.append(float(line[1]))
            self.std.append(float(line[2]))

        self.accuracy = np.array(self.accuracy)
        self.std = np.array(self.std)

    def plot(self, color=None, show_std=True):
        if not color:
            color = np.random.choice(COLOR)
        y1 = self.accuracy - self.std
        y2 = self.accuracy + self.std
        plt.xticks(self.acquisition_step, self.acquisition_step)
        # Plot datapoints
        plt.scatter(self.acquisition_step, self.accuracy, s=40, color=color)
        plt.plot(self.acquisition_step, self.accuracy, 'k-', label=self.strategy_name, color=color)
        if show_std:
            plt.plot(self.acquisition_step, y1, 'r--', label=f'{self.strategy_name} mean$\pm$std', linewidth=1, color=color)
            plt.plot(self.acquisition_step, y2, 'r--', linewidth=1, color=color)

    def plot_dotted_line(self, color='red'):
        plt.axhline(y=self.accuracy[-1], color=color, linestyle='--', label='Upper bound performance')


def plot_result(plot_entity_list, show_std, formatter, plot_baseline=False):
    plt.figure(figsize=(8, 7))
    for i, pe in enumerate(plot_entity_list):
        pe.plot(COLOR[i], show_std)
    if formatter:
        plt.gca().xaxis.set_major_formatter(formatter)

    dataset_name = plot_entity_list[0].folder_name
    if plot_baseline:
        baseline = glob.glob(f'./results/{dataset_name}/B_*.txt')
        if len(baseline) > 0:
            baseline = plot_entity(baseline[0])
            baseline.plot_dotted_line()
    plt.xlabel('Number of Labelled Samples')
    plt.ylabel('Mean Absolute Error')
    plt.title(plot_entity_list[0].dataset_name)
    plt.legend()
    plt.grid()
    plt.savefig(f'./results/{dataset_name}/my_figure.png')
    plt.show()


def plot_from_dataset_name(dataset_name, show_std=True, formatter=None, plot_baseline=False, ignore_first=True):
    plot_entity_list = []
    file_name_list = glob.glob(f'./results/{dataset_name}/P_*.txt')
    for file_name in file_name_list:
        plot_entity_list.append(plot_entity(file_name, ignore_first))
    plot_result(plot_entity_list, show_std, formatter, plot_baseline)


thousands_formatter = FuncFormatter(thousands)
if __name__ == '__main__':
    plot_from_dataset_name('temp', show_std=False, plot_baseline=True, ignore_first=True)