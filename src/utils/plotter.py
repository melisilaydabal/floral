import sys
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib.colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from cycler import cycler
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy.ndimage.filters import gaussian_filter1d
import ast
from matplotlib.lines import Line2D

plt.style.use('fast')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.serif'] = ['Computer Modern'] + plt.rcParams['font.serif']
plt.rc('axes', prop_cycle=(cycler('color', [
                                            '#0023FF', '#CB0003','#13B000', '#FFA200', '#F500DF', '#E2D402',
                                            '#72EEFF', '#FD74C3', '#7BA0E9', '#1f78b4', '#F97511','#dbdb40',
                                            '#37E593', '#3CEAF6', '#9D0505','#f781bf','#a65628', '#984ea3',
                                            '#4daf4a', '#ff7f00','#377eb8','#e41a1c'
                                            ])
                           + cycler('linestyle',['-', '-', '-', '-', '-', '-','-', '-', '-', '-', '-', '-', '-', '-', '-',
                                                 '-', '-', '-', '-', '-', '-','-'])))
repl_perturb = 1

colors = {'Loss': 'red', 'Accuracy': 'blue',
          'MSE': 'blue', 'MAE': 'green', 'RSquared': 'red'}
plot_lines = {'test': 'solid', 'train': 'dashed', 'validation': 'dotted'}
alphas = {'test': 1.0, 'train': 1.0, 'validation': 1.0}
line_width = 10

def _create_fig(width=16, height=10):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111)
    return fig, ax

def custom_colormap():
    red_cmap = plt.cm.Reds
    blue_cmap = plt.cm.Blues
    cmap_colors = np.vstack((blue_cmap(np.linspace(0, 1, 256)),
                             red_cmap(np.linspace(0, 1, 256))))
    custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', cmap_colors)
    return custom_cmap

def _plot_prior_style(fig, ax):
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#0023FF', '#CB0003', '#FFA200',
                                                                    '#13B000', '#F500DF', '#E2D402','#72EEFF',
                                                                    '#FD74C3', '#7BA0E9', '#1f78b4', '#F97511',
                                                                    '#dbdb40', '#37E593', '#3CEAF6', '#9D0505',
                                                                    '#f781bf', '#a65628', '#984ea3', '#4daf4a',
                                                                    '#ff7f00', '#377eb8', '#e41a1c'])
    return cmap

def plot_perf(config, dir_exp_folder, round, list_performance, plot_type, data_type, num_epochs):
    print(f"Plotting {data_type}_{plot_type} with list {list_performance}")
    fig, ax = _create_fig()
    ax.plot(range(1, num_epochs + 1), list_performance,
            label=data_type + plot_type,
            linewidth=2, marker=".", markevery=10, markersize=20, linestyle='solid')
    fig.suptitle('Classification ' + plot_type +': ' + data_type, fontsize=55)
    ax.set_ylim(min(0, np.min(list_performance)), np.max(list_performance))
    _plot_posterior_style(fig, ax, "epoch", plot_type)
    if config.get('training').get('is_perturbed'):
        plot_path = os.path.join(config.get('dump').get('dir_dump'),
                                f"{dir_exp_folder}round{round}_{data_type}_{plot_type}"
                                f"_{config.get('data').get('dataset')}"
                                f"_isperturbed_{config.get('training').get('is_perturbed')}_{repl_perturb}_plot.png")
    else:
        plot_path = os.path.join(config.get('dump').get('dir_dump'),
                                f"{dir_exp_folder}round{round}_{data_type}_{plot_type}"
                                f"_{config.get('data').get('dataset')}"
                                f"_model{config.get('model').get('name')}"
                                f"_{config.get('optim').get('optimizer')}optim"
                                f"_epoch{num_epochs}"
                                f"_lr{config.get('optim').get('lr')}"
                                f"_isperturbed_{config.get('training').get('is_perturbed')}_plot.png")
    fig.savefig(plot_path)
    plt.close()

def plot_perf_training(config, dir_exp_folder, round, dict_performances, plot_type, data_type, num_epochs):
    fig, ax = _create_fig()
    for key in dict_performances.keys():
        ax.plot(range(1, num_epochs + 1), dict_performances[key],
                label=data_type + ' ' + key + ' ' + plot_type,
                linewidth=5, marker=".", markevery=10, markersize=10, linestyle='solid')
    fig.suptitle('Classification ' + plot_type + ' ' + data_type, fontsize=55)
    # Set the range of y-axis
    all_values = [value for values in dict_performances.values() for value in values]
    ax.set_ylim(min(0, min(all_values)), max(all_values))
    _plot_posterior_style(fig, ax, "Epoch", plot_type)
    if config.get('training').get('is_perturbed'):
        plot_path = os.path.join(config.get('dump').get('dir_dump'),
                                f"{dir_exp_folder}round{round}_perf_{data_type}_{plot_type}"
                                f"_{config.get('data').get('dataset')}"
                                f"_isperturbed_{config.get('training').get('is_perturbed')}_{repl_perturb}_plot.png")
    else:
        plot_path = os.path.join(config.get('dump').get('dir_dump'),
                                f"{dir_exp_folder}round{round}_perf_{data_type}_{plot_type}"
                                f"_{config.get('data').get('dataset')}"
                                f"_model{config.get('model').get('name')}"
                                f"_{config.get('optim').get('optimizer')}optim"
                                f"_epoch{num_epochs}"
                                f"_lr{config.get('optim').get('lr')}"
                                f"_isperturbed_{config.get('training').get('is_perturbed')}_plot.png")
    fig.savefig(plot_path)
    plt.close()

def _plot_posterior_style(fig, ax, x_label, y_label):
    ax.xaxis.set_tick_params(labelsize=40)
    ax.yaxis.set_tick_params(labelsize=40)
    ax.set_xlabel(x_label, fontsize=45)
    ax.set_ylabel(y_label, fontsize=45)
    # ax.legend(fontsize='xx-large')
    fig.tight_layout()

def read_parameters_and_performance(file_path, metric='Loss'):
    metrics = ['Loss', 'Accuracy']
    is_header = False
    with open(file_path, 'r') as file:
        lines = file.readlines()

    is_perturbed = None
    for line in lines:
        if line.startswith('#training: is_perturbed'):
            is_perturbed = line.split('=')[1].strip()
    if is_perturbed is None:
        raise ValueError('Parameter "training: is_perturbed" not found in the file.')

    # Extract the metric values
    values = []
    for line in lines:
        if is_header == False and not line.startswith("#"):
            is_header = True
            continue
        if line.startswith("#"):
            continue
        values.append(float(line.split()[metrics.index(metric)]))

    return is_perturbed, values


def plot_and_save_performance(file_pattern, output_filename, metric='Loss'):
    fig, ax = _create_fig()
    # Get a list of file paths matching the pattern
    file_paths = glob.glob(file_pattern)

    # Lists to store performance values and parameter values
    all_values = []
    all_is_perturbed = []
    x_labels = []

    # Read and plot individual performances
    for file_path in file_paths:
        is_perturbed, values = read_parameters_and_performance(file_path, metric)
        print(values)
        if is_perturbed == "True":
            all_is_perturbed.append(values)
        label = f'test_{metric}_isperturbed_{is_perturbed}'
        # plt.plot(values, label=label)
        if is_perturbed == "False":
            print(label)
            ax.bar(0, values, label=label)
            x_labels.append(label)

    # Read and plot average performance
    if all_is_perturbed:
        average_values = [sum(x) / len(x) for x in zip(*all_is_perturbed)]
        error = np.std(all_is_perturbed)
        print(all_is_perturbed)
        print(error)
        label = f'Average_test_{metric}_isperturbed_{is_perturbed}'
        ax.bar(1, average_values, yerr=error, label=label, align='center', alpha=0.5, ecolor='black', capsize=10)
        x_labels.append(label)

    plt.legend()
    plt.xlabel(x_labels)
    plt.ylabel(metric)
    plt.title(f'{metric} Performance')

    # Save the figure
    plt.savefig('./workbench/' + metric + "_" + output_filename)

    # Show the plot
    plt.show()
    _plot_posterior_style(fig, ax, "epoch", "accuracy")

def specific_plot(file_path, output_filename, metric='Loss'):
    fig, ax = _create_fig()
    cmap = _plot_prior_style(fig, ax)

    colors = {'Loss' : 'red', 'Accuracy' : 'blue'}

    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()

        # Initialize empty lists to store rounds, loss, and accuracy
        rounds = []
        values = []

        # Iterate over each line starting from start_line
        for line in lines:
            if line.startswith('#') or line.startswith('Round'):
                continue
            # Split the line by whitespace
            parts = line.strip().split()
            if len(parts) == 3:
                # Extract round number and metric value
                round_num = int(parts[0])
                metric_value = float(parts[1]) if metric.lower() == 'loss' else float(parts[2])

                rounds.append(round_num)
                values.append(metric_value)

    # Plot the performance metric over rounds
    plt.plot(rounds, values, color=colors[metric], linewidth = 5, marker = "o", markevery = 1, markersize = 10, linestyle = 'solid')
    plt.xlabel('Round')
    plt.ylabel(metric)
    plt.grid(True)
    plt.show()

    fig.suptitle(f'{metric} Performance', fontsize=40)
    _plot_posterior_style(fig, ax, "Round", f"{metric}")
    fig.savefig("./workbench/" + "_" + output_filename.format(metric))


def general_plot(folder_path, output_filename, metric='Loss'):
    fig, ax = _create_fig()
    cmap = _plot_prior_style(fig, ax)

    colors = {'Loss': 'red', 'Accuracy': 'blue'}
    plot_lines = {'test': 'solid', 'train': 'dashed', 'validation': 'dotted'}

    # Initialize empty lists to store rounds, loss, and accuracy
    rounds = []
    values = []

    files = [f for f in os.listdir(folder_path) if f.endswith('.out') and f.startswith(('test', 'validation', 'train'))]

    # Iterate over each file
    for file in files:
        rounds = []
        values = []

        file_path = os.path.join(folder_path, file)
        # Read the file
        with open(file_path, 'r') as f:
            lines = f.readlines()

            # Iterate over each line starting from start_line
            for line in lines:
                if line.startswith('#') or line.startswith('Round'):
                    continue
                # Split the line by whitespace
                parts = line.strip().split()
                if len(parts) == 3:
                    # Extract round number and metric value
                    round_num = int(parts[0])
                    metric_value = float(parts[1]) if metric.lower() == 'loss' else float(parts[2])

                    # Append round number and metric value to lists
                    rounds.append(round_num)
                    values.append(metric_value)

        # Extract label for the plotted line
        label = file.split('_')[0] + '_' + metric
        perf_type = file.split('_')[0]

        # Plot the performance metric over rounds
        plt.plot(rounds, values, label=label, color=colors[metric], linewidth = 3, linestyle = plot_lines[perf_type], alpha=alphas[perf_type])
        plt.xlabel('Round')
        plt.ylabel(metric)
        plt.grid(True)

    fig.suptitle(f'{metric} Performance', fontsize=40)
    _plot_posterior_style(fig, ax, "Round", f"{metric}")

    # Save the figure
    fig.savefig(folder_path + "/_" + output_filename.format(metric))


def general_plot(folder_path, output_filename, metric='Loss'):
    fig, ax = _create_fig()
    cmap = _plot_prior_style(fig, ax)

    # Initialize empty lists to store rounds, loss, and accuracy
    rounds = []
    values = []

    # Find all .out files inside the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.out') and f.startswith(('test_acc', 'validation_acc', 'train_acc'))]

    # Iterate over each file
    for file in files:
        rounds = []
        values = []

        file_path = os.path.join(folder_path, file)
        print(f"Reading {file_path}")
        # Read the file
        with open(file_path, 'r') as f:
            lines = f.readlines()

            # Iterate over each line starting from start_line
            for line in lines:
                if line.startswith('#') or line.startswith('Round') or line.startswith('Epoch'):
                    continue
                # Split the line by whitespace
                parts = line.strip().split()
                if len(parts) == 3:
                    # Extract round number and metric value
                    round_num = int(parts[0])
                    metric_value = float(parts[1]) if metric.lower() == 'loss' else float(parts[2])

                    # Append round number and metric value to lists
                    rounds.append(round_num)
                    values.append(metric_value)

        # Extract label for the plotted line
        label = file.split('_')[0] + '_' + metric
        perf_type = file.split('_')[0]

        # Plot the performance metric over rounds
        plt.plot(rounds, values,
                 label=label, color=colors[metric], linewidth = 3, linestyle = plot_lines[perf_type], alpha=alphas[perf_type])

        plt.xlabel('Round')
        plt.ylabel(metric)
        plt.grid(True)

    fig.suptitle(f'{metric} Performance', fontsize=40)
    _plot_posterior_style(fig, ax, "Round", f"{metric}")
    fig.savefig(folder_path + "/_" + output_filename.format(metric))


def general_reg_plot(folder_path, output_filename, metric='Loss'):
    fig, ax = _create_fig()
    cmap = _plot_prior_style(fig, ax)

    colors = {'Loss': 'red', 'Accuracy': 'blue',
              'MSE': 'blue', 'MAE': 'green', 'RSquared': 'red'}
    plot_lines = {'test': 'solid', 'train': 'dashed', 'validation': 'dotted'}

    # Initialize empty lists to store rounds, loss, and accuracy
    values = []

    # Find all .out files inside the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.out') and f.startswith(('test_reg', 'validation_reg', 'train_reg'))]

    # Iterate over each file
    for file in files:
        rounds = []
        values = []

        file_path = os.path.join(folder_path, file)
        print(f"Reading {file_path}")
        # Read the file
        with open(file_path, 'r') as f:
            lines = f.readlines()

            # Iterate over each line starting from start_line
            for line in lines:
                if line.startswith('#') or line.startswith('Round'):
                    continue
                # Split the line by whitespace
                parts = line.strip().split()
                if len(parts) == 3:
                    # Extract round number and metric value
                    if metric.lower() == 'mse':
                        metric_value = float(parts[0])
                    elif metric.lower() == 'mae':
                        metric_value = float(parts[1])
                    elif metric.lower() == 'rsquared':
                        metric_value = float(parts[2])

                    # Append round number and metric value to lists
                    values.append(metric_value)

        # Extract label for the plotted line
        label = file.split('_')[0] + '_' + metric
        perf_type = file.split('_')[0]

        # Plot the performance metric over rounds
        plt.plot(list(range(len(values))), values, label=label, color=colors[metric], linewidth = 3, linestyle = plot_lines[perf_type], alpha=alphas[perf_type])

        plt.xlabel('Round')
        plt.ylabel(metric)
        plt.grid(True)

    fig.suptitle(f'{metric} Performance', fontsize=40)
    _plot_posterior_style(fig, ax, "Round", f"{metric}")

    # Save the figure
    fig.savefig(folder_path + "/_" + output_filename.format(metric))

def extract_info(file_name):
    match = re.search(r'train_adv_\d+', file_name)
    if match:
        return match.group(0)
    else:
        return 'train_clean'


def nn_motivation_plot(folder_path, output_filename, metric='Loss'):
    fig, ax = _create_fig(10, 10)
    cmap = _plot_prior_style(fig, ax)

    line_colors={'train_clean': '#0023FF',
                 'train_adv_20': '#CB0003', 'train_adv_5': '#CB0003',
                 'train_adv_40': '#13B000', 'train_adv_10': '#13B000',
                 'train_adv_100': '#FFA200', 'train_adv_25': '#FFA200'}

    # Initialize empty lists to store rounds, loss, and accuracy
    rounds = []
    values = []

    # Find all .out files inside the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.out') and f.startswith(('test_'))]

    # Iterate over each file
    for file in files:
        rounds = []
        values = []

        file_path = os.path.join(folder_path, file)
        print(f"Reading {file_path}")
        # Read the file
        with open(file_path, 'r') as f:
            lines = f.readlines()

            # Iterate over each line starting from start_line
            for line in lines:
                if line.startswith('#') or line.startswith('Round') or line.startswith('Epoch'):
                    continue
                # Split the line by whitespace
                parts = line.strip().split()
                if len(parts) == 3:
                    # Extract round number and metric value
                    round_num = int(parts[0])
                    metric_value = float(parts[1]) if metric.lower() == 'accuracy' else float(parts[2])

                    # Append round number and metric value to lists
                    rounds.append(round_num)
                    values.append(metric_value)

        # Extract label for the plotted line
        label = extract_info(file)
        perf_type = file.split('_')[0]

        # Plot the performance metric over rounds
        Z = plt.plot(rounds, values, label=label, linewidth=5, linestyle=plot_lines[perf_type],
                 alpha=alphas[perf_type], color=line_colors[label])
        plt.xlabel('Round')
        plt.ylabel(metric)
        plt.grid(True)

        # # # Setup zoom window
        # axins = zoomed_inset_axes(ax, 2, loc="center", bbox_to_anchor=(0.5, 0.5))
        # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.1")
        # axins.set_xlim([1500, 2000])
        # axins.set_ylim([0.95, 1.0])
        # axins.plot(rounds, values, label=label, linewidth=5, linestyle=plot_lines[perf_type],
        #              alpha=alphas[perf_type])

    fig.suptitle(f'Test {metric}', fontsize=40)
    _plot_posterior_style(fig, ax, "Epoch", f"{metric}")
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles=[h[2], h[0], h[3], h[1]], labels=['Clean', 'Adv_5%', 'Adv_10%', 'Adv_25%'], loc='best', fontsize=25)
    fig.savefig(folder_path + "/_" + output_filename.format(metric))

def plot_roberta_motivation(folder_path, output_filename, metric='Accuracy'):
    file_path = "PATH_TO_CSV_FILE"

    fig, ax = _create_fig(15, 10)
    cmap = _plot_prior_style(fig, ax)

    line_colors = {
        'clean': '#FB00FF',
        '0.01': '#b200ff',
        '0.02': '#b200ff',
        '0.05': '#500073',
        '0.1': '#500000',
        '0.15': '#0004ff',
        '0.2': '#7374b2',
        '0.25': '#0004ff',
        '0.3': '#ABA9A9',
        '0.35': '#515151',
        '0.4': '#000000'
                   }
    line_styles = {'clean': 'solid', '0.01': 'solid', '0.02': 'solid', '0.05': 'solid', '0.1': 'solid',
                   '0.15': 'solid', '0.2': 'solid', '0.25': 'solid', '0.3': 'solid', '0.35': 'solid', '0.4': 'solid'}
    df = pd.read_csv(file_path)

    steps = df['steps']
    for column in df.columns:
        if '__MIN' in column or '__MAX' in column or 'steps' == column or '_step' in column:
            continue

        y_value = df[column]
        y_min = df[column + '__MIN']
        y_max = df[column + '__MAX']

        if 'flip' not in column:
            label = 'clean'
        else:
            pattern = r'_flip([\d.]+)_'
            match = re.search(pattern, column)
            if match:
                number = match.group(1)
                label = str(number)
            else:
                print("No match found.")
        y_value = y_value.to_numpy()
        y_min = y_min.to_numpy()
        y_max = y_max.to_numpy()

        print(f"TYPE: {metric} | Method: {label} | AVG: {y_value[-1]} | STDEV: {y_value[-1]}")
        if 'accuracy' in metric or 'Accuracy' in metric:
            print(f"TYPE: {metric} | Method: {label} | MAX-AVG: {np.max(y_value)}")
        elif 'loss' in metric or 'Loss' in metric:
            print(f"TYPE: {metric} | Method: {label} | MIN-AVG: {np.min(y_value)}")

        # if smoothness is to be applied for better visibility
        # y_value = gaussian_filter1d(y_value, sigma=0.01)
        # y_min = gaussian_filter1d(y_min, sigma=0.01)
        # y_max = gaussian_filter1d(y_max, sigma=0.01)

        plt.plot(steps.to_numpy(), y_value,
                 label=label, color=line_colors[label], linewidth=line_width, linestyle=line_styles[label])
        plt.fill_between(steps.to_numpy(), y_min, y_max, alpha=0.3, color=line_colors[label])

    plt.xlabel('Rounds')
    plt.ylabel(metric)
    plt.grid(True)
    fig.suptitle(f'Test {metric}', fontsize=50)
    _plot_posterior_style(fig, ax, "Rounds", f"{metric}")

    # figl, axl = plt.subplots(figsize=(15, 1))
    # h, l = ax.get_legend_handles_labels()
    # axl.legend(handles=[h[4], h[0], h[2], h[1], h[3], h[5]],
    #           # labels=['RoBERTa_Clean', 'Adv_10%', 'Adv_25%', 'Adv_30%', 'Adv_35%', 'Adv_40%'],
    #            labels=['Clean', 'Adv_10%', 'Adv_25%', 'Adv_30%', 'Adv_35%', 'Adv_40%'],
    #            loc="center", ncol=7, bbox_to_anchor=(0.5, 0.5), prop={"size": 15}, fontsize='xx-large')
    # axl.axis(False)
    # legend_path = os.path.join(folder_path, 'legend_.pdf')
    # figl.savefig(legend_path)
    # legend_path = os.path.join(folder_path, 'legend_.png')
    # figl.savefig(legend_path)

    fig.savefig(folder_path + "/_" + output_filename.format(metric))


def plot_adv_labels(config, dir_exp_folder, round, X, y, y_adv):
    if config.get('player_flip').get('is_set_flip'):
        for idx, y_adv_item in enumerate(y_adv):
            fig, ax = _create_fig()
            cmap = _plot_prior_style(fig, ax)
            plot_path = os.path.join(f"{dir_exp_folder}round{round}_{idx}_train_adv_dataset_plot.png")
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, marker='o', s=150, edgecolors='k')
            ax.scatter(X[:, 0], X[:, 1], c=y_adv_item, cmap=cmap, marker='x', s=40, alpha=1)
            fig.suptitle(f'Train_adv Dataset Dataset | Round {round} | Set {idx}', fontsize=40)
            _plot_posterior_style(fig, ax, "Feature 0", "Feature 1")
            legendElements = [
                Line2D([0], [0], linestyle='none', marker='o', color='blue', markersize=7, markeredgecolor='k'),
                Line2D([0], [0], linestyle='none', marker='o', color='red', markersize=7, markeredgecolor='k')]
            myLegend = plt.legend(legendElements,
                                  ['Negative -1', 'Positive +1'],
                                  fontsize="15", loc='upper right')
            myLegend.get_frame().set_linewidth(0.3)
            fig.savefig(plot_path)
            plt.close()
    else:
        fig, ax = _create_fig()
        cmap = _plot_prior_style(fig, ax)

        plot_path = os.path.join(f"{dir_exp_folder}round{round}_train_adv_dataset_plot.png")

        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, marker='o', s=150, edgecolors='k')
        ax.scatter(X[:, 0], X[:, 1], c=y_adv, cmap=cmap, marker='x', s=40, alpha=1)
        fig.suptitle(f'Train_adv Dataset Dataset | Round {round}', fontsize=40)

        _plot_posterior_style(fig, ax, "Feature 0", "Feature 1")
        legendElements = [
            Line2D([0], [0], linestyle='none', marker='o', color='blue', markersize=7, markeredgecolor='k'),
            Line2D([0], [0], linestyle='none', marker='o', color='red', markersize=7, markeredgecolor='k')]
        myLegend = plt.legend(legendElements,
                              ['Negative -1', 'Positive +1'],
                              fontsize="15", loc='upper right')
        myLegend.get_frame().set_linewidth(0.3)

        fig.savefig(plot_path)
        plt.close()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit("usage: python plotter.py <file_path> <metric>")
    else:
        file_path = sys.argv[1]
        metric = sys.argv[2]

    plot_roberta_motivation(file_path, 'roberta_motivation_{}_plot.png', metric=metric)
