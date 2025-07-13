import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_rel, wilcoxon, shapiro
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.style.use('fast')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

MODEL_COLOR_MAP = {
    'svm': '#FB00FF',
    'nn': '#0004ff',
    'nn_pgd': '#8100c6',
    'ln-robust-svm': '#7f0000',
    'roberta': '#00c293',
    'vanilla-svm': '#000000',
    'curie-svm': '#00bec4',
    'ls-svm': '#4ca700',
    'k-lid-svm': '#d08400',
}
orders = {
    'svm': 8,
    'nn': 1,
    'nn_pgd': 2,
    'ln-robust-svm': 5,
    'roberta': 1,
    'vanilla-svm': 7,
    'curie-svm': 6,
    'ls-svm': 4,
    'k-lid-svm': 3,
    'unknown_model': 0,
}
line_width = 10
sigma = 0.01  # Standard deviation for Gaussian kernel

def _create_fig(width=15, height=15):
    """Create a figure and axis with the specified size."""
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111)
    return fig, ax

def _plot_posterior_style(fig, ax, x_label, y_label):
    """Apply the posterior style to the plot."""
    ax.xaxis.set_tick_params(labelsize=40)
    ax.yaxis.set_tick_params(labelsize=40)
    ax.set_xlabel(x_label, fontsize=50)
    ax.set_ylabel(y_label, fontsize=50)
    ax.grid(True)
    fig.tight_layout()

def extract_model_info(folder_name):
    """Extract model information from the folder name."""
    # Extract the part after 'model' and before the next '_'
    model_match = re.search(r'model([^_]+)', folder_name)
    if model_match:
        model_info = model_match.group(1).lower()
        if model_info.startswith('nn'):
            pgd_match = re.search(r'modelnn_([^_isbaseline]+)', folder_name)
            if pgd_match:
                return 'nn_pgd'
            return 'nn'
        return model_info
    return "unknown_model"

def read_performance_file(file_path, measure='accuracy'):
    """Read performance data from a file and return rounds and the chosen measure (accuracy or loss)."""
    rounds = []
    measure_values = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if re.match(r"^\d+ ", line):
                parts = line.split()
                rounds.append(int(parts[0]))
                if measure == 'loss':
                    measure_values.append(float(parts[1]))
                elif measure == 'accuracy':
                    measure_values.append(float(parts[2]))
    return rounds, measure_values


def read_nn_performance_file(file_path, measure='accuracy'):
    """Read NN performance data from a file formatted as round0_test_accuracy_loss...out."""
    epochs = []
    measure_values = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        header_found = False
        for line in lines:
            if header_found:
                if re.match(r"^\d+", line):
                    parts = line.split()
                    if len(parts) >= 3:
                        epoch = int(parts[0])
                        accuracy = float(parts[1])
                        loss = float(parts[2])
                        epochs.append(epoch)
                        if measure == 'accuracy':
                            measure_values.append(accuracy)
                        elif measure == 'loss':
                            measure_values.append(loss)
            if "Epoch Accuracy Loss" in line:
                header_found = True
    return epochs, measure_values

def perform_statistical_tests(model_data, perf_type, reference_model='svm'):
    """Perform statistical tests to check for significance."""
    reference_results = np.mean(model_data[reference_model][perf_type], axis=0)[1]

    if reference_results.size == 0:
        print(f"Reference model '{reference_model}' not found in the results.")
        return

    for model, results in model_data.items():
        if model == reference_model:
            continue

        model_results = np.mean(results[perf_type], axis=0)[1]
        print(f"Statistical test: for {perf_type}")
        print(f"Comparison: {reference_model} vs {model}")
        # Check normality for both sets of results
        _, p_ref = shapiro(reference_results)
        _, p_model = shapiro(model_results)

        if reference_results.shape[0] != model_results.shape[0]:
            if reference_results.shape[0] < model_results.shape[0]:
                extension_rounds = model_results.shape[0] - reference_results.shape[0]
                reference_results = np.append(reference_results, [reference_results[-1] * extension_rounds], axis=0)
            else:
                extension_rounds = reference_results.shape[0] - model_results.shape[0]
                model_results = np.append(model_results, [model_results[-1] * extension_rounds], axis=0)

        if p_ref > 0.05 and p_model > 0.05:
            # If both distributions are normal, use paired t-test
            t_stat, p_value = ttest_rel(reference_results, model_results)
            test_type = "Paired t-test"
        else:
            # If not normal, use non-parametric Wilcoxon test
            t_stat, p_value = wilcoxon(reference_results, model_results)
            test_type = "Wilcoxon signed-rank test"

        print(f"Comparison w {test_type}: {reference_model} vs {model} | p-value = {p_value:.5f}")
        if p_value < 0.05:
            print(f"Result: {reference_model} significantly outperforms {model}")
        else:
            print(f"Result: No significant difference detected between {reference_model} and {model}")

def extend_performance_to_max_round(rounds, values, max_round):
    """Align the results to a common x-axis length (max rounds) if needed."""
    assert len(rounds) == len(values), "Rounds and values must be the same length."
    current_max_round = rounds[-1]
    if current_max_round < max_round:
        extension_rounds = list(range(current_max_round + 1, max_round + 1))
        extended_values = values + [values[-1]] * len(extension_rounds)
        extended_rounds = rounds + extension_rounds
        return extended_rounds, extended_values
    return rounds, values

def plot_performance_with_std(rounds, avg_values, std_values, title, ax, color, label, inset_ax=None):
    """Plot and save the performance data with standard deviation shaded, using a specified color."""
    x_max = len(rounds)
    y_max = len(avg_values)

    avg_values_ymax = avg_values[:y_max]
    std_values_ymax = std_values[:y_max]

    print(f"TYPE: {title} | Method: {label} | AVG: {avg_values_ymax[-1]} | STDEV: {std_values_ymax[-1]}")
    if 'accuracy' in title or 'Accuracy' in title:
        print(f"TYPE: {title} | Method: {label} | MAX-AVG: {np.max(avg_values[:y_max])}")
    elif 'loss' in title or 'Loss' in title:
        print(f"TYPE: {title} | Method: {label} | MIN-AVG: {np.min(avg_values[:y_max])}")

    # # Apply Gaussian smoothing for better visibility
    # avg_values = gaussian_filter1d(avg_values, sigma=sigma)
    # std_values = gaussian_filter1d(std_values, sigma=sigma)

    ax.plot(rounds[:x_max], avg_values[:y_max],
            linewidth=line_width, linestyle='solid', color=color, label=label, zorder=orders[label.lower()])
    ax.fill_between(rounds[:x_max], avg_values[:y_max] - std_values[:y_max], avg_values[:y_max] + std_values[:y_max],
                    color=color, alpha=0.15)

    if inset_ax is not None:
        # Define the zoom range
        x1, x2 = 0, 1000  # X-axis range for zoom
        y1, y2 = 0.8, 0.9  # Y-axis range for zoom

        # Plot the zoomed-in area
        inset_ax.plot(rounds[:x_max], avg_values[:y_max],
                      linewidth=line_width, linestyle='solid', color=color, label=label, zorder=orders[label.lower()])
        inset_ax.fill_between(rounds[:x_max], avg_values[:y_max] - std_values[:y_max], avg_values[:y_max] + std_values[:y_max],
                              color=color, alpha=0.15)
        inset_ax.set_xlim(x1, x2)
        inset_ax.set_ylim(y1, y2)
        inset_ax.grid(True)
        inset_ax.xaxis.set_tick_params(labelsize=30)
        inset_ax.yaxis.set_tick_params(labelsize=30)

    # # Add an arrow pointing to the zoomed area
    # ax.annotate('', xytext=(0, 0.8), xy=(0.1, 0.1),  # (x, y) coordinates for arrow's tail and head
    #             textcoords='axes fraction',  # Interpret the coordinates as a fraction of the axes
    #             arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=10))

    ylabel = title.split()[-1].capitalize()
    ax.set_xlabel("Epochs", fontsize=50)
    ax.set_ylabel(ylabel, fontsize=50)
    ax.xaxis.set_tick_params(labelsize=40)
    # ax.set_ylim([0.85, 1.0])  # adjust if needed
    ax.yaxis.set_tick_params(labelsize=40)
    ax.grid(True)

def plot_experiment_results(base_folder, measure='accuracy', dump_folder='plots_output'):
    """Main function to traverse folders, aggregate results, and generate plots with mean and std."""
    if not os.path.exists(dump_folder):
        os.makedirs(dump_folder)

    model_data = {}
    model_names = set()
    max_round = 0

    for root, dirs, files in os.walk(base_folder):
        for folder in dirs:
            subfolder_path = os.path.join(root, folder)
            model_name = extract_model_info(folder)
            model_names.add(model_name)

            # Initialize storage for performance data
            if model_name not in model_data:
                model_data[model_name] = {'test': [], 'train': [], 'validation': []}

            # Traverse and read all .out files in the subfolder
            for file in os.listdir(subfolder_path):
                if file.endswith(".out"):
                    file_path = os.path.join(subfolder_path, file)
                    if '_test_perf' in file:
                        rounds, values = read_performance_file(file_path, measure)
                        perf_type = 'test'
                    elif '_train_perf' in file:
                        rounds, values = read_performance_file(file_path, measure)
                        perf_type = 'train'
                    elif '_validation_perf' in file:
                        rounds, values = read_performance_file(file_path, measure)
                        perf_type = 'validation'
                    else:
                        continue

                    if perf_type not in model_data[model_name]:
                        model_data[model_name][perf_type] = []
                    model_data[model_name][perf_type].append((rounds, values))

                    max_round = max(max_round, rounds[-1])

    # Plot the averages with standard deviations
    for perf_type in ['test']:   # only do for test performance
        fig, ax = _create_fig()
        inset_ax = None

        # # # Add zoomed-in inset
        # # # Define the location and size of the inset plot
        # inset_ax = inset_axes(ax, width="60%", height="60%",
        #                       # loc='lower right')  # Adjust the size and position as needed
        #                     bbox_to_anchor=(0.16, 0.1, 0.8, 0.8),  # (x0, y0, width, height) in axes fraction
        #                     bbox_transform = ax.transAxes,  # transform relative to the parent axes
        #                     loc = 'right')  # 'loc' determines which part of the bbox_to_anchor is aligned to the axes

        for model_name, perf_dict in model_data.items():
            all_rounds_values = perf_dict.get(perf_type, [])
            result_values_list = []

            for rounds, values in all_rounds_values:
                # Align the results to a common x-axis length (max rounds) if needed
                result_rounds, result_values = extend_performance_to_max_round(rounds, values, max_round)
                result_values_list.append(result_values)

            if result_values_list:
                all_values = np.array(result_values_list)
                avg_values = np.mean(all_values, axis=0)
                std_values = np.std(all_values, axis=0)
                color = MODEL_COLOR_MAP.get(model_name, "#000000")  # Default
                label = model_name.upper()
                plot_performance_with_std(result_rounds, avg_values, std_values,
                                          f"{perf_type.capitalize()} {measure.capitalize()}", ax, color, label, inset_ax)
        # Finalize and save the plot
        fig.suptitle(f"{perf_type.capitalize()} {measure.capitalize()}", fontsize=50)
        _plot_posterior_style(fig, ax, "Rounds", measure.capitalize())

        output_path = os.path.join(dump_folder, f"{perf_type}_{measure}.png")
        fig.savefig(output_path)
        output_path = os.path.join(dump_folder, f"{perf_type}_{measure}.pdf")
        fig.savefig(output_path)
        plt.close()

        perform_statistical_tests(model_data, perf_type, reference_model='svm')

    # # Create a separate legend plot
    # if model_names:
    #     fig_legend, ax_legend = _create_fig(15, 1)
    #     for model_name in model_names:
    #         ax_legend.plot([], [], color=MODEL_COLOR_MAP.get(model_name, "#000000"),
    #                        linewidth=line_width, label=model_name.upper())
    #
    #     h, l = ax_legend.get_legend_handles_labels()
    #     ax_legend.legend(h, l, loc='center', ncol=len(model_names), fontsize=30, frameon=False)
    #     ax_legend.axis('off')
    #     legend_path_png = os.path.join(dump_folder, "legend.png")
    #     legend_path_pdf = os.path.join(dump_folder, "legend.pdf")
    #     fig_legend.savefig(legend_path_png)
    #     fig_legend.savefig(legend_path_pdf)
    #     plt.close()

    print(f"Plots with averages, standard deviations, and legend have been saved to {dump_folder}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot experiment results from nested folders with mean and standard deviation.")
    parser.add_argument("base_folder", type=str, help="Path to the outermost folder containing experiment results.")
    parser.add_argument("-m", "--measure", type=str, choices=['accuracy', 'loss'], default='accuracy',
                        help="Performance measure to plot (accuracy or loss).")
    parser.add_argument("-d", "--dump_folder", type=str, default='plots_output', help="Folder to save the plots.")

    args = parser.parse_args()

    plot_experiment_results(args.base_folder, measure=args.measure, dump_folder=args.dump_folder)


if __name__ == "__main__":
    main()
