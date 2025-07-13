import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Apply the desired plot style globally
plt.style.use('fast')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
# latex font
plt.rcParams['font.serif'] = ['Computer Modern'] + plt.rcParams['font.serif']
plt('text', usetex=True)
line_colors = {'clean': '#FB00FF',
               '0.1': '#820084', 'Adv_0.1': '#820084',
               '0.25': '#00e1a0', 'Adv_0.25': '#00e1a0',
               '0.3': '#007a57', 'Adv_0.3': '#007a57',
               '0.35': '#515151', 'Adv_0.35': '#515151',
               '0.4': '#000000', 'Adv_0.4': '#000000'}

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

def read_point_index(file_path):
    """Reads the PointIndex column from a result file and returns it as a list of integers."""
    point_indices = []
    with open(file_path, 'r') as f:
        for line in f:
            if re.match(r'\d+\s+\d+', line):
                point_index = int(line.split()[1])
                point_indices.append(point_index)
    return point_indices


def get_seed_from_filename(filename):
    """Extracts the seed from the filename."""
    match = re.search(r'seed(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def compute_common_entries(lists):
    """Computes the percentage of common entries across multiple lists."""
    if not lists:
        return 0
    # Find common entries in all lists using set intersection
    common_entries = set(lists[0])
    for l in lists[1:]:
        common_entries.intersection_update(l)
    # Return the percentage of common entries
    total_entries = len(lists[0])
    if total_entries == 0:
        return 0
    return len(common_entries) / total_entries * 100


def plot_common_entries(common_entries_dict, save_path):
    """Plots a bar plot of the average number of common entries."""
    if common_entries_dict:
        avg_common_percentage = sum(common_entries_dict.values()) / len(common_entries_dict)
        stdev_common_percentage = np.std(list(common_entries_dict.values()))
    else:
        avg_common_percentage = 0

    # Plot
    fig, ax = _create_fig()
    # plt.bar(seeds, common_percentages, color='blue')
    ax.bar(['Clean'], [avg_common_percentage], yerr= stdev_common_percentage, color=line_colors['clean'], alpha=1, linewidth=20, width=0.5, error_kw=dict(elinewidth=5, capsize=15, capthick=5))
    ax.bar(['10%'], [avg_common_percentage], color=line_colors['0.1'], alpha=1, linewidth=20, width=0.5, error_kw=dict(elinewidth=5, capsize=15, capthick=5))
    ax.bar(['25%'], [avg_common_percentage], color=line_colors['0.25'], alpha=1, linewidth=20, width=0.5, error_kw=dict(elinewidth=5, capsize=15, capthick=5))
    ax.bar(['30%'], [avg_common_percentage], color=line_colors['0.3'], alpha=1, linewidth=20, width=0.5, error_kw=dict(elinewidth=5, capsize=15, capthick=5))
    ax.bar(['35%'], [avg_common_percentage], color=line_colors['0.35'], alpha=1, linewidth=20, width=0.5, error_kw=dict(elinewidth=5, capsize=15, capthick=5))
    ax.bar(['40%'], [avg_common_percentage], color=line_colors['0.4'], alpha=1, linewidth=20, width=0.5, error_kw=dict(elinewidth=5, capsize=15, capthick=5))


    ax.xaxis.set_tick_params(labelsize=40)
    ax.yaxis.set_tick_params(labelsize=40)
    ax.grid(True)
    fig.suptitle(f"Common Influential Points: FLORAL vs RoBERTa", fontsize=40)
    _plot_posterior_style(fig, ax, "Label Poisoning Level", 'Common Influential Points (%)'.capitalize())

    fig.savefig(save_path)
    plt.close()


def process_directory(directory):
    """Processes all .out files in the directory, grouping by seeds and computing common entries."""
    seed_dict = {}

    # Group files by seeds
    for filename in os.listdir(directory):
        if filename.endswith('.out'):
            seed = get_seed_from_filename(filename)
            if seed is not None:
                file_path = os.path.join(directory, filename)
                point_indices = read_point_index(file_path)

                if seed not in seed_dict:
                    seed_dict[seed] = []
                seed_dict[seed].append(point_indices)

    print(f"Grouped results with seeds: {seed_dict.keys()}")
    # Compute percentage of common entries for each seed
    common_entries_dict = {}
    for seed, lists in seed_dict.items():
        common_percentage = compute_common_entries(lists)
        common_entries_dict[seed] = common_percentage

    return common_entries_dict

def extract_dataset_name(subfolder_name):
    """Extracts the dataset name from the subfolder name, after the '_D' prefix."""
    if 'adv' in subfolder_name:
        return "Adv_" + subfolder_name.split('_Dadv')[1]
    else:
        return subfolder_name.split('_D')[1]
    return subfolder_name

def plot_multiple_datasets(main_folder, save_path, desired_order=None):
    """Processes each subfolder, computes the average common entries for all datasets, and plots them."""
    dataset_results = {}
    dataset_results_stdev = {}

    # Process each subfolder (dataset)
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)

        if os.path.isdir(subfolder_path):
            dataset_name = extract_dataset_name(subfolder)
            print(f"Processing {subfolder_path}")
            # Process the subfolder and compute common entries
            common_entries_dict = process_directory(subfolder_path)

            print(common_entries_dict)
            # Calculate the average percentage of common entries for the dataset
            if common_entries_dict:
                avg_common_percentage = sum(common_entries_dict.values()) / len(common_entries_dict)
                stdev_common_percentage = np.std(list(common_entries_dict.values()))
            else:
                avg_common_percentage = 0
                stdev_common_percentage = 0

            # Store the results using dataset_name as key
            dataset_results[dataset_name] = avg_common_percentage
            dataset_results_stdev[dataset_name] = stdev_common_percentage

    # print(dataset_results)
    # Prepare data for plotting in the desired order
    if desired_order:
        dataset_names = []
        avg_common_percentages = []
        stdev_common_percentages = []

        # Add the datasets to the lists according to the desired order
        for dataset in desired_order:
            if dataset in dataset_results:
                dataset_names.append(dataset)
                avg_common_percentages.append(dataset_results[dataset])
                stdev_common_percentages.append(dataset_results_stdev[dataset])
    else:
        # If no desired order is provided, just use the default order from processing
        dataset_names = list(dataset_results.keys())
        avg_common_percentages = list(dataset_results.values())


    # Plot
    fig, ax = _create_fig()
    for i in range(len(dataset_names)):
        ax.bar(dataset_names[i].capitalize(), avg_common_percentages[i], yerr=stdev_common_percentages[i],
               color=line_colors[dataset_names[i]], alpha=1, linewidth=20, width=0.5,
               error_kw=dict(elinewidth=5, capsize=15, capthick=5, ecolor='red'))

    ax.xaxis.set_tick_params(labelsize=40)
    plt.xticks(rotation=45, ha='right')  # Rotate dataset names for better readability
    ax.yaxis.set_tick_params(labelsize=40)
    ax.grid(True)
    fig.suptitle(f"Common Influential Training Points (%): \n Identified by FLORAL and RoBERTa", fontsize=40)
    _plot_posterior_style(fig, ax, "Dataset", '(%) Common Points'.capitalize())

    fig.savefig(save_path)
    save_path = save_path + ".pdf"
    fig.savefig(save_path)
    plt.close()



if __name__ == "__main__":
    directory = "PATH_TO_RESULTS"
    save_path = directory + "/common_influential_points_plot.png"  # Change this to your desired save path

    # common_entries_dict = process_directory(directory)
    # # Plot the results
    # plot_common_entries(common_entries_dict, save_path)

    # Define the desired order of datasets for the plot
    desired_order = ['clean', 'Adv_0.1', 'Adv_0.25', 'Adv_0.3', 'Adv_0.35', 'Adv_0.4']  # Modify as needed (names after '_D')
    # Plot the results for multiple datasets and save the figure
    plot_multiple_datasets(directory, save_path, desired_order)
