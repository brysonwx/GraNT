import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

"""
    For plotting loss_val, mae_val, loss_train and other curves from wandb recorded data.
"""

def plot_mean_var_curve(time_bins, mean_curve, min_curve, max_curve, label_name):
    line, = plt.plot(time_bins, mean_curve, alpha=0.75, linewidth=13.0, label=label_name)
    # line, = plt.plot(time_bins, mean_curve, alpha=0.75, linewidth=13.0, label=label_name, zorder=1, marker=None)
    plt.scatter(time_bins[-1], mean_curve[-1], color=line.get_color(), s=240, zorder=2)


def plot_mean_var_curve_1(time_bins, mean_curve, min_curve, max_curve, label_name):
    line, = plt.plot(time_bins, mean_curve, alpha=0.75, linewidth=10.0, label=label_name)
    # plt.plot(time_bins, mean_curve, alpha=0.75, linewidth=2.7, label=label_name)
    plt.scatter(time_bins[-1], mean_curve[-1], color=line.get_color(), s=240, zorder=5)


def plot_mean_var_curve_2(time_bins, mean_curve, min_curve, max_curve, label_name):
    line, = plt.plot(time_bins, mean_curve, alpha=0.75, linewidth=7.0, label=label_name)
    # plt.plot(time_bins, mean_curve, alpha=0.75, linewidth=2.7, label=label_name)
    plt.scatter(time_bins[-1], mean_curve[-1], color=line.get_color(), s=240, zorder=5)


def plt_init_fig():
    plt.figure(figsize=(20, 15))
    plt.style.use('seaborn-darkgrid')


def plt_save_fig(curve_save_path, y_label):
    plt.xlabel('Wallclock Time (s)', fontsize=65)
    plt.ylabel(y_label, fontsize=65)
    plt.legend(fontsize=70)
    plt.tick_params(axis='both', which='major', labelsize=40)
    plt.tight_layout()
    plt.savefig(curve_save_path, bbox_inches='tight')


def smooth_curve(curve):
    return curve


def wallclock_curve(path, column_name, step_name, num_bins):
    df = pd.read_csv(path)

    start_time, end_time = get_start_end_time(df, step_name)
    time_bins = split_timespan(start_time, end_time, num_bins)

    mean_curve = []
    min_curve = []
    max_curve = []
    for i in tqdm(range(len(time_bins) - 1)):
        point_sum = summary_at_bin(df, time_bins[i], time_bins[i + 1], column_name)
        mean_curve.append(point_sum[0])
        min_curve.append(point_sum[1])
        max_curve.append(point_sum[2])

    mean_curve = pd.Series(mean_curve).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    min_curve = pd.Series(min_curve).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    max_curve = pd.Series(max_curve).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

    time_bins = time_bins[:-1]
    mean_curve = smooth_curve(np.array(mean_curve), window_size=3)
    min_curve = smooth_curve(np.array(min_curve), window_size=3)
    max_curve = smooth_curve(np.array(max_curve), window_size=3)

    return time_bins, mean_curve, min_curve, max_curve


def average_endtime(df, step_name):
    max_step_row = df[df[step_name] == df[step_name].max()]
    # max_step_row = df[df[step_name] == 250.0]  # gen-reg
    max_time = max_step_row['wallclock_time'].max()
    print(max_time)
    return max_time


def get_start_end_time(df, step_name):
    start_time = df['wallclock_time'].iloc[0]
    end_time = average_endtime(df, step_name)
    return start_time, end_time


def split_timespan(start_time, end_time, num_bins=2000):
    return np.linspace(start_time, end_time, num_bins)


def summary_at_bin(df, start_time, end_time, column_name):
    filtered_rows = df[(df['wallclock_time'] >= start_time) & (df['wallclock_time'] <= end_time)]
    mean_val = filtered_rows[column_name].mean()
    min_val = filtered_rows[column_name].min()
    max_val = filtered_rows[column_name].max()
    return (mean_val, min_val, max_val)


if __name__ == "__main__":
    plt_init_fig()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='qm9')
    # loss_val / mae_val / loss_train / acc_val / mae_train / acc_train
    parser.add_argument('--plot_type', type=str, default='loss_val')
    # nvidia / amd
    parser.add_argument('--device_type', type=str, default='amd')
    args = parser.parse_args()
    dataset_name = args.dataset.lower()
    plot_type = args.plot_type.lower()
    assert plot_type in ('loss_val', 'mae_val', 'loss_train', 'acc_val', 'mae_train', 'acc_train')
    device_type = args.device_type.lower()
    assert device_type in ('nvidia', 'amd')
    # csv_path = f'{dataset_name}/{plot_type}_{dataset_name}.csv'
    csv_path = f'{dataset_name}/{plot_type}_{dataset_name}_{device_type}.csv'
    if dataset_name == 'qm9':  # graph regression, sampling_type: 1, 2
        num_bins = 2000
        if device_type == 'nvidia':
            curve_name_1 = f'full_{dataset_name}_750_0.00005_{device_type} - {plot_type}'
            step_name_1 = f'full_{dataset_name}_750_0.00005_{device_type} - _step'

            curve_name_2 = f's1_{dataset_name}_750_0.00005_{device_type} - {plot_type}'
            step_name_2 = f's1_{dataset_name}_750_0.00005_{device_type} - _step'

            curve_name_3 = f's2_{dataset_name}_750_0.00005_{device_type} - {plot_type}'
            step_name_3 = f's2_{dataset_name}_750_0.00005_{device_type} - _step'

        label_name_1 = 'without GraNT'
        time_bins_1, mean_curve_1, min_curve_1, max_curve_1 = wallclock_curve(csv_path, curve_name_1, step_name_1, num_bins)
        plot_mean_var_curve(time_bins_1, mean_curve_1, min_curve_1, max_curve_1, label_name_1)

        label_name_2 = 'with GraNT (B)'
        time_bins_2, mean_curve_2, min_curve_2, max_curve_2 = wallclock_curve(csv_path, curve_name_2, step_name_2, num_bins)
        plot_mean_var_curve(time_bins_2, mean_curve_2, min_curve_2, max_curve_2, label_name_2)

        label_name_3 = 'with GraNT (S)'
        time_bins_3, mean_curve_3, min_curve_3, max_curve_3 = wallclock_curve(csv_path, curve_name_3, step_name_3, num_bins)
        plot_mean_var_curve(time_bins_3, mean_curve_3, min_curve_3, max_curve_3, label_name_3)
    elif dataset_name == 'zinc':  # graph regression, sampling_type: 1, 2
        num_bins = 2000
        if device_type == 'nvidia':
            curve_name_1 = f'full_{dataset_name}_1000_0.0004_{device_type} - {plot_type}'
            step_name_1 = f'full_{dataset_name}_1000_0.0004_{device_type} - _step'

            curve_name_2 = f's1_{dataset_name}_1000_0.0004_{device_type} - {plot_type}'
            step_name_2 = f's1_{dataset_name}_1000_0.0004_{device_type} - _step'

            curve_name_3 = f's2_{dataset_name}_1000_0.0004_{device_type} - {plot_type}'
            step_name_3 = f's2_{dataset_name}_1000_0.0004_{device_type} - _step'
        else:  # amd
            raise NotImplementedError

        label_name_1 = 'without GraNT'
        time_bins_1, mean_curve_1, min_curve_1, max_curve_1 = wallclock_curve(csv_path, curve_name_1, step_name_1, num_bins)
        plot_mean_var_curve(time_bins_1, mean_curve_1, min_curve_1, max_curve_1, label_name_1)

        label_name_2 = 'with GraNT (B)'
        time_bins_2, mean_curve_2, min_curve_2, max_curve_2 = wallclock_curve(csv_path, curve_name_2, step_name_2, num_bins)
        plot_mean_var_curve(time_bins_2, mean_curve_2, min_curve_2, max_curve_2, label_name_2)

        label_name_3 = 'with GraNT (S)'
        time_bins_3, mean_curve_3, min_curve_3, max_curve_3 = wallclock_curve(csv_path, curve_name_3, step_name_3, num_bins)
        plot_mean_var_curve_2(time_bins_3, mean_curve_3, min_curve_3, max_curve_3, label_name_3)
    elif dataset_name == 'gen-reg':  # node regression, sampling_type: 3, 4
        num_bins = 1000
        # num_bins = 500
        if device_type == 'amd':
            raise NotImplementedError
        else:  # nvidia
            curve_name_1 = f'gd-reg-full_0.0002 - {plot_type}'
            step_name_1 = f'gd-reg-full_0.0002 - _step'

            curve_name_2 = f'gd-reg-s3_0.0002 - {plot_type}'
            step_name_2 = f'gd-reg-s3_0.0002 - _step'

            curve_name_3 = f'gd-reg-s4_0.0002 - {plot_type}'
            step_name_3 = f'gd-reg-s4_0.0002 - _step'

        label_name_1 = 'without GraNT'
        time_bins_1, mean_curve_1, min_curve_1, max_curve_1 = wallclock_curve(csv_path, curve_name_1, step_name_1, num_bins)
        plot_mean_var_curve_1(time_bins_1, mean_curve_1, min_curve_1, max_curve_1, label_name_1)

        label_name_2 = 'with GraNT (B)'
        time_bins_2, mean_curve_2, min_curve_2, max_curve_2 = wallclock_curve(csv_path, curve_name_2, step_name_2, num_bins)
        plot_mean_var_curve(time_bins_2, mean_curve_2, min_curve_2, max_curve_2, label_name_2)

        label_name_3 = 'with GraNT (S)'
        time_bins_3, mean_curve_3, min_curve_3, max_curve_3 = wallclock_curve(csv_path, curve_name_3, step_name_3, num_bins)
        plot_mean_var_curve_2(time_bins_3, mean_curve_3, min_curve_3, max_curve_3, label_name_3)
    elif dataset_name == 'gen-cls':
        num_bins = 500
        if device_type == 'amd':
            curve_name_1 = f'{dataset_name}_full_0.02 - {plot_type}'
            step_name_1 = f'{dataset_name}_full_0.02 - _step'

            curve_name_2 = f'{dataset_name}_s3_0.02 - {plot_type}'
            step_name_2 = f'{dataset_name}_s3_0.02 - _step'

            curve_name_3 = f'{dataset_name}_s4_0.02 - {plot_type}'
            step_name_3 = f'{dataset_name}_s4_0.02 - _step'
        else:  # nvidia
            curve_name_1 = f'{dataset_name}_full_nvidia_0.0002 - {plot_type}'
            step_name_1 = f'{dataset_name}_full_nvidia_0.0002 - _step'

            curve_name_2 = f'{dataset_name}_s3_nvidia_0.0002 - {plot_type}'
            step_name_2 = f'{dataset_name}_s3_nvidia_0.0002 - _step'

            curve_name_3 = f'{dataset_name}_s4_nvidia_0.0002 - {plot_type}'
            step_name_3 = f'{dataset_name}_s4_nvidia_0.0002 - _step'

        label_name_1 = 'without GraNT'
        time_bins_1, mean_curve_1, min_curve_1, max_curve_1 = wallclock_curve(csv_path, curve_name_1, step_name_1, num_bins)
        plot_mean_var_curve_1(
            time_bins_1, mean_curve_1, min_curve_1, max_curve_1, label_name_1)

        label_name_2 = 'with GraNT (B)'
        time_bins_2, mean_curve_2, min_curve_2, max_curve_2 = wallclock_curve(csv_path, curve_name_2, step_name_2, num_bins)
        plot_mean_var_curve(
            time_bins_2, mean_curve_2, min_curve_2, max_curve_2, label_name_2)

        label_name_3 = 'with GraNT (S)'
        time_bins_3, mean_curve_3, min_curve_3, max_curve_3 = wallclock_curve(csv_path, curve_name_3, step_name_3, num_bins)
        plot_mean_var_curve_2(
            time_bins_3, mean_curve_3, min_curve_3, max_curve_3, label_name_3)
    elif dataset_name == 'ogbg-molhiv':
        num_bins = 1000
        if device_type == 'nvidia':
            curve_name_1 = f'full_0.1_x - {plot_type}'
            step_name_1 = f'full_0.1_x - _step'

            curve_name_2 = f's1_0.1_x - {plot_type}'
            step_name_2 = f's1_0.1_x - _step'

            curve_name_3 = f's2_0.1_x - {plot_type}'
            step_name_3 = f's2_0.1_x - _step'
        else:
            raise NotImplementedError
        label_name_1 = 'without GraNT'
        time_bins_1, mean_curve_1, min_curve_1, max_curve_1 = wallclock_curve(csv_path, curve_name_1, step_name_1, num_bins)
        plot_mean_var_curve_2(
            time_bins_1, mean_curve_1, min_curve_1, max_curve_1, label_name_1)

        label_name_2 = 'with GraNT (B)'
        time_bins_2, mean_curve_2, min_curve_2, max_curve_2 = wallclock_curve(csv_path, curve_name_2, step_name_2, num_bins)
        plot_mean_var_curve_2(
            time_bins_2, mean_curve_2, min_curve_2, max_curve_2, label_name_2)

        label_name_3 = 'with GraNT (S)'
        time_bins_3, mean_curve_3, min_curve_3, max_curve_3 = wallclock_curve(csv_path, curve_name_3, step_name_3, num_bins)
        plot_mean_var_curve_2(
            time_bins_3, mean_curve_3, min_curve_3, max_curve_3, label_name_3)
    elif dataset_name == 'ogbg-molpcba':
        num_bins = 2000
        if device_type == 'nvidia':
            curve_name_1 = f'molpcba-3,407-full - {plot_type}'
            step_name_1 = f'molpcba-3,407-full - _step'

            curve_name_2 = f'molpcba-3,407-s1 - {plot_type}'
            step_name_2 = f'molpcba-3,407-s1 - _step'

            curve_name_3 = f'molpcba-3,407-s2 - {plot_type}'
            step_name_3 = f'molpcba-3,407-s2 - _step'

        label_name_1 = 'without GraNT'
        time_bins_1, mean_curve_1, min_curve_1, max_curve_1 = wallclock_curve(csv_path, curve_name_1, step_name_1,
                                                                              num_bins)
        plot_mean_var_curve_1(time_bins_1, mean_curve_1, min_curve_1, max_curve_1, label_name_1)

        label_name_2 = 'with GraNT (B)'
        time_bins_2, mean_curve_2, min_curve_2, max_curve_2 = wallclock_curve(csv_path, curve_name_2, step_name_2,
                                                                              num_bins)
        plot_mean_var_curve(time_bins_2, mean_curve_2, min_curve_2, max_curve_2, label_name_2)

        label_name_3 = 'with GraNT (S)'
        time_bins_3, mean_curve_3, min_curve_3, max_curve_3 = wallclock_curve(csv_path, curve_name_3, step_name_3,
                                                                              num_bins)
        plot_mean_var_curve_2(time_bins_3, mean_curve_3, min_curve_3, max_curve_3, label_name_3)

    if plot_type == 'loss_val':
        y_label = 'Validation Loss'
    elif plot_type == 'mae_val':
        y_label = 'Validation MAE'
    elif plot_type == 'loss_train':
        y_label = 'Training Loss'
    elif plot_type == 'mae_train':
        y_label = 'Training MAE'
    elif plot_type == 'acc_val':
        if dataset_name in ('gen-cls', 'ogbg-molhiv'):
            y_label = 'Validation ROC-AUC'
        elif dataset_name == 'ogbg-molpcba':
            y_label = 'Validation AP'
        else:
            y_label = 'Validation Acc'
    elif plot_type == 'acc_train':
        if dataset_name in ('gen-cls', 'ogbg-molhiv'):
            y_label = 'Training ROC-AUC'
        elif dataset_name == 'ogbg-molpcba':
            y_label = 'Training AP'
        else:
            y_label = 'Training Acc'

    fig_path = f'{dataset_name}/wandb_wallclock_{plot_type}_{dataset_name}_{device_type}.pdf'
    print(fig_path)
    plt_save_fig(fig_path, y_label)
