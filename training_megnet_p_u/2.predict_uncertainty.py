from __future__ import annotations

import json
import math
import os
import warnings

import matgl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


import numpy as np
import torch
from matplotlib import gridspec, ticker

from pymatgen.core import Structure
from sklearn.model_selection import train_test_split
from matgl.layers import BondExpansion

from my_megnet import MEGNet

warnings.simplefilter("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if torch.cuda.is_available():
    # torch.set_default_device("cuda:0")
    torch.cuda.set_device(0)

init_seed = 42
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def delete_cache():
    for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
        try:
            os.remove(fn)
        except FileNotFoundError:
            pass
    print("删除缓存完成")

def get_test_dataset():
    with open('../datasets/all_data.json', 'r') as fp:
        d = json.load(fp)
    # 高保真度数据集
    Exp_structures = [Structure.from_dict(x['structure']) for x in d['ordered_exp'].values()]
    Exp_targets = [torch.tensor(x['band_gap']) for x in d['ordered_exp'].values()]

    return Exp_structures, Exp_targets

def get_mae_mse_and_coverage_rate(model, test_x, test_y):
    uncertainties = []
    targets = []

    for s in test_x:
        pre = model.predict_structure(s)
        uncertainties.append(pre[0])
        targets.append(pre[1])

    uncertainties = torch.tensor(uncertainties)
    targets = torch.tensor(targets)

    # mae mse
    mae = torch.mean(torch.abs(targets - test_y))  # 将 test_y 转换为张量
    mse = torch.mean((targets - test_y) ** 2)

    # 计算覆盖率百分比
    lower_bound_2std = targets.numpy() - 2 * uncertainties.numpy()
    upper_bound_2std = targets.numpy() + 2 * uncertainties.numpy()
    is_inside_2std = (test_y >= torch.tensor(lower_bound_2std)) & (
            test_y <= torch.tensor(upper_bound_2std))  # 将 lower_bound 和 upper_bound 转换为张量
    coverage_percentage_2std = torch.mean(is_inside_2std.float()) * 100

    lower_bound_1std = targets.numpy() - uncertainties.numpy()
    upper_bound_1std = targets.numpy() + uncertainties.numpy()
    is_inside_1std = (test_y >= torch.tensor(lower_bound_1std)) & (
            test_y <= torch.tensor(upper_bound_1std))  # 将 lower_bound 和 upper_bound 转换为张量
    coverage_percentage_1std = torch.mean(is_inside_1std.float()) * 100

    return mae, mse, coverage_percentage_1std, coverage_percentage_2std, np.mean(uncertainties.numpy())



def predict_p_and_u(name):
    test_x, test_y = get_test_dataset()
    test_y = torch.tensor(test_y)

    bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=100, width=0.4)

    model = MEGNet(
        dim_node_embedding=64,
        dim_edge_embedding=100,
        dim_state_embedding=2,
        nblocks=3,
        hidden_layer_sizes_input=(64, 32),
        hidden_layer_sizes_conv=(64, 32, 32),
        nlayers_set2set=1,
        niters_set2set=3,
        hidden_layer_sizes_output=(32, 16),
        is_classification=False,
        activation_type="softplus",
        bond_expansion=bond_expansion,
        cutoff=5.0,
        gauss_width=0.4,
    )

    random_seed_list = [42 * i for i in range(1, 11)]
    print(f"{name}:")

    if name == 'scan':
        print(f"{name}_42:")
        model = matgl.load_model(path=f"./saved_models/{name}_42_p_u")
        mae, mse, cov1, cov2, u = get_mae_mse_and_coverage_rate(model, test_x, test_y)
        print(f"MAE:{mae.item():.3f} MSE:{mse.item():.3f}")
        print(f"1*u Coverage Percentage:{cov1.item():.3f}%")
        print(f"2*u Coverage Percentage:{cov2.item():.3f}%")
        print(f"avg 1*u: {u:.3f}")

    else:
        all_mae = []
        all_mse = []
        cov1_list = []
        cov2_list = []
        all_avg_uncertainty = []
        for seed in random_seed_list:
            print(f"{name}_{seed}:")
            model = matgl.load_model(path=f"./saved_models/{name}_{seed}_p_u")
            mae, mse, cov1, cov2, u = get_mae_mse_and_coverage_rate(model, test_x, test_y)
            all_mae.append(round(mae.item(), 3))
            all_mse.append(round(mse.item(), 3))
            cov1_list.append(round(cov1.item(), 3))
            cov2_list.append(round(cov2.item(), 3))
            all_avg_uncertainty.append(round(u,3))
            print(f"MAE:{mae.item():.3f} MSE:{mse.item():.3f}")
            print(f"1*u Coverage Percentage:{cov1.item():.3f}%")
            print(f"2*u Coverage Percentage:{cov2.item():.3f}%")
            print(f"avg 1*u:{u:.3f}")

        print(f"all mae: {all_mae} \naverage mae: {np.mean(all_mae):.3f}")
        print(f"all mse: {all_mse} \naverage mse: {np.mean(all_mse):.3f}")
        print(f"cov1_list: {cov1_list} \naverage cov1: {np.mean(cov1_list):.3f}%")
        print(f"cov2_list: {cov2_list} \naverage cov2: {np.mean(cov2_list):.3f}%")
        print(f"1*u_list: {all_avg_uncertainty} \navg 1*u: {np.mean(all_avg_uncertainty):.3f}")


    return 0


def plot_results(uncertainties, targets, test_y, name):
    targets = targets.numpy()
    u = uncertainties.numpy()
    test_y = test_y.numpy()
    upper = targets + u
    lower = targets - u

    flag = (test_y - upper) * (test_y - lower)

    y = np.array(flag)

    count_positive = sum(1 for val in y if val > 0)
    count_non_positive = len(y) - count_positive

    fig, axs = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
    plt.subplots_adjust(hspace=0.1)

    # Plot for y > 0
    axs[0].scatter(np.where(y > 0)[0], y[y > 0], c=test_y[y > 0], alpha=0.4, cmap='rainbow')
    axs[0].set_ylabel('Evaluation value (> 0)', fontname='Times New Roman', fontsize=20)
    axs[0].set_title(f'{name} Prediction Results', fontname='Times New Roman', fontsize=20)

    # Plot for y <= 0
    if name == 'PBE':
        # 为了更好展示这些点，这里添加了一些抖动，使得点不会过于密集，百分比计算结果不受影响
        np.random.seed(42)
        y_jittered = y.copy()
        mask = (y >= -0.002) & (y <= 0)
        y_jittered[mask] = y[mask] - abs(np.random.normal(0, 0.0005, size=y[mask].shape))

        axs[1].scatter(np.where(y <= 0)[0], y_jittered[y <= 0], c=test_y[y <= 0], alpha=0.4, cmap='rainbow')
        axs[1].set_xlabel('Data Index', fontname='Times New Roman', fontsize=20)
        axs[1].set_ylabel('Evaluation value (≤ 0)', fontname='Times New Roman', fontsize=20)
    elif name == 'Exp':
        np.random.seed(42)
        y_jittered = y.copy()
        mask = (y >= -0.0005) & (y <= 0)
        y_jittered[mask] = y[mask] - abs(np.random.normal(0, 0.001, size=y[mask].shape))

        axs[1].scatter(np.where(y <= 0)[0], y_jittered[y <= 0], c=test_y[y <= 0], alpha=0.4, cmap='rainbow')
        axs[1].set_xlabel('Data Index', fontname='Times New Roman', fontsize=20)
        axs[1].set_ylabel('Evaluation value (≤ 0)', fontname='Times New Roman', fontsize=20)
    elif name == 'HSE':
        np.random.seed(42)
        y_jittered = y.copy()
        mask = (y >= -0.005) & (y <= 0)
        y_jittered[mask] = y[mask] - abs(np.random.normal(0, 0.001, size=y[mask].shape))

        axs[1].scatter(np.where(y <= 0)[0], y_jittered[y <= 0], c=test_y[y <= 0], alpha=0.4, cmap='rainbow')
        axs[1].set_xlabel('Data Index', fontname='Times New Roman', fontsize=20)
        axs[1].set_ylabel('Evaluation value (≤ 0)', fontname='Times New Roman', fontsize=20)
    else:
        axs[1].scatter(np.where(y <= 0)[0], y[y <= 0], c=test_y[y <= 0], alpha=0.4, cmap='rainbow')
        axs[1].set_xlabel('Data Index', fontname='Times New Roman', fontsize=20)
        axs[1].set_ylabel('Evaluation value (≤ 0)', fontname='Times New Roman', fontsize=20)

    # Color bar for both subplots
    cbar = fig.colorbar(axs[0].collections[0], ax=axs, pad=0.02)
    cbar.set_label('True value', rotation=270, labelpad=15, fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    # Adding percentage text
    axs[0].text(0.07, 0.92, f'{(count_positive / len(y)) * 100:.2f}% > 0', ha='center', va='center', transform=axs[0].transAxes,
                 fontsize=20, fontname='Times New Roman', bbox=dict(facecolor='white', alpha=1.0, edgecolor='none'))
    axs[1].text(0.07, 0.07, f'{(count_non_positive / len(y)) * 100:.2f}% ≤ 0', ha='center', va='center', transform=axs[1].transAxes,
                 fontsize=20, fontname='Times New Roman', bbox=dict(facecolor='white', alpha=1.0, edgecolor='none'))

    # Adjust tick font sizes
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')

    # y_clip_pos = np.clip(y, 0, 25)
    # y_clip_neg = np.clip(y, -0.02, 0)

    xcustom_ticks = [0, 5, 10, 15, 20, 25]  # 自定义刻度值
    xcustom_ticklabels = ['0.0', '5.0', '10.0', '15.0', '20.0', '25.0']  # 自定义刻度标签
    axs[0].set_yticks(xcustom_ticks)
    axs[0].set_yticklabels(xcustom_ticklabels,
                           fontname='Times New Roman',
                           fontsize=20)

    if name == "HSE": # ok
        axs[0].set_ylim(-1, 25.6)
        axs[1].set_ylim(-0.005, 0.0001)
    elif name == "Exp":
        axs[0].set_ylim(-1, 25.5)
        axs[1].set_ylim(-0.015, 0.0005)
    elif name == "PBE":
        axs[0].set_ylim(-1, 25.3)
        axs[1].set_ylim(-0.005, 0.0001)
    elif name == "GLLB":
        axs[0].set_ylim(-1, 25.1)
        axs[1].set_ylim(-0.1, 0.005)
    else: # SCAN
        axs[0].set_ylim(-1, 26.6)
        axs[1].set_ylim(-0.033, 0.0008)


    def format_y_tick(x, pos):
        if np.isclose(x, 0.0):
            return '0'
        else:
            if name == 'SCAN' or name == 'GLLB':
                return f'{x:.2f}'
            else:
                return f'{x:.3f}'

    def format_y_tick_upper(x, pos):
        if np.isclose(x, 0.0):
            return '0'
        else:
            return f'{x:.0f}'

    # Apply the custom formatter to y-axis
    axs[0].yaxis.set_major_formatter(ticker.FuncFormatter(format_y_tick_upper))
    axs[1].yaxis.set_major_formatter(ticker.FuncFormatter(format_y_tick))

    # plt.tight_layout()
    # plt.savefig(f'../saved_fig/MEGNet/MEGNet_{name}_prediction.pdf', bbox_inches='tight')
    plt.show()
# 'scan', 'hse', 'gllb-sc', 'pbe'
datasets_name = ['scan', 'hse', 'gllb-sc', 'pbe']
# 'SCAN', 'Exp', 'GLLB', 'HSE', 'PBE'
for name in datasets_name:
    delete_cache()
    predict_p_and_u(name)

######################################################################
# scan
# MAE:1.685 MSE:4.356
# 1*u Coverage Percentage:18.683%
# 2*u Coverage Percentage:50.092%
# avg 1*u: 0.850

# hse
# all mae: [1.381, 1.431, 1.342, 1.137, 1.354, 1.434, 1.338, 1.203, 1.37, 1.305]
# average mae: 1.330
# all mse: [2.732, 4.659, 3.918, 2.559, 4.037, 4.124, 3.538, 2.93, 3.377, 2.564]
# average mse: 3.444
# cov1_list: [41.398, 48.909, 41.62, 45.838, 43.655, 31.04, 43.322, 45.801, 44.21, 77.58]
# average cov1: 46.337%
# cov2_list: [98.002, 73.437, 68.923, 67.777, 63.966, 63.3, 71.291, 79.097, 78.542, 96.448]
# average cov2: 76.078%
# 1*u_list: [1.6, 0.856, 0.702, 0.617, 0.601, 0.677, 0.798, 0.829, 0.971, 1.499]
# avg 1*u: 0.915

# gllb-sc
# all mae: [1.875, 2.439, 2.812, 2.731, 2.376, 2.169, 2.371, 2.317, 2.368, 2.396]
# average mae: 2.385
# all mse: [5.197, 9.695, 12.365, 9.811, 8.108, 6.923, 8.596, 8.824, 8.065, 9.423]
# average mse: 8.701
# cov1_list: [25.009, 17.795, 16.389, 18.054, 18.091, 24.75, 17.536, 13.282, 18.794, 16.611]
# average cov1: 18.631%
# cov2_list: [50.647, 29.819, 29.634, 40.03, 34.184, 43.877, 47.355, 24.602, 33.999, 43.655]
# average cov2: 37.780%
# 1*u_list: [0.897, 0.853, 1.101, 1.392, 1.041, 1.039, 0.981, 0.658, 0.972, 0.97]
# avg 1*u: 0.990

# pbe
# all mae: [1.118, 1.198, 1.107, 1.224, 1.131, 1.36, 1.131, 1.079, 1.113, 1.165]
# average mae: 1.163
# all mse: [2.802, 2.641, 2.786, 3.129, 2.468, 3.955, 3.05, 2.362, 2.777, 2.335]
# average mse: 2.831
# cov1_list: [37.699, 48.021, 49.168, 38.291, 54.31, 35.849, 44.247, 47.355, 28.339, 38.846]
# average cov1: 42.212%
# cov2_list: [62.301, 77.285, 71.735, 60.747, 78.727, 51.461, 59.674, 76.471, 57.048, 89.271]
# average cov2: 68.472%
# 1*u_list: [0.443, 0.824, 0.626, 0.513, 0.837, 0.393, 0.403, 0.701, 0.388, 0.902]
# avg 1*u: 0.603
######################################################################

# scan:
# MAE:1.685 MSE:4.356
# 1*u Coverage Percentage:18.683%
# 2*u Coverage Percentage:50.092%
# 删除缓存完成
# hse:
# MAE:1.381 MSE:2.732
# 1*u Coverage Percentage:41.398%
# 2*u Coverage Percentage:98.002%
# MAE:1.431 MSE:4.659
# 1*u Coverage Percentage:48.909%
# 2*u Coverage Percentage:73.437%
# MAE:1.342 MSE:3.918
# 1*u Coverage Percentage:41.620%
# 2*u Coverage Percentage:68.923%
# MAE:1.137 MSE:2.559
# 1*u Coverage Percentage:45.838%
# 2*u Coverage Percentage:67.777%
# MAE:1.354 MSE:4.037
# 1*u Coverage Percentage:43.655%
# 2*u Coverage Percentage:63.966%
# MAE:1.434 MSE:4.124
# 1*u Coverage Percentage:31.040%
# 2*u Coverage Percentage:63.300%
# MAE:1.338 MSE:3.538
# 1*u Coverage Percentage:43.322%
# 2*u Coverage Percentage:71.291%
# MAE:1.203 MSE:2.930
# 1*u Coverage Percentage:45.801%
# 2*u Coverage Percentage:79.097%
# MAE:1.370 MSE:3.377
# 1*u Coverage Percentage:44.210%
# 2*u Coverage Percentage:78.542%
# MAE:1.305 MSE:2.564
# 1*u Coverage Percentage:77.580%
# 2*u Coverage Percentage:96.448%
# all mae: [1.381, 1.431, 1.342, 1.137, 1.354, 1.434, 1.338, 1.203, 1.37, 1.305]
# average mae: 1.330
# all mse: [2.732, 4.659, 3.918, 2.559, 4.037, 4.124, 3.538, 2.93, 3.377, 2.564]
# average mse: 3.444
# cov1_list: [41.398, 48.909, 41.62, 45.838, 43.655, 31.04, 43.322, 45.801, 44.21, 77.58]
# average cov1: 46.337%
# cov2_list: [98.002, 73.437, 68.923, 67.777, 63.966, 63.3, 71.291, 79.097, 78.542, 96.448]
# average cov2: 76.078%
# 删除缓存完成
# gllb-sc:
# MAE:1.875 MSE:5.197
# 1*u Coverage Percentage:25.009%
# 2*u Coverage Percentage:50.647%
# MAE:2.439 MSE:9.695
# 1*u Coverage Percentage:17.795%
# 2*u Coverage Percentage:29.819%
# MAE:2.812 MSE:12.365
# 1*u Coverage Percentage:16.389%
# 2*u Coverage Percentage:29.634%
# MAE:2.731 MSE:9.811
# 1*u Coverage Percentage:18.054%
# 2*u Coverage Percentage:40.030%
# MAE:2.376 MSE:8.108
# 1*u Coverage Percentage:18.091%
# 2*u Coverage Percentage:34.184%
# MAE:2.169 MSE:6.923
# 1*u Coverage Percentage:24.750%
# 2*u Coverage Percentage:43.877%
# MAE:2.371 MSE:8.596
# 1*u Coverage Percentage:17.536%
# 2*u Coverage Percentage:47.355%
# MAE:2.317 MSE:8.824
# 1*u Coverage Percentage:13.282%
# 2*u Coverage Percentage:24.602%
# MAE:2.368 MSE:8.065
# 1*u Coverage Percentage:18.794%
# 2*u Coverage Percentage:33.999%
# MAE:2.396 MSE:9.423
# 1*u Coverage Percentage:16.611%
# 2*u Coverage Percentage:43.655%
# all mae: [1.875, 2.439, 2.812, 2.731, 2.376, 2.169, 2.371, 2.317, 2.368, 2.396]
# average mae: 2.385
# all mse: [5.197, 9.695, 12.365, 9.811, 8.108, 6.923, 8.596, 8.824, 8.065, 9.423]
# average mse: 8.701
# cov1_list: [25.009, 17.795, 16.389, 18.054, 18.091, 24.75, 17.536, 13.282, 18.794, 16.611]
# average cov1: 18.631%
# cov2_list: [50.647, 29.819, 29.634, 40.03, 34.184, 43.877, 47.355, 24.602, 33.999, 43.655]
# average cov2: 37.780%
# 删除缓存完成
# pbe:
# MAE:1.118 MSE:2.802
# 1*u Coverage Percentage:37.699%
# 2*u Coverage Percentage:62.301%
# MAE:1.198 MSE:2.641
# 1*u Coverage Percentage:48.021%
# 2*u Coverage Percentage:77.285%
# MAE:1.107 MSE:2.786
# 1*u Coverage Percentage:49.168%
# 2*u Coverage Percentage:71.735%
# MAE:1.224 MSE:3.129
# 1*u Coverage Percentage:38.291%
# 2*u Coverage Percentage:60.747%
# MAE:1.131 MSE:2.468
# 1*u Coverage Percentage:54.310%
# 2*u Coverage Percentage:78.727%
# MAE:1.360 MSE:3.955
# 1*u Coverage Percentage:35.849%
# 2*u Coverage Percentage:51.461%
# MAE:1.131 MSE:3.050
# 1*u Coverage Percentage:44.247%
# 2*u Coverage Percentage:59.674%
# MAE:1.079 MSE:2.362
# 1*u Coverage Percentage:47.355%
# 2*u Coverage Percentage:76.471%
# MAE:1.113 MSE:2.777
# 1*u Coverage Percentage:28.339%
# 2*u Coverage Percentage:57.048%
# MAE:1.165 MSE:2.335
# 1*u Coverage Percentage:38.846%
# 2*u Coverage Percentage:89.271%
# all mae: [1.118, 1.198, 1.107, 1.224, 1.131, 1.36, 1.131, 1.079, 1.113, 1.165]
# average mae: 1.163
# all mse: [2.802, 2.641, 2.786, 3.129, 2.468, 3.955, 3.05, 2.362, 2.777, 2.335]
# average mse: 2.831
# cov1_list: [37.699, 48.021, 49.168, 38.291, 54.31, 35.849, 44.247, 47.355, 28.339, 38.846]
# average cov1: 42.212%
# cov2_list: [62.301, 77.285, 71.735, 60.747, 78.727, 51.461, 59.674, 76.471, 57.048, 89.271]
# average cov2: 68.472%