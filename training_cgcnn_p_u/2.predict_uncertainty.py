from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import warnings
from random import sample

from matplotlib import ticker
from torch.utils.data import DataLoader

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pymatgen.core import Structure
from sklearn.model_selection import train_test_split

from data import StruData, collate_pool
from model import CrystalGraphConvNet
import model

warnings.simplefilter("ignore")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('--max_epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run (default: 1000)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')
parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
# emb dim
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')

parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')

args = parser.parse_args(sys.argv[1:])

if torch.cuda.is_available():
    # torch.set_default_device("cuda:0")
    torch.cuda.set_device(0)

init_seed = 42
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def get_test_dataset():
    with open('../datasets/all_data.json', 'r') as fp:
        d = json.load(fp)
    # 高保真度数据集
    Exp_structures = [Structure.from_dict(x['structure']) for x in d['ordered_exp'].values()]
    Exp_targets = [torch.tensor(x['band_gap']) for x in d['ordered_exp'].values()]
    # print(f"测试集大小：{len(Exp_structures)}")

    return Exp_structures, Exp_targets

def load_dataset(dict_targets):
    with open("../datasets/mp.2019.04.01.json") as f:
        structure_data = {i["material_id"]: i["structure"] for i in json.load(f)}
    structures = []
    targets = []
    for mid in dict_targets.keys():
        dict_targets[mid] = torch.tensor(dict_targets[mid])
        struct = Structure.from_str(structure_data[mid], fmt="cif")
        structures.append(struct)
        targets.append(dict_targets[mid])
    return structures, targets

def get_low_fidelity_dataset(name):
    with gzip.open("../datasets/data_no_structs.json.gz", "rb") as f:
        bandgap = json.loads(f.read())
    # 低保真度数据集
    now_bandgap = bandgap[name]
    now_structures, now_targets = load_dataset(now_bandgap)
    # print(f"{name}数据集大小：{len(now_structures),len(now_targets)}")
    return now_structures, now_targets

def get_high_fidelity_dataset():
    with open('../datasets/all_data.json', 'r') as fp:
        d = json.load(fp)
    # 高保真度数据集
    Exp_structures = [Structure.from_dict(x['structure']) for x in d['ordered_exp'].values()]
    Exp_targets = [torch.tensor(x['band_gap']) for x in d['ordered_exp'].values()]
    combined = list(zip(Exp_structures, Exp_targets))
    exp_train_data, exp_test_data = train_test_split(combined, test_size=0.5, random_state=init_seed)
    exp_structures = [e[0] for e in exp_train_data]
    exp_targets = [e[1] for e in exp_train_data]
    exp_test_structures = [e[0] for e in exp_test_data]
    exp_test_targets = [e[1] for e in exp_test_data]

    # print(f"Exp数据集大小：{len(exp_structures)}")
    return exp_structures, exp_targets, exp_test_structures, exp_test_targets

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

class Cgcnn_lightning(pl.LightningModule):

    def __init__(self, crystalGraphConvNet, normalizer):
        super().__init__()
        self.crystalGraphConvNet = crystalGraphConvNet
        self.normalizer = normalizer

    def forward(self, *input):
        return self.crystalGraphConvNet(*input)

    def training_step(self, batch):
        x, y = batch

        input_var = (x[0], x[1], x[2], x[3])

        target_var = self.normalizer.norm(y)

        zeros = torch.zeros_like(target_var)
        now_target_var = torch.cat((zeros, target_var), dim=1)

        y_hat = self.forward(*input_var)

        delta = torch.abs(torch.squeeze(target_var) - y_hat[:, 1])
        now_target_var[:,0] = delta

        # loss_fn = nn.MSELoss()
        diff = torch.abs(now_target_var[:, 1] - y_hat[:, 1])  # |Ptrue - P|
        mse_loss = nn.MSELoss()
        u_loss = mse_loss(diff, y_hat[:, 0])  # (|Ptrue - P| - u)²
        # print("u_loss:",u_loss)
        total_loss = mse_loss(now_target_var[:, 1], y_hat[:, 1]) + u_loss  # 计算总损失 (Ptrue -P)² + (|Ptrue - P| - u)²

        # loss = loss_fn(y_hat, target_var)

        self.log("train_loss", total_loss, on_epoch=True, prog_bar=True, batch_size=128)
        return total_loss

    def validation_step(self, batch):
        x, y = batch
        input_var = (x[0], x[1], x[2], x[3])

        target_var = self.normalizer.norm(y)

        zeros = torch.zeros_like(target_var)
        now_target_var = torch.cat((zeros, target_var), dim=1)

        y_hat = self.forward(*input_var)

        delta = torch.abs(torch.squeeze(target_var) - y_hat[:, 1])
        now_target_var[:,0] = delta

        # loss_fn = nn.L1Loss()  # mae
        # val_loss = loss_fn(y_hat, target_var)

        mae = nn.L1Loss()
        total_loss = mae(now_target_var[:, 1], y_hat[:, 1]) + mae(now_target_var[:, 0], y_hat[:, 0])

        self.log('val_MAE', total_loss, on_epoch=True, prog_bar=True, batch_size=128)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

def load_model(train_inputs,train_outputs,path,seed):
    random.seed(seed)
    combined = list(zip(train_inputs, train_outputs))
    sampled_data = random.sample(combined, 472)
    train_inputs, train_outputs = zip(*sampled_data)

    train_inputs = pd.Series(train_inputs)
    train_outputs = pd.Series(train_outputs)
    # print("数据集大小: ", len(train_inputs), len(train_outputs))

    dataset = StruData(train_inputs, train_outputs)

    # obtain target value normalizer
    if len(dataset) < 500:
        sample_data_list = [dataset[i] for i in range(len(dataset))]
    else:
        sample_data_list = [dataset[i] for i in
                            sample(range(len(dataset)), 500)]
    _, sample_target = collate_pool(sample_data_list)
    normalizer = Normalizer(sample_target)

    # build model
    structures, _, = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]

    model = Cgcnn_lightning(CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                                atom_fea_len=args.atom_fea_len,
                                                n_conv=3,
                                                h_fea_len=128,
                                                n_h=1,
                                                classification=False), normalizer)
    # 加载模型权重
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def predict(test_loader, model):
    model.eval()
    predictions = []

    with torch.no_grad():
        for input, target in test_loader:
            output = model(*input)
            predictions.extend(output.tolist())

    return predictions

def get_mae_mse_and_coverage_rate(model, test_loader, test_y):
    pre = predict(test_loader, model)
    target = np.array([result[1] for result in pre])
    u = np.array([result[0] for result in pre])
    uncertainties = torch.tensor(u)

    test_y_values = test_y.values.astype(float)  # 将 tensor 转换为数值类型
    targets = torch.tensor(target)
    # mae mse
    mae = torch.mean(torch.abs(targets - test_y_values))  # 将 test_y 转换为张量
    mse = torch.mean((targets - test_y_values) ** 2)

    # 计算覆盖率百分比
    test_y_values = torch.tensor(test_y_values)
    lower_bound_2std = targets - 2 * uncertainties
    upper_bound_2std = targets + 2 * uncertainties

    is_inside_2std = (test_y_values >= lower_bound_2std) & (
            test_y_values <= upper_bound_2std)  # 将 lower_bound 和 upper_bound 转换为张量
    coverage_percentage_2std = torch.mean(is_inside_2std.float()) * 100

    lower_bound_1std = targets - uncertainties
    upper_bound_1std = targets + uncertainties
    is_inside_1std = (test_y_values >= lower_bound_1std) & (
            test_y_values <= upper_bound_1std)  # 将 lower_bound 和 upper_bound 转换为张量
    coverage_percentage_1std = torch.mean(is_inside_1std.float()) * 100

    return mae, mse, coverage_percentage_1std, coverage_percentage_2std, np.mean(uncertainties.numpy())


def predict_p_and_u(name):
    test_x, test_y = get_test_dataset()

    test_x = pd.Series(test_x)
    test_y = pd.Series(test_y)
    dataset = StruData(test_x, test_y)
    test_loader = DataLoader(dataset=dataset,
                             batch_size=128,  # 设置批处理大小
                             shuffle=False,  # 关闭数据洗牌
                             num_workers=0,  # 设置数据加载器的工作线程数量
                             collate_fn=collate_pool)

    structures, targets = get_low_fidelity_dataset(name)

    random_seed_list = [42 * i for i in range(1, 11)]

    if name == 'scan' :
        print(f"{name}_42:")
        model= load_model(structures, targets, path=f"./saved_models/{name}_42_p_u.ckpt", seed=42)
        mae, mse, cov1, cov2, u = get_mae_mse_and_coverage_rate(model, test_loader, test_y)
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
            model= load_model(structures, targets, path=f"./saved_models/{name}_{seed}_p_u.ckpt", seed=seed)
            mae, mse, cov1, cov2, u = get_mae_mse_and_coverage_rate(model, test_loader, test_y)
            all_mae.append(round(mae.item(), 3))
            all_mse.append(round(mse.item(), 3))
            cov1_list.append(round(cov1.item(), 3))
            cov2_list.append(round(cov2.item(), 3))
            all_avg_uncertainty.append(round(u,3))
            print(f"MAE:{mae.item():.3f} MSE:{mse.item():.3f}")
            print(f"1*u Coverage Percentage:{cov1.item():.3f}%")
            print(f"2*u Coverage Percentage:{cov2.item():.3f}%")
            print(f"avg 1*u: {u:.3f}")

        print(f"all mae: {all_mae} \naverage mae: {np.mean(all_mae):.3f}")
        print(f"all mse: {all_mse} \naverage mse: {np.mean(all_mse):.3f}")
        print(f"cov1_list: {cov1_list} \naverage cov1: {np.mean(cov1_list):.3f}%")
        print(f"cov2_list: {cov2_list} \naverage cov2: {np.mean(cov2_list):.3f}%")
        print(f"1*u_list: {all_avg_uncertainty} \navg 1*u: {np.mean(all_avg_uncertainty):.3f}")


    return 0


def plot_results(uncertainties, targets, test_y, name):
    targets = targets
    u = uncertainties
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
    axs[1].text(0.07, 0.06, f'{(count_non_positive / len(y)) * 100:.2f}% ≤ 0', ha='center', va='center', transform=axs[1].transAxes,
                 fontsize=20, fontname='Times New Roman', bbox=dict(facecolor='white', alpha=1.0, edgecolor='none'))

    # Adjust tick font sizes
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')

    # xcustom_ticks = [0, 5, 10, 15, 20, 25]  # 自定义刻度值
    # xcustom_ticklabels = ['0.0', '5.0', '10.0', '15.0', '20.0', '25.0']  # 自定义刻度标签
    # axs[0].set_yticks(xcustom_ticks)
    # axs[0].set_yticklabels(xcustom_ticklabels,
    #                        fontname='Times New Roman',
    #                        fontsize=20)

    # if name == "HSE": # ok
    #     axs[0].set_ylim(-1, 25.6)
    #     axs[1].set_ylim(-0.005, 0.0001)
    # elif name == "Exp":
    #     axs[0].set_ylim(-1, 25.5)
    #     axs[1].set_ylim(-0.015, 0.0005)
    # elif name == "PBE":
    #     axs[0].set_ylim(-1, 25.3)
    #     axs[1].set_ylim(-0.005, 0.0001)
    # elif name == "GLLB":
    #     axs[0].set_ylim(-1, 25.1)
    #     axs[1].set_ylim(-0.1, 0.005)
    # else: # SCAN
    #     axs[0].set_ylim(-1, 26.6)
    #     axs[1].set_ylim(-0.033, 0.0008)

    # Custom tick formatting function
    def format_y_tick_upper(x, pos):
        if np.isclose(x, 0.0):
            return '0'
        else:
            return f'{x:.0f}'

    def format_y_tick_lower(x, pos):
        if np.isclose(x, 0.0):
            return '0'
        else:
            if name == 'GLLB':
                return f'{x:.0f}'
            else:
                return f'{x:.1f}'

    # Apply the custom formatters to y-axes
    axs[0].yaxis.set_major_formatter(ticker.FuncFormatter(format_y_tick_upper))
    axs[1].yaxis.set_major_formatter(ticker.FuncFormatter(format_y_tick_lower))

    plt.savefig(f'../saved_fig/CGCNN/CGCNN_{name}_prediction.pdf', bbox_inches='tight')
    plt.show()

# 'scan', 'gllb-sc', 'hse', 'pbe'
datasets_name = ['scan','hse', 'gllb-sc', 'pbe']

for name in datasets_name:
    predict_p_and_u(name)

# scan:
# MAE:1.458 MSE:3.620
# 1*u Coverage Percentage:4.920%
# 2*u Coverage Percentage:23.011%
# avg 1*u: 0.283

# hse
# all mae: [1.6, 1.552, 1.497, 1.262, 1.415, 1.506, 1.608, 1.529, 1.545, 1.51]
# average mae: 1.502
# all mse: [4.161, 3.924, 3.714, 3.109, 3.48, 3.836, 4.187, 3.927, 3.89, 3.784]
# average mse: 3.801
# cov1_list: [6.807, 2.109, 3.33, 10.803, 3.848, 4.735, 4.994, 3.515, 4.218, 2.368]
# average cov1: 4.673%
# cov2_list: [12.246, 3.848, 6.807, 29.893, 8.287, 11.025, 10.877, 7.769, 8.583, 5.327]
# average cov2: 10.466%
# 1*u_list: [0.287, 0.133, 0.169, 0.316, 0.23, 0.27, 0.289, 0.192, 0.218, 0.13]
# avg 1*u: 0.223

# gllb
# all mae: [2.04, 1.95, 1.85, 1.919, 2.044, 1.96, 1.929, 1.849, 1.903, 1.832]
# average mae: 1.928
# all mse: [5.643, 5.329, 5.057, 5.245, 5.747, 5.357, 5.284, 4.871, 5.179, 5.016]
# average mse: 5.273
# cov1_list: [2.22, 1.443, 3.034, 1.073, 1.998, 4.07, 1.48, 1.85, 1.739, 4.329]
# average cov1: 2.324%
# cov2_list: [4.772, 3.922, 6.215, 2.146, 4.329, 7.473, 3.7, 3.404, 3.811, 8.028]
# average cov2: 4.780%
# 1*u_list: [0.216, 0.151, 0.146, 0.113, 0.2, 0.219, 0.101, 0.216, 0.144, 0.196]
# avg 1*u: 0.170

# pbe
# all mae: [1.663, 1.638, 1.607, 1.486, 1.711, 1.663, 1.682, 1.71, 1.592, 1.629]
# average mae: 1.638
# all mse: [4.474, 4.279, 4.352, 3.824, 4.819, 4.585, 4.765, 4.782, 4.141, 4.679]
# average mse: 4.470
# cov1_list: [3.145, 4.661, 3.293, 3.811, 2.294, 5.66, 5.771, 3.663, 2.294, 3.589]
# average cov1: 3.818%
# cov2_list: [6.955, 10.063, 6.77, 8.916, 5.808, 10.581, 12.505, 6.252, 4.698, 11.876]
# average cov2: 8.442%
# 1*u_list: [0.189, 0.304, 0.221, 0.231, 0.234, 0.271, 0.306, 0.214, 0.159, 0.373]
# avg 1*u: 0.250