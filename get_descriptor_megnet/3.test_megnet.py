from __future__ import annotations

import gzip
import json
import os
import random
import warnings

import numpy as np
import torch
from matgl.layers import BondExpansion
from pymatgen.core import Structure

import my_megnet as megnet
from my_megnet import MEGNet

warnings.simplefilter("ignore")

if torch.cuda.is_available():
    torch.cuda.set_device(0)

init_seed = 42
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

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

def get_sampled_low_fidelity_dataset(name, randomseed):
    with gzip.open("../datasets/data_no_structs.json.gz", "rb") as f:
        bandgap = json.loads(f.read())
    now_bandgap = bandgap[name]
    now_structures, now_targets = load_dataset(now_bandgap)
    # print(f"{name}数据集大小：{len(now_structures)}")
    random.seed(randomseed)
    combined = list(zip(now_structures, now_targets))
    sampled_data = random.sample(combined, 472)
    sampled_structures, sampled_targets = zip(*sampled_data)
    # print(f"{name}数据集随机采样后大小：{len(sampled_structures)} 随机种子：{randomseed}")
    return sampled_structures, sampled_targets


def get_low_fidelity_dataset(name):
    with gzip.open("../datasets/data_no_structs.json.gz", "rb") as f:
        bandgap = json.loads(f.read())
    now_bandgap = bandgap[name]
    now_structures, now_targets = load_dataset(now_bandgap)
    # print(f"{name}数据集大小：{len(now_structures)}")

    return now_structures, now_targets

def get_high_fidelity_dataset():
    with open('../datasets/all_data.json', 'r') as fp:
        d = json.load(fp)
    # 高保真度数据集
    exp_structures = [Structure.from_dict(x['structure']) for x in d['ordered_exp'].values()]
    exp_targets = [torch.tensor(x['band_gap']) for x in d['ordered_exp'].values()]

    # print(f"Exp大小：{len(exp_structures)}")
    return exp_structures, exp_targets

def load_my_model(name, seed):
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

    megnet_model = model.load(path=f"./saved_models/{name}_{seed}")

    return megnet_model


def get_test_results(name, seed):
    # 加载训练好的模型
    megnet_model = load_my_model(name, seed)

    megnet.now_descriptor = []

    test_structures, test_targets = get_high_fidelity_dataset()
    pre_targets = []
    for s in test_structures:
        pre_target = megnet_model.predict_structure(s)
        pre_targets.append(pre_target)

    pre_targets_np = np.array(pre_targets)
    test_targets_np = np.array(test_targets)

    mae = np.mean(np.abs(pre_targets_np - test_targets_np))
    mse = np.mean((pre_targets_np - test_targets_np) ** 2)

    print(f"{name}_{seed}  MAE: {mae}, MSE: {mse}")

    return mae, mse

datasets_name = ['scan', 'hse', 'gllb-sc', 'pbe']
random_seed_list = [42*i for i in range(1,11)]
all_result = []

for name in datasets_name:
    if name == 'scan':
        mae, mse = get_test_results('scan', 42)
        result_dict = {
            'name': name,
            'all_mae': [mae],
            'all_mse': [mse]
        }

        all_result.append(result_dict)
    else:
        all_mae = []
        all_mse = []
        for seed in random_seed_list:
            mae, mse = get_test_results(name, seed)
            all_mae.append(mae)
            all_mse.append(mse)

        result_dict = {
            'name': name,
            'all_mae': all_mae,
            'all_mse': all_mse
        }

        all_result.append(result_dict)


print("===================================================")
for dict in all_result:
    print(dict['name'])
    print('all_mae:', [round(mae,3) for mae in dict['all_mae']])
    print('all_mse:', [round(mse,3) for mse in dict['all_mse']])
    print("avg_mae:", round(np.mean(dict['all_mae']),3))
    print("avg_mse:", round(np.mean(dict['all_mse']),3))

# scan_42  MAE: 1.4048759937286377, MSE: 3.644975185394287
# hse_42  MAE: 1.1046119928359985, MSE: 2.8464653491973877
# hse_84  MAE: 1.4556797742843628, MSE: 4.972673416137695
# hse_126  MAE: 1.2363667488098145, MSE: 3.384244680404663
# hse_168  MAE: 1.125429391860962, MSE: 2.5992467403411865
# hse_210  MAE: 1.2868062257766724, MSE: 3.5944035053253174
# hse_252  MAE: 1.291329264640808, MSE: 3.6771693229675293
# hse_294  MAE: 1.3198446035385132, MSE: 4.05251407623291
# hse_336  MAE: 1.2828577756881714, MSE: 3.7606749534606934
# hse_378  MAE: 1.3412216901779175, MSE: 3.989583969116211
# hse_420  MAE: 1.2850427627563477, MSE: 3.476064920425415
# gllb-sc_42  MAE: 1.9483529329299927, MSE: 5.824422836303711
# gllb-sc_84  MAE: 2.134463310241699, MSE: 7.486562728881836
# gllb-sc_126  MAE: 2.5041563510894775, MSE: 9.138734817504883
# gllb-sc_168  MAE: 2.0155134201049805, MSE: 5.9739766120910645
# gllb-sc_210  MAE: 2.269641160964966, MSE: 8.407370567321777
# gllb-sc_252  MAE: 2.6253206729888916, MSE: 9.585771560668945
# gllb-sc_294  MAE: 2.2420361042022705, MSE: 7.929876327514648
# gllb-sc_336  MAE: 1.8619531393051147, MSE: 5.916292190551758
# gllb-sc_378  MAE: 2.1434619426727295, MSE: 7.940128803253174
# gllb-sc_420  MAE: 2.022753953933716, MSE: 6.649469375610352
# pbe_42  MAE: 1.201643705368042, MSE: 3.1392698287963867
# pbe_84  MAE: 1.0685514211654663, MSE: 2.618723154067993
# pbe_126  MAE: 1.1586710214614868, MSE: 3.0697615146636963
# pbe_168  MAE: 1.2446928024291992, MSE: 3.551240921020508
# pbe_210  MAE: 1.104888677597046, MSE: 2.6630096435546875
# pbe_252  MAE: 1.2332099676132202, MSE: 3.2369155883789062
# pbe_294  MAE: 1.1481133699417114, MSE: 3.2490859031677246
# pbe_336  MAE: 1.2169395685195923, MSE: 3.2732677459716797
# pbe_378  MAE: 1.0926661491394043, MSE: 2.8724355697631836
# pbe_420  MAE: 1.0743827819824219, MSE: 2.7052290439605713
# ===================================================
# scan
# all_mae: [1.405]
# all_mse: [3.645]
# avg_mae: 1.405
# avg_mse: 3.645
# hse
# all_mae: [1.105, 1.456, 1.236, 1.125, 1.287, 1.291, 1.32, 1.283, 1.341, 1.285]
# all_mse: [2.846, 4.973, 3.384, 2.599, 3.594, 3.677, 4.053, 3.761, 3.99, 3.476]
# avg_mae: 1.273
# avg_mse: 3.635
# gllb-sc
# all_mae: [1.948, 2.134, 2.504, 2.016, 2.27, 2.625, 2.242, 1.862, 2.143, 2.023]
# all_mse: [5.824, 7.487, 9.139, 5.974, 8.407, 9.586, 7.93, 5.916, 7.94, 6.649]
# avg_mae: 2.177
# avg_mse: 7.485
# pbe
# all_mae: [1.202, 1.069, 1.159, 1.245, 1.105, 1.233, 1.148, 1.217, 1.093, 1.074]
# all_mse: [3.139, 2.619, 3.07, 3.551, 2.663, 3.237, 3.249, 3.273, 2.872, 2.705]
# avg_mae: 1.154
# avg_mse: 3.038
