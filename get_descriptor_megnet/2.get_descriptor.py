from __future__ import annotations

import gzip
import json
import os
import random
import warnings

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

def get_train_datasets_descriptors(name, seed):
    megnet_model = load_my_model(name, seed)

    megnet.now_descriptor = []

    if name == 'scan':
        train_structures, train_targets = get_low_fidelity_dataset(name)
    else:
        train_structures, train_targets = get_sampled_low_fidelity_dataset(name, seed)

    # 通过预测方法得到数据集所对应的描述子
    for s in train_structures:
        train_target = megnet_model.predict_structure(s)

    train_descriptors = megnet.now_descriptor.copy()

    file_path = f"./saved_descriptors/{name}_{seed}/train_data.pt"
    # 创建保存路径的父目录
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    train_data_dict = {
        'descriptors': train_descriptors,
        'targets': train_targets
    }
    torch.save(train_data_dict, file_path)

    # 读取
    # loaded_data = torch.load(file_path)
    # descriptors1 = loaded_data['descriptors']
    # targets1 = loaded_data['targets']

    print(f"train_data_dict:{name}_{seed}保存完成！", len(train_descriptors), len(train_targets))




def get_test_datasets_descriptors(name, seed):
    # 加载训练好的模型
    megnet_model = load_my_model(name, seed)

    megnet.now_descriptor = []

    test_structures, test_targets = get_high_fidelity_dataset()

    for s in test_structures:
        target = megnet_model.predict_structure(s)

    test_descriptors = megnet.now_descriptor.copy()

    file_path = f"./saved_descriptors/{name}_{seed}/test_data.pt"
    # 创建保存路径的父目录
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    train_data_dict = {
        'descriptors': test_descriptors,
        'targets': test_targets
    }
    torch.save(train_data_dict, file_path)

    print(f"test_data_dict:{name}_{seed}保存完成！", len(test_descriptors), len(test_targets))



datasets_name = ['hse','gllb-sc','pbe']
random_seed_list = [42*i for i in range(1,11)]

for name in datasets_name:
    if name == 'scan':
        get_train_datasets_descriptors('scan', '42')
        get_test_datasets_descriptors('scan', '42')
    else:
        for seed in random_seed_list:
            get_train_datasets_descriptors(name, seed)
            get_test_datasets_descriptors(name, seed)

