
from __future__ import annotations

import gc
import gzip
import json
import os
import random
import shutil
import warnings
import zipfile

import matgl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from dgl.data.utils import split_dataset
from pymatgen.core import Structure
from pytorch_lightning.loggers import CSVLogger
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MEGNetDataset, MGLDataLoader, collate_fn
from matgl.layers import BondExpansion
# from matgl.data.transformer import
from matgl.models import MEGNet
from matgl.utils.io import RemoteFile
# from my_training import ModelLightningModule
from matgl.utils.training import ModelLightningModule

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

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

def training_model(structures,targets,saved_name,now_seed):
    delete_cache()
    targets = torch.tensor(targets)
    elem_list = get_element_list(structures)
    converter = Structure2Graph(element_types=elem_list, cutoff=5.0)
    bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=100, width=0.4)

    print("构建数据集...")
    dataset = MEGNetDataset(
        structures=structures,
        labels={"bandgap": targets},
        converter=converter,
        name="bandgap"
    )
    print("构建完成")

    train_data, val_data, test_data = split_dataset(
        dataset,
        frac_list=[0.8, 0.2, 0.0],
        shuffle=True,
        random_state=42,
    )

    train_loader, val_loader, test_loader = MGLDataLoader(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        collate_fn=collate_fn,
        batch_size=128,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

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

    lit_module = ModelLightningModule(model=model, loss="mse_loss")
    logger = CSVLogger("logs", name=f"train_sampled_{saved_name}_{now_seed}")

    early_stopping_callback = EarlyStopping(
        monitor='val_MAE',  # 选择监控的指标，例如验证集损失
        min_delta=0.0,  # 定义监控指标的变化阈值
        patience=500,  # 在没有改善的情况下等待停止的epoch数
        verbose=True,  # Print early stopping messages
        mode='min'  # 监控指标的模式，'min'表示越小越好，'max'表示越大越好
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_MAE',
        save_top_k=1,
        mode='min',
        dirpath=f"checkpoints/{saved_name}_{now_seed}",  # 临时保存最佳模型的路径
        filename='{epoch:02d}'
    )

    print("开始训练...")
    trainer = pl.Trainer(max_epochs=10000, logger=logger, callbacks=[early_stopping_callback, checkpoint_callback])
    trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("训练结束")

    # 获取最佳模型
    best_model_path = checkpoint_callback.best_model_path  # 替换为实际的检查点路径
    best_model = ModelLightningModule.load_from_checkpoint(checkpoint_path=best_model_path)
    best_model = best_model.model # 获取其中具体的模型，类型是MEGNet
    # 保存模型到指定路径
    print(f"开始保存 {saved_name} 模型...")
    save_path = f"./saved_models/{saved_name}_{now_seed}"
    metadata = {"description": f"train sampled {saved_name} datasets", "training_set": f"{saved_name}", "seed": f"{now_seed}"}
    best_model.save(save_path, metadata=metadata)
    print("保存完成")
    delete_cache()
    return model

def delete_cache():
    for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
        try:
            os.remove(fn)
        except FileNotFoundError:
            pass
    print("删除缓存完成")

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
    print(f"{name}数据集大小：{len(now_structures)}")
    random.seed(randomseed)
    combined = list(zip(now_structures, now_targets))
    sampled_data = random.sample(combined, 472)
    sampled_structures, sampled_targets = zip(*sampled_data)
    print(f"{name}数据集随机采样后大小：{len(sampled_structures)} 随机种子：{randomseed}")
    print(sampled_targets[:10])
    return sampled_structures, sampled_targets


def get_low_fidelity_dataset(name):
    with gzip.open("../datasets/data_no_structs.json.gz", "rb") as f:
        bandgap = json.loads(f.read())
    now_bandgap = bandgap[name]
    now_structures, now_targets = load_dataset(now_bandgap)
    print(f"{name}数据集大小：{len(now_structures)}")

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

    print(f"Exp训练集大小：{len(exp_structures)}, 测试集大：{len(exp_test_data)}")
    return exp_structures, exp_targets, exp_test_data

# PBE、HSE、GLLB、SCAN、(E/2, shuffle, random_state=42)
# 获取数据集
# pbe_structures, pbe_targets = get_low_fidelity_dataset('pbe')
# hse_structures, hse_targets = get_low_fidelity_dataset('hse')
# gllb_structures, gllb_targets = get_low_fidelity_dataset('gllb-sc')
# scan_structures, scan_targets = get_low_fidelity_dataset('scan')
# exp_structures, exp_targets, exp_test_data = get_high_fidelity_dataset()

def get_10_data():
    with open('../datasets/10_data_structures.json', 'r') as file:
        all_structures = json.load(file)
    structures = [Structure.from_str(structure, fmt="cif") for structure in all_structures]
    with open('../datasets/10_data_targets.json', 'r') as file:
        all_targets = json.load(file)
    return structures, all_targets


# , 'hse', 'gllb-sc', 'pbe'
# , 'hse', 'gllb-sc', 'pbe'
# 'scan',
all_datasets_name = ['pbe']
random_seed_list = [42*i for i in range(1,11)]

for name in all_datasets_name:
    print(f"START===================================train {name} for p&u=======================================")
    if name == 'scan':
        structures, targets = get_low_fidelity_dataset(name)
        model = training_model(structures, targets, name, 42)
    else:
        for sample_seed in random_seed_list:
            structures, targets = get_sampled_low_fidelity_dataset(name, sample_seed)
            model = training_model(structures, targets, name, sample_seed)
    print(f"END===================================train {name} for p&u=======================================")





