import argparse
import gzip
import json
import os
import random
import shutil
import sys
import warnings

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from pymatgen.core import Structure
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from random import sample, seed
from model import CrystalGraphConvNet
from data import StruData, get_train_loader, collate_pool

# 处理anaconda和torch重复文件
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


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))

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

        # 原本维度为1的真值
        target_var = self.normalizer.norm(y)

        zeros = torch.zeros_like(target_var)
        # 扩展维度，包含不确定性u和预测值的真值
        now_target_var = torch.cat((zeros, target_var), dim=1)

        # 预测值
        y_hat = self.forward(*input_var)

        delta = torch.abs(torch.squeeze(target_var) - y_hat[:, 1])
        now_target_var[:,0] = delta

        # loss_fn = nn.MSELoss()
        diff = torch.abs(now_target_var[:, 1] - y_hat[:, 1])  # |Ptrue - P|
        mse_loss = nn.MSELoss()
        u_loss = mse_loss(diff, y_hat[:, 0])  # (|Ptrue - P| - u)²
        print("u_loss:",u_loss)
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

warnings.simplefilter("ignore")

if torch.cuda.is_available():
    torch.cuda.set_device(0)

init_seed = 42
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
np.random.seed(init_seed)  # 用于numpy的随机数
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
seed(init_seed)

def load_dataset(dict_targets):
    # with open("/data0/MF_megnet/my_BO/datasets/mp.2019.04.01.json") as f:
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
    now_bandgap = bandgap[name]
    now_structures, now_targets = load_dataset(now_bandgap)
    print(f"{name}数据集大小：{len(now_structures)}")
    return now_structures, now_targets

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

def get_10_data():
    with open('../datasets/10_data_structures.json', 'r') as file:
        all_structures = json.load(file)
    structures = [Structure.from_str(structure, fmt="cif") for structure in all_structures]
    with open('../../datasets/10_data_targets.json', 'r') as file:
        all_targets = json.load(file)
    return structures, all_targets

def get_test_dataset():
    with open('../datasets/all_data.json', 'r') as fp:
        d = json.load(fp)
    # 高保真度数据集
    Exp_structures = [Structure.from_dict(x['structure']) for x in d['ordered_exp'].values()]
    Exp_targets = [torch.tensor(x['band_gap']) for x in d['ordered_exp'].values()]
    combined = list(zip(Exp_structures, Exp_targets))
    exp_train_data, exp_test_data = train_test_split(combined, test_size=0.5, random_state=init_seed)

    exp_test_structures = [e[0] for e in exp_test_data]
    exp_test_targets = [e[1] for e in exp_test_data]
    print(f"测试集大小：{len(exp_test_data)}")

    return exp_test_structures, exp_test_targets

def training_model(train_inputs,train_outputs,saved_name, now_seed):

    train_inputs = pd.Series(train_inputs)
    train_outputs = pd.Series(train_outputs)
    print("数据集大小: ", len(train_inputs), len(train_outputs))

    dataset = StruData(train_inputs, train_outputs)

    # collate_fn = collate_fn()
    # 训练 验证资料
    train_loader, val_loader = get_train_loader(dataset=dataset,
                                                collate_fn=collate_pool,
                                                batch_size=128,
                                                train_ratio=0.8,
                                                val_ratio=0.2
                                                )

    # obtain target value normalizer
    if len(dataset) < 500:
        # warnings.warn('Dataset has less than 500 data points. '
        #               'Lower accuracy is expected. ')
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

    early_stop_callback = EarlyStopping(monitor="val_MAE", min_delta=0.00, patience=500, verbose=True,
                                        mode="min")

    logger = CSVLogger("./logs", name=f"{saved_name}_{now_seed}")

    # 定义ModelCheckpoint回调，用于保存模型参数
    checkpoint_callback = ModelCheckpoint(
        monitor='val_MAE',
        dirpath=f"./saved_models",  # 临时保存最佳模型的路径
        filename=f'{saved_name}_{now_seed}_p_u',
        save_top_k=1,
        mode='min'
    )

    print("开始训练...")
    trainer = pl.Trainer(max_epochs=100000, callbacks=[early_stop_callback, checkpoint_callback], logger=logger)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("训练结束")

    return model, normalizer

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def predict(val_loader, model, normalizer):
    model.eval()
    mae_errors = AverageMeter()
    predictions = []

    with torch.no_grad():
        for input, target in val_loader:
            # Forward pass
            output = model(*input)
            print("output的形状：", output.shape)
            print("target.shape:", target.shape)
            now_targets = torch.squeeze(target, dim=1)
            print("now_targets.shape:", now_targets.shape)
            # Compute MAE
            mae_error = mae(normalizer.denorm(output[:,1].cpu()), now_targets.cpu())
            mae_errors.update(mae_error.item(), target.size(0))
            # Save predictions
            predictions.extend(output.tolist())

    return mae_errors.avg, predictions


all_datasets_name = ['scan', 'hse', 'gllb-sc', 'pbe']
random_seed_list = [42*i for i in range(1,11)]

for name in all_datasets_name:
    print(f"START===================================train {name} for p&u=======================================")
    if name == 'scan':
        structures, targets = get_low_fidelity_dataset(name)
        model = training_model(structures, targets, name, 42)
    else:
        for sample_seed in random_seed_list:
            structures, targets = get_sampled_low_fidelity_dataset(name, sample_seed)
                # structures, targets= get_10_data()
            model = training_model(structures,targets, name, sample_seed)
    print(f"END===================================train {name} for p&u=======================================")
