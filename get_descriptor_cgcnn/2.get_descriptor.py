import argparse
import gzip
import json
import os
import random
import sys
import warnings
from random import sample

import lightning.pytorch as pl
import pandas as pd
import torch
import torch.nn as nn
from pymatgen.core import Structure
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

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
    torch.cuda.set_device(0)

init_seed = 42
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

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

        # y_hat = self.crystalGraphConvNet(*input_var)
        y_hat = self.forward(*input_var)  # 使用 forward 方法进行前向传播

        loss_fn = nn.MSELoss()

        loss = loss_fn(y_hat, target_var)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=128)
        return loss

    def validation_step(self, batch):
        x, y = batch
        input_var = (x[0], x[1], x[2], x[3])
        target_var = self.normalizer.norm(y)

        # y_hat = self.crystalGraphConvNet(*input_var)
        y_hat = self.forward(*input_var)

        loss_fn = nn.L1Loss()  # mae
        val_loss = loss_fn(y_hat, target_var)

        self.log('val_MAE', val_loss, on_epoch=True, prog_bar=True, batch_size=128)
        return val_loss

    def test_step(self, batch):
        x, y = batch
        input_var = (x[0], x[1], x[2], x[3])
        target_var = y

        # y_hat = self.crystalGraphConvNet(*input_var)
        y_hat = self.forward(*input_var)
        # loss
        loss_fn = nn.L1Loss()
        test_loss = loss_fn(self.normalizer.denorm(y_hat), target_var)
        self.log('test_MAE', test_loss, on_epoch=True, prog_bar=True, batch_size=128)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

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
    now_bandgap = bandgap[name]
    now_structures, now_targets = load_dataset(now_bandgap)
    # print(f"{name}数据集大小：{len(now_structures)}")
    return now_structures, now_targets

def get_sampled_low_fidelity_dataset(name, randomseed):
    with gzip.open("../datasets/data_no_structs.json.gz", "rb") as f:
        bandgap = json.loads(f.read())
    now_bandgap = bandgap[name]
    now_structures, now_targets = load_dataset(now_bandgap)
    random.seed(randomseed)
    combined = list(zip(now_structures, now_targets))
    sampled_data = random.sample(combined, 472)
    sampled_structures, sampled_targets = zip(*sampled_data)
    return sampled_structures, sampled_targets

def get_high_fidelity_dataset():
    with open('../datasets/all_data.json', 'r') as fp:
        d = json.load(fp)
    exp_structures = [Structure.from_dict(x['structure']) for x in d['ordered_exp'].values()]
    exp_targets = [torch.tensor(x['band_gap']) for x in d['ordered_exp'].values()]
    return exp_structures, exp_targets


def load_model(train_inputs,train_outputs,path):
    train_inputs = pd.Series(train_inputs)
    train_outputs = pd.Series(train_outputs)

    dataset = StruData(train_inputs, train_outputs)

    if len(dataset) < 500:
        sample_data_list = [dataset[i] for i in range(len(dataset))]
    else:
        sample_data_list = [dataset[i] for i in
                            sample(range(len(dataset)), 500)]
    _, sample_target = collate_pool(sample_data_list)
    normalizer = Normalizer(sample_target)

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


def predict(structures, targets, model):

    test_inputs = pd.Series(structures)
    test_outputs = pd.Series(targets)
    dataset = StruData(test_inputs, test_outputs)
    test_loader = DataLoader(dataset=dataset,
                             batch_size=len(test_inputs),
                             shuffle=False,
                             num_workers=0,
                             collate_fn=collate_pool)

    model.eval()
    predictions = []

    with torch.no_grad():
        for input, target in test_loader:
            output = model(*input)
            predictions.extend(output.tolist())

    # mae_errors.avg,
    return predictions

def get_train_datasets_descriptors(name, seed):
    model.now_descriptor = []
    if name == 'scan':
        structures, targets = get_low_fidelity_dataset(name)
    else:
        structures, targets = get_sampled_low_fidelity_dataset(name, seed)
    cgcnn_model = load_model(structures, targets, path=f"./saved_models/{name}_{seed}.ckpt")
    pre_targets = predict(structures, targets, cgcnn_model)
    descriptors = model.now_descriptor.copy()
    file_path = f"./saved_descriptors/{name}_{seed}/train_data.pt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    train_data_dict = {
        'descriptors': descriptors,
        'targets': targets
    }
    torch.save(train_data_dict, file_path)

    print(f"train_data_dict:{name}_{seed}保存完成！", len(descriptors), len(targets))


def get_test_datasets_descriptors(name, seed):
    model.now_descriptor = []

    test_structures, test_targets = get_high_fidelity_dataset()
    # 加载训练好的模型
    cgcnn_model = load_model(test_structures, test_targets, path=f"./saved_models/{name}_{seed}.ckpt")
    # 通过预测方法得到数据集所对应的描述子
    pre_targets = predict(test_structures, test_targets, cgcnn_model)

    test_descriptors = model.now_descriptor.copy()

    file_path = f"./saved_descriptors/{name}_{seed}/test_data.pt"
    # 创建保存路径的父目录
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    train_data_dict = {
        'descriptors': test_descriptors,
        'targets': test_targets
    }
    torch.save(train_data_dict, file_path)
    print(f"test_data_dict:{name}_{seed}保存完成！", len(test_descriptors), len(test_targets))


# , 'hse', 'gllb-sc', 'pbe'
datasets_name = ['hse', 'gllb-sc', 'pbe']
random_seed_list = [42*i for i in range(1,11)]

for name in datasets_name:
    if name == 'scan':
        get_train_datasets_descriptors('scan', '42')
        get_test_datasets_descriptors('scan', '42')
    else:
        for seed in random_seed_list:
            get_train_datasets_descriptors(name, seed)
            get_test_datasets_descriptors(name, seed)

