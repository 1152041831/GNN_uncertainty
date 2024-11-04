import argparse
import gzip
import json
import os
import random
import sys
import warnings
from random import sample

import lightning.pytorch as pl
import numpy as np
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

def get_test_results(name, seed):
    model.now_descriptor = []

    test_structures, test_targets = get_high_fidelity_dataset()
    # 加载训练好的模型
    cgcnn_model = load_model(test_structures, test_targets, path=f"./saved_models/{name}_{seed}.ckpt")

    pre_targets = predict(test_structures, test_targets, cgcnn_model)
    pre_targets = [target[0] for target in pre_targets]

    pre_targets_np = np.array(pre_targets)
    test_targets_np = np.array(test_targets)

    mae = np.mean(np.abs(pre_targets_np - test_targets_np))
    mse = np.mean((pre_targets_np - test_targets_np) ** 2)

    # mae = round(mae, 3)
    # mse = round(mse, 3)

    print(f"{name}_{seed}  MAE: {mae}, MSE: {mse}")

    return mae, mse




# , 'hse', 'gllb-sc', 'pbe'
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


# scan_42  MAE: 1.7050615014881172, MSE: 4.264065030989129
# hse_42  MAE: 1.4636225262260574, MSE: 3.7698122131840677
# hse_84  MAE: 1.509241040398429, MSE: 3.8531520921845366
# hse_126  MAE: 1.5187717042229394, MSE: 3.7725888907890552
# hse_168  MAE: 1.3579344848653938, MSE: 3.4882456665003576
# hse_210  MAE: 1.4180446467271037, MSE: 3.608863003962397
# hse_252  MAE: 1.4822815212014893, MSE: 3.824668695517472
# hse_294  MAE: 1.5519500971708724, MSE: 3.962033966031478
# hse_336  MAE: 1.534213701697894, MSE: 3.8595966759309412
# hse_378  MAE: 1.5170079799816205, MSE: 3.9982299533556986
# hse_420  MAE: 1.5364365992138023, MSE: 3.9991433255673905
# gllb-sc_42  MAE: 1.9845789148199364, MSE: 5.413416976731237
# gllb-sc_84  MAE: 2.027064793094403, MSE: 5.6625320470882325
# gllb-sc_126  MAE: 1.9536379687377823, MSE: 5.309088711078983
# gllb-sc_168  MAE: 1.990242599108254, MSE: 5.61408332603326
# gllb-sc_210  MAE: 2.028236332586789, MSE: 5.69187683116603
# gllb-sc_252  MAE: 2.0257457967611914, MSE: 5.540474210844864
# gllb-sc_294  MAE: 1.9248274751042012, MSE: 5.173330964658907
# gllb-sc_336  MAE: 2.1430362460875, MSE: 6.098624703147501
# gllb-sc_378  MAE: 1.887790133347621, MSE: 5.096555425506184
# gllb-sc_420  MAE: 1.9748345831980363, MSE: 5.577683289259435
# pbe_42  MAE: 1.6555360524969942, MSE: 4.329097887033196
# pbe_84  MAE: 1.7850095082895814, MSE: 4.964266049566598
# pbe_126  MAE: 1.570892542648869, MSE: 4.16198727031053
# pbe_168  MAE: 1.5355795275101585, MSE: 3.9918899244278023
# pbe_210  MAE: 1.7334675473366739, MSE: 4.948298772202712
# pbe_252  MAE: 1.6278440361271247, MSE: 4.27312292558283
# pbe_294  MAE: 1.6194592794590181, MSE: 4.446474017189774
# pbe_336  MAE: 1.6234880146597035, MSE: 4.286306286898044
# pbe_378  MAE: 1.634809367756986, MSE: 4.464523022658803
# pbe_420  MAE: 1.6038339294664097, MSE: 4.178526772632248
# ===================================================
# scan
# all_mae: [1.705]
# all_mse: [4.264]
# avg_mae: 1.705
# avg_mse: 4.264
# hse
# all_mae: [1.464, 1.509, 1.519, 1.358, 1.418, 1.482, 1.552, 1.534, 1.517, 1.536]
# all_mse: [3.77, 3.853, 3.773, 3.488, 3.609, 3.825, 3.962, 3.86, 3.998, 3.999]
# avg_mae: 1.489
# avg_mse: 3.814
# gllb-sc
# all_mae: [1.985, 2.027, 1.954, 1.99, 2.028, 2.026, 1.925, 2.143, 1.888, 1.975]
# all_mse: [5.413, 5.663, 5.309, 5.614, 5.692, 5.54, 5.173, 6.099, 5.097, 5.578]
# avg_mae: 1.994
# avg_mse: 5.518
# pbe
# all_mae: [1.656, 1.785, 1.571, 1.536, 1.733, 1.628, 1.619, 1.623, 1.635, 1.604]
# all_mse: [4.329, 4.964, 4.162, 3.992, 4.948, 4.273, 4.446, 4.286, 4.465, 4.179]
# avg_mae: 1.639
# avg_mse: 4.404