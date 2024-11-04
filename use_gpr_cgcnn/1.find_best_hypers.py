import math
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import gpytorch
from tqdm import tqdm

from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ExactGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

warnings.simplefilter("ignore")

init_seed = 42
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if torch.cuda.is_available():
    torch.cuda.set_device(0)

def get_train_dataset(name, seed):
    train_data = torch.load(f'../get_descriptor_cgcnn/saved_descriptors/{name}_{seed}/train_data.pt')

    X = np.array([e.detach().numpy() for e in train_data['descriptors']])
    Y = np.array([e.detach().numpy() for e in train_data['targets']])

    # 组合
    combined = list(zip(X, Y))
    # print("Length of dataset before cleaning: ", len(combined))

    # 遍历每一条数据，检查是否有 NaN 值
    nan_indices = []
    for i, data in enumerate(X):
        if np.isnan(data).any():
            nan_indices.append(i)

    # 打印包含 NaN 值的数据索引
    # print(f"Length of samples with NaN values: {len(nan_indices)}")

    combined_cleaned = [data for i, data in enumerate(combined) if i not in nan_indices]
    # print("Length of dataset after cleaning: ", len(combined_cleaned))

    # 拆分清洗后的数据
    X_cleaned, Y_cleaned = zip(*combined_cleaned)

    # 将清洗后的数据转换为 NumPy 数组
    X_cleaned = np.array(X_cleaned)
    Y_cleaned = np.array(Y_cleaned)

    # 划分训练集和验证集
    train_x, val_x, train_y, val_y = train_test_split(X_cleaned, Y_cleaned, test_size=0.2, random_state=42)

    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    val_x = torch.tensor(val_x, dtype=torch.float32)
    val_y = torch.tensor(val_y, dtype=torch.float32)

    # print(f"{name}训练集输入的形状:{train_x.shape} 标签train_y形状:{train_y.shape}")
    # print(f"{name}验证集输入的形状:{val_x.shape} 标签val_y形状:{val_y.shape}")

    return train_x,train_y,val_x,val_y

# 定义高斯过程回归模型
class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# 训练超参数
def train_with_hypers(name, seed, hypers, early_stopping_patience):
    # print("当前超参数: ", hypers)
    train_x, train_y, val_x, val_y = get_train_dataset(name, seed)

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_dataset = TensorDataset(val_x, val_y)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 10% datasets
    inducing_points = train_x[:135, :]

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # print("原本的noise: ", likelihood.noise_covar.noise)
    likelihood.noise_covar.noise = hypers['likelihood.noise_covar.noise']

    # print("修改后的noise: ", likelihood.noise_covar.noise)

    # model = ExactGPModel(train_x_fold, train_y_fold, likelihood)
    model = GPModel(inducing_points=inducing_points)

    # print("原本的length: ",model.covar_module.base_kernel.lengthscale)
    model.covar_module.base_kernel.lengthscale = hypers['covar_module.base_kernel.lengthscale']
    # print("修改后的length: ",model.covar_module.base_kernel.lengthscale)

    model.train()
    likelihood.train()

    max_epochs = 10000
    no_improve_count = 0
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=hypers['lr'])

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    score = float('inf')  # mae
    patience = early_stopping_patience

    for epoch in range(max_epochs):
        model.train()

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()

        # 在验证集上评估模型性能
        model.eval()
        val_predictions = []
        val_true = []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                output_val = model(x_val)
                mean_val = output_val.mean
                val_predictions.extend(mean_val.numpy())
                val_true.extend(y_val.numpy())

        # 计算验证集上的MAE
        val_mae = mean_absolute_error(val_true, val_predictions)

        # 保存最佳模型
        if val_mae < score:
            score = val_mae
            no_improve_count = 0
        else:
            no_improve_count += 1

        # 检查是否需要早停
        if no_improve_count >= patience:
            print(f'Early stopping at epoch {epoch + 1}, best validation MAE: {score}')
            break

    return score


def find_best_hypers(name, seed):
    best_score = float('inf')
    best_lengthscale = None
    best_noise = None
    best_lr = None

    # 使用 LOO-CV 寻找最优超参数
    for lr in lrs:
        for lengthscale in lengthscale_range:
            for noise in noise_range:
                hypers = {
                    'likelihood.noise_covar.noise': torch.tensor(noise),
                    'covar_module.base_kernel.lengthscale': torch.tensor(lengthscale),
                    'lr': lr
                }
                try:
                    # 对于每一种超参数组合，都会有一个score，score: MAE
                    score = train_with_hypers(name, seed, hypers, early_stopping_patience=50)
                except Exception as e:
                    print(f"出现异常，跳过=> lr: {lr}, lengthscale: {lengthscale}, noise: {noise}")
                    continue  # 继续下一个超参数组合

                if score < best_score:
                    best_score = score
                    best_lengthscale = lengthscale
                    best_noise = noise
                    best_lr = lr

    # print(f"Best Lr: {best_lr}, Best Lengthscale: {best_lengthscale}, Best Noise: {best_noise}, Best MAE: {best_score}")

    file_path = f"./saved_best_hypers/{name}_{seed}.pt"
    # 创建保存路径的父目录
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    best_hypers_dict = {
        'lr': best_lr,
        'lengthscale': best_lengthscale,
        'noise': best_noise,
        'mae': best_score
    }
    torch.save(best_hypers_dict, file_path)
    print(f"best_hypers_dict:{name}_{seed}保存完成！")

# 准备数据
torch.manual_seed(42)

# 设置超参数范围 [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 20.0]
lengthscale_range = [math.pow(10,i) for i in range(-4,2)]
lengthscale_range.extend([20.0])
noise_range = [math.pow(10,i) for i in range(-4,2)]
noise_range.extend([20.0])
# [0.1, 0.01, 0.001]
lrs = [10 ** (-i) for i in range(1, 4)]
# 'scan', 'hse', 'gllb-sc'
datasets_name = ['pbe']
random_seed_list = [42*i for i in range(1,11)]

for name in datasets_name:
    if name =='scan':
        find_best_hypers(name, 42)
    else:
        for seed in random_seed_list:
            find_best_hypers(name, seed)



# scan_42 lengthscale: 1.0  lr: 0.1  noise: 0.01  mae: 0.25699237
# hse_42 lengthscale: 1.0  lr: 0.1  noise: 0.001  mae: 0.29995313
# hse_84 lengthscale: 1.0  lr: 0.01  noise: 0.001  mae: 0.3700166
# hse_126 lengthscale: 10.0  lr: 0.01  noise: 10.0  mae: 0.41608936
# hse_168 lengthscale: 20.0  lr: 0.1  noise: 1.0  mae: 0.37816298
# hse_210 lengthscale: 10.0  lr: 0.01  noise: 0.1  mae: 0.41307116
# hse_252 lengthscale: 1.0  lr: 0.01  noise: 1.0  mae: 0.60420424
# hse_294 lengthscale: 1.0  lr: 0.1  noise: 0.1  mae: 0.41206416
# hse_336 lengthscale: 0.1  lr: 0.001  noise: 0.1  mae: 0.43831125
# hse_378 lengthscale: 1.0  lr: 0.1  noise: 0.0001  mae: 0.41188198
# hse_420 lengthscale: 20.0  lr: 0.01  noise: 0.0001  mae: 0.5103061
# gllb-sc_42 lengthscale: 20.0  lr: 0.1  noise: 10.0  mae: 0.37897375
# gllb-sc_84 lengthscale: 1.0  lr: 0.1  noise: 0.1  mae: 0.42806682
# gllb-sc_126 lengthscale: 10.0  lr: 0.01  noise: 0.0001  mae: 0.41805825
# gllb-sc_168 lengthscale: 20.0  lr: 0.1  noise: 1.0  mae: 0.35727274
# gllb-sc_210 lengthscale: 20.0  lr: 0.01  noise: 0.01  mae: 0.36339614
# gllb-sc_252 lengthscale: 10.0  lr: 0.1  noise: 0.0001  mae: 0.4177252
# gllb-sc_294 lengthscale: 10.0  lr: 0.01  noise: 0.001  mae: 0.4532394
# gllb-sc_336 lengthscale: 20.0  lr: 0.01  noise: 10.0  mae: 0.34869266
# gllb-sc_378 lengthscale: 1.0  lr: 0.01  noise: 20.0  mae: 0.3177524
# gllb-sc_420 lengthscale: 20.0  lr: 0.1  noise: 0.001  mae: 0.5223807
# pbe_42 lengthscale: 0.1  lr: 0.01  noise: 0.1  mae: 0.41553894
# pbe_84 lengthscale: 20.0  lr: 0.1  noise: 0.0001  mae: 0.66000426
# pbe_126 lengthscale: 20.0  lr: 0.1  noise: 1.0  mae: 0.4381546
# pbe_168 lengthscale: 1.0  lr: 0.1  noise: 0.1  mae: 0.3895405
# pbe_210 lengthscale: 1.0  lr: 0.1  noise: 0.1  mae: 0.5221208
# pbe_252 lengthscale: 10.0  lr: 0.01  noise: 0.0001  mae: 0.3163635
# pbe_294 lengthscale: 0.1  lr: 0.001  noise: 0.001  mae: 0.48925954
# pbe_336 lengthscale: 20.0  lr: 0.1  noise: 0.0001  mae: 0.30638704
# pbe_378 lengthscale: 1.0  lr: 0.01  noise: 0.001  mae: 0.48995963
# pbe_420 lengthscale: 1.0  lr: 0.1  noise: 0.001  mae: 0.71996605





