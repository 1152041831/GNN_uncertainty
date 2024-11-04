from __future__ import annotations

import gpytorch
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from sklearn.model_selection import train_test_split

init_seed = 42
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if torch.cuda.is_available():
    torch.cuda.set_device(0)

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

def get_train_val_dataset(name, seed):
    train_data = torch.load(f'../get_descriptor_megnet/saved_descriptors/{name}_{seed}/train_data.pt')

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

def get_test_dataset(name, seed):
    train_data = torch.load(f'../get_descriptor_megnet/saved_descriptors/{name}_{seed}/test_data.pt')

    X = np.array([e.detach().numpy() for e in train_data['descriptors']])
    Y = np.array([e.detach().numpy() for e in train_data['targets']])

    combined = list(zip(X, Y))

    nan_indices = []
    for i, data in enumerate(X):
        if np.isnan(data).any():
            nan_indices.append(i)

    combined_cleaned = [data for i, data in enumerate(combined) if i not in nan_indices]

    X_cleaned, Y_cleaned = zip(*combined_cleaned)

    # 将清洗后的数据转换为 NumPy 数组
    X_cleaned = np.array(X_cleaned)
    Y_cleaned = np.array(Y_cleaned)

    test_x = torch.tensor(X_cleaned, dtype=torch.float32)
    test_y = torch.tensor(Y_cleaned, dtype=torch.float32)

    return test_x, test_y

def predict_p_and_u(name, seed):
    print(f"{name}_{seed}:")

    train_x, train_y, _, _ = get_train_val_dataset(name, seed)

    state_dict = torch.load(f'./saved_gpr_models/{name}_{seed}.pth')
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    test_x, test_y = get_test_dataset(name, seed)

    inducing_points = train_x[:135, :]
    model = GPModel(inducing_points=inducing_points)

    model.load_state_dict(state_dict)

    model.eval()
    likelihood.eval()

    # test_x = test_x[:100]
    # test_y = test_y[:100]

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))

        # 计算 MAE
        mean = observed_pred.mean.numpy()
        mae = torch.mean(torch.abs(torch.tensor(mean) - test_y))  # 将 test_y 转换为张量
        mse = torch.mean((torch.tensor(mean) - test_y) ** 2)

        print(f"MAE:{mae.item():.3f} MSE:{mse.item():.3f}")

        # 计算覆盖率百分比
        std = observed_pred.stddev.numpy()
        lower_bound_2std = mean - 2 * std
        upper_bound_2std = mean + 2 * std

        lower_bound_2std = torch.tensor(lower_bound_2std)
        upper_bound_2std = torch.tensor(upper_bound_2std)

        count_2std = 0
        for i,e in enumerate(test_y):
            if e <= upper_bound_2std[i] and e >= lower_bound_2std[i]:
                count_2std += 1

        is_inside_2std = (test_y >= lower_bound_2std) & (
                    test_y <= upper_bound_2std)  # 将 lower_bound 和 upper_bound 转换为张量
        coverage_percentage_2std = torch.mean(is_inside_2std.float()) * 100
        print(f"2std Coverage Percentage:{coverage_percentage_2std.item():.3f}%")
        lower_bound_1std = mean - std
        upper_bound_1std = mean + std
        is_inside_1std = (test_y >= torch.tensor(lower_bound_1std)) & (
                test_y <= torch.tensor(upper_bound_1std))  # 将 lower_bound 和 upper_bound 转换为张量
        coverage_percentage_1std = torch.mean(is_inside_1std.float()) * 100
        print(f"1std Coverage Percentage:{coverage_percentage_1std.item():.3f}%")

        # 将 不确定性 设置为和 修改loss函数得到的不确定性一样的大小
        lower_bound_loss = mean - megnet_loss_uncertainty[name]
        upper_bound_loss = mean + megnet_loss_uncertainty[name]
        is_inside_loss = (test_y >= torch.tensor(lower_bound_loss)) & (
                test_y <= torch.tensor(upper_bound_loss))  # 将 lower_bound 和 upper_bound 转换为张量
        coverage_percentage_loss = torch.mean(is_inside_loss.float()) * 100
        print(f"u=loss Coverage Percentage:{coverage_percentage_loss.item():.3f}%")

    # return observed_pred, test_y
    # return mae, mse, round(coverage_percentage_2std.item(), 3), np.mean(2*std)
    return mae, mse, round(coverage_percentage_loss.item(), 3), megnet_loss_uncertainty[name]

# 测试集：1352 'scan', 'exp', 'gllb-sc', 'hse', 'pbe'
datasets_name = ['scan', 'hse', 'gllb-sc', 'pbe']
random_seed_list = [42*i for i in range(1,11)]
all_result = []
megnet_loss_uncertainty = {'scan': 0.850,
                           'hse': 0.915,
                           'pbe': 0.603,
                           'gllb-sc': 0.990}

for n in datasets_name:
    if n == 'scan':
        all_mae = []
        all_mse = []
        all_coverage = []
        all_avg_uncertainty = []
        mae, mse, coverage, std = predict_p_and_u(n, 42)
        all_mae.append(mae)
        all_mse.append(mse)
        all_coverage.append(coverage)
        all_avg_uncertainty.append(std)
        result_dict = {
            'name': n,
            'all_mae': all_mae,
            'all_mse': all_mse,
            'all_coverage': all_coverage,
            'all_avg_std': all_avg_uncertainty
        }
        all_result.append(result_dict)
    else:
        all_mae = []
        all_mse = []
        all_coverage = []
        all_avg_uncertainty = []
        for seed in random_seed_list:
            mae, mse, coverage, std = predict_p_and_u(n, seed)
            all_mae.append(mae)
            all_mse.append(mse)
            all_coverage.append(coverage)
            all_avg_uncertainty.append(std)

        result_dict = {
            'name': n,
            'all_mae': all_mae,
            'all_mse': all_mse,
            'all_coverage': all_coverage,
            'all_avg_std': all_avg_uncertainty
        }

        all_result.append(result_dict)


print("===================================================")
for dict in all_result:
    print(dict['name'])
    print('all_mae:', dict['all_mae'])
    print('all_mse:', dict['all_mse'])
    print('all_coverage(%):', dict['all_coverage'])
    print('all_avg_uncertainty:', dict['all_avg_std'])
    print("avg_mae:", round(np.mean(dict['all_mae']),3))
    print("avg_mse:", round(np.mean(dict['all_mse']),3))
    print("avg_coverage(%):", round(np.mean(dict['all_coverage']),3))
    # print("all_avg_uncertainty(2std):", round(np.mean(dict['all_avg_std']),3))
    print("all_avg_uncertainty(u=loss):", round(np.mean(dict['all_avg_std']),3))



# ===================================================
# scan
# all_mae: [tensor(1.2964)]
# all_mse: [tensor(2.4400)]
# all_coverage(%): [86.201]
# all_avg_uncertainty: [1.7672168]
# avg_mae: 1.296
# avg_mse: 2.44
# avg_coverage(%): 86.201
# all_avg_uncertainty(2std): 1.767
# hse
# all_mae: [tensor(1.0425), tensor(1.0826), tensor(1.2032), tensor(1.1132), tensor(1.1029), tensor(1.0588), tensor(1.3935), tensor(1.2042), tensor(1.3351), tensor(1.0933)]
# all_mse: [tensor(2.2500), tensor(2.2386), tensor(2.9277), tensor(2.7674), tensor(2.2453), tensor(2.3482), tensor(2.8124), tensor(3.0062), tensor(3.4279), tensor(2.4699)]
# all_coverage(%): [82.501, 86.385, 75.583, 80.355, 85.942, 82.353, 79.319, 78.32, 77.063, 80.059]
# all_avg_uncertainty: [1.7164015, 1.9108692, 1.7631121, 1.8435386, 1.800647, 1.7141342, 1.9942515, 1.839461, 1.8851186, 1.7461379]
# avg_mae: 1.163
# avg_mse: 2.649
# avg_coverage(%): 80.788
# all_avg_uncertainty(2std): 1.821
# gllb-sc
# all_mae: [tensor(3.0539), tensor(2.4260), tensor(2.7316), tensor(2.1401), tensor(2.4740), tensor(2.4778), tensor(2.1689), tensor(2.3707), tensor(2.6553), tensor(2.3476)]
# all_mse: [tensor(11.7170), tensor(7.3275), tensor(9.5102), tensor(6.7535), tensor(7.9305), tensor(8.2291), tensor(7.2936), tensor(7.5124), tensor(9.2609), tensor(7.2383)]
# all_coverage(%): [21.902, 37.662, 25.86, 43.766, 30.559, 31.706, 50.277, 32.445, 37.662, 38.809]
# all_avg_uncertainty: [1.7725794, 2.1831424, 1.7816973, 1.8099636, 1.8346184, 1.6861513, 1.961978, 1.7070673, 2.4159362, 2.0034306]
# avg_mae: 2.485
# avg_mse: 8.277
# avg_coverage(%): 35.065
# all_avg_uncertainty(2std): 1.916
# pbe
# all_mae: [tensor(1.2026), tensor(1.0971), tensor(1.1007), tensor(1.2292), tensor(1.1917), tensor(1.0818), tensor(1.3153), tensor(1.1117), tensor(1.0411), tensor(1.1655)]
# all_mse: [tensor(2.7293), tensor(2.3128), tensor(2.6191), tensor(2.9210), tensor(2.4557), tensor(2.4965), tensor(3.2725), tensor(2.4339), tensor(2.3434), tensor(2.4149)]
# all_coverage(%): [74.954, 81.946, 79.245, 77.802, 77.395, 79.8, 75.842, 85.535, 79.504, 83.241]
# all_avg_uncertainty: [1.7134336, 1.7138938, 1.739663, 1.7510488, 1.721684, 1.7411574, 1.9021664, 2.016048, 1.7238077, 1.7122542]
# avg_mae: 1.154
# avg_mse: 2.6
# avg_coverage(%): 79.526
# all_avg_uncertainty(2std): 1.774


# 不确定性 = loss 的不确定性：
# ===================================================
# scan
# all_mae: [tensor(1.2964)]
# all_mse: [tensor(2.4400)]
# all_coverage(%): [26.378]
# all_avg_uncertainty: [0.85]
# avg_mae: 1.296
# avg_mse: 2.44
# avg_coverage(%): 26.378
# all_avg_uncertainty(u=loss): 0.85
# hse
# all_mae: [tensor(1.0425), tensor(1.0826), tensor(1.2032), tensor(1.1132), tensor(1.1029), tensor(1.0588), tensor(1.3935), tensor(1.2042), tensor(1.3351), tensor(1.0933)]
# all_mse: [tensor(2.2500), tensor(2.2386), tensor(2.9277), tensor(2.7674), tensor(2.2453), tensor(2.3482), tensor(2.8124), tensor(3.0062), tensor(3.4279), tensor(2.4699)]
# all_coverage(%): [58.417, 59.378, 53.274, 58.269, 51.387, 57.936, 30.485, 54.791, 47.059, 59.526]
# all_avg_uncertainty: [0.915, 0.915, 0.915, 0.915, 0.915, 0.915, 0.915, 0.915, 0.915, 0.915]
# avg_mae: 1.163
# avg_mse: 2.649
# avg_coverage(%): 53.052
# all_avg_uncertainty(u=loss): 0.915
# gllb-sc
# all_mae: [tensor(3.0539), tensor(2.4260), tensor(2.7316), tensor(2.1401), tensor(2.4740), tensor(2.4778), tensor(2.1689), tensor(2.3707), tensor(2.6553), tensor(2.3476)]
# all_mse: [tensor(11.7170), tensor(7.3275), tensor(9.5102), tensor(6.7535), tensor(7.9305), tensor(8.2291), tensor(7.2936), tensor(7.5124), tensor(9.2609), tensor(7.2383)]
# all_coverage(%): [10.174, 12.098, 14.613, 24.713, 15.279, 16.389, 26.785, 17.203, 14.909, 18.128]
# all_avg_uncertainty: [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
# avg_mae: 2.485
# avg_mse: 8.277
# avg_coverage(%): 17.029
# all_avg_uncertainty(u=loss): 0.99
# pbe
# all_mae: [tensor(1.2026), tensor(1.0971), tensor(1.1007), tensor(1.2292), tensor(1.1917), tensor(1.0818), tensor(1.3153), tensor(1.1117), tensor(1.0411), tensor(1.1655)]
# all_mse: [tensor(2.7293), tensor(2.3128), tensor(2.6191), tensor(2.9210), tensor(2.4557), tensor(2.4965), tensor(3.2725), tensor(2.4339), tensor(2.3434), tensor(2.4149)]
# all_coverage(%): [35.479, 37.514, 42.915, 35.183, 32.852, 44.913, 33.74, 37.329, 47.577, 28.413]
# all_avg_uncertainty: [0.603, 0.603, 0.603, 0.603, 0.603, 0.603, 0.603, 0.603, 0.603, 0.603]
# avg_mae: 1.154
# avg_mse: 2.6
# avg_coverage(%): 37.592
# all_avg_uncertainty(u=loss): 0.603