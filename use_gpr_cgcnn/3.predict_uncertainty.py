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
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

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

def get_test_dataset(name, seed):
    train_data = torch.load(f'../get_descriptor_cgcnn/saved_descriptors/{name}_{seed}/test_data.pt')

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

        # percentage = (count_2std / len(test_y)) * 100
        # print("真值在预测范围内的百分比1:", percentage)

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
        lower_bound_loss = mean - cgcnn_loss_uncertainty[name]
        upper_bound_loss = mean + cgcnn_loss_uncertainty[name]
        is_inside_loss = (test_y >= torch.tensor(lower_bound_loss)) & (
                test_y <= torch.tensor(upper_bound_loss))  # 将 lower_bound 和 upper_bound 转换为张量
        coverage_percentage_loss = torch.mean(is_inside_loss.float()) * 100
        print(f"u=loss Coverage Percentage:{coverage_percentage_loss.item():.3f}%")

    # return observed_pred, test_y
    # return mae, mse, round(coverage_percentage_2std.item(), 3), np.mean(2*std)
    return mae, mse, round(coverage_percentage_loss.item(), 3), cgcnn_loss_uncertainty[name]

def plot_results(observed_pred, test_y, name):
    mean = observed_pred.mean.numpy()
    std = observed_pred.stddev.numpy()
    test_y = test_y.numpy()

    upper = mean + 2*std
    lower = mean - 2*std

    flag = (test_y - upper)*(test_y - lower)

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
    if name == 'PBE' or name == 'HSE' or name == 'Exp':
        # 为了更好展示这些点，这里添加了一些抖动，使得点不会过于密集，百分比计算结果不受影响
        np.random.seed(42)
        y_jittered = y.copy()
        mask = (y >= -3.0) & (y <= -2.6)
        y_jittered[mask] = y[mask] + np.random.normal(0, 0.1, size=y[mask].shape)

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



    axs[0].set_ylim(-0.5, 25.5)

    if name == "HSE":
        axs[0].set_ylim(-0.5, 20.5)
        axs[1].set_ylim(-3.5, 0.1)
    if name == "PBE":
        axs[1].set_ylim(-3.5, 0.1)
    if name == "GLLB":
        axs[1].set_ylim(-6.5, 0.2)


    plt.savefig(f'../saved_fig/GPR/CGCNN/GPR_CGCNN_{name}_prediction.pdf', bbox_inches='tight')
    plt.show()

# 测试集：1352 'scan', 'exp', 'gllb-sc', 'hse', 'pbe'
datasets_name = ['scan', 'hse', 'gllb-sc', 'pbe']
random_seed_list = [42*i for i in range(1,11)]
all_result = []
cgcnn_loss_uncertainty = {'scan': 0.283,
                           'hse': 0.223,
                           'pbe': 0.250,
                           'gllb-sc': 0.170}
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
# all_mae: [tensor(0.9332)]
# all_mse: [tensor(1.9165)]
# all_coverage(%): [86.016]
# all_avg_uncertainty: [1.7740499]
# avg_mae: 0.933
# avg_mse: 1.916
# avg_coverage(%): 86.016
# all_avg_uncertainty(2std): 1.774
# hse
# all_mae: [tensor(0.8532), tensor(0.8486), tensor(0.9032), tensor(0.8740), tensor(0.8885), tensor(0.8269), tensor(0.7837), tensor(0.8198), tensor(0.8519), tensor(0.8974)]
# all_mse: [tensor(1.8901), tensor(1.8373), tensor(2.0815), tensor(1.8114), tensor(2.0370), tensor(1.8408), tensor(1.6311), tensor(1.8475), tensor(1.8932), tensor(1.8372)]
# all_coverage(%): [84.61, 85.35, 84.795, 86.533, 85.239, 88.309, 87.458, 87.754, 86.127, 84.98]
# all_avg_uncertainty: [1.7064123, 1.7283173, 1.7312455, 1.7317761, 1.7000291, 1.8226147, 1.7315996, 1.734023, 1.6790723, 1.6662077]
# avg_mae: 0.855
# avg_mse: 1.871
# avg_coverage(%): 86.116
# all_avg_uncertainty(2std): 1.723
# gllb-sc
# all_mae: [tensor(1.2771), tensor(1.4045), tensor(1.4764), tensor(1.4599), tensor(1.1633), tensor(1.3283), tensor(1.3322), tensor(1.3155), tensor(1.3484), tensor(1.3895)]
# all_mse: [tensor(3.3212), tensor(3.9553), tensor(4.5140), tensor(3.8704), tensor(2.9823), tensor(3.8550), tensor(3.6456), tensor(3.3954), tensor(3.6294), tensor(3.7302)]
# all_coverage(%): [78.764, 71.365, 71.402, 72.845, 81.613, 74.621, 75.213, 76.767, 73.881, 73.696]
# all_avg_uncertainty: [1.8955534, 1.8811415, 1.6818771, 1.8468473, 1.7991496, 1.6879945, 1.8587648, 1.7945755, 1.7422782, 1.8123367]
# avg_mae: 1.35
# avg_mse: 3.69
# avg_coverage(%): 75.017
# all_avg_uncertainty(2std): 1.8
# pbe
# all_mae: [tensor(0.8832), tensor(0.8225), tensor(0.9034), tensor(0.8180), tensor(0.9378), tensor(0.8198), tensor(0.8994), tensor(0.8063), tensor(0.8161), tensor(0.8786)]
# all_mse: [tensor(1.9112), tensor(1.6180), tensor(1.9321), tensor(1.7716), tensor(2.2851), tensor(1.7067), tensor(1.9342), tensor(1.8126), tensor(1.6819), tensor(1.8006)]
# all_coverage(%): [82.057, 85.387, 82.686, 85.905, 79.8, 85.757, 81.65, 84.906, 86.053, 84.721]
# all_avg_uncertainty: [1.7275741, 1.6750683, 1.7321842, 1.6999066, 1.7051492, 1.6699584, 1.7355667, 1.6679637, 1.7445102, 1.7838202]
# avg_mae: 0.858
# avg_mse: 1.845
# avg_coverage(%): 83.892
# all_avg_uncertainty(2std): 1.714


# u = loss的不确定性
# ===================================================
# scan
# all_mae: [tensor(0.9332)]
# all_mse: [tensor(1.9165)]
# all_coverage(%): [24.047]
# all_avg_uncertainty: [0.283]
# avg_mae: 0.933
# avg_mse: 1.916
# avg_coverage(%): 24.047
# all_avg_uncertainty(u=loss): 0.283
# hse
# all_mae: [tensor(0.8532), tensor(0.8486), tensor(0.9032), tensor(0.8740), tensor(0.8885), tensor(0.8269), tensor(0.7837), tensor(0.8198), tensor(0.8519), tensor(0.8974)]
# all_mse: [tensor(1.8901), tensor(1.8373), tensor(2.0815), tensor(1.8114), tensor(2.0370), tensor(1.8408), tensor(1.6311), tensor(1.8475), tensor(1.8932), tensor(1.8372)]
# all_coverage(%): [32.889, 31.188, 27.821, 22.789, 27.71, 32.297, 34.776, 33.0, 27.377, 25.268]
# all_avg_uncertainty: [0.223, 0.223, 0.223, 0.223, 0.223, 0.223, 0.223, 0.223, 0.223, 0.223]
# avg_mae: 0.855
# avg_mse: 1.871
# avg_coverage(%): 29.512
# all_avg_uncertainty(u=loss): 0.223
# gllb-sc
# all_mae: [tensor(1.2771), tensor(1.4045), tensor(1.4764), tensor(1.4599), tensor(1.1633), tensor(1.3283), tensor(1.3322), tensor(1.3155), tensor(1.3484), tensor(1.3895)]
# all_mse: [tensor(3.3212), tensor(3.9553), tensor(4.5140), tensor(3.8704), tensor(2.9823), tensor(3.8550), tensor(3.6456), tensor(3.3954), tensor(3.6294), tensor(3.7302)]
# all_coverage(%): [10.803, 10.729, 9.138, 7.769, 12.542, 11.506, 10.803, 10.396, 9.064, 9.397]
# all_avg_uncertainty: [0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17]
# avg_mae: 1.35
# avg_mse: 3.69
# avg_coverage(%): 10.215
# all_avg_uncertainty(u=loss): 0.17
# pbe
# all_mae: [tensor(0.8832), tensor(0.8225), tensor(0.9034), tensor(0.8180), tensor(0.9378), tensor(0.8198), tensor(0.8994), tensor(0.8063), tensor(0.8161), tensor(0.8786)]
# all_mse: [tensor(1.9112), tensor(1.6180), tensor(1.9321), tensor(1.7716), tensor(2.2851), tensor(1.7067), tensor(1.9342), tensor(1.8126), tensor(1.6819), tensor(1.8006)]
# all_coverage(%): [39.808, 31.927, 31.558, 36.7, 41.805, 38.735, 36.256, 41.842, 38.957, 32.63]
# all_avg_uncertainty: [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
# avg_mae: 0.858
# avg_mse: 1.845
# avg_coverage(%): 37.022
# all_avg_uncertainty(u=loss): 0.25