import numpy as np
import torch

file_path = f"./saved_descriptors/pbe_420/train_data.pt"
# 读取
loaded_data = torch.load(file_path)

descriptors1 = loaded_data['descriptors']
targets1 = list(loaded_data['targets'])

print(len(descriptors1),len(descriptors1[0]))



# 将 descriptors1 转换为 NumPy 数组以便处理 NaN 值
descriptors_array = np.array(descriptors1)
# 检查并统计 descriptors1 中的 NaN 值
nan_count_descriptors = np.isnan(descriptors_array).sum()
# 检查 targets1 中的 NaN 值（假设 targets1 是一个一维数组）
targets_array = np.array(targets1)
nan_count_targets = np.isnan(targets_array).sum()
# 输出 NaN 值的统计结果
print(f"Number of NaN values in descriptors1: {nan_count_descriptors}")
print(f"Number of NaN values in targets1: {nan_count_targets}")