import torch

file_path = f"./saved_descriptors/scan_42/train_data.pt"
# 读取
loaded_data = torch.load(file_path)
descriptors1 = loaded_data['descriptors']
targets1 = loaded_data['targets']

file_path2 = f"./saved_descriptors/pbe_42/train_data.pt"
# 读取
loaded_data1 = torch.load(file_path2)
descriptors2 = loaded_data1['descriptors']
targets2 = loaded_data1['targets']

print(len(descriptors1),len(descriptors1[0]))

print(targets1[:10])
print(targets2[:10])

