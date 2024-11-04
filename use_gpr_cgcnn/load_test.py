import torch


name = ['scan', 'hse', 'gllb-sc', 'pbe']
random_seed_list = [42*i for i in range(1,11)]

for n in name:
    if n == 'scan':
        file_path = f"./saved_best_hypers/{n}_42.pt"
        loaded_data = torch.load(file_path)
        lengthscale = loaded_data['lengthscale']
        lr = loaded_data['lr']
        noise = loaded_data['noise']
        mae = loaded_data['mae']

        print(f"{n}_42 lengthscale:", lengthscale, " lr:", lr, " noise:", noise, " mae:", mae)
    else:
        for seed in random_seed_list:
            file_path = f"./saved_best_hypers/{n}_{seed}.pt"
            loaded_data = torch.load(file_path)
            lengthscale = loaded_data['lengthscale']
            lr = loaded_data['lr']
            noise = loaded_data['noise']
            mae = loaded_data['mae']

            print(f"{n}_{seed} lengthscale:",lengthscale, " lr:",lr, " noise:",noise, " mae:",mae)