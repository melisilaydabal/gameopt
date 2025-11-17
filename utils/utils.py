import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import os
import os.path
import gpytorch
import wandb


def update_datasets(arglist, device, data, train_data, test_data, sampled_points, train_x, train_y, test_x, test_y, oracle, esm_embedding_model):
    for variant in sampled_points.keys():
        # If sampled point is not already in the training data
        if variant not in list(train_data.keys()):
            if arglist.dataset == 'gb1_4':
                train_data[variant] = data[variant]
                sampled_point_x = torch.from_numpy(np.array([data[variant]['e' + str(i)] for i in range(1, 61)], dtype=np.float64)).float().unsqueeze(0)
                sampled_point_y = torch.tensor([float(data[variant]['LogFitness'])])

                train_x = torch.cat((train_x, sampled_point_x.to(device)), dim=0)
                train_y = torch.cat((train_y, sampled_point_y.to(device)), dim=0)

                try:
                    index = list(test_data.keys()).index(variant)
                    print(f"The index of key '{variant}' is: {index}")
                    test_x = torch.cat([test_x[:index], test_x[index + 1:]])
                    test_y = torch.cat([test_y[:index], test_y[index + 1:]])
                    test_data.pop(variant, None)
                except ValueError:
                    print(f"Key '{variant}' not found in the dictionary.")
            else:
                train_data[variant] = {}
                seq_list = []
                seq_list.append(variant)
                # compute train_y via oracle
                oracle = oracle.to(device)
                oracle.eval()
                with torch.no_grad():
                    # compute train x via esm_embedding_model
                    sampled_point_x = esm_embedding_model.embed(seq_list, device)
                    # Forward pass on the embedding to get label via oracle model
                    sampled_point_y = oracle(sampled_point_x.to(device))
                train_data[variant]['embedding'] = sampled_point_x
                train_data[variant]['fitness'] = sampled_point_y.item()

                train_x = torch.cat((train_x, sampled_point_x.to(device)), dim=0)
                train_y = torch.cat((train_y, sampled_point_y.squeeze(0).to(device)), dim=0)

                # Free GPU memory at the end of each round
                torch.cuda.empty_cache()

        print(f'Updated train data size {len(train_data)}')
        # print('Size of train_x : {}'.format(train_x.size()))
        # print('Size of sampled_point_x : {}'.format(sampled_point_x.size()))
        # print('Updated size of sampled_point_x : {}'.format(sampled_point_x.unsqueeze(0).size()))
        # print('Size of train_y : {}'.format(train_y.size()))
        # print('Size of sampled_point_y : {}'.format(sampled_point_y.size()))

    return train_data, train_x, train_y, test_x, test_y
