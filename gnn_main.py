# %%
import numpy as np 
import matplotlib.pyplot as plt

import torch

from torch import nn
import torch.nn.functional as F
import dgl
from cv_dataset import CVDataset
from egnn import EGNN
from utils import periodic_difference_torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import sys

import os
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loss_fcn(pred_coord, true_coord, nbody_coord=None):
    dist_hydro = torch.mean(periodic_difference_torch(pred_coord, true_coord)**2)
    loss = dist_hydro
    if nbody_coord is not None:
        dist_nbody = torch.mean(periodic_difference_torch(pred_coord, nbody_coord))
        loss -= dist_nbody
    return 1e6 * loss

def setup_wandb(cfg):
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'project': cfg.wandb.project, 'entity': cfg.wandb.entity,
             'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True),
              'reinit': True,
              'mode': cfg.wandb.mode}
    wandb.init(**kwargs)
    wandb.save('*.txt')
    return cfg


@hydra.main(config_path='configs/', config_name='config')
def main(cfg: DictConfig):
    #%%
    cfg = setup_wandb(cfg)

    full_dataset = CVDataset(data_path=cfg.dataset.path, threshold=cfg.dataset.threshold)
    split_fracs = [1 - cfg.dataset.frac_val - cfg.dataset.frac_test, cfg.dataset.frac_val, cfg.dataset.frac_test]
    train_data, val_data, test_data = dgl.data.utils.split_dataset(full_dataset, split_fracs)

    true_diff_sum = 0
    for graph_i, graph in enumerate(train_data):
        print(f'Graph {graph_i}')
        true_loss = loss_fcn(graph.ndata['nbody_pos'], graph.ndata['hydro_pos'])
        print(f'True difference: {true_loss.item()}')
        true_diff_sum += true_loss.item()
    print(f'True difference mean: {true_diff_sum / len(train_data)}')


    # %%
    node_feat_dim = 2
    edge_feat_dim = 1
    if cfg.model.model == 'egnn':
        model = EGNN(node_feat_dim, cfg.model.width, cfg.model.width,
                     n_layers=cfg.model.n_layers, edge_feat_size=edge_feat_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    epochs = cfg.training.epochs

    loop = tqdm(range(epochs))
    for epoch in loop:
        loss_sum = 0
        # Training
        for graph_i, graph in enumerate(train_data):
            optimizer.zero_grad()
            # Random node features
            #node_features = torch.rand(graph.number_of_nodes(), node_feat_dim, device=device)
            nbody_norm_log_mass = graph.ndata['nbody_norm_log_mass']
            nbody_vel_sqr = graph.ndata['nbody_norm_log_vel_sqr']
            node_features = torch.cat([nbody_norm_log_mass, nbody_vel_sqr], dim=1)
            edge_features = graph.edata['nbody_norm_vel_dot_prod']
            _, x = model(graph, node_features, graph.ndata['nbody_pos'], edge_features)

            loss = loss_fcn(x, graph.ndata['hydro_pos'])
            loss.backward()
            loss_sum += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        epoch_loss = loss_sum / len(train_data)
        wandb.log({'epoch': epoch, 'loss': epoch_loss})
        loop.set_description(f"Training Epoch {epoch}")
        loop.set_postfix(loss=epoch_loss)

        if epoch % cfg.training.val_every == 0:
            with torch.no_grad():
                loss_sum = 0
                # Validation
                for graph_i, graph in enumerate(val_data):
                    nbody_norm_log_mass = graph.ndata['nbody_norm_log_mass']
                    nbody_vel_sqr = graph.ndata['nbody_norm_log_vel_sqr']
                    node_features = torch.cat([nbody_norm_log_mass, nbody_vel_sqr], dim=1)
                    edge_features = graph.edata['nbody_norm_vel_dot_prod']
                    _, x = model(graph, node_features, graph.ndata['nbody_pos'], edge_features)
                    loss = loss_fcn(x, graph.ndata['hydro_pos'])
                    loss_sum += loss.item()
                wandb.log({'val_loss': loss_sum / len(val_data)})

if __name__ == "__main__":
    main()