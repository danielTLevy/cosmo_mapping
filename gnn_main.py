# %%
import numpy as np 
import matplotlib.pyplot as plt

import torch

from torch import nn
import torch.nn.functional as F
import dgl
from camels_dataset import CamelsDataset
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

def model_pred(graph, model, loss_fcn, use_extra_feats=False):
    graph_feature_list = [graph.Omega_m, graph.sigma_8, graph.A_SN1, graph.A_AGN1, graph.A_SN2, graph.A_AGN2]
    graph_features = torch.cat(graph_feature_list)[None, :].to(device)
    nbody_norm_log_mass = graph.ndata['nbody_norm_log_mass']
    nbody_vel_sqr = graph.ndata['nbody_norm_log_vel_sqr']
    nbody_masscrit200 = graph.ndata['nbody_masscrit200']
    nbody_masscrit500 = graph.ndata['nbody_masscrit500']
    nbody_masstophat200 = graph.ndata['nbody_masstophat200']
    nbody_rcrit200 = graph.ndata['nbody_rcrit200']
    nbody_rcrit500 = graph.ndata['nbody_rcrit500']

    if use_extra_feats:
        node_features = torch.cat([nbody_norm_log_mass, nbody_vel_sqr, nbody_masscrit200,
                                nbody_masscrit500, nbody_masstophat200,
                                nbody_rcrit200, nbody_rcrit500], dim=1).to(device)
    else:
        node_features = torch.cat([nbody_norm_log_mass, nbody_vel_sqr], 1).to(device)
    edge_features = graph.edata['nbody_norm_vel_dot_prod']
    h, x, u = model(graph, node_features, graph.ndata['nbody_pos'], edge_features, graph_features)
    loss = loss_fcn(x, graph.ndata['hydro_pos'])
    return x, loss


@hydra.main(config_path='configs/', config_name='config')
def main(cfg: DictConfig):
    cfg = setup_wandb(cfg)
    if not cfg.dataset.load:
        full_dataset = CamelsDataset(data_path=cfg.dataset.path, threshold=cfg.dataset.threshold,
                                    suite=cfg.dataset.suite, sim_set=cfg.dataset.sim_set, debug=cfg.training.debug)
    else:
        full_dataset = torch.load('/home/mila/d/daniel.levy/scratch/cosmo_mapping/processed_data/threshold{}/full_dataset.pt'.format(str(int(100*cfg.dataset.threshold))))
    split_fracs = [1 - cfg.dataset.frac_val - cfg.dataset.frac_test, cfg.dataset.frac_val, cfg.dataset.frac_test]
    train_data, val_data, test_data = dgl.data.utils.split_dataset(full_dataset, split_fracs)

    true_diff_sum = 0
    for graph_i, graph in enumerate(train_data):
        true_loss = loss_fcn(graph.ndata['nbody_pos'], graph.ndata['hydro_pos'])
        true_diff_sum += true_loss.item()
    true_difference_mean = true_diff_sum / len(train_data)
    print(f'True difference mean (train): {true_difference_mean}')
    wandb.run.summary['true_difference_mean'] = true_difference_mean
    true_val_diff_sum = 0
    for graph_i, graph in enumerate(val_data):
        true_val_loss = loss_fcn(graph.ndata['nbody_pos'], graph.ndata['hydro_pos'])
        true_val_diff_sum += true_val_loss.item()
    true_val_diff_mean = true_val_diff_sum / len(val_data)
    print(f'True difference mean (validation): {true_val_diff_mean}')
    wandb.run.summary['true_val_diff_mean'] = true_val_diff_mean

    node_feat_dim = 7 if cfg.model.use_extra_feats else 2
    edge_feat_dim = 1
    graph_feat_dim = 6
    if cfg.model.model == 'egnn':
        model = EGNN(node_feat_dim, cfg.model.width, cfg.model.width,
                     n_layers=cfg.model.n_layers, edge_feat_size=edge_feat_dim, graph_feat_size=graph_feat_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    epochs = cfg.training.epochs

    loop = tqdm(range(epochs))
    best_val_loss = 1e10
    for epoch in loop:
        train_loss_sum = 0
        # Training
        for graph_i, graph in enumerate(train_data):
            model.train()
            optimizer.zero_grad()
            x, loss = model_pred(graph.to(device), model, loss_fcn, use_extra_feats=cfg.model.use_extra_feats)
            loss.backward()
            train_loss_sum += loss.item()
            if cfg.training.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.clip_grad_norm)
            optimizer.step()
        epoch_loss = train_loss_sum / len(train_data)
        wandb.log({'epoch': epoch, 'loss': epoch_loss})
        loop.set_description(f"Training Epoch {epoch}")
        loop.set_postfix(loss=epoch_loss)

        if epoch % cfg.training.val_every == 0:
            with torch.no_grad():
                val_loss_sum = 0
                # Validation
                for graph_i, graph in enumerate(val_data):
                    model.eval()
                    x, loss = model_pred(graph.to(device), model, loss_fcn, use_extra_feats=cfg.model.use_extra_feats)
                    val_loss_sum += loss.item()
                val_loss = val_loss_sum / len(val_data)
                wandb.log({'val_loss': val_loss})
                if val_loss < best_val_loss:
                    print("New best val loss; Saving")
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'best_model.pt'))
                wandb.log({'best_val_loss': best_val_loss})

if __name__ == "__main__":
    main()