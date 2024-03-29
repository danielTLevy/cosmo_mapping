
import h5py
from dgl.data import DGLDataset
import torch
import dgl
import numpy as np
import csv
from tqdm import tqdm
from utils import periodic_difference_numpy
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_SIMS = {
    'CV': 27,
    'LH': 1000
}

COSMO_PARAM_KEYS = {
    "Omega_m",
    "sigma_8",
    "A_SN1",
    "A_AGN1",
    "A_SN2",
    "A_AGN2",
    "seed"
}

eps = 1e-8

class CamelsDataset(DGLDataset):
    def __init__(self, data_path, threshold=0.1,
                 suite='SIMBA', sim_set='CV', debug=False, overfit=False, threshold_type='dist'):
        self.box_size = 25000
        self.data_path = data_path
        self.suite = suite
        self.sim_set = sim_set
        self.threshold = threshold
        self.threshold_type = threshold_type
        self.debug = debug # Train/test/validate on 10 different sims
        self.overfit = overfit # Train/test/validate on the same sim
        super().__init__(name='self.suite' + '_' + 'self.sim_set')

    def match_indices(self, nbody_dict, hydro_dict, simulation):
        '''
        Matches nbody and hydro haloes based on halo matching data.
        Returns nbody_dict and hydro_dict with only matched haloes.
        '''
        #loading halo matching data
        halo_matcher = self.data_path + f'halo_matcher/Nbody_{simulation}_{self.suite}_{simulation}.hdf5'
        hm = h5py.File(halo_matcher, 'r')

        #loading in arrays (all length N, where N is the number of matched haloes)
        nbody_halo_index = hm['nbody_index'][:]        #indices of matched n-body haloes
        hydro_halo_index = hm['hydro_index'][:]        #indices of corresponding hydro haloes
        percent_matched = hm['percent_matched'][:]     #percent of shared particles between matched haloes
        cross_match = hm["cross_match"][:]             #either 0 (didn't match both ways) or 1 (matched both ways)
        cross_match_frac_mask = (percent_matched>60) & (cross_match == 1)

        #selecting only matched haloes


        for key in nbody_dict.keys():
            nbody_dict[key] = nbody_dict[key][nbody_halo_index]# apply indices arrays to both nbody and hydro so that
            #                                                       the arrays are the same length
            #further selecting only haloes that cross matched and share >60% of particles
            nbody_dict[key] = nbody_dict[key][cross_match_frac_mask]

        for key in hydro_dict.keys():
            # And again for haloes
            hydro_dict[key] = hydro_dict[key][hydro_halo_index]
            hydro_dict[key] = hydro_dict[key][cross_match_frac_mask]

        return nbody_dict, hydro_dict

    def get_data(self, simulation):
        #loading halo CoMs
        nbody_halo_filename = f'{self.data_path}/{self.suite}_{self.sim_set}_data/nbody_sim/{simulation}_fof_subhalo_tab_033.hdf5'
        hydro_halo_filename = f'{self.data_path}/{self.suite}_{self.sim_set}_data/hydro_sim/{simulation}_fof_subhalo_tab_033.hdf5'

        nbody_haloes = h5py.File(nbody_halo_filename, 'r')
        hydro_haloes = h5py.File(hydro_halo_filename, 'r')

        # Store features in dicts
        nbody_dict = {}
        nbody_dict['Pos'] = nbody_haloes['Group/GroupPos'][:,:]
        nbody_dict['Mass'] = nbody_haloes['Group/GroupMass'][:]
        nbody_dict['MassCrit200'] = nbody_haloes['Group/Group_M_Crit200'][:]
        nbody_dict['MassCrit500'] = nbody_haloes['Group/Group_M_Crit500'][:]
        nbody_dict['MassTopHat200'] = nbody_haloes['Group/Group_M_TopHat200'][:]
        nbody_dict['RCrit200'] = nbody_haloes['Group/Group_R_Crit200'][:]
        nbody_dict['RCrit500'] = nbody_haloes['Group/Group_R_Crit500'][:]
        
        nbody_dict['Vel'] = nbody_haloes['Group/GroupVel'][:,:]
        hydro_dict = {}
        hydro_dict['Pos'] = hydro_haloes['Group/GroupPos'][:,:]
        hydro_dict['Mass'] = hydro_haloes['Group/GroupMass'][:]
        hydro_dict['Vel'] = hydro_haloes['Group/GroupVel'][:,:]

        # Select only matced haloes that share >60% of particles
        nbody_dict, hydro_dict = self.match_indices(nbody_dict, hydro_dict, simulation)

        # Normalize positions
        #self.box_size = nbody_haloes["Header"].attrs["BoxSize"]
        nbody_dict['Pos'] = nbody_dict['Pos'] / self.box_size
        hydro_dict['Pos'] = hydro_dict['Pos'] / self.box_size

        #dist = np.linalg.norm(selected_nbody_Pos - selected_hydro_Pos, axis = 1)
        #selected_nbody_Pos = np.delete(selected_nbody_Pos, np.where(dist>sim_max*0.25), axis=0)
        #selected_hydro_Pos = np.delete(selected_hydro_Pos, np.where(dist>sim_max*0.25), axis=0)

        return nbody_dict, hydro_dict

    def get_edges_from_pos(self, pos_matrix):
        differences = periodic_difference_numpy(pos_matrix[:, None],  pos_matrix[None,:], 1)#self.box_size)
        dist_matrix = np.linalg.norm(differences, axis=-1)
        neigh_matrix =  dist_matrix < self.threshold
        edges_src, edges_dst = neigh_matrix.nonzero()
        return edges_src, edges_dst
    
    def get_edges_from_force(self, pos_matrix, masses):
        '''
        Returns edges based on the force between particles
        Forces calculated as F = G*m1*m2/r^2
        '''
        differences = periodic_difference_numpy(pos_matrix[:, None],  pos_matrix[None,:], 1)
        dist_matrix = np.linalg.norm(differences, axis=-1)
        force_matrix = masses[:, None]*masses[None, :]/(dist_matrix**2)
        force_matrix[force_matrix==np.inf] = 0
        force_matrix[force_matrix==np.nan] = 0
        neigh_matrix =  force_matrix > self.threshold
        edges_src, edges_dst = neigh_matrix.nonzero()
        return edges_src, edges_dst

    def norm_log(self, x):
        # For normalizing values with a long tail
        # any values set to zero are set to the minimum non-zero value
        log_x = torch.log(x)
        log_x = torch.log(x)
        min_val = log_x[~torch.isinf(log_x)].min()
        log_x[torch.isinf(log_x)] = min_val
        return (log_x - (log_x.mean())) / (eps + log_x.std())

    def norm(self, x):
        # For normalizing datat that's roughly normally distributed
        return (x - (x.mean())) / (eps + x.std())

    def make_graph_from_dicts(self, nbody_dict, hydro_dict):
        nbody_pos = nbody_dict['Pos']
        hydro_pos = hydro_dict['Pos']
        nbody_masses = nbody_dict['Mass']
        n_nodes = nbody_pos.shape[0]
        assert n_nodes == hydro_pos.shape[0]
        # Get edges from nbody positions
        if self.threshold_type == 'dist':
            edges_src, edges_dst = self.get_edges_from_pos(nbody_pos)
        elif self.threshold_type == 'force':
            edges_src, edges_dst = self.get_edges_from_force(nbody_pos, nbody_masses)
        # Create graph with positions and masses as features
        graph = dgl.graph((edges_src, edges_dst), num_nodes=n_nodes)
        graph.ndata['nbody_pos'] = torch.from_numpy(nbody_pos).float()
        graph.ndata['hydro_pos'] = torch.from_numpy(hydro_pos).float()
        graph.ndata['nbody_mass'] = torch.from_numpy(nbody_dict['Mass']).float()[:,None]
        graph.ndata['hydro_mass'] = torch.from_numpy(hydro_dict['Mass']).float()[:,None]
        graph.ndata['nbody_norm_log_mass'] = self.norm_log(graph.ndata['nbody_mass'])
        graph.ndata['nbody_vel_sqr'] = torch.from_numpy(np.sum(nbody_dict['Vel']**2, axis=1)).float()[:,None]
        graph.ndata['nbody_norm_log_vel_sqr'] = self.norm_log(graph.ndata['nbody_vel_sqr'])
        graph.edata['nbody_vel_dot_prod'] = torch.from_numpy(np.sum(nbody_dict['Vel'][edges_src] * nbody_dict['Vel'][edges_dst], axis=1))[:,None].float()
        graph.edata['nbody_norm_vel_dot_prod'] = self.norm(graph.edata['nbody_vel_dot_prod'])
        graph.ndata['nbody_masscrit200'] = self.norm_log(torch.from_numpy(nbody_dict['MassCrit200']).float()[:,None])
        graph.ndata['nbody_masscrit500'] = self.norm_log(torch.from_numpy(nbody_dict['MassCrit500']).float()[:,None])
        graph.ndata['nbody_masstophat200'] = self.norm_log(torch.from_numpy(nbody_dict['MassTopHat200']).float()[:,None])
        graph.ndata['nbody_rcrit200'] = self.norm_log(torch.from_numpy(nbody_dict['RCrit200']).float()[:,None])
        graph.ndata['nbody_rcrit500'] = self.norm_log(torch.from_numpy(nbody_dict['RCrit500']).float()[:,None])
        graph = dgl.remove_self_loop(graph)

        return graph

    def get_cosmo_params(self):
        cosmo_param_filename = f'{self.data_path}/{self.suite}_{self.sim_set}_data/CosmoAstroSeed_{self.suite}.txt'
        self.cosmo_param_dict = {}
        with open(cosmo_param_filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=' ', skipinitialspace=True)
            for row in reader:
                self.cosmo_param_dict[row['#Name']] = row

    def add_cosmo_features(self, graph, simulation):
        for param_name in COSMO_PARAM_KEYS:
            graph_property = float(self.cosmo_param_dict[simulation][param_name])
            setattr(graph, param_name, torch.Tensor([graph_property]))
        return graph

    def process(self):
        self.graphs = []
        self.get_cosmo_params()
        print("Processing graphs...")
        num_sims = NUM_SIMS[self.sim_set]
        if self.debug:
            num_sims = 10
        if self.overfit:
            num_sims = 1
        for i in tqdm(range(num_sims)):
            simulation = self.sim_set + '_' + str(i)
            nbody_dict, hydro_dict = self.get_data(simulation)
            graph = self.make_graph_from_dicts(nbody_dict, hydro_dict)
            # Add in cosmological parameters
            graph = self.add_cosmo_features(graph, simulation)
            self.graphs.append(graph)

    def __getitem__(self, i):
        return self.graphs[i]
    
    def __len__(self):
        return len(self.graphs)


class CamelsDatasetRaw(Dataset):
    def __init__(self, data_path, suite='SIMBA', sim_set='CV', sim_type ='hydro', debug=False):
        self.box_size = 25000
        self.data_path = data_path
        self.suite = suite
        self.sim_set = sim_set
        assert sim_type == 'nbody' or sim_type == 'hydro'
        self.sim_type = sim_type
        self.debug = debug
        self.process()

    def get_data(self, simulation, sim_type):
        # Load in data
        halo_filename = f'{self.data_path}/{self.suite}_{self.sim_set}_data/{sim_type}_sim/{simulation}_fof_subhalo_tab_033.hdf5'

        haloes = h5py.File(halo_filename, 'r')

        # Store features in dicts
        sim_dict = {}
        sim_dict['Pos'] = haloes['Group/GroupPos'][:,:]
        sim_dict['Mass'] = haloes['Group/GroupMass'][:]
        sim_dict['Vel'] = haloes['Group/GroupVel'][:,:]

        # Normalize positions
        #self.box_size = nbody_haloes["Header"].attrs["BoxSize"]
        sim_dict['Pos'] = sim_dict['Pos'] / self.box_size

        return sim_dict


    def get_cosmo_params(self):
        cosmo_param_filename = f'{self.data_path}/{self.suite}_{self.sim_set}_data/CosmoAstroSeed_{self.suite}.txt'
        self.cosmo_param_dict = {}
        with open(cosmo_param_filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=' ', skipinitialspace=True)
            for row in reader:
                self.cosmo_param_dict[row['#Name']] = row

    def add_cosmo_features(self, sim_dict, simulation):
        for param_name in COSMO_PARAM_KEYS:
            graph_property = float(self.cosmo_param_dict[simulation][param_name])
            sim_dict[param_name] = graph_property
        return sim_dict

    def process(self):
        self.sim_dicts = []
        self.get_cosmo_params()
        num_sims = NUM_SIMS[self.sim_set]
        if self.debug:
            num_sims = 10
        for i in tqdm(range(num_sims)):
            simulation = self.sim_set + '_' + str(i)
            sim_dict = self.get_data(simulation, self.sim_type)
            self.sim_dicts.append(sim_dict)

    def __getitem__(self, i):
        return self.sim_dicts[i]
    
    def __len__(self):
        return len(self.sim_dicts)