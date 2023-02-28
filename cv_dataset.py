
import h5py
from dgl.data import DGLDataset
import torch
import dgl
import numpy as np
from utils import periodic_difference_numpy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CVDataset(DGLDataset):
    def __init__(self, data_path, threshold = 0.1):
        self.box_size = 25000
        self.data_path = data_path
        self.threshold = threshold
        super().__init__(name='CV')

    
    def get_data(self, simulation, suite='SIMBA'):
        PATH = self.data_path

        #loading halo matching data
        halo_matcher = PATH + f'halo_matcher/Nbody_{simulation}_{suite}_{simulation}.hdf5'
        hm = h5py.File(halo_matcher, 'r')

        #loading in arrays (all length N, where N is the number of matched haloes)
        nbody_halo_index = hm['nbody_index'][:]        #indices of matched n-body haloes
        hydro_halo_index = hm['hydro_index'][:]        #indices of corresponding hydro haloes
        percent_matched = hm['percent_matched'][:]     #percent of shared particles between matched haloes
        cross_match = hm["cross_match"][:]             #either 0 (didn't match both ways) or 1 (matched both ways)

        #loading halo CoMs
        nbody_halo_filename = PATH + 'SIMBA_CV_data/' + 'nbody_sim/' + simulation + '_fof_subhalo_tab_033.hdf5'
        hydro_halo_filename = PATH + 'SIMBA_CV_data/' + 'hydro_sim/' + simulation + '_fof_subhalo_tab_033.hdf5'

        nbody_haloes = h5py.File(nbody_halo_filename, 'r')
        hydro_haloes = h5py.File(hydro_halo_filename, 'r')

        nbody_Pos = nbody_haloes['Group/GroupPos'][:,:]
        hydro_Pos = hydro_haloes['Group/GroupPos'][:,:]
        nbody_Mass = nbody_haloes['Group/GroupMass'][:]
        hydro_Mass = hydro_haloes['Group/GroupMass'][:]

        #selecting only matched haloes
        matched_nbody_Pos = nbody_Pos[nbody_halo_index]    # apply indices arrays to both nbody and hydro so that
        matched_hydro_Pos = hydro_Pos[hydro_halo_index]    #          the arrays are the same length
        matched_nbody_Mass = nbody_Mass[nbody_halo_index]
        matched_hydro_Mass = hydro_Mass[hydro_halo_index]

        #further selecting only haloes that cross matched and share >60% of particles
        cross_match_frac_mask = (percent_matched>60) & (cross_match == 1)
        selected_nbody_Pos = matched_nbody_Pos[cross_match_frac_mask]
        selected_hydro_Pos = matched_hydro_Pos[cross_match_frac_mask]
        selected_nbody_Mass = matched_nbody_Mass[cross_match_frac_mask]
        selected_hydro_Mass = matched_hydro_Mass[cross_match_frac_mask]
        
        #self.box_size = nbody_haloes["Header"].attrs["BoxSize"]
        selected_nbody_Pos =  selected_nbody_Pos / self.box_size
        selected_hydro_Pos = selected_hydro_Pos / self.box_size
        
        #dist = np.linalg.norm(selected_nbody_Pos - selected_hydro_Pos, axis = 1)
        #selected_nbody_Pos = np.delete(selected_nbody_Pos, np.where(dist>sim_max*0.25), axis=0)
        #selected_hydro_Pos = np.delete(selected_hydro_Pos, np.where(dist>sim_max*0.25), axis=0)

        return selected_nbody_Pos, selected_hydro_Pos, selected_nbody_Mass, selected_hydro_Mass

    def get_edges_from_pos(self, pos_matrix):
        differences = periodic_difference_numpy(pos_matrix[:, None],  pos_matrix[None,:], 1)#self.box_size)
        dist_matrix = np.linalg.norm(differences, axis=-1)
        neigh_matrix =  dist_matrix < self.threshold
        edges_src, edges_dst = neigh_matrix.nonzero()
        return edges_src, edges_dst

    def process(self):
        self.graphs = []
        remove_self_loops = dgl.RemoveSelfLoop()
        for i in range(27):
            simulation = 'CV_' + str(i)
            nbody_pos, hydro_pos, nbody_mass, hydro_mass = self.get_data(simulation)
            n_nodes = nbody_pos.shape[0]
            assert n_nodes == hydro_pos.shape[0]
            # Get edges from nbody positions
            edges_src, edges_dst = self.get_edges_from_pos(nbody_pos)
            # Create graph with positions as features
            graph = dgl.graph((edges_src, edges_dst), num_nodes=n_nodes).to(device)
            graph = remove_self_loops(graph)
            graph.ndata['nbody_pos'] = torch.tensor(nbody_pos, device=device)
            graph.ndata['hydro_pos'] = torch.tensor(hydro_pos, device=device)
            graph.ndata['nbody_mass'] = torch.tensor(nbody_mass, device=device)
            graph.ndata['hydro_mass'] = torch.tensor(hydro_mass, device=device)
            self.graphs.append(graph)

    def __getitem__(self, i):
        return self.graphs[i]
    
    def __len__(self):
        return len(self.graphs)