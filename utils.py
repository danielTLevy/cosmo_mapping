import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import dgl

def is_graph_connected(graph):
    """Check if graph is connected"""
    nx_graph = nx.Graph(dgl.to_networkx(graph).to_undirected())
    return nx.is_connected(nx_graph)


def periodic_difference_numpy(x, y, boundary=1.0):
    ''' 
    Given two 3-dimensional coordinates x and y, compute a vector between them under periodic boundary conditions.
    '''
    dx = x - y
    dx = boundary * np.round(dx / boundary) - dx 
    return dx


def periodic_difference_torch(x, y, boundary=1.0):
    ''' 
    Given two 3-dimensional coordinates x and y, compute a vector between them under periodic boundary conditions.
    '''
    dx = x - y
    dx = boundary * torch.round(dx / boundary) - dx 
    return dx

def compare_haloes(x1, x2):
    """Compare haloes between two graphs"""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (20, 10))

    #color map to show matched haloes based on color
    num_haloes = 100  
    norm = mpl.colors.Normalize(0, num_haloes)
    c_m = mpl.cm.gist_rainbow
    s_m = mpl.cm.ScalarMappable(cmap=c_m, norm=norm)

    #ax0.text(7500, 21000, "N-body", fontsize = 30)
    #ax1.text(7500, 21000, "Hydro", fontsize = 30)
    #looping so we can get the matching colors
    for i in range(num_haloes):
        ax0.scatter(x1[i,0], x1[i,1])
        ax1.scatter(x2[i,0], x2[i,1])