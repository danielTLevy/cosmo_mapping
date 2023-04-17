"""Torch Module for E(n) Equivariant Graph Convolutional Layer"""
import torch
import torch.nn as nn
from utils import periodic_difference_torch
from dgl import function as fn

class EGNNConv(nn.Module):
    r"""Equivariant Graph Convolutional Layer from `E(n) Equivariant Graph
    Neural Networks <https://arxiv.org/abs/2102.09844>`__

    .. math::

        m_{ij}=\phi_e(h_i^l, h_j^l, ||x_i^l-x_j^l||^2, a_{ij})

        x_i^{l+1} = x_i^l + C\sum_{j\in\mathcal{N}(i)}(x_i^l-x_j^l)\phi_x(m_{ij})

        m_i = \sum_{j\in\mathcal{N}(i)} m_{ij}

        h_i^{l+1} = \phi_h(h_i^l, m_i)

    where :math:`h_i`, :math:`x_i`, :math:`a_{ij}` are node features, coordinate
    features, and edge features respectively. :math:`\phi_e`, :math:`\phi_h`, and
    :math:`\phi_x` are two-layer MLPs. :math:`C` is a constant for normalization,
    computed as :math:`1/|\mathcal{N}(i)|`.

    Parameters
    ----------
    in_size : int
        Input feature size; i.e. the size of :math:`h_i^l`.
    hidden_size : int
        Hidden feature size; i.e. the size of hidden layer in the two-layer MLPs in
        :math:`\phi_e, \phi_x, \phi_h`.
    out_size : int
        Output feature size; i.e. the size of :math:`h_i^{l+1}`.
    edge_feat_size : int, optional
        Edge feature size; i.e. the size of :math:`a_{ij}`. Default: 0.

    Example
    -------
    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import EGNNConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> node_feat, coord_feat, edge_feat = th.ones(6, 10), th.ones(6, 3), th.ones(6, 2)
    >>> conv = EGNNConv(10, 10, 10, 2)
    >>> h, x = conv(g, node_feat, coord_feat, edge_feat)
    """
    def __init__(self, in_size, hidden_size, out_size, edge_feat_size=0, graph_feat_size=0):
        super(EGNNConv, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_feat_size = edge_feat_size
        self.graph_feat_size = graph_feat_size
        self.eps = 1e-30
        act_fn = nn.SiLU()
        tanh = nn.Tanh()

        # \phi_e
        self.edge_mlp = nn.Sequential(
            # +1 for the radial feature: ||x_i - x_j||^2
            nn.Linear(in_size * 2 + edge_feat_size + 1, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn
        )

        # \phi_h
        self.node_mlp = nn.Sequential(
            nn.Linear(in_size + hidden_size + self.graph_feat_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, out_size)
        )

        # \phi_x
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, 1, bias=False),
            #tanh
        )

        # \phi_u
        self.graph_mlp = nn.Sequential(
            nn.Linear(graph_feat_size + hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size, bias=False),
        )

    def u_periodic_sub_v(self, src_name, dst_name, edge_name):
        """periodic difference of two vectors"""
        def func(edges):
            return {edge_name: periodic_difference_torch(edges.src[src_name], edges.dst[dst_name])}
        return func

    def message(self, edges):
        """message function for EGNN"""
        # concat features for edge mlp
        if self.edge_feat_size > 0:
            f = torch.cat(
                [edges.src['h'], edges.dst['h'], edges.data['radial'], edges.data['a']],
                dim=-1
            )
        else:
            f = torch.cat([edges.src['h'], edges.dst['h'], edges.data['radial']], dim=-1)

        msg_h = self.edge_mlp(f)

        msg_x = self.coord_mlp(msg_h) * edges.data['x_diff']
        return {'msg_x': msg_x, 'msg_h': msg_h}

    def forward(self, graph, node_feat, coord_feat, edge_feat=None, graph_feat=None):
        r"""
        Description
        -----------
        Compute EGNN layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        node_feat : torch.Tensor
            The input feature of shape :math:`(N, h_n)`. :math:`N` is the number of
            nodes, and :math:`h_n` must be the same as in_size.
        coord_feat : torch.Tensor
            The coordinate feature of shape :math:`(N, h_x)`. :math:`N` is the
            number of nodes, and :math:`h_x` can be any positive integer.
        edge_feat : torch.Tensor, optional
            The edge feature of shape :math:`(M, h_e)`. :math:`M` is the number of
            edges, and :math:`h_e` must be the same as edge_feat_size.

        Returns
        -------
        node_feat_out : torch.Tensor
            The output node feature of shape :math:`(N, h_n')` where :math:`h_n'`
            is the same as out_size.
        coord_feat_out: torch.Tensor
            The output coordinate feature of shape :math:`(N, h_x)` where :math:`h_x`
            is the same as the input coordinate feature dimension.
        """
        with graph.local_scope():
            n_nodes = node_feat.shape[0]
            # node feature
            graph.ndata['h'] = node_feat
            # coordinate feature
            graph.ndata['x'] = coord_feat
            # graph feature
            u = graph_feat
            # edge feature
            if self.edge_feat_size > 0:
                assert edge_feat is not None, "Edge features must be provided."
                graph.edata['a'] = edge_feat
            # get coordinate diff & radial features
            graph.apply_edges(self.u_periodic_sub_v('x', 'x', 'x_diff'))
            graph.edata['radial'] = graph.edata['x_diff'].square().sum(dim=1).unsqueeze(-1)
            # normalize coordinate difference
            #print("graph.edata['radial']", graph.edata['radial'])

            graph.edata['x_diff'] = graph.edata['x_diff'] #/ ((graph.edata['radial'] +  self.eps).sqrt() + self.eps)
            graph.apply_edges(self.message)
            graph.update_all(fn.copy_e('msg_x', 'm'), fn.mean('m', 'x_neigh'))
            graph.update_all(fn.copy_e('msg_h', 'm'), fn.sum('m', 'h_neigh'))

            h_neigh, x_neigh = graph.ndata['h_neigh'], graph.ndata['x_neigh']

            if u is not None:
                h = self.node_mlp(
                    torch.cat([node_feat, h_neigh, u.expand(n_nodes, -1)], dim=-1)
                )
            else:
                h = self.node_mlp(
                    torch.cat([node_feat, h_neigh], dim=-1)
                )
            if u is not None:
                u = self.graph_mlp(torch.cat([u, h.mean(dim=0, keepdim=True)], dim=-1))
            x = coord_feat + x_neigh
            
            return h, x, u


class EGNN(nn.Module):
    def __init__(self, in_node_dim, mlp_h_dim, hidden_node_dim, n_layers=3, edge_feat_size=0, graph_feat_size=0):
        super().__init__()
        self.egnnconvs = nn.ModuleList()
        self.egnnconvs.append(EGNNConv(in_node_dim, mlp_h_dim, hidden_node_dim,
                                       edge_feat_size=edge_feat_size,
                                       graph_feat_size=graph_feat_size))
        for i in range(n_layers - 1):
            self.egnnconvs.append(EGNNConv(hidden_node_dim, mlp_h_dim, hidden_node_dim, edge_feat_size=0, graph_feat_size=hidden_node_dim))
    def forward(self, graph, node_feat, coords, edge_feat=None, graph_feat=None):
        h, x, u = node_feat, coords, graph_feat
        for egnnconv in self.egnnconvs:
            h, x, u = egnnconv(graph, h, x, edge_feat=edge_feat, graph_feat=u)
        return h, x, u
