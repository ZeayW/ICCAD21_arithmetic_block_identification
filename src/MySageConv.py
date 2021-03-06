"""Torch Module for GraphSAGE layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch
from torch import nn
from torch.nn import functional as F
from options import get_options
from dgl import function as fn
from dgl.utils import expand_as_pair, check_eq_shape
from time import time

class SAGEConv(nn.Module):

    def __init__(self,
                 ntypes,
                 in_feats,          # the dim of input feature
                 out_feats,         # the dim of output
                 combine_type,      # the combine operator
                 aggregator_type,   # the aggregate operator
                 include = False,   # differentiate asychronous / synchronous message passing
                 feat_drop=0.,      # dropout rate
                 bias=True,         # bias for linear function
                 norm=None,         # normalization
                 activation=None):  # activation function
        super(SAGEConv, self).__init__()
        self.include = include
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        if combine_type == 'concat':
            out_feats = int(out_feats/2)
        self.combine_type = combine_type

        if include:self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        else: self.fc_self = nn.Linear(ntypes,out_feats,bias=bias)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=bias)

    def forward(self,act_flag, graph, feat):
        r"""

        Description
        -----------
        Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        with graph.local_scope():

            feat_src = feat
            feat_dst = graph.dstdata['ntype']
            if self.include:
                feat_dst = feat_src[:graph.number_of_dst_nodes()]

            h_self = feat_dst
            graph.srcdata['h'] = feat_src
            # aggregate messages from neighbors and save to mailbox 'neighbor'
            graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']  # aggregated neighborhood message
            # combine the neighborhood message h_neigh with central node's previous embedding h_self
            rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)
            # apply activation function
            if act_flag and self.activation is not None:
                rst = self.activation(rst)

            return rst
