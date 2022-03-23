import torch as th
import torch.nn as nn
import torch.nn.functional as F

from MySageConv import SAGEConv

from time import time


class ABGNN(nn.Module):
    r"""
                    Description
                    -----------
                    our Asynchronous Bidirectional Graph Nueral Network (ABGNN) model

                    Note that this model is ony for one direction
        """
    def __init__(
        self,
        ntypes,
        hidden_dim,    # dim of the hidden layers
        out_dim,       # dim of the last layer
        dropout,       # dropout rate
        n_layers=None,  # number of layers
        in_dim=16,      # dim of the input layer
        activation=th.relu, #activation function
    ):
        super(ABGNN, self).__init__()
        self.activation = activation
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()
        self.fc_init = nn.Linear(in_dim,hidden_dim)
        in_dim = hidden_dim

        self.conv = SAGEConv(
            hidden_dim,
            hidden_dim,
            include=False,
            combine_type='sum',
            aggregator_type='mean',
            activation=activation,
        )

    def forward(self, blocks, features):
        r"""

                Description
                -----------
                forward computation of FGNN

                Parameters
                ----------
                blocks : [dgl_block]
                    blocks gives the sampled neighborhood for a batch of target nodes.
                    Given a target node n, its sampled neighborhood is organized in layers
                    depending on the distance to n.
                    A block is a graph that describes the part between two succesive layers,
                    consisting of two sets of nodes: the *input* nodes and *output* nodes.
                    The output nodes of the last block are the target nodes (POs), and
                    the input nodes of the first block are the PIs.
                feature : torch.Tensor
                    It represents the input (PI) feature of shape :math:`(N, D_{in})`
                    where :math:`D_{in}` is size of input feature, :math:`N` is the number of target nodes (POs).

                Returns
                -------
                torch.Tensor
                    The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
                    is size of output (PO) feature.
                """

        r"""
        The message passes through the blocks layer by layer, from the PIs of the blocks to the POs
        In each iteration, messages are only passed between two successive layers (blocks)
        """
        depth = len(blocks)
        h = self.activation(self.fc_init(features))
        for i in range(depth):
            if i != 0:
                h = self.dropout(h)
            # if ac_flag is True, then we apply an activation function in the next layer;
            # else not.
            act_flag = (i != depth - 1)
            h = self.conv(act_flag, blocks[i], h) # the generated node embeddings of current layer
        return h.squeeze(1)


class MLP(nn.Module):
    r"""
                Description
                -----------
                a simple multilayer perceptron
    """
    def __init__(self,in_dim,out_dim,nlayers,activation =nn.ReLU() ,dropout=0.5):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nlayers = nlayers
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.layers= nn.Sequential()
        dim1 = in_dim
        for i in range(nlayers-1):
            self.layers.add_module('dropout_{}'.format(i+1),self.dropout)
            self.layers.add_module('activation_{}'.format(i+1), self.activation)
            self.layers.add_module('linear_{}'.format(i+1),nn.Linear(dim1, int(dim1/2)))
            dim1 = int(dim1 / 2)
        self.layers.add_module('linear_{}'.format(nlayers),nn.Linear(dim1, out_dim))
    def forward(self,embedding):
        return self.layers(embedding).squeeze(-1)

class BiClassifier(nn.Module):
    r"""
                    Description
                    -----------
                    the model used to classify
                    consits of two GNN models and one MLP model
        """
    def __init__(
        self, GCN1,GCN2,mlp
    ):
        super(BiClassifier, self).__init__()

        self.GCN1 = GCN1
        self.GCN2 = GCN2
        self.mlp=mlp
        # print(self.layers)
    def forward(self, in_blocks, in_features,out_blocks,out_features):
        if self.GCN2 is None:
            h = self.GCN1(in_blocks,in_features)
        elif self.GCN1 is None:
            h = self.GCN2(out_blocks, out_features)
        else:
            h = self.GCN1(in_blocks, in_features)
            rh = self.GCN2(out_blocks,out_features)
            # combine the information from both direction
            h = th.cat((h,rh),1)
        h = self.mlp(h)

        return h