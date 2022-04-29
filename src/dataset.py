import sys

sys.path.append("..")

import dgl
from dgl.data import DGLDataset
import networkx as nx
import torch as th

import os
import random
from verilog_parser import DcParser
from options import get_options


def parse_single_file(parser,vfile_pair,hier_report):
    r"""

    generate the DAG for a circuit design

    :param parser: DCParser
        the parser used to transform neetlist to DAG
    :param vfile_pair: (str,str)
        the netlists for current circuit, including a hierarchical one and a non-hierarchical one
    :param hier_report: str
        the report file for current circuit
    :return: dglGraph
        the result DAG
    """

    # gate types

    nodes, edges = parser.parse(vfile_pair,hier_report)
    print('#node: {}, #edges:{}'.format(len(nodes),len(edges)))
    
    ctype2id = {
        "1'b0":0,
        "1'b1":1,
        "PI":2,
        "AND":3,
        "OR":4,
        "XOR":5,
        "MUX":6,
        "INV":7,
        "NAND":8,
        "NOR":9,
        "NXOR":10
    }

    print('--- Transforming to dgl graph...')
    # build the dgl graph
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    print('\tassign nid to each node')
    # assign an id to each node
    node2id = {}
    for n in nodes:
        if node2id.get(n[0]) is None:
            nid = len(node2id)
            node2id[n[0]] = nid

    # init the label tensors
    is_input = th.zeros((len(node2id), 1), dtype=th.long)
    is_output = th.zeros((len(node2id), 1), dtype=th.long)


    # collect the label information
    print('\tlabel the nodes')
    for n in nodes:
        nid = node2id[n[0]]
        is_input[nid][0] = n[1]["is_input"]
        is_output[nid][0] = n[1]["is_output"]



    print('\tgenerate type-relative initial features')
    # collect the node type information
    ntype = th.zeros((len(node2id), len(ctype2id)), dtype=th.float)
    for n in nodes:
        nid = node2id[n[0]]
        ntype[nid][ctype2id[n[1]["type"]]] = 1


    src_nodes = []
    dst_nodes = []
    is_reverted = []

    for src, dst, edict in edges:
        src_nodes.append(node2id[src])
        dst_nodes.append(node2id[dst])
        is_reverted.append([0, 1] if edict["is_reverted"] else [1, 0])

    # create the graph
    graph = dgl.graph(
        (th.tensor(src_nodes), th.tensor(dst_nodes)), num_nodes=len(node2id)
    )

    graph.ndata["ntype"] = ntype

    # add label information
    graph.ndata['label_i'] = is_input
    graph.ndata['label_o'] = is_output

    graph.edata["r"] = th.FloatTensor(is_reverted)

    print('--- Transforming is done!')
    print('Processing is Accomplished!')
    return graph


class Dataset(DGLDataset):
    def __init__(self, top_module,data_paths,report_folders,target_block,keywords):
        self.data_paths = data_paths
        self.report_folders = report_folders
        self.parser = DcParser(top_module,target_block,keywords,report_folders[0])
        super(Dataset, self).__init__(name="dac")

    def process(self):
        r"""

        transform the netlists to DAGs

        :return:

        """

        self.batch_graphs = []
        self.graphs = []
        self.len = 0
        vfile_pairs = {}
        for i,path in enumerate(self.data_paths):
            files = os.listdir(path)
            for v in files:
                if not v.endswith('v'):
                    continue
                if v.startswith('hier'):
                    vname = v[5:-2]
                    vfile_pairs[vname] = vfile_pairs.get(vname, [])
                    vfile_pairs[vname].insert(0, v)
                else:
                    vname = v[:-2]
                    vfile_pairs[vname] = vfile_pairs.get(vname, [])
                    vfile_pairs[vname].append(v)
            vfile_pairs = vfile_pairs.values()
            # each circuit has 2 netlists: a hierarchical one and a non-hierarchical one
            for vfile_pair in vfile_pairs:
                hier_vf, vf = vfile_pair[0], vfile_pair[1]
                # the report file is also needed to label the target arithmetic blocks
                hier_report = os.path.join(self.report_folders[i], hier_vf[:-1] + 'rpt')
                hier_vf = os.path.join(path, hier_vf)
                vf = os.path.join(path, vf)

                print("Processing file {}".format(vfile_pair[1]))
                self.len += 1
                # parse single file
                graph = parse_single_file(self.parser, (hier_vf,vf), hier_report)

                self.graphs.append(graph)

        # combine all the graphs into a batch graph
        self.batch_graph = dgl.batch(self.graphs)

    def __len__(self):
        return self.len




