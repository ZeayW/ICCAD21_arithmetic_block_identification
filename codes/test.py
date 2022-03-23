r"""
this script is used to train/fine-tune and validate/test the models
"""

from dataset import *
from options import get_options
from model import *
import dgl
import pickle
import numpy as np
import os
from MyDataLoader2 import *
from time import time
from random import shuffle
import itertools


def DAG2UDG(g):
    r"""

    used to transform a (directed acyclic graph)DAG into a undirected graph

    :param g: dglGraph
        the target DAG

    :return:
        dglGraph
            the output undirected graph
    """
    edges = g.edges()
    reverse_edges = (edges[1],edges[0])
    # add the reversed edges
    new_edges = (th.cat((edges[0],reverse_edges[0])),th.cat((edges[1],reverse_edges[1])))
    udg =  dgl.graph(new_edges,num_nodes=g.num_nodes())

    # copy the node features
    for key, value in g.ndata.items():
        # print(key,value)
        udg.ndata[key] = value
    # copy the edge features
    udg.edata['direction'] = th.cat((th.zeros(size=(1,g.number_of_edges())).squeeze(0),th.ones(size=(1,g.number_of_edges())).squeeze(0)))

    return udg

def get_reverse_graph(g):
    edges = g.edges()
    reverse_edges = (edges[1], edges[0])

    rg = dgl.graph(reverse_edges, num_nodes=g.num_nodes())
    for key, value in g.ndata.items():
        # print(key,value)
        rg.ndata[key] = value
    for key, value in g.edata.items():
        # print(key,value)
        rg.edata[key] = value
    return rg

def type_count(ntypes,count):
    for tp in ntypes:
        tp = tp.item()
        count[tp] +=1

def cal_ratios(count1,count2):
    ratios = []
    for i in range(len(count1)):
        if count2[i] == 0:
            ratios.append(-1)
        else:
            ratio = count1[i] / count2[i]
            ratios.append(round(ratio,4))
    return ratios

def oversample(g,options,in_dim):
    r"""

    oversample the postive nodes when the dataset is imbalanced

    :param g:
        the target graph
    :param options:
        some args
    :param in_dim:
        number of different node types
    :return:
    """
    print("oversampling dataset......")

    print("total number of nodes: ", g.num_nodes())



    if options.label == 'in':
        labels = g.ndata['label_i']
    elif options.label == 'out':
        labels = g.ndata['label_o']

    else:
        print("wrong label type")
        return
    lowbit_mask = g.ndata['position']<=3
    # unlabel the nodes in muldiv
    no_muldiv_mask = labels.squeeze(-1)!=-1
    print('no_mul',len(labels[no_muldiv_mask]))
    nodes = th.tensor(range(g.num_nodes()))
    nodes = nodes[no_muldiv_mask]
    labels = labels[no_muldiv_mask]
    print(len(nodes))

    mask_pos = (labels ==1).squeeze(1)

    mask_neg = (labels == 0).squeeze(1)
    pos_nodes = nodes[mask_pos].numpy().tolist()
    neg_nodes = nodes[mask_neg].numpy().tolist()
    shuffle(pos_nodes)
    shuffle(neg_nodes)
    pos_size = len(pos_nodes)
    neg_size = len(neg_nodes)

    ratio = float(neg_size) / float(pos_size)
    print("ratio=", ratio)


    pos_count = th.zeros(size=(1, in_dim)).squeeze(0).numpy().tolist()
    neg_count = th.zeros(size=(1, in_dim)).squeeze(0).numpy().tolist()
    pos_types = g.ndata['ntype'][pos_nodes]
    neg_types = g.ndata['ntype'][neg_nodes]
    pos_types = th.argmax(pos_types, dim=1)
    neg_types = th.argmax(neg_types, dim=1)
    type_count(pos_types, pos_count)
    type_count(neg_types, neg_count)

    print("train pos count:", pos_count)
    print("train neg count:", neg_count)
    rates = cal_ratios(neg_count, pos_count)
    print(rates)

    train_nodes = pos_nodes.copy()
    train_nodes.extend(neg_nodes)

    ratios = []
    for type in range(in_dim):
        pos_mask = pos_types == type
        neg_mask = neg_types == type
        pos_nodes_n = th.tensor(pos_nodes)[pos_mask].numpy().tolist()
        neg_nodes_n = th.tensor(neg_nodes)[neg_mask].numpy().tolist()

        if len(pos_nodes_n) == 0: ratio = 0
        else: ratio = len(neg_nodes_n) / len(pos_nodes_n)
        ratios.append(ratio)
        if ratio >options.os_rate : ratio = options.os_rate

        if options.balanced and ratio!=0:
            if ratio > 1:
                short_nodes = pos_nodes_n
            else:
                short_nodes = neg_nodes_n
                ratio = 1 / ratio
            short_len = len(short_nodes)
            while ratio > 1:
                shuffle(short_nodes)
                train_nodes.extend(short_nodes[:int(short_len * min(1, ratio - 1))])
                ratio -= 1

    print("ratios:",ratios)
    return train_nodes,pos_count, neg_count




def load_model(device,options):
    r"""
    Load the model

    :param device:
        the target device that the model is loaded on
    :param options:
        some additional parameters
    :return:
        param: new options
        model : loaded model
        mlp: loaded mlp
    """
    model_dir = options.model_saving_dir
    if os.path.exists(os.path.join(model_dir, 'model.pkl')) is False:
        return None,None


    with open(os.path.join(model_dir,'model.pkl'), 'rb') as f:
        param, classifier = pickle.load(f)
        param.model_saving_dir = options.model_saving_dir
        classifier = classifier.to(device)

        # make some changes to the options
        if options.change_lr:
            param.learning_rate = options.learning_rate
        if options.change_alpha:
            param.alpha = options.alpha
    return param,classifier



def test(options):

    th.multiprocessing.set_sharing_strategy('file_system')
    device = th.device("cuda:"+str(options.gpu) if th.cuda.is_available() else "cpu")

    data_path = options.datapath
    print(data_path)
    test_data_file = os.path.join(data_path,'rocket.pkl')

    print(options)
    # load the model
    options, model = load_model(device, options)
    if model is None:
        print("No model, please prepocess first , or choose a pretrain model")
        return
    print(model)

    in_nlayers = options.in_nlayers if isinstance(options.in_nlayers,int) else options.in_nlayers[0]
    out_nlayers = options.out_nlayers if isinstance(options.out_nlayers,int) else options.out_nlayers[0]

    label_name = 'label_o'
    print("Loading data...")
   
    with open(test_data_file,'rb') as f:
        test_g = pickle.load(f)

    if in_nlayers == -1:
        in_nlayers = 0
    if out_nlayers == -1:
        out_nlayers = 0
    in_sampler = Sampler([None] * (in_nlayers + 1), include_dst_in_src=options.include)
    out_sampler = Sampler([None] * (out_nlayers + 1), include_dst_in_src=options.include)

    with open(os.path.join('../models/tp6_new1/ln4_bs1024_7/', 'test_nids.pkl'), 'rb') as f:
        test_nids = pickle.load(f)
    # create dataloader for training/validate dataset
    if options.sage:
        graph_function = DAG2UDG
        out_sampler.include_dst_in_src = True
    else:
        graph_function = get_reverse_graph

    testdataloader = MyNodeDataLoader(
        True,
        test_g,
        graph_function(test_g),
        test_nids,
        in_sampler,
        out_sampler,
        batch_size=test_g.num_nodes(),
        shuffle=True,
        drop_last=False,
    )
    print(len(test_nids))

    beta = options.beta
    # set loss function
    Loss = nn.CrossEntropyLoss()
    # set the optimizer
    optim = th.optim.Adam(
        model.parameters(), options.learning_rate, weight_decay=options.weight_decay
    )
    model.train()
    if model.GCN1 is not None: model.GCN1.train()
    if model.GCN2 is not None: model.GCN2.train()

    print('testing...')
    total_num, total_loss, correct, fn, fp, tn, tp = 0, 0.0, 0, 0, 0, 0, 0
    runtime = 0

    with th.no_grad():
        for ni, (in_blocks, out_blocks) in enumerate(testdataloader):
            start = time()
            in_blocks = [b.to(device) for b in in_blocks]
            out_blocks = [b.to(device) for b in out_blocks]
            # print(out_blocks)
            # get in input features
            if not options.function:
                in_input_features = in_blocks[0].srcdata["ntype"]
                out_input_features = out_blocks[0].srcdata["ntype"]
            else:
                in_input_features = in_blocks[0].srcdata["f_input"]
                out_input_features = out_blocks[0].srcdata["f_input"]
            # the central nodes are the output of the final block
            output_labels = in_blocks[-1].dstdata[label_name].squeeze(1)
            total_num += len(output_labels)
            # predict the labels of central nodes
            label_hat = model(in_blocks, in_input_features, out_blocks, out_input_features)
            pos_prob = nn.functional.softmax(label_hat, 1)[:, 1]
            # adjust the predicted labels based on a given thredshold beta
            pos_prob[pos_prob >= beta] = 1
            pos_prob[pos_prob < beta] = 0
            predict_labels = pos_prob

            end = time()
            runtime += end - start

            # calculate the loss
            val_loss = Loss(label_hat, output_labels)
            total_loss += val_loss.item() * len(output_labels)

            correct += (
                    predict_labels == output_labels
            ).sum().item()

            # count fake negatives (fn), true negatives (tp), true negatives (tn), true postives (tp)
            fn += ((predict_labels == 0) & (output_labels != 0)).sum().item()
            tp += ((predict_labels != 0) & (output_labels != 0)).sum().item()
            tn += ((predict_labels == 0) & (output_labels == 0)).sum().item()
            fp += ((predict_labels != 0) & (output_labels == 0)).sum().item()

    loss = total_loss / total_num
    acc = correct / total_num

    # calculate recall, precision and F1-score
    recall = 0
    precision = 0
    if tp != 0:
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
    F1_score = 0
    if precision != 0 or recall != 0:
        F1_score = 2 * recall * precision / (recall + precision)

    print("\ttp:", tp, " fp:", fp, " fn:", fn, " tn:", tn, " precision:", round(precision, 3))
    print("\tloss:{:.3f}, acc:{:.3f}, recall:{:.3f}, F1 score:{:.3f}".format(loss, acc, recall, F1_score))



if __name__ == "__main__":
    seed = 1234
    # th.set_deterministic(True)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    test(get_options())
