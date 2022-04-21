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
    # unlabel the nodes in muldiv
    no_muldiv_mask = labels.squeeze(-1)!=-1
    nodes = th.tensor(range(g.num_nodes()))
    nodes = nodes[no_muldiv_mask]
    labels = labels[no_muldiv_mask]
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

    return train_nodes,pos_count, neg_count


def preprocess(data_path,device,options):
    r"""

    do some preprocessing work: generate dataset / initialize the model

    :param data_path:
        the path to save the dataset
    :param device:
        the device to load the model
    :param options:
        some additional parameters
    :return:
        no return
    """
    print('----------------Preprocessing----------------')
    if os.path.exists(data_path) is False:
        os.makedirs(data_path)
    train_data_file = os.path.join(data_path, 'train.pkl')
    if options.test_id ==0:
        test_save_file = 'test.pkl'
    else:
        test_save_file = 'test_{}.pkl'.format(options.test_id)
    val_data_file = os.path.join(data_path, test_save_file)
    if not os.path.exists(os.path.join(data_path,'ctype2id.pkl')):
        ctype2id = {"1'b0": 0, "1'b1": 1, 'DFF': 2, 'DFFSSR': 3, 'DFFAS': 4,'NAND': 5, 'AND': 6,
                    'OR': 7, 'DELLN': 8, 'INV': 9, 'NOR': 10, 'XOR': 11, 'MUX': 12, 'XNOR': 13,
                    'MAJ': 14, 'PI': 15, 'NA': 16, 'NR': 17, 'OA': 18, 'AO': 19}
        with open(os.path.join(data_path,'ctype2id.pkl'),'wb') as f:
            pickle.dump(ctype2id,f)

        # with open(train_data_file,'rb') as f:
        #     train_g = pickle.load(f)
        # with open(train_data_file,'wb') as f:
        #     pickle.dump((ctype2id,train_g),f)
        # with open(train_data_file,'rb') as f:
        #     temp,train_g = pickle.load(f)
        #     print(temp,train_g)
        # with open(val_data_file,'rb') as f:
        #     val_g = pickle.load(f)
        # with open(val_data_file,'wb') as f:
        #     pickle.dump((ctype2id,val_g),f)
    else:
        with open(os.path.join(data_path,'ctype2id.pkl'),'rb') as f:
            ctype2id = pickle.load(f)

    if type(options.keywords) == str:
        keywords = [options.keywords]
    else:
        keywords = options.keywords
    # generate and save the test dataset if missing
    if os.path.exists(val_data_file) is False:
        print('Validation dataset does not exist. Generating validation dataset... ')
        datapaths = [os.path.join(options.val_netlist_path,'implementation')]
        report_folders = [os.path.join(options.val_netlist_path,'report')]
        th.multiprocessing.set_sharing_strategy('file_system')
        dataset = Dataset(options.val_top,datapaths,report_folders,ctype2id,options.target_block,keywords)

        ctype2id = dataset.ctype2id
        ntypes = len(ctype2id)
        print(ctype2id)
        with open(os.path.join(data_path,'ctype2id.pkl'),'wb') as f:
            pickle.dump(ctype2id,f)
        g = dataset.batch_graph
        with open(val_data_file, 'wb') as f:
            pickle.dump((ctype2id,g), f)

    print('Validation dataset is ready!')
    # generate and save the train dataset if missing
    if os.path.exists(train_data_file) is False:
        print('Training dataset does not exist. Generating training dataset... ')
        datapaths = [os.path.join(options.train_netlist_path, 'implementation')]
        report_folders = [os.path.join(options.train_netlist_path, 'report')]
        th.multiprocessing.set_sharing_strategy('file_system')
        dataset = Dataset(options.train_top, datapaths, report_folders, ctype2id,options.target_block,options.keywords)

        ctype2id = dataset.ctype2id
        ntypes = len(ctype2id)
        print(ctype2id)
        with open(os.path.join(data_path, 'ctype2id.pkl'), 'wb') as f:
            pickle.dump(ctype2id, f)
        g = dataset.batch_graph
        with open(train_data_file, 'wb') as f:
            pickle.dump((ctype2id, g), f)
    print('Training dataset is ready!')
    print(ctype2id)
    # initialize the bidirectional model
    print('Intializing models...')
    network = ABGNN

    # initialize the model for collecting fanin information
    # options.in_nlayers gives the number of graph convolution layers for fanin model.
    # options.in_nlayers = 0 means no fanin model is used (that we only collect information from fanout direction)
    if options.in_nlayers!=0:
        model1 = network(
            ntypes = len(ctype2id),
            hidden_dim=options.hidden_dim,
            out_dim=options.out_dim,
            n_layers = options.in_nlayers,
            in_dim = len(ctype2id),
            dropout=options.gcn_dropout,
        )
        out_dim1 = model1.out_dim
    else:
        model1 = None
        out_dim1 = 0

    # initialize the model for collecting fanout information
    # options.out_nlayers gives the number of graph convolution layers for fanout model.
    # options.out_nlayers = 0 means no fanout model is used (that we only collect information from fanin direction)
    if options.out_nlayers!=0:
        model2 = network(
            ntypes=len(ctype2id),
            hidden_dim=options.hidden_dim,
            out_dim=options.out_dim,
            n_layers=options.out_nlayers,
            in_dim=len(ctype2id),
            dropout=options.gcn_dropout,
        )
        out_dim2 = model2.out_dim
    else:
        model2 = None
        out_dim2 =0

    # options.out_nlayers and options.in_nlayers can not be zero at the same time
    assert model1 is not None or model2 is not None

    # we feed the output of ABGNN to a multlayer perceptron (MLP) to make prediction based on the node embeddings generate by ABGNN
    # Here is the code to initialze the MLP
    mlp = MLP(
        in_dim = out_dim1+out_dim2,
        out_dim = options.nlabels,
        nlayers = options.n_fcn,
        dropout = options.mlp_dropout
    ).to(device)

    # The whole model is composed of 2 GNN models (one for fanin direction, and one for fanout), and 1 MLP
    classifier = BiClassifier(model1,model2,mlp)
    print(classifier)
    print("creating model in:",options.model_saving_dir)

    # save the model and create a file to save the results
    if os.path.exists(options.model_saving_dir) is False:
        os.makedirs(options.model_saving_dir)
        with open(os.path.join(options.model_saving_dir, 'model.pkl'), 'wb') as f:
            parameters = options
            pickle.dump((parameters, classifier), f)
        with open(os.path.join(options.model_saving_dir, 'res.txt'), 'w') as f:
            pass
    print('Preprocessing is accomplished!')

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
    print('----------------Loading the model and hyper-parameters----------------')
    model_dir = options.model_saving_dir
    # if there is no model in the target directory, break
    if os.path.exists(os.path.join(model_dir, 'model.pkl')) is False:
        print("No model, please prepocess first , or choose a pretrain model")
        assert False

    # read the pkl file that saves the hype-parameters and the model.
    with open(os.path.join(model_dir,'model.pkl'), 'rb') as f:
        # param: hyper-parameters, e.g., learning rate;
        # classifier: the model
        param, classifier = pickle.load(f)
        param.model_saving_dir = options.model_saving_dir
        classifier = classifier.to(device)

        # make some changes to the options
        if options.change_lr:
            param.learning_rate = options.learning_rate
        if options.change_alpha:
            param.alpha = options.alpha
    print('Model and hyper-parameters successfully loaded!')
    return param,classifier



def validate(loader,label_name,device,model,Loss,beta,options):
    r"""

    validate the model

    :param loader:
        the data loader to load the validation dataset
    :param label_name:
        target label name
    :param device:
        device
    :param model:
        trained model
    :param mlp:
        trained mlp
    :param Loss:
        used loss function
    :param beta:
        a hyperparameter that determines the thredshold of binary classification
    :param options:
        some parameters
    :return:
        result of the validation: loss, acc,recall,precision,F1_score
    """

    total_num, total_loss, correct, fn, fp, tn, tp = 0, 0.0, 0, 0, 0, 0, 0
    runtime = 0

    with th.no_grad():
        # load validation data, one batch at a time
        # each time we sample some central nodes, together with their input neighborhoods (in_blocks) \
        # and output neighborhoods (out_block).
        # The dst_nodes of the last block of in_block/out_block is the central nodes.
        for ni, (in_blocks, out_blocks) in enumerate(loader):
            start = time()
            # transfer the data to GPU
            in_blocks = [b.to(device) for b in in_blocks]
            out_blocks = [b.to(device) for b in out_blocks]

            # get features
            in_features = in_blocks[0].srcdata["ntype"]
            out_features = out_blocks[0].srcdata["ntype"]
            # the central nodes are the dst_nodes of the final block
            output_labels = in_blocks[-1].dstdata[label_name].squeeze(1)
            total_num += len(output_labels)
            # predict the labels of central nodes
            label_hat = model(in_blocks, in_features, out_blocks, out_features)
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

            # count for the correctly predicted samples
            correct += (
                    predict_labels == output_labels
            ).sum().item()

            # count fake negatives (fn), true negatives (tp), true negatives (tn), true postives (tp)
            fn += ((predict_labels == 0) & (output_labels != 0)).sum().item()
            tp += ((predict_labels != 0) & (output_labels != 0)).sum().item()
            tn += ((predict_labels == 0) & (output_labels == 0)).sum().item()
            fp += ((predict_labels != 0) & (output_labels == 0)).sum().item()

    # calculate the overall loss / accuracy
    loss = total_loss / total_num
    acc = correct / total_num

    # calculate overall recall, precision and F1-score
    recall = 0
    precision = 0
    if tp != 0:
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
    F1_score = 0
    if precision != 0 or recall != 0:
        F1_score = 2 * recall * precision / (recall + precision)

    print("\ttp:", tp, " fp:", fp, " fn:", fn, " tn:", tn, " precision:", round(precision, 3))
    print("\tloss:{:.3f}, acc:{:.3f}, recall:{:.3f}, F1 score:{:.3f}".format(loss, acc,recall, F1_score))

    return [loss, acc,recall,precision,F1_score]


def load_data(data_path,latest_ctype2id_file):
    with open(latest_ctype2id_file, 'rb') as f:
        latest_ctype2id = pickle.load(f)
    with open(data_path,'rb') as f:
        graph_ctype2id,graph = pickle.load(f)
        graph_id2ctype = {}
        for key,value in graph_ctype2id.items():
            graph_id2ctype[value] = key
        ntypes = len(graph_ctype2id)
        graph_ntypes = graph.ndata['ntype'].shape[1]

    latest_ntypes = len(latest_ctype2id)

    if graph_ntypes < latest_ntypes:
        for n in graph.nodes():
            type_id = th.argmax(graph.ndata['ntype'][n])
            type = graph_id2ctype[type_id.item()]
            if latest_ctype2id.get(type,None) is None:
                latest_ctype2id[type] = len(latest_ctype2id)
                print('unknown cell type, extend the ctype2id!')
        updated_ntype = th.zeros((graph.number_of_nodes(), len(latest_ctype2id)), dtype=th.float)
        for n in graph.nodes():
            type_id = th.argmax(graph.ndata['ntype'][n])
            type = graph_id2ctype[type_id.item()]
            type_id = latest_ctype2id[type]
            updated_ntype[n][type_id] = 1
        graph.ndata['ntype'] = updated_ntype
        with open(data_path, 'wb') as f:
            pickle.dump((latest_ctype2id,graph),f)
    elif graph_ntypes>ntypes:
        assert False, 'too many cell types!'
    print(latest_ctype2id)
    with open(latest_ctype2id_file, 'wb') as f:
        pickle.dump(latest_ctype2id,f)

    in_dim = len(latest_ctype2id)
    return graph,in_dim
def train(options):

    th.multiprocessing.set_sharing_strategy('file_system')
    device = th.device("cuda:"+str(options.gpu) if th.cuda.is_available() else "cpu")

    # you can define your dataset file here
    data_path = options.datapath
    print(data_path)
    ctype2id_file = os.path.join(data_path,'ctype2id.pkl')
    train_data_file = os.path.join(data_path,'train.pkl')
    val_data_file = os.path.join(data_path,'test.pkl')

    # preprocess: generate dataset / initialize the model
    if options.preprocess :
        preprocess(data_path,device,options)
        return

    # load the model
    options, model = load_model(device, options)
    print('Hyperparameters are listed as follows:')
    print(options)
    print('The model architecture is shown as follow:')
    print(model)
    in_nlayers = options.in_nlayers if isinstance(options.in_nlayers,int) else options.in_nlayers[0]
    out_nlayers = options.out_nlayers if isinstance(options.out_nlayers,int) else options.out_nlayers[0]

    if options.label =='in':
        label_name = 'label_i'
    elif options.label == 'out':
        label_name = 'label_o'
    else:
        print('Error: wrong label!')
        exit()
    print("----------------Loading data----------------")
    if not os.path.exists(ctype2id_file) :
        assert False, 'No ctype2id file! Please run the data generating procedure or copy a existed ctype2id file to the data path!'

    print('train ctypes:')
    train_g,in_dim = load_data(train_data_file,ctype2id_file)
    print('test ctypes:')
    val_g,in_dim = load_data(val_data_file,ctype2id_file)
    print('Data successfully loaded!')

    # apply the over-samplying strategy to deal with data imbalance
    train_nodes, pos_count, neg_count = oversample(train_g, options, in_dim)

    if options.label == 'in':
        labels = val_g.ndata['label_i']
    elif options.label == 'out':
        labels = val_g.ndata['label_o']
    no_muldiv_mask = labels.squeeze(-1) != -1
    nodes = th.tensor(range(val_g.num_nodes()))
    nodes = nodes[no_muldiv_mask]
    labels = labels[no_muldiv_mask]
    mask_pos = (labels == 1).squeeze(1)
    mask_neg = (labels == 0).squeeze(1)
    pos_nodes = nodes[mask_pos].numpy().tolist()
    neg_nodes = nodes[mask_neg].numpy().tolist()
    shuffle(pos_nodes)
    shuffle(neg_nodes)
    pos_count = th.zeros(size=(1, in_dim)).squeeze(0).numpy().tolist()
    neg_count = th.zeros(size=(1, in_dim)).squeeze(0).numpy().tolist()
    pos_types = val_g.ndata['ntype'][pos_nodes]
    neg_types = val_g.ndata['ntype'][neg_nodes]
    pos_types = th.argmax(pos_types, dim=1)
    neg_types = th.argmax(neg_types, dim=1)
    type_count(pos_types, pos_count)
    type_count(neg_types, neg_count)
    print("test pos count:", pos_count)
    print("test neg count:", neg_count)

    # initialize the data sampler
    in_nlayers = max(1,in_nlayers)
    out_nlayers = max(1,out_nlayers)
    in_sampler = Sampler([None] * in_nlayers, include_dst_in_src=False)
    out_sampler = Sampler([None] * out_nlayers, include_dst_in_src=False)

    # split the validation set and test set
    if os.path.exists(os.path.join(options.datapath, 'val_nids.pkl')):
        with open(os.path.join(options.datapath, 'val_nids.pkl'), 'rb') as f:
            val_nids = pickle.load(f)
        with open(os.path.join(options.datapath, 'test_nids.pkl'), 'rb') as f:
            test_nids = pickle.load(f)
    else:
        val_nids = th.tensor(range(val_g.number_of_nodes()))
        val_nids = val_nids[val_g.ndata['label_o'].squeeze(-1) != -1]
        val_nids1 = val_nids.numpy().tolist()
        shuffle(val_nids1)
        val_nids = val_nids1[:int(len(val_nids1) / 10)]
        test_nids = val_nids1[int(len(val_nids1) / 10):]

        with open(os.path.join(options.datapath, 'val_nids.pkl'), 'wb') as f:
            pickle.dump(val_nids, f)
        with open(os.path.join(options.datapath, 'test_nids.pkl'), 'wb') as f:
            pickle.dump(test_nids, f)

    # create dataloader for training/validate dataset
    graph_function = get_reverse_graph

    # initialize the dataloaders
    traindataloader = MyNodeDataLoader(
        False,
        train_g,
        graph_function(train_g),
        train_nodes,
        in_sampler,
        out_sampler,
        batch_size=options.batch_size,
        shuffle=True,
        drop_last=False,
    )
    valdataloader = MyNodeDataLoader(
        True,
        val_g,
        graph_function(val_g),
        val_nids,
        in_sampler,
        out_sampler,
        batch_size=val_g.num_nodes(),
        shuffle=True,
        drop_last=False,
    )
    testdataloader = MyNodeDataLoader(
        True,
        val_g,
        graph_function(val_g),
        test_nids,
        in_sampler,
        out_sampler,
        batch_size=val_g.num_nodes(),
        shuffle=True,
        drop_last=False,
    )

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


    print("----------------Start training---------------")
    pre_loss = 100
    stop_score = 0
    max_recall = 0

    # start training
    for epoch in range(options.num_epoch):
        runtime = 0

        total_num,total_loss,correct,fn,fp,tn,tp = 0,0.0,0,0,0,0,0
        pos_count , neg_count =0, 0
        pos_embeddings= th.tensor([]).to(device)
        # each time we sample some central nodes, together with their input neighborhoods (in_blocks) \
        # and output neighborhoods (out_block).
        # The dst_nodes of the last block of in_block/out_block is the cent
        for ni, (in_blocks,out_blocks) in enumerate(traindataloader):
            if ni == len(traindataloader)-1:
                continue
            start_time = time()

            # transfer the data to GPU
            in_blocks = [b.to(device) for b in in_blocks]
            out_blocks = [b.to(device) for b in out_blocks]
            # get features
            in_features = in_blocks[0].srcdata["ntype"]

            out_features = out_blocks[0].srcdata["ntype"]
            # the central nodes are the dst_nodes of the final block
            output_labels = in_blocks[-1].dstdata[label_name].squeeze(1)
            total_num += len(output_labels)
            # predict the labels of central nodes
            label_hat = model(in_blocks,in_features,out_blocks,out_features)

            if get_options().nlabels != 1:
                pos_prob = nn.functional.softmax(label_hat, 1)[:, 1]
            else:
                pos_prob = th.sigmoid(label_hat)
            # adjust the predicted labels based on a given thredshold beta
            pos_prob[pos_prob >= beta] = 1
            pos_prob[pos_prob < beta] = 0
            predict_labels = pos_prob

            # calculate the loss
            train_loss = Loss(label_hat, output_labels)
            total_loss += train_loss.item() * len(output_labels)
            endtime = time()
            runtime += endtime - start_time
            # count the correctly predicted samples
            correct += (
                    predict_labels == output_labels
            ).sum().item()

            # count fake negatives (fn), true negatives (tp), true negatives (tn), true post
            fn += ((predict_labels == 0) & (output_labels != 0)).sum().item()
            tp += ((predict_labels != 0) & (output_labels != 0)).sum().item()
            tn += ((predict_labels == 0) & (output_labels == 0)).sum().item()
            fp += ((predict_labels != 0) & (output_labels == 0)).sum().item()

            start_time = time()
            # back propagation
            optim.zero_grad()
            train_loss.backward()
            optim.step()
            endtime = time()
            runtime += endtime-start_time

        Train_loss = total_loss / total_num


        # calculate accuracy, recall, precision and F1-score
        Train_acc = correct / total_num
        Train_recall = 0
        Train_precision = 0
        if tp != 0:
            Train_recall = tp / (tp + fn)
            Train_precision = tp / (tp + fp)
        Train_F1_score = 0
        if Train_precision != 0 or Train_recall != 0:
            Train_F1_score = 2 * Train_recall * Train_precision / (Train_recall + Train_precision)

        print("epoch[{:d}]".format(epoch))
        print("training runtime: ",runtime)
        print("  train:")
        print("\ttp:", tp, " fp:", fp, " fn:", fn, " tn:", tn, " precision:", round(Train_precision,3))
        print("\tloss:{:.8f}, acc:{:.3f}, recall:{:.3f}, F1 score:{:.3f}".format(Train_loss,Train_acc,Train_recall,Train_F1_score))

        # validate
        print("  validate:")
        val_loss, val_acc, val_recall, val_precision, val_F1_score = validate(valdataloader,label_name, device, model,
                                                                       Loss, beta,options)
        print("  test:")
        validate(testdataloader, label_name, device, model,
                 Loss, beta, options)

        # save the result of current epoch
        with open(os.path.join(options.model_saving_dir, 'res.txt'), 'a') as f:
            f.write(str(round(Train_loss, 8)) + " " + str(round(Train_acc, 3)) + " " + str(
                round(Train_recall, 3)) + " " + str(round(Train_precision,3))+" " + str(round(Train_F1_score, 3)) + "\n")
            f.write(str(round(val_loss, 3)) + " " + str(round(val_acc, 3)) + " " + str(
                round(val_recall, 3)) + " "+ str(round(val_precision,3))+" " + str(round(val_F1_score, 3)) + "\n")
            f.write('\n')

        judgement = val_recall > max_recall
        if judgement:
           stop_score = 0
           max_recall = val_recall
           print("Saving model.... ", os.path.join(options.model_saving_dir))
           if os.path.exists(options.model_saving_dir) is False:
              os.makedirs(options.model_saving_dir)
           with open(os.path.join(options.model_saving_dir, 'model.pkl'), 'wb') as f:
              parameters = options
              pickle.dump((parameters, model), f)
           print("Model successfully saved")
        else:
            stop_score += 1
            if stop_score >= 5:
                print('Early Stop!')
                exit()




if __name__ == "__main__":
    seed = 1234
    # th.set_deterministic(True)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    train(get_options())
