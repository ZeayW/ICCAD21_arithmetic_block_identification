import argparse


def get_options(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, help = 'the learning rate for training. Type: float.',default=1e-3)
    parser.add_argument("--batch_size", type=int, help = 'the number of samples in each training batch. Type: int',default=1024)
    parser.add_argument("--num_epoch", type=float, help='Type: int; number of epoches that the training procedure runs. Type: int',default=1500)
    parser.add_argument("--in_dim", type=int, help='the dimension of the input feature. Type: int',default=9)
    parser.add_argument("--out_dim", type=int, help='the dimension of the output embedding. Type: int',default=256)
    parser.add_argument("--hidden_dim", type=int, help='the dimension of the intermediate GNN layers. Type: int',default=256)
    parser.add_argument("--out_nlayers", type=int,help='number of GNN layers for the fanout direction. Type: int',default=2)
    parser.add_argument("--in_nlayers", type=int,help='number of GNN layers for the fanin direction. Type: int',default=2)
    parser.add_argument("--gcn_dropout", type=float,help='dropout rate for GNN layers. Type: float', default=0)
    parser.add_argument("--mlp_dropout", type=float, help='dropout rate for mlp. Type: float',default=0)
    parser.add_argument("--weight_decay", type=float, help='weight decay. Type: float',default=0)
    parser.add_argument("--model_saving_dir", type=str, help='the directory to save the trained model. Type: str',default='../models/example')
    parser.add_argument("--preprocess",help='decide whether to run the preprocess procedure or not. If set True, then a preprocess procedure'
                                            ' (generating dataset + initialize model)will be carried out; '
                                            'Else a normal training procedure will be carried out.'
                                            'Type: sote_true',action='store_true')
    parser.add_argument("--n_fcn",type=int,help='the number of full connected layers of the mlp. Type: int',default=3)
    parser.add_argument("--alpha",type=float,help='the weight of the cost-sensitive learning. Type: float',default=1.0)
    parser.add_argument("--change_lr",help='Decide to change to learning rate. Type: float',action='store_true')
    parser.add_argument("--change_alpha",help='Decide to change alpha. Type: float',action='store_true')
    parser.add_argument("--gpu",type=int,help='index of gpu. Type: int',default=0)
    parser.add_argument('--balanced',action='store_true',help = 'decide whether to balance the training dataset (using oversampling) or not; Type: store_true')
    parser.add_argument('--label',type=str,help='The target label. Valid values: ["in","out"]. Type: str',default='in')
    parser.add_argument('--nlabels',type=int,help='number of prediction classes. Type: int',default=2)
    parser.add_argument('--os_rate',help='the oversampling rate. Type: int',type=int,default=20)
    parser.add_argument('--beta',type=float,default=0.5,help='choose the threshold for binary classification to make a trade-off between recall and precision. Type: float')
    parser.add_argument('--datapath',type=str,help='the directory that contains the dataset. Type: str',default='../dataset')
    parser.add_argument('--val_netlist_path',type=str,help='the directory that contains the validation netlists. Type: str',default='../dc/rocket')
    parser.add_argument('--train_netlist_path',type=str,help='the directory that contains the training netlists. Type: str',default='../dc/boom')
    parser.add_argument('--val_top',type=str,help='the top module name of the validation netlist. Type: str',default='Rocket')
    parser.add_argument('--train_top',type=str, help='the top module name of the training netlist. Type: str',default='BoomCore')
    parser.add_argument('--predict_path',type=str,help='the directory used to save the prediction result. Type: str',default='../prediction/example')
    parser.add_argument('--target_block',type=str,help='the target block to label. Type: str',default='add')
    parser.add_argument('--keywords',type=str, nargs='+',help='the instance keyword of target block. Type: list(str)',default='add_x')
    parser.add_argument('--test_id',type=int,default=0)
    options = parser.parse_args(args)

    return options
