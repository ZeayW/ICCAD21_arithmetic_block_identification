import dgl
import pickle 
import argparse


def get_options(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--files ',nargs='+',type=str)
    options = parser.parse_args(args)

    return options

options = get_options()

all_graphs = []
for file in options.files:
    with open(file,'rb') as f:
        batch_graph = pickle.load(f)
        graphs = dgl.unbatch(batch_graph)
        all_graphs.extend(graphs)

res_graph = dgl.batch(all_graphs)
with open('merge.pkl','wb') as f:
    pickle.dump(res_graph,f)
