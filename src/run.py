import sys
import os

labels = ['out','in']
lr_rates = [1e-3, 3e-3, 3e-4]
in_nlayers = [0,1,2,3,4,5]
out_nlayers = [0,1,2,3,4,5]

os_params = {'':1,'--balanced --os_rate 5':5}

datapath = sys.argv[1]
model_saving_dir = sys.argv[2]
for lr_rate in lr_rates:
    for in_nlayer in in_nlayers:
        for out_nlayer in out_nlayers:
            if in_nlayer==0 and out_nlayer==0:
                continue
            for label in labels:
                for os_param in os_params.keys():
                    prefix = '{}/lr{}_in{}_out{}_os{}'.format(label,lr_rate,in_nlayer,out_nlayer,os_params[os_param])
                    model_saving_path = os.path.join(model_saving_dir,prefix)
                    os.system("python train.py --label {} --datapath {} --model_saving_dir {} --in_nlayers {} --out_nlayers {} {} --preprocess"
                            .format(label,datapath,model_saving_path,in_nlayer,out_nlayer, os_param))
                    os.system(
                        "python train.py --label {} --datapath {} --model_saving_dir {} --in_nlayers {} --out_nlayers {} {}"
                        .format(label, datapath, model_saving_path, in_nlayer, out_nlayer, os_param))
