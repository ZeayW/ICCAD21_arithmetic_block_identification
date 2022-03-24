#项目描述
	本仓库包含一个电路算术模块识别框架及相关数据集，相关成果已发表在ICCAD21会议上。

## 环境配置
运行环境：Linux

Python环境创建：在根目录下执行如下命令：

		conda env create -f dgl.yaml
Python环境激活：

		conda activate dgl

## 数据集生成
输入：测试电路RTL Design文件（如verilog）

输出：测试电路的有向无环图（DAG）表示形式（dgl graph），并以pkl文件的格式进行存储。

步骤：
1.	切换到目录dc/test，将design文件拷贝到该目录下
2.	在dc/test目录下，修改dc.tcl文件，将第二行替换为 ‘set rtl “FILE” ’，其中FILE为实际设计文件名，同时将第四行替换为‘set top_module “MODULE” ’，其中MODULE为实际顶层模块名。
3.	在dc/test目录下，启动design complier (dc)进行逻辑综合，命令如下：

			./run.sh
4.	切换到目录src，执行如下命令：

		python train.py --val_netlist_path ../dc/test --val_top TOP --datapath PATH1 --preprocess 
其中TOP是测试网表顶层模块名（e.g., BoomCore）。生成的数据集以pkl文件的形式存放在指定路径PATH1下。
	注：仓库中包含两个综合好的网表的例子boom/rocket，在根目录下的压缩包dc.zip中。
	注：仓库中包含已经生成好的训练/测试数据集，在根目录下的压缩包dataset.zip中。
## 训练模型
	输出：训练/验证数据集所在路径
	输出：训练好的模型，以pkl文件的格式存放在指定路径。
	步骤：以预测输出边界的模型为例。
1.	切换到目录src，首先执行如下命令初始化模型/生成数据集（如果还未生成）。其中PATH1是数据集所在路径，PATH2是模型保存路径。

		python train.py --label out --datapath PATH1 --model_saving_dir PATH2 --in_nlayers 3 --out_nlayers 0 –preprocess
2.	然后执行如下命令即可开始训练：

		python train.py --label out --datapath PATH1 --model_saving_dir PATH2
	注：训练脚本有多个参数，可以使用如下命令查看：
		python train.py --help
	
## 测试模型
输出：训练/验证数据集所在路径
输出：算术块边界的预测结果，以pkl格式进行存储，给出预测为边界的cell节点编号列表 (list)。
步骤：以预测输出边界的模型为例。
1.	切换到目录src下，执行如下命令：

		python test.py --datapath PATH1 --model_saving_dir PATH2 –predict_path PATH3
	其中 PATH1是测试数据集所在路径，PATH2是训练好的模型所在路径，PATH3是预测输出路径。仓库中提供两个已经训练好的模型，路径分别为 ”models/in_nl14” 和 “models/out_nl30”，分别对应加法器输入边界预测以及输出边界预测。

