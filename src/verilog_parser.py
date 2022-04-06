import os
import collections

from typing import List, Dict, Tuple, Optional
import pyverilog
from pyverilog.vparser.parser import parse
import re

import networkx as nx

import cProfile

def parse_arg(arg,port_info,ios,wires):
    r"""

    parse the information of an arg

    :param arg:
        the arg
    :param port_info: PortInfo
        the port that the arg belongs to
    :param ios:
        io information of the current top module
    :param wires:
        wire information of current top module
    :return:

    """

    # identifier, e.g., a
    if type(arg) == pyverilog.vparser.ast.Identifier:
        if wires.get(arg.name,None) is not None:
            high_bit, low_bit = wires[arg.name]
        # if the arg is an io of the current top module, then it need chain update latter
        elif ios.get(arg.name,None) is not None:
            high_bit, low_bit = ios[arg.name]
            port_info.flag_update = True
            port_info.args_need_update.add(arg.name)
        else:
            assert False

        # add the current arg to the port's arg_list
        width = high_bit-low_bit+1
        if width == 1:
            port_info.arg_list.append(arg.name)
        else:
            for i in range(high_bit,low_bit-1,-1):
                port_info.arg_list.append("{}_{}".format(arg,i))
    # const, e.g., 1'b0
    elif type(arg) == pyverilog.vparser.ast.IntConst:
        port_info.arg_list.append(arg.value)
    # parselect, e.g., a[n1:n2]
    elif type(arg) == pyverilog.vparser.ast.Partselect:
        arg_nm,high_bit,low_bit = arg.children()
        arg_nm = arg_nm.name
        # get the highest/lowest bit
        high_bit, low_bit = int(str(high_bit)),int(str(low_bit))
        if high_bit < low_bit:
            temp = high_bit
            high_bit = low_bit
            low_bit = temp
        # add the arg to arglist
        for i in range(high_bit,low_bit-1,-1):
            port_info.arg_list.append("{}_{}".format(arg_nm,i))
        if ios.get(arg_nm,None) is not None:
            port_info.flag_update = True
            port_info.args_need_update.add(arg_nm)
    # pointer, e.g., a[n]
    elif type(arg) == pyverilog.vparser.ast.Pointer:
        arg_nm, position = arg.children()
        arg_nm = arg_nm.name
        port_info.arg_list.append("{}_{}".format(arg_nm,position))
        if ios.get(arg_nm,None) is not None:
            port_info.flag_update = True
            port_info.args_need_update.add(arg_nm)
    else:
        print(arg)
        assert False

class ModuleInfo:

    cell_name:str   #
    cell_type:str
    instance_name:str
    ports:dict
    index:int
    def __init__(self,cell_name,cell_type,instance_name):
        self.cell_name = cell_name
        self.cell_type = cell_type
        self.instance_name = instance_name
        self.ports = {}
        self.index = -1

class PortInfo:
    ptype:str
    portname: str
    argname: str
    argcomp: str
    is_output: bool
    is_input: bool

    input_comp: str
    output_comp: str
    arg_list: list

    flag_update:bool
    args_need_update:set

    def __init__(self, portname,argname, argcomp):
        self.ptype = None
        self.portname = portname
        self.argname = argname
        self.argcomp = argcomp
        self.arg_list = []
        self.flag_update = False
        self.args_need_update = set()

class DcParser:
    def __init__(
        self, top_module: str, target_block,keywords: List[str]
    ):
        self.top_module = top_module
        self.target_block = target_block
        self.keywords = keywords

    def is_input_port(self, port: str) -> bool:
        return not self.is_output_port(port)

    def is_output_port(self, port: str) -> bool:
        return port in ("Y", "S", "SO", "CO", "C1", "Q", "QN")

    def parse_report(self,fname):
        print('\t###  parsing the report file...')
        r"""

        parse the sythesis report to find information about the target arithmetic blocks (cells)

        here gives the information of an example block in the report:
            ****************************************
            Design : MulAddRecFNToRaw_preMul
            ****************************************

            ......

            Datapath Report for DP_OP_279J57_124_314
            ==============================================================================
            | Cell                 | Contained Operations                                |
            ==============================================================================
            | DP_OP_279J57_124_314 | mult_292515 add_292517                              |
            ==============================================================================

            ==============================================================================
            |       |      | Data     |       |                                          |
            | Var   | Type | Class    | Width | Expression                               |
            ==============================================================================
            | I1    | PI   | Signed   | 9     |                                          |
            | I2    | PI   | Signed   | 65    |                                          |
            | I3    | PI   | Signed   | 65    |                                          |
            | T7    | IFO  | Signed   | 73    | I1 * I2                                  |
            | O1    | PO   | Signed   | 73    | T7 + I3                                  |
            ==============================================================================

            Implementation Report
            ......

        :param fname: str
            the path of the report file
        :return:
            dp_target_blocks:
                {block_name:(block_type,{input_port:position},{output_port:position})}
        """
        if not os.path.exists(fname):
            print('\tError: report file doest not exist!')
            exit()
        with open(fname,'r') as f:
            text = f.read()
        print('\treport file is read.')
        blocks  = text.split('Datapath Report for')
        blocks = blocks[1:]
        dp_target_blocks = {}
        print('\tscanning the report file to find target blocks...')
        for block in blocks:
            block = block.split('Implementation Report')[0]
            block = block[:block.rfind('\n==============================================================================')]
            block_name = block.split('\n')[0].replace(' ','')
            if '*' in block and '+' in block:
                continue
            vars = block[block.rfind('=============================================================================='):]
            vars = vars.split('\n')[1:] # the vars in the datapath report, e.g., | I1    | PI   | Signed   | 9     |                                          |

            var_types = {}   # record the port type of the vars, valid types include: PI, PO, IFO
            for var in vars:
                var = var.replace(' ','')
                _,var_name,type,data_class,width,expression,_ =var.split('|')
                var_types[var_name] = (type)
                # find a multiply operation
                if self.target_block == 'mul' and '*' in expression:
                    dp_target_blocks[block_name] = dp_target_blocks.get(block_name, ('mul', {}, {}))
                    # get the operants (inputs)
                    operants = expression.split('*')
                    for operant in operants:
                        dp_target_blocks[block_name][1][operant] = 2
                # find an add operation
                if self.target_block == 'add' and '+' in expression and '-' not in expression:
                    dp_target_blocks[block_name] = dp_target_blocks.get(block_name, ('add', {}, {}))
                    # record the output port of the block
                    dp_target_blocks[block_name][2][var_name] = 1
                    # get the operants
                    operants = expression.split('+')
                    for operant in operants:
                        dp_target_blocks[block_name][1][operant] = 1
                # find a subtract operation
                if  self.target_block == 'sub' and '-' in expression and '+' not in expression:
                    dp_target_blocks[block_name] = dp_target_blocks.get(block_name, ('sub', {}, {}))
                    # record the output port of the block
                    dp_target_blocks[block_name][2][var_name] = 1
                    # get the operants
                    operants = expression.split('-')
                    for i,operant in enumerate(operants):
                        dp_target_blocks[block_name][1][operant] = 1 if i==0 else 2
        print('\tscanning is done! Found target blocks in report file: ',dp_target_blocks)
        return dp_target_blocks

    def parse_port_hier(
        self, ios:dict,wires:dict, port: pyverilog.vparser.parser.Portlist,
    ) -> PortInfo:
        r"""

        parse a given port

        :param ios: dict
            io information of the modules
        :param wires: dict
            wires information
        :param port: pyverilog.vparser.parser.Portlist
            port
        :return:
            portInfo: PortInfo
                the information of the port
        """

        portname, argname = port.portname, port.argname
        port_info = PortInfo(portname,None,None)

        # if the arg is concated (in the form of {a,b})
        if type(argname) == pyverilog.vparser.ast.Concat:
            args = argname.children()
            for arg in args:
                parse_arg(arg,port_info,ios,wires)
        else:
            parse_arg(argname,port_info,ios,wires)

        return port_info

    def parse_port(
        self, mcomp: str,port: pyverilog.vparser.parser.Portlist,index01:list,dp_inputs:list,dp_outputs:list
    ) -> PortInfo:
        r"""

        :param mcomp: str
            the module of the port
        :param target_cells: list
            target blocks information
        :param port: port
        :param index01: list
            the index of 1'b0 and 1'b1
        :param dp_inputs: list
        :param dp_outputs: list
        :return:
            port_info: PortInfo
                the information of the port
        """
        portname, argname = port.portname, port.argname
        if type(argname) == pyverilog.vparser.ast.Partselect:
            print(argname)

        if type(argname) == pyverilog.vparser.ast.Pointer:
            argname = str(argname.var) + "_" + str(argname.ptr)
        elif type(argname) == pyverilog.vparser.ast.IntConst:
            argname = argname.__str__()
        else:  # just to show there could be various types!
            argname = argname.__str__()
        argcomp = argname[: argname.rfind("_")]

        if argname == "1'b0" :
            argname = "{}_{}".format(argname,index01[0])
            index01[0] += 1
        elif argname =="1'b1":
            argname = "{}_{}".format(argname, index01[1])
            index01[1] += 1

        port_info = PortInfo(portname, argname, argcomp)

        if portname in ("CLK"):  # clock
            port_info.ptype = "CLK"
            return port_info
        elif self.is_output_port(portname):
            port_info.ptype = "fanout"
        else:
            port_info.ptype = "fanin"

        is_target = False
        for kw in self.keywords:
            if kw in mcomp :
                is_target = True
                break
        if len(dp_inputs)!=0 or len(dp_outputs)!=0:
            is_target = True

        # label the io of target blocks
        if is_target and mcomp != argcomp:
            module_ports = None
            # for cases that instance_name is not unique, e.g, have several add_x_1，each is instance of different cell,
            # in theses cases, mcomp contains both cell information and instance information
            cell_type = None

            # label the ouput wires
            if self.is_output_port(portname):
                port_info.is_output = True
                port_info.output_comp = mcomp
            # label the input wires
            else:
                port_info.is_input = True
                port_info.input_comp = mcomp

        elif is_target and argcomp != mcomp:
            assert False

        return port_info

    def parse_nonhier(self, fname,dp_target_blocks):
        r"""

        parse the non-hierarchical netlist with block information extracted from report and hier_netlist

        :param fname: str
            nonhier netlist filepath
        :param dp_target_blocks: dict
            the information of some target arithmetic blocks extraced from the hier_netlist
        :param target_blocks:  dict
            the information of all the target arithmetic blocks extraced from the report

        :return:
            nodes: list
                the labeled nodes of the transformed DAG
            edges:  list
                the edges of the transformed DAG
        """
        print('\t###  parsing the flatten netlist...')
        nodes: List[Tuple[str, Dict[str, str]]] = [
            ("1'b0", {"type": "1'b0"}),
            ("1'b1", {"type": "1'b1"}),
        ]  # a list of (node, {"type": type})
        edges: List[
            Tuple[str, str, Dict[str, bool]]
        ] = []  # a list of (src, dst, {"is_reverted": is_reverted})
        print('\tgenerating the abstract syntax tree...')
        ast, directives = parse([fname])
        index01 = [0,0]
        inputs = set()
        outputs = set()

        buff_replace = {}
        top_module = None

        print('\tsearching for the top module...')
        for module in ast.description.definitions:
            if module.name == self.top_module:
                top_module = module
                break
        assert top_module is not None, "top module {} not found".format(self.top_module)
        # parse the information of each cell/block
        print('\tsequentially parsing the modules in the generated abstract syntax tree label the ios of the target blocks...')
        for item in top_module.items:
            if type(item) != pyverilog.vparser.ast.InstanceList:
                continue
            instance = item.instances[0]

            # we extract the following parts:
            # mcell: cell name in SAED, e.g. AND2X1
            # mtype: cell type with input shape, e.g. AND2
            # mfunc: cell function, e.g. AND
            # mname: module name, e.g. ALU_DP_OP_J23_U233
            # mcomp: module component, e.g. ALU_DP_OP_J23
            mcell = instance.module  # e.g. AND2X1
            mname = instance.name
            ports = instance.portlist
            mtype = mcell[0 : mcell.rfind("X")]  # e.g. AND2
            mfunc = mtype  # e.g. AND
            mcomp = mname[: mname.rfind("_")]
            if mcell.startswith("SNPS_CLOCK") or mcell.startswith("PlusArgTimeout"):
                continue

            # fanins / fanouts the the cell
            fanins: List[PortInfo] = []
            fanouts: List[PortInfo] = []

            dp_inputs,dp_outputs = [],[]

            # judge if the current cell a target
            for dp_cell in dp_target_blocks.keys():
                if dp_cell in mcomp:
                    dp_inputs = dp_target_blocks[dp_cell][1]
                    dp_outputs = dp_target_blocks[dp_cell][2]
                    break

            # parse the port information
            for p in ports:
                port_info = self.parse_port(mcomp, p,index01,dp_inputs,dp_outputs)
                if port_info.ptype == "fanin":
                    fanins.append(port_info)
                elif port_info.ptype == "fanout":
                    fanouts.append(port_info)

                if port_info.is_output:
                    outputs.add(port_info.argname)
                if port_info.is_input:
                    inputs.add(port_info.argname)
            if not fanouts:
                item.show()
                print("***** warning, the above gate has no fanout recognized! *****")
                # do not assert, because some gates indeed have no fanout...
                # assert False, "no fanout recognized"
            inputs = {}

            for fo in fanouts:
                # the nodes are the fanouts of cells
                # do some replacement, replace some of the cell to some fix cell type, e.g., AO221 -> AND + OR
                if mfunc == "HADD":
                    if fo.portname == "SO":
                        ntype = 'XOR'
                    elif fo.portname == "C1":
                        ntype = 'AND'
                    else:
                        print(fo.portname)
                        assert False
                elif mfunc == "FADD":
                    if fo.portname == "S":
                        ntype = 'XOR'
                    elif fo.portname == "CO":
                        ntype = 'MAJ'
                    else:
                        print(fo.portname)
                        assert False
                else:
                    ntype = mfunc
                if 'DFF' in ntype and ntype not in ['DFF','DFFSSR','DFFAS']:
                    ntype = 'DFF'
                if 'AO' in ntype or 'OA' in ntype:

                    num_inputs = ntype[re.search('\d',ntype).start():]
                    ntype1 = 'AND' if 'AO' in ntype else 'OR'
                    ntype2 = 'OR' if 'AO' in ntype else 'AND'
                    if 'I' in ntype:
                        output_name = '{}_i'.format(fo.argname)
                        nodes.append((output_name,{"type":ntype2}))
                        nodes.append((fo.argname, {"type": 'INV'}))
                        inputs[fo.argname] = [output_name]
                    else:
                        output_name = fo.argname
                        nodes.append((output_name,{"type":ntype2}))
                    inputs[output_name] = inputs.get(output_name,[])
                    for i,num_input in enumerate(num_inputs):
                        if num_input == '2':
                            h_node_name = '{}_h{}'.format(fo.argname,i)
                            nodes.append( (h_node_name,{"type":ntype1}) )
                            inputs[h_node_name] = inputs.get(h_node_name,[])
                            inputs[h_node_name].append(fanins[2*i].argname)
                            inputs[h_node_name].append(fanins[2*i+1].argname)

                            inputs[output_name].append(h_node_name)

                        elif num_input =='1':
                            inputs[output_name].append(fanins[2*i].argname)
                        else:
                            print(ntype,i,num_input)
                            assert  False

                else:
                    pos = re.search("\d", mtype)
                    if pos:
                        ntype = ntype[: pos.start()]

                    inputs[fo.argname] = inputs.get(fo.argname,[])
                    for fi in fanins:

                        if ntype == 'NBUFF':
                            buff_replace[fo.argname] = fi.argname
                        else:
                            inputs[fo.argname].append(fi.argname)
                    if ntype == 'IBUFF':
                        ntype = 'INV'

                    if buff_replace.get(fo.argname,None) is None:
                        nodes.append((fo.argname, {"type": ntype}))
            # the edges represents the connection between the fanins and the fanouts
            for output,input in inputs.items():

                for fi in input:
                    edges.append(
                        (
                            fi,
                            output,
                            {"is_reverted": False, "is_sequencial": "DFF" in mtype},
                        )
                    )

        # remove the buffers
        new_edges = []
        for edge in edges:
            if buff_replace.get(edge[0],None) is not None:
                new_edges.append((buff_replace[edge[0]],edge[1],edge[2]) )
            else:
                new_edges.append(edge)
        edges = new_edges
        print(
            "\tlabelling is done! #inputs:{}, #outputs:{}".format(len(inputs), len(outputs)),
            flush=True,
        )

        # add the edges that connect PIs
        gate_names = set([n[0] for n in nodes])
        pis = []
        for (src, _, _) in edges:
            if src not in gate_names and src not in pis:
                nodes.append((src, {"type": "PI"}))
                pis.append(src)

        # label the nodes
        for n in nodes:
            n[1]["is_input"] = n[0] in inputs
            n[1]["is_output"] = n[0] in outputs
        #
        # print('num adder inputs:', len(adder_inputs))
        # print('num adder outputs:', len(adder_outputs))
        #
        # print('num muldiv inputs1:', len(muldiv_inputs1))
        # print('num muldiv inputs2:', len(muldiv_inputs2))
        # print('num muldiv outputs:', len(multdiv_outputs))
        #
        # print('num sub inputs1:', len(sub_inputs1))
        # print('num sub inputs2:', len(sub_inputs2))
        # print('num sub outputs:', len(sub_outputs))

        return nodes, edges

    def parse(self,vfile_pair,hier_report):
        R"""

        :param vfile_pair: (str, str)
            The netlist files of the target circuit, containing a hierarchical one where the hierarchy (boundary of modules) is preserved
            and a non-hierarchical one where the hirerarchy is cancelled
        :param hier_report: str
            the report file given by DC
        :return: (nodes:list, edges:list)
            return the nodes and edges of the transformed DAG
        """
        print('--- Start parsing the netlist...')
        hier_vf, nonhier_vf = vfile_pair[0], vfile_pair[1]
        dp_target_blocks = self.parse_report(hier_report)
        nodes, edges = self.parse_nonhier(nonhier_vf, dp_target_blocks=dp_target_blocks)
        print('--- Parsing is done!')
        return nodes,edges

