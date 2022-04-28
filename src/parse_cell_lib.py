import os
import pickle
import re

class CellInfo:
    outputs : {}

    def __init__(self):
        self.outputs = {}
        pass
def get_ntype(operator):
    if operator == '!':
        return 'INV'
    elif operator == '+':
        return 'OR'
    elif operator == '&':
        return 'AND'
    elif operator == '^':
        return 'XOR'
    else:
        assert False, 'wrong operator: '.format(operator)

def merge_inv(inputs:dict,nodes,output):

    #print(nodes,inputs)
    nd2type = {}
    for nd in nodes:
        nd2type[nd[0]] = nd[1]['type']
    fathers = {}
    new_inputs = {}
    output_nd = output
    remove_nodes = {}
    replace_nodes = []
    for nd,chilren in inputs.items():
        for child in chilren:
            fathers[child] = fathers.get(child,[])
            fathers[child].append(nd)
        if chilren is None:
            continue
        if nd2type[nd] == 'INV' and nd2type.get(chilren[0],None) is not None:
            remove_nodes[nd] = chilren[0]
            nd2type[chilren[0]] = 'N'+nd2type[chilren[0]]
            replace_nodes.append(chilren[0])
            #inputs[chilren[0]] = None
            if nd == output:
                output_nd = chilren[0]
    for nd,replace_nd in remove_nodes.items():
        nd2type[nd] = None
        inputs[nd] = None
        if nd!=output:
            for father in fathers[nd]:
                inputs[father].remove(nd)
                inputs[father].append(replace_nd)

    nodes = [(item[0], {'type': item[1]}) for item in nd2type.items() if item[1] is not None]

    remove_nd = None
    new_nd = None
    if output_nd!=output:
        inputs[output] = inputs[output_nd]
        inputs[output_nd] = None
        for nd in nodes:
            if nd[0] == output_nd:
                remove_nd = nd
                new_nd  = (output,nd[1])
        nodes.remove(remove_nd)
        nodes.append(new_nd)
    for key, value in inputs.items():
        if value is not None:
            new_inputs[key] = value

    return nodes,new_inputs
def merge_same(inputs:dict,nodes):

    nd2type = {}
    for nd in nodes:
        nd2type[nd[0]] = nd[1]['type']
    #print(nodes, inputs)
    new_inputs = {}

    for nd, children in inputs.items():
        remove_nodes = []
        if children is None:
            continue
        father_type = nd2type[nd]
        sub_children = []
        for child in children:
            if nd2type.get(child,None) is not None and nd2type[child]==father_type:
                sub_children.extend(inputs[child])
                remove_nodes.append(child)
        if len(sub_children)!=0:
            sub_children = list(set(sub_children))
            inputs[nd].extend(sub_children)
            for rm_nd in remove_nodes:
                inputs[nd].remove(rm_nd)
                nd2type[rm_nd] = None
                inputs[rm_nd] = None

    nodes = [(item[0],{'type':item[1]}) for item in nd2type.items() if item[1] is not None]

    for key,value in inputs.items():
        if value is not None:
            new_inputs[key] = value
    return nodes,new_inputs

def parse_expression_withoutbracket(expression:str,output):
    expression = expression.replace(' ','')
    sub_expressions = expression.split('+')

    inputs = {}
    nodes = {}
    or_fis = []
    nid = 0
    for sub_express in sub_expressions:
        and_operands = sub_express.split('&')
        and_fis = []

        for operand in and_operands:
            if operand.startswith('!') and nodes.get(operand, None) is None:
                nodes[operand] = (nid, {'type': 'INV'})
                value = operand[1:]
                inputs[nid] = [value]
                and_fis.append(nid)
                nid += 1
            elif nodes.get(operand, None) is not None:
                and_fis.append(nodes[operand][0])
            else:
                and_fis.append(operand)

        if len(and_fis) == 1:
            or_fis.extend(and_fis)
        else:
            nodes['AND_{}'.format(nid)] = (nid, {'type': 'AND'})
            inputs[nid] = and_fis
            or_fis.append(nid)
            nid += 1

    if len(or_fis) != 1:
        nodes[output] = (output, {'type': 'OR'})
        inputs[output] = or_fis
        nodes = list(nodes.values())
    else:

        nodes = list(nodes.values())
        if len(nodes)!=0:
            out_nd = nodes.pop()
            out_name = out_nd[0]
            nodes.append((output,out_nd[1]))
            new_inputs = {}
            for key,value in inputs.items():
                if key == out_name:
                    new_inputs[output] = value
                else:
                    new_inputs[key] = value
            inputs = new_inputs
        # if nodes.get('AND_1',None) is not None:
        #     nodes['AND_1'] = ()

    nodes, inputs = merge_same(inputs,nodes)
    nodes, inputs = merge_inv(inputs, nodes, output)
    return nodes,inputs


def parse_expression_withbracket(expression:str,output):
    expression = expression.replace(' ','&')
    operator_stack  = []
    value_stack = []
    nodes = []
    inputs = {}
    skip = 0
    nid = 0
    for i in range(len(expression)):
        if skip != 0:
            skip -= 1
            continue
        c = expression[i]

        if c in ['+','&','!','^']:
            operator_stack.append(c)
        elif c == '(':
            value_stack.append(c)
        elif c != ')':
            var = ''
            while c not in ['+','&','!','^','(',')']:
                var += c
                i += 1
                skip += 1
                c = expression[i]
            skip -= 1
            value_stack.append(var)
        elif c == ')':
            if len(operator_stack)==0:
                break
            operator = operator_stack.pop()
            # if len(operator_stack)==0:
            #     nname = output
            # else:
            nname = nid
            pis = []
            while len(value_stack)!=0:
                topValue = value_stack.pop()
                pis.append(topValue)
                if topValue == '(':
                    break

            ntype =get_ntype(operator)
            nodes.append((nname,{'type':ntype}))
            inputs[nname] = [pi for pi in pis if pi!='(']
            value_stack.append(nid)
            nid += 1

    if len(nodes)!=0:
        output_nd = nodes.pop()
        nodes.append((output,output_nd[1]))
        inputs[output] = inputs[output_nd[0]]
        inputs[output_nd[0]] = None
    nodes, inputs = merge_same(inputs, nodes)
    nodes,inputs = merge_inv(inputs,nodes,output)
    return nodes,inputs


def parse_cell_lib(file):
    cell_info_map = {}

    with open(file,'r') as f:
        text = f.read()

    cell_list = text.split('cell')

    for cell_text in cell_list[1:]:
        cell_name = cell_text.split('\n')[0]
        cell_name = cell_name[cell_name.find('(')+1:cell_name.find(')')]

        if cell_name.startswith('ANTE') or cell_name.startswith('BHD') or cell_name.startswith('TIE') or cell_name.startswith('DCAP') or cell_name.startswith('GCK'):
            continue

        if cell_name.startswith('ND'):
            idx = re.search('((EEQM|OPT|CCB|SK)\w*|)((D|X)\d+\w*COT)',cell_name[2:])
        elif cell_name.startswith('IND'):
            idx = re.search('((EEQM|OPT|CCB|SK)\w*|)((D|X)\d+\w*COT)', cell_name[3:])
        else:
            idx = re.search('((EEQM|OPT|CCB|SK)\w*|)((D|X)\d+\w*COT)',cell_name)
        if idx is None:
            print(cell_name)
            assert False

        if cell_name.startswith('MUX'):
            idx = re.search('MUX\d+',cell_name)
            cell_name = cell_name[:idx.end()]
        elif cell_name.startswith('MXI'):
            idx = re.search('MXI\d+',cell_name)
            cell_name = cell_name[:idx.end()]
        else:
            if cell_name.startswith('ND'):
                cell_name = cell_name[:idx.start()+2]
            elif cell_name.startswith('IND'):
                cell_name = cell_name[:idx.start()+3]
            else:
                cell_name = cell_name[:idx.start()]

        cell_info_map[cell_name] = CellInfo()
        pin_text = cell_text[cell_text.find('pin'):]
        pins = pin_text.split('pin')[1:]
        fanouts , fanins = [],[]
        for pin in pins:
            if 'function' in pin:
                pinname, function = pin.split('\n')[:-1]
                pinname = pinname[pinname.find('(')+1:pinname.find(")")]
                fanouts.append(pinname)
                function = function[function.find('"')+1:function.rfind('"')]
                if cell_name.startswith('MUX') or cell_name.startswith('MXI'):
                    continue
                if '(' in function:
                    nodes,inputs = parse_expression_withbracket(function,pinname)
                else:
                    nodes,inputs = parse_expression_withoutbracket(function,pinname)
                cell_info_map[cell_name].outputs[pinname] = (nodes,inputs)
            else:
                pinname = pin[pin.find('(') + 1:pin.find(")")]
                fanins.append(pinname)

        if cell_name.startswith('MUX'):
            output_pins = ['Z','ZN']
            for output_pin in output_pins:
                nodes = [((output_pin,{'type':'MUX'}))]
                inputs = {}
                inputs[output_pin] = fanins
                cell_info_map[cell_name].outputs[output_pin] = (nodes, inputs)
        elif cell_name.startswith('MXI'):
            output_pins = ['Z', 'ZN']
            for output_pin in output_pins:
                nodes, inputs = [],{}
                nodes.append( (1,{'type':'INV'}) )
                nodes.append( (output_pin,{'type':'MUX'}) )
                inputs[1] = [output_pin]
                inputs[output_pin] = fanins
                cell_info_map[cell_name].outputs[output_pin] = (nodes, inputs)
    return cell_info_map

def main():
    cell_info_map = parse_cell_lib('comb_cell.txt')

    os.makedirs('../data',exist_ok=True)
    with open('../data/comb_cell_lib.pkl','wb') as f:
        pickle.dump(cell_info_map,f)
    # exit()

    for key,value in cell_info_map.items():
        print(key)
        for output,output_v in value.outputs.items():
            print('\t',output)
            print('\t\t',output_v[0])
            print('\t\t', output_v[1])

    cell_info_map = parse_cell_lib('sequential_cell.txt')
    #os.makedirs('../data', exist_ok=True)
    with open('../data/seq_cell_lib.pkl', 'wb') as f:
        pickle.dump(cell_info_map, f)
    # exit()

    for key, value in cell_info_map.items():
        print(key)
        for output, output_v in value.outputs.items():
            print('\t', output)
            print('\t\t', output_v[0])
            print('\t\t', output_v[1])

    exit()
    expressions = [
        '(!((A1 A2)+(B1 B2)))',
        '(A1 A2)',
        '(A1 A2 A3)',
        '((A1 A2) !(B1))',
        '(((A2+A3)+(!A1)))',
        '(!I)',
        '(I)',
        '((A1^A2)^A3)',
        '(!((A1+A2)+(B1 A1)))',
        '(!(A1 B1)+!(A1+!B))',
    ]

    for express in expressions:
        nodes,inputs = parse_expression_withbracket(express,'ZN')
        print(express)
        print('\t',nodes)
        print('\t',inputs)
        print('\n')

    expressions = [
        '!A1&!B&!C&!D + !A2&!B&!C&!D + !C',
        'A1',
        'A1&!A2',
        '!A1'
    ]

    for express in expressions:
        nodes,inputs = parse_expression_withoutbracket(express,'ZN')
        print(express)
        print('\t',nodes)
        print('\t',inputs)
        print('\n')

if __name__ == '__main__':
    main()