import os
import pickle

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

def merge_same(inputs:dict,nodes):

    nd2type = {}
    for nd in nodes:
        nd2type[nd[0]] = nd[1]['type']
    new_inputs = inputs.copy()
    new_inputs2 = {}
    for nd, children in inputs.items():
        flag = True
        is_trivial = True
        father_type = nd2type[nd]
        sub_children = []
        for child in children:
            if nd2type.get(child, None) is None or nd2type[child]=='INV':
                sub_children.append(child)
            elif nd2type[child]!=father_type:
                flag =False
                is_trivial = False
                break
            else:
                is_trivial = False
                sub_children.extend(inputs[child])

        if flag and not is_trivial:
            sub_children = list(set(sub_children))
            new_inputs[nd] = sub_children
            #print('children',children,'sub',sub_children)
            for child in children:
                if not child in sub_children:
                    new_inputs[child] = None
                    nd2type[child] = None

    nodes = [(item[0],{'type':item[1]}) for item in nd2type.items() if item[1] is not None]
    for key,value in new_inputs.items():
        if value is not None:
            new_inputs2[key] = value
    return nodes,new_inputs2

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
            if len(operator_stack)==0:
                nname = output
            else:
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

    nodes, inputs = merge_same(inputs, nodes)
    return nodes,inputs


def parse_cell_lib(file):
    cell_info_map = {}

    with open(file,'r') as f:
        text = f.read()

    cell_list = text.split('cell')

    for cell_text in cell_list:
        cell_name = cell_text.split('\n')[0]
        cell_name = cell_name[cell_name.find('(')+1:cell_name.find(')')]
        cell_info_map[cell_name] = CellInfo()
        pin_text = cell_text[cell_text.find('pin'):]
        pins = pin_text.split('pin')[1:]
        for pin in pins:
            if 'function' in pin:
                pinname, function = pin.split('\n')[:-1]
                pinname = pinname[pinname.find('(')+1:pinname.find(")")]
                function = function[function.find('"')+1:function.rfind('"')]
                if '(' in function:
                    nodes,inputs = parse_expression_withbracket(function,pinname)
                else:
                    nodes,inputs = parse_expression_withoutbracket(function,pinname)
                cell_info_map[cell_name].outputs[pinname] = (nodes,inputs)

    return cell_info_map

cell_info_map = parse_cell_lib('cell_lib.txt')

os.makedirs('../data',exist_ok=True)
with open('../data/cell_lib.pkl','wb') as f:
    pickle.dump(cell_info_map,f)
# exit()

for key,value in cell_info_map.items():
    print(key)
    for output,output_v in value.outputs.items():
        print('\t',output)
        print('\t\t',output_v[0])
        print('\t\t', output_v[1])


expressions = [
    '(!((A1 A2)+(B1 B2)))',
    '(A1 A2)',
    '(A1 A2 A3)',
    '((A1 A2) !(B1))',
    '(((A2+A3)+(!A1)))',
    '(!I)',
    '(I)'
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
