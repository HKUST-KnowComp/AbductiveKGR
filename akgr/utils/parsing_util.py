import json

import queue

def recur_parse_str(qry_list:list) -> dict:
    """
    Input: The current query list

    Supported output format:
    - pattern: the orginal pattern
    - qry_nestedlist: a nested tuple representing the query
    - qry_str: basically replaces indices of ent/rel by the real names
    """
    operator, *args = qry_list

    # pattern_str = ''
    # qry_nestedlist = None
    qry_str = ''

    if operator == 'e':# ['e', [node]]
        try:
            [[id]] = args
        except:
            # print('# Warning: invalid syntax detected in recur_parse_str, return None')
            return None

        # pattern_str = '(e)'
        token_raw = ['(', operator, '(', id, ')', ')']
        # qry_nestedlist = ('e', (id))
        qry_str = f'(e,({id}))'


    elif operator == 'p':# ['p', [rel], [query]]
        try:
            [rel], u_qry = args
        except:
            # print('# Warning: invalid syntax detected in recur_parse_str, return None')
            return None

        rel = -rel

        sub_parsed = recur_parse_str(u_qry)
        if sub_parsed == None: return None

        # pattern_str = f'(p,{sub_parsed["pattern"]})'
        token_raw = ['(', operator, '(', rel, ')', *sub_parsed['token_raw'], ')']
        # qry_nestedlist = ('p', (rel), sub_parsed['qry_nestedlist'])
        qry_str = f'(p,({rel}),{sub_parsed["qry_str"]})'


    elif operator == 'i':# ['i', [query1], ..., [query2]]
        try:
            sub_qrys = args
        except:
            # print('# Warning: invalid syntax detected in recur_parse_str, return None')
            return None
        # sub_patterns = []
        token_raw = ['(', operator]
        # sub_qry_nestedlists = ['i']
        sub_qry_strs = []
        for sub_qry in sub_qrys:
            sub_parsed = recur_parse_str(sub_qry)
            if sub_parsed == None: return None

            token_raw.extend(sub_parsed['token_raw'])
            # sub_patterns.append(sub_parsed['pattern'])
            sub_qry_strs.append(sub_parsed['qry_str'])
            # sub_qry_nestedlists.append(sub_parsed['qry_nestedlist'])
        # pattern_str = f'(i,{",".join(sub_patterns)})'
        token_raw.append(')')
        qry_str = f'(i, {",".join(sub_qry_strs)})'
        # qry_nestedlist = tuple(sub_qry_nestedlists)

    else:
        print(f'Operator "{operator}" not supported')
        exit()

    return {
        # 'pattern':pattern_str,
        'token_raw':token_raw,
        # 'qry_nestedlist':qry_nestedlist,
        'qry_str':qry_str,
        'max_var_id': None,
    }

def create_variable(n: int) -> int:
    return -n
def is_anchor(u):
    return u >= 0
def is_strint(s):
    try:
        int(s)
    except ValueError:
        return False
    else:
        return True
def qry_nestedlist_2_graph(qry_nestedlist) -> list:
    if qry_nestedlist == None:
        return None
    q = queue.Queue()

    n_indices = 0
    edge_list_id = []
    var2ent = {}

    def push(nestedlist, var_id=None):
        if var_id == None:
            nonlocal n_indices
            n_indices += 1
            var_id = create_variable(n_indices)
            var2ent[var_id] = var_id
        q.put((nestedlist, var_id))
        return var_id

    push(qry_nestedlist)

    while not q.empty():

        qry_nestedlist, tail_id = q.get()
        # print('now:', qry_nestedlist, tail_id)
        try:
            operator, *args = qry_nestedlist
        except:
            # print('# Warning: invalid syntax detected in qry_nestedlist_2_graph, return None')
            return None

        if operator == 'e': # ['e', [node]]
            try:
                [[head_ent]] = args
                if not isinstance(head_ent, int) or head_ent < 0:
                    return None
            except:
                # print('# Warning: invalid syntax detected in qry_nestedlist_2_graph, return None')
                return None
            var2ent[tail_id] = head_ent

        elif operator == 'p': # ['p', [rel], [query]]
            try:
                [rel], u_qry_nestedlist = args
                if not isinstance(rel, int) or rel > 0:
                    return None
            except:
                # print('# Warning: invalid syntax detected in qry_nestedlist_2_graph, return None')
                return None
            head_id = push(u_qry_nestedlist)
            edge_list_id.append((head_id, rel, tail_id))

        elif operator == 'i': # ['i', [query1], ..., [query2]]
            try:
                sub_queries = args
                if len(sub_queries) != 2: return None
            except:
                # print('# Warning: invalid syntax detected in recur_parse_str, return None')
                return None
            for sub_qry_nestedlist in sub_queries:
                # push(sub_qry_nestedlist, tail_id)
                head_id = push(sub_qry_nestedlist)
                edge_list_id.append((head_id, 'i', tail_id))

        elif operator == 'u':
            try:
                sub_queries = args
                if len(sub_queries) != 2: return None
            except:
                # print('# Warning: invalid syntax detected in recur_parse_str, return None')
                return None
            for sub_qry_nestedlist in sub_queries:
                head_id = push(sub_qry_nestedlist)
                edge_list_id.append((head_id, 'u', tail_id))

        elif operator == 'n': # ['n', [query]]
            try:
                if len(args) != 1: return None
                [head_qry_nestedlist] = args
            except:
                # print('# Warning: invalid syntax detected in qry_nestedlist_2_graph, return None')
                return None
            head_id = push(head_qry_nestedlist)
            edge_list_id.append((head_id, 'n', tail_id))

        else:
            # print(f'# Warning: Operator "{operator}" not supported')
            return None

    fixed_list = [ent for ent in var2ent.values() if ent >= 0]
    var_list = [ent for ent in var2ent.values() if ent < 0]
    edge_list = []
    for h, r, t in edge_list_id:
        # tran = lambda x : f'v{x}' if x in var_list else var2ent[x]
        hent = var2ent[h]
        tent = var2ent[t]
        if tent in var_list: # fixed -> var or var -> var
            edge_list.append((hent, r, tent))
        else:
            return None

    return {
        'edges': edge_list,
        'fixed': fixed_list,
        'var': var_list
    }



# def pattern_nestedlist_2_edge_dict(input_pattern):
#     pattern = input_pattern
#     q = queue.Queue()

#     n_indices = 0
#     def push(obj, id=None):
#         if id == None:
#             nonlocal n_indices
#             n_indices += 1
#             id = n_indices
#         q.put((obj, id))
#         return id

#     edge_dict = {}
#     push(pattern)
#     while not q.empty():
#         pattern, id = q.get()

#         operator, *operands = pattern
#         if operator == 'i':
#             push(operands[0], id) # same id
#             push(operands[1], id) # same id
#         elif operator == 'p':
#             succ_id = push(operands[0])
#             if not id in edge_dict:
#                 edge_dict[id] = [succ_id]
#             else:
#                 edge_dict[id].append(succ_id)
#     return edge_dict

# def qry_str_2_qry_nestedlist(qry_str):
#     operator_list = ['p', 'i', 'u', 'n', 'e']
#     for op in operator_list:
#         qry_str = qry_str.replace(op, f'"{op}"')
#     qry_str = qry_str.replace('(', '[')
#     qry_str = qry_str.replace(')', ']')
#     try:
#         qry_list = json.loads(qry_str)
#         return qry_list
#     except:
#         # print('Warning: qry_str cannot be parsed by json, return None')
#         return None

def qry_wordlist_2_nestedlist(qry_list):
    # print(qry_list)
    operator_list = ['p', 'i', 'u', 'n', 'e']
    jsonstr = ""
    for i, elem in enumerate(qry_list):
        if elem in operator_list:
            qry_list[i] = f'"{elem}"'
        elif elem == '(':
            qry_list[i] = '['
        elif elem == ')':
            qry_list[i] = ']'
    for i, elem in enumerate(qry_list):
        jsonstr += elem if type(elem) == str else str(elem)
        if elem == '[' or i == len(qry_list) - 1: continue
        if i+1 <= len(qry_list) - 1 and qry_list[i+1] == ']': continue
        jsonstr += ','
    # print(jsonstr)
    try:
        qry_nested = json.loads(jsonstr)
        # print(qry_nested)
        return qry_nested
    except:
        return None

import torch
def shift_entity_index(x: int) -> int:
    return abs(int(x)) + 1
def shift_relation_index(x: int) -> int:
    return -(abs(int(x)) + 1)
def unshift_entity_index(x: int) -> int:
    return abs(int(x)) - 1
def unshift_relation_index(x: int) -> int:
    return -(abs(int(x)) - 1)
def qry_shift_indices(qry_list: list) -> list:
    now_operator = None
    qry_list = qry_list[:]
    for i, s in enumerate(qry_list):
        if isinstance(s, int):
            # print('before', s)
            if now_operator == 'p': # relation
                qry_list[i] = shift_relation_index(s)
            else:
                qry_list[i] = shift_entity_index(s)
            # print('after', qry_list)
        elif s in ['p', 'e']:
            now_operator = s
    return qry_list
def qry_unshift_indices(qry_list: list) -> list:
    now_operator = None
    qry_list = qry_list[:]
    for i, s in enumerate(qry_list):
        if isinstance(s, int):
            if now_operator == 'p': # relation
                qry_list[i] = unshift_relation_index(s)
            else:
                qry_list[i] = unshift_entity_index(s)
        elif s in ['p', 'e']:
            now_operator = s
    return qry_list
def ans_shift_indices(ans_list: list) -> list:
    # ans_list = ans.split(' ')
    return [shift_entity_index(s) for s in sorted(ans_list)]
def ans_unshift_indices(ans_list: list) -> list:
    # ans_list = ans.split(' ')
    return [unshift_entity_index(s) for s in sorted(ans_list)]

# def original_qry_str_2_edges(qry_str, tokenizer=None)->dict:
#     """
#     not the qry_str in the Tokenizer class
#     """
#     # print(qry_str)
#     token_word = qry_str_2_wordlist(qry_str)
#     # print(token_word)
#     # print(qry_wordlist_2_graph(token_word, tokenizer))
#     return qry_wordlist_2_graph(token_word)

def qry_wordlist_2_graph(input_tokens:list)->dict:
    # (S)
    if input_tokens == None: return None
    token_word = input_tokens[:]
    # qry_str, _ = tokenizer.detokenize(token_word=True)
    # qry_str = qry_wordlist_2_str(token_word)
    # nestedlist = qry_str_2_qry_nestedlist(qry_str)
    nested_list = qry_wordlist_2_nestedlist(token_word)
    graph = qry_nestedlist_2_graph(nested_list)
    # print(token_word)
    # print(qry_str)
    # print(nestedlist)
    # print(edges)
    return graph
def qry_str_2_wordlist(s:str) -> list:
    wordlist = s.split()
    return [int(w) if is_strint(w) else w for w in wordlist]

# def qry_str_2_original_qry_str(qry_str):
#     """
#     the qry_str produced by the detokenize() function represents rel by neg int.
#     the GraphSampler class represents rel by pos int
#     """
#     return qry_str.replace('-', '')


def qry_graph_2_actions(graph, do_ordering:bool=False):
    def dfs(in_edges, tail):
        if is_anchor(tail):
            actions.append(('e', tail))
        else:
            for (h, r, t) in in_edges[tail]:
                if is_anchor(h):
                    actions.append((r, 'anchor'))
                else:
                    nonlocal var
                    var -= 1
                    if do_ordering:
                        actions.append((r, var))
                    else:
                        actions.append((r, h))
                dfs(in_edges, h)
    in_edges = {}
    for h, r, t in graph['edges']:
        if not t in in_edges:
            in_edges[t] = []
        in_edges[t].append((h, r, t))
    actions = []
    var = -1
    dfs(in_edges, -1)
    # var = -1
    # for i, (r, h) in enumerate(actions):
    #     if h != 'anchor':
    #         var -= 1
    #         actions[i] = (r, var)
    return actions
def qry_wordlist_2_actions(wordlist:list, do_ordering:bool=False):
    graph = qry_wordlist_2_graph(wordlist)
    actions = qry_graph_2_actions(graph, do_ordering=do_ordering)
    return actions

def qry_wordlist_2_actions_v2(wordlist:list):
    wordlist = [w for w in wordlist if w not in ['(', ')']]
    actions = []
    for i, w in enumerate(wordlist):
        if w in ['i', 'u', 'n']:
            actions.append(w)
        elif w == 'p':
            # actions.append((w, wordlist[i+1]))
            actions.append(str(wordlist[i+1]))
        elif w == 'e':
            # actions.append((w, wordlist[i+1]))
            actions.append(str(wordlist[i+1]))
        # elif isinstance(w, int):
            # continue
    return actions
def qry_str_2_actionstr(qry:str):
    actionstr = qry.replace('(', '')
    actionstr = actionstr.replace(')', '')
    actionstr = actionstr.replace('p', '')
    actionstr = actionstr.replace('e', '')
    return ' '.join(actionstr.split())



def qry_actionlist_2_wordlist_v2(actions:list, return_stack:bool=False):
    # print('actions')
    # print(actions)
    stack = []
    deg = {
        'i': 2,
        'u': 2,
        'n': 1,
        'p': 1,
        'e': 0
    }
    wordlist = []
    for i, w in enumerate(actions):
        # if (not isinstance(w, tuple)) and (not w in ['i', 'u', 'n']):
        if (not isinstance(w, int)) and (not w in ['i', 'u', 'n']):
            return None
        # operator = w[0] if isinstance(w, tuple) else w
        operator = w
        # print(stack)
        wordlist.append('(')
        if operator in ['i', 'u', 'n']:
            wordlist.append(w)
        # elif operator in ['p', 'e']:
        else:
            operand = w
            if is_anchor(operand): operator = 'e'
            else: operator = 'p'
            # wordlist.extend([w[0], '(', w[1], ')'])
            wordlist.extend([operator, '(', operand, ')'])
        stack.append((operator, deg[operator]))

        if operator == 'e':
            wordlist.append(')')
            stack.pop()
            while stack:
                stack[-1] = (stack[-1][0], stack[-1][1] - 1)
                if stack[-1][1] == 0:
                    wordlist.append(')')
                    stack.pop()
                else:
                    break
    return wordlist if not return_stack else stack
def qry_actionstr_2_actionlist(s:str) -> list:
    actionlist = s.split()
    return [int(a) if is_strint(a) else a for a in actionlist]
def qry_actionstr_2_wordlist(actionstr, return_stack:bool=False):
    return qry_actionlist_2_wordlist_v2(
        actions=qry_actionstr_2_actionlist(actionstr),
        return_stack=return_stack
    )
def qry_tokenizer_2_kg_act(actionstr):
    return qry_unshift_indices(qry_actionstr_2_wordlist(actionstr))
def ans_tokenizer_2_kg(ans):
    return ans_unshift_indices(qry_actionstr_2_actionlist(ans))

def map_idlist_2_idnamelist(kg, idlist):
    idnamelist = []
    now_operator = 'e' # Compatible with both qry ans ans
    for id in idlist:
        if isinstance(id, int):
            # print('before', s)
            if now_operator == 'p': # relation
                idnamelist.append(str(id) + ':' + kg.rel_id2name[-id])
            else:
                idnamelist.append(str(id) + ':' + kg.ent_id2name[id])
            # print('after', qry_list)
        else:
            idnamelist.append(id)
            if id in ['p', 'e']:
                now_operator = id
    return idnamelist

def qry_actionprefix_get_branching(action_prefix: list):
    stack = qry_actionstr_2_wordlist(
        actionstr=action_prefix,
        return_stack=True)
    return 'EMPTY' if not stack else stack[-1]

def qry_actions_2_graph_wordlist(actions:list) -> list:
    if actions == None or None in actions:
        return (None, None)
    deg = {}
    stack = []
    V = []; E = []
    root = -1
    V.append(root)
    stack.append(root)
    last_rel = None
    wordlist = []
    for (operator, operand) in actions:
        top = stack[-1]
        if operator != 'e':
            if not top in deg:
                deg[top] = 2 if operator in ['i', 'u'] else 1
                wordlist.extend(
                    ['(', operator] if operator in ['i', 'u', 'n'] \
                        else ['(', 'p', '(', operator, ')'])
            if type(operator) != int or operand != 'anchor':
                E.append((operand, operator, top))
                V.append(operand)
                stack.append(operand)
            else:
                last_rel = operator
        else:
            if last_rel == None: # invalid representation
                return (None, None)
            wordlist.extend(['(', 'e', '(', operand, ')', ')'])
            E.append((operand, last_rel, top))
            V.append(operand)
            while stack:
                top = stack[-1]
                if (not top in deg) or deg[top] <= 0: # invalid representation
                    return (None, None)
                deg[top] -= 1
                if deg[top] > 0:
                    break
                if deg[top] == 0:
                    stack.pop()
                    wordlist.append(')')
    # print('parsed')
    if stack:
        return (None, None)
    return ({
        'edges': E,
        'fixed': [v for v in V if is_anchor(v)],
        'var': [v for v in V if not is_anchor(v)]
    }, wordlist)
def list_to_str(l: list) -> str:
    # print('before', l)
    # print('after', ' '.join([str(x) if isinstance(x, int) else x for x in l]))
    return ' '.join([str(x) if isinstance(x, int) else x for x in l])

"""
to extract:
{'edges': [(-2, -1, -27), (4918, -1, 0), (10033, -1, -200), (13958, -2, 0)], 'fixed': [4918, 10033, 13958], 'var': [-1, -2]}

to extract:
{'edges': [('v1', 'v0', -27), (4918, 'v0', 0), (10033, 'v0', -200), (13958, 'v1', 0)], 'fixed': [4918, 10033, 13958], 'var': [0, 1]}
"""

def debug():
    sample = {"answers":[5842,6987],"query":["(","i","(","i","(","n","(","p","(",-230,")","(","e","(",650,")",")",")",")","(","p","(",-230,")","(","e","(",9239,")",")",")",")","(","p","(",-230,")","(","e","(",11554,")",")",")",")"],"pattern_str":"(i,(i,(n,(p,(e))),(p,(e))),(p,(e)))"}
    # sample = {"answers":[17536,10400,2439,12328,9904,14360,20185,13978,13755,13756,24607],"query":["(","u","(","p","(",-659,")","(","e","(",18516,")",")",")","(","p","(",-194,")","(","e","(",22846,")",")",")",")"],"pattern_str":"(u,(p,(e)),(p,(e)))"}
    # sample = {"answers":[8576,6415,5627,5021,3234,2212,11943,10152,7977,13997,5171,3380,2745,322,2505,3803,9691,5469,11743,5472,4449,5606,6125,3053,367,3185,8050,1530,2683,8188],"query":["(","p","(",-137,")","(","u","(","p","(",-136,")","(","e","(",6125,")",")",")","(","p","(",-131,")","(","e","(",4539,")",")",")",")",")"],"pattern_str":"(p,(u,(p,(e)),(p,(e))))"}
    # print('Method 2')
    # qry_wordlist_2_graph2(sample['query'])
    print('Query wordlist:')
    print(sample['query'])
    graph = qry_wordlist_2_graph(sample['query'])
    print('Query graph')
    print(graph)
    actions = qry_graph_2_actions(graph, do_ordering=True)
    print('Actions')
    print(actions)

    # actions = actions[:-1]
    from akgr.tokenizer import ActionTokenizer, QueryTokenizer
    from akgr.utils.load_util import load_yaml
    config_dataloader = load_yaml('akgr/configs/config-dataloader.yml')
    act_tokenizer = ActionTokenizer(300000, 3000, config_dataloader, is_shared_ent=True)
    act_tokenized = act_tokenizer.tokenize(actions)['token_uid']
    print(act_tokenized)
    act_detokenized = act_tokenizer.detokenize(act_tokenized)
    print(act_detokenized)

    # qry_tokenizer = QueryTokenizer(300000, 3000, config_dataloader)
    # qry_tokenized = qry_tokenizer.tokenize(sample['query'])
    # print(qry_tokenized)
    wrong_actions = actions
    print('Recovered query graph and wordlist')
    graph_parsed, wordlist = qry_actions_2_graph_wordlist(wrong_actions)
    print(graph_parsed)
    print(wordlist)
    for k in graph.keys():
        print(sorted(graph_parsed[k]) == sorted(graph[k]))
    print(wordlist == sample['query'])

    print('=' * 50)
    wordlist = sample['query']
    act_v2 = qry_wordlist_2_actions_v2(wordlist)
    print(act_v2)

    act_tokenizer_v2 = ActionTokenizer(300000, 3000, config_dataloader, is_v2=True)
    act_v2_tokenized = act_tokenizer_v2.tokenize(act_v2)['token_uid']
    print(act_v2_tokenized)
    act_v2_detokenized = act_tokenizer_v2.detokenize(act_v2_tokenized)
    print(act_v2_detokenized)
    act_v2_recovered = qry_actionlist_2_wordlist_v2(act_v2)
    print(act_v2_recovered)

    print('=' * 50)
    action_prefix = act_v2[:]
    print(action_prefix)
    branching = qry_actionprefix_get_branching(action_prefix=action_prefix)
    print(branching)

if __name__ == '__main__':
    debug()