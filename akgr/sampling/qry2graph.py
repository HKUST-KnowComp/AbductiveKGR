import pandas as pd
import json

import queue



def BFS_qry2graph(query):
    q = queue.Queue()

    n_indices = 0
    q.put((query, n_indices)); n_indices += 1
    edge_list_id = []
    id2ent = {}

    while not q.empty():
        query, v_id = q.get()
        # print('now:', query, v_id)
        operator, *args = query

        if operator == 'e': # ['e', [node]]
            [[v_ent]] = args
            id2ent[v_id] = v_ent

        elif operator == 'p': # ['p', [rel], [query]]
            [rel], u_query = args
            u_id = n_indices
            q.put((u_query, u_id)); n_indices += 1
            edge_list_id.append((u_id, v_id, rel))

        elif operator == 'i': # ['i', [query1], ..., [query2]]
            sub_queries = args
            for sub_query in sub_queries:
                q.put((sub_query, v_id))

        elif operator == 'u':
            print('Union not supported')
            exit()

        elif operator == 'n':
            print('Negation not supported')
            exit()

        else:
            print('Negation not supported')
            exit()
    
    fixed_list = list(id2ent.values())
    var_list = [id for id in range(n_indices) if not id in id2ent.keys()]
    edge_list = []
    for u, v, rel in edge_list_id:
        tran = lambda x : f'v{x}' if x in var_list else f'f{id2ent[x]}'
        edge_list.append((tran(u), tran(v), rel))
    
    return {'edges': edge_list, 
            'fixed': fixed_list, 
            'var': var_list}



# def main():
#     qrystr = '(p,(9),(i,(p,(11),(e,(12652))),(p,(276),(e,(7702)))))'
#     qrylist = qrystr_2_qrylist(qrystr)
#     print(qrylist)
#     print(BFS_qry2graph(qrylist))

# if __name__ == '__main__':
#     main()