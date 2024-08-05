import pandas as pd
import json
import os

import argparse

from akgr.kgdata import load_kg

# from nltk.corpus import wordnet as wn

# from qry2graph import *

id2ent = None
id2rel = None

isWordNet = False

def output(sample, input_file):
    df = pd.DataFrame.from_records(sample)
    output_file = input_file.removesuffix('.jsonl') + '-checked.jsonl'
    df.to_json(output_file, orient='records', indent=2, force_ascii=False)

def recur_qry2graph(query):
    """
    Input: The current query list

    Output:
    - pattern: the orginal pattern
    - qry_tuple: a nested tuple representing the query
    - qry_str: basically replaces indices of ent/rel by the real names
    """
    operator, *args = query

    pattern_str = ''
    qry_tuple = None
    qry_str = ''

    # print(operator)
    # print(args)

    if operator == 'e':# ['e', [node]]
        pattern_str = '(e)'
        [[id]] = args

        entity_str = str(id2ent[id])

        qry_str = f'(e, ({id}, {entity_str}))'
        # qry_tuple = ('e', (id, id2ent[id]))
        # simplified
        qry_tuple = (f'e, ({id}, {entity_str})', )

    elif operator == 'p':# ['p', [rel], [query]]
        [rel], u_qry = args
        sub_parsed = recur_qry2graph(u_qry)
        pattern_str = f'(p,{sub_parsed["pattern"]})'
        qry_str = f'(p, ({rel} {str(id2rel[rel])}), {sub_parsed["qry_str"]})'
        # qry_tuple = ('p', (rel, id2rel[rel]), sub_parsed['qry_tuple'])
        # simplified
        qry_tuple = (f'p, ({rel} {id2rel[rel]})', sub_parsed['qry_tuple'], )

    elif operator == 'i':# ['i', [query1], ..., [query2]]
        sub_qrys = args
        sub_qry_list_list = ['i']
        sub_pattern_list = []
        sub_qry_str_list = []
        for sub_qry in sub_qrys:
            sub_parsed = recur_qry2graph(sub_qry)
            sub_pattern_list.append(sub_parsed['pattern'])
            sub_qry_str_list.append(sub_parsed['qry_str'])
            sub_qry_list_list.append(sub_parsed['qry_tuple'])
        pattern_str = f'(i,{",".join(sub_pattern_list)})'
        qry_str = f'(i, {", ".join(sub_qry_str_list)})'
        qry_tuple = tuple(sub_qry_list_list)

    elif operator == 'u':# ['u', [query1], ..., [query2]]
        sub_qrys = args
        sub_qry_list_list = ['u']
        sub_pattern_list = []
        sub_qry_str_list = []
        for sub_qry in sub_qrys:
            sub_parsed = recur_qry2graph(sub_qry)
            sub_pattern_list.append(sub_parsed['pattern'])
            sub_qry_str_list.append(sub_parsed['qry_str'])
            sub_qry_list_list.append(sub_parsed['qry_tuple'])
        pattern_str = f'(u,{",".join(sub_pattern_list)})'
        qry_str = f'(u, {", ".join(sub_qry_str_list)})'
        qry_tuple = tuple(sub_qry_list_list)

    elif operator == 'n':# ['n', [query]]
        u_qry = args[0]
        sub_parsed = recur_qry2graph(u_qry)
        pattern_str = f'(n,{sub_parsed["pattern"]})'
        qry_str = f'(n, {sub_parsed["qry_str"]})'
        # qry_tuple = ('p', (rel, id2rel[rel]), sub_parsed['qry_tuple'])
        # simplified
        qry_tuple = (f'n,', sub_parsed['qry_tuple'], )

    else:
        print("Error", operator)
        exit()

    return {'pattern':pattern_str, 'qry_tuple':qry_tuple, 'qry_str':qry_str}

def qry_wordlist_2_nestedlist(qry_list):
    print(qry_list)
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
    print(jsonstr)
    qry_nested = json.loads(jsonstr)
    print(qry_nested)
    return qry_nested

def qrystr_2_graph(qrystr):
    #return BFS_qry2graph(qry_wordlist_2_nestedlist(qrystr))
    return recur_qry2graph(qry_wordlist_2_nestedlist(qrystr))

def my_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataname', default='FB15k-237')
    parser.add_argument('--scale', default='debug')
    parser.add_argument('-a', '--max-answer-size', type=int, default=32)
    parser.add_argument('--data_root', default='./sampled_data/')
    parser.add_argument('--sample_rate', type=float, default=1)
    args = parser.parse_args()
    return args

def main():
    args = my_parse_args()

    global id2ent, id2rel, isWordNet

    if args.dataname == "WN18RR":
        isWordNet = True

    kg = load_kg(args.dataname)#, id_map_only=True)
    id2ent = kg.ent_id2name
    id2rel = kg.rel_id2name

    input_files = [
        os.path.join(args.data_root, args.dataname,
            f'{args.dataname}-{args.scale}-{args.max_answer_size}-{mode}-a2q.jsonl') \
            for mode in ['train', 'valid', 'test']
        ]
    for input_file in input_files:
        df = pd.read_json(input_file, orient='records', lines=True)
        sample = df.sample(frac=args.sample_rate, axis='index').reset_index()
        print('Input file:', input_file)
        print(sample)

        sample_decoded = []
        for i, row in sample.iterrows():
            answers, query = row['answers'], row['query']

            parsed = qrystr_2_graph(query)
            # edges = parsed['edges']

            answers_dec = [f'{a} {id2ent[a]}' for a in answers]
            # edges_dec = decode_edge(edges)
            # sample_decoded.append({'answers': answers_dec, 'edges': edges_dec, 'query': query})
            parsed['answers'] = answers_dec
            parsed['ans_size'] = len(answers)
            parsed['qry_ori'] = query
            parsed = dict(sorted(parsed.items()))
            sample_decoded.append(parsed)
            #print(answers, query)

        output(sample_decoded, input_file)


if __name__ == '__main__':
    main()