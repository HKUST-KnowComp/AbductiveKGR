import argparse

import pandas as pd
import csv
import os

import pykeen.datasets as pk_datasets
import pykeen.utils as pk_utils
import torch
# import datasets as hg_datasets

import networkx as nx
from akgr.utils.nx_util import df_to_graph

def df_concat(df_list: list):
    return pd.concat(df_list, ignore_index=True)

def update_inverse_edges(rel_id2name: dict, raw_df: pd.DataFrame):
    """
    Input: the rel_id2name map and the single-direction data raw_df

    Process: For each split, create inverse edges for existing edges. New
        edges and original edges are copied into a separate dataframe. The
        relation id maps are updated accordingly.

    Output: new rel_id2name map, rel_id2inv map, and the new data new_df.
    """
    new_id2name = {}
    rel_id2inv  = {}
    for id, name in rel_id2name.items():
        new_id2name[id * 2] = f'+{name}'
        new_id2name[id * 2 + 1] = f'-{name}'
        rel_id2inv[id * 2] = id * 2 + 1
        rel_id2inv[id * 2 + 1] = id * 2
    new_df = {}
    for split, df in raw_df.items():
        df_inv = pd.DataFrame(data=df, copy=True)
        # inverse edges
        df_inv.loc[:, ['head_id', 'tail_id']] = (df_inv.loc[:, ['tail_id', 'head_id']].values)
        # reindex rel id
        df['relation_id'] = df['relation_id'].apply(lambda x: x * 2)
        df_inv['relation_id'] = df_inv['relation_id'].apply(lambda x: x * 2 + 1)
        df_all = df_concat([df, df_inv])
        new_df[split] = df_all.sort_values(by=['relation_id'])
    return new_id2name, rel_id2inv, new_df

def load_kg_common(dataname: str, reverse_edges_flag: bool, id_map_only: bool):
    """
    :param dataname:
    :return: a dict, see return
    """
    # https://pykeen.readthedocs.io/en/stable/reference/datasets.html
    # No matter whether reverse_edges_flag is True or not, we load single-direction data first and reallocate splits.
    if dataname == 'YAGO310':
        ds = pk_datasets.YAGO310(create_inverse_triples=False)
    elif dataname == 'FB15k-237':
        ds = pk_datasets.FB15k237(create_inverse_triples=False)
    elif dataname == 'DBpedia50':
        ds = pk_datasets.DBpedia50(create_inverse_triples=False)
    elif dataname == 'BioKG':
        ds =  pk_datasets.BioKG(create_inverse_triples=False)
    elif dataname == 'PharmKG8k':
        ds = pk_datasets.PharmKG8k(create_inverse_triples=False)
    elif dataname == 'WN18RR':
        ds = pk_datasets.WN18RR(create_inverse_triples=False)
    elif dataname == 'OGBWikiKG2':
        ds = pk_datasets.OGBWikiKG2(create_inverse_triples=False)
    else:
        print(f'# Dataset "{dataname}" not supported, return None')
        return None
    # https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.Dataset.html#pykeen.datasets.Dataset
    # print('# summarize()')
    # print(ds.summarize())
    # print('#' + '=' * 50)

    # https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.Dataset.html#pykeen.datasets.Dataset
    num_ent = ds.num_entities
    num_rel = ds.num_relations

    # https://pykeen.readthedocs.io/en/latest/_modules/pykeen/utils.html#invert_mapping
    ent_id2name = pk_utils.invert_mapping(ds.entity_to_id)
    rel_id2name = pk_utils.invert_mapping(ds.relation_to_id)
    rel_id2inv = {}

    print('# During loading raw kg:')
    raw_df = {}
    # https://pykeen.readthedocs.io/en/latest/reference/triples.html#pykeen.triples.TriplesFactory
    for split in ['training', 'validation', 'testing']:
        # print(f'# Split: {split}')
        if id_map_only == True: continue
        # https://pykeen.readthedocs.io/en/latest/reference/triples.html#pykeen.triples.TriplesFactory
        # print(f'# loading .factory_dict[{split}]')
        factory = ds.factory_dict[split]
        # print(factory)
        # https://pykeen.readthedocs.io/en/latest/reference/triples.html#pykeen.triples.CoreTriplesFactory
        # print('# loading .mapped_triples')
        mapped_triples = factory.mapped_triples
        # print(f'# Split shape of pykeen:', mapped_triples.shape)

        # https://pykeen.readthedocs.io/en/latest/reference/triples.html#pykeen.triples.CoreTriplesFactory.get_inverse_relation_id
        # https://pykeen.readthedocs.io/en/latest/reference/triples.html#pykeen.triples.TriplesFactory.tensor_to_df
        # print('# convertning tensor to df')
        # select (u, v, k) columns
        triples_df = factory.tensor_to_df(mapped_triples)[['head_id', 'tail_id', 'relation_id']]
        raw_df[split] = triples_df
        # print('#' + '=' * 50)

    # Merge all splits
    raw_df_all = df_concat([raw_df['training'], raw_df['validation'], raw_df['testing']])
    # Reallocate splits
    raw_df['training'] = raw_df_all.sample(frac=0.8, replace=False)
    raw_df_remaining = raw_df_all.drop(raw_df['training'].index)
    raw_df['validation'] = raw_df_remaining.sample(frac=0.5, replace=False)
    raw_df['testing'] = raw_df_remaining.drop(raw_df['validation'].index)

    if reverse_edges_flag == True:
        rel_id2name, rel_id2inv, raw_df = update_inverse_edges(rel_id2name, raw_df)
        num_rel *= 2

    print('# Sizes after adding inverse edges')
    print(raw_df['training'].shape)
    print(raw_df['validation'].shape)
    print(raw_df['testing'].shape)

    if id_map_only == True:
        return {
            'ent_id2name': ent_id2name,
            'rel_id2name': rel_id2name
        }
    # creating graphs
    our_df = {
        'train': raw_df['training'],
        'valid': df_concat([raw_df['training'], raw_df['validation']]),
        'test': df_concat([raw_df['training'], raw_df['validation'], raw_df['testing']]),
        'test_only': raw_df['testing']
    }
    graphs = {}
    for split, df in our_df.items():
        graphs[split] = df_to_graph(df)

    print('# Checking id ranges (in graphs)')
    print(f'ent id: {min(ent_id2name.keys()), max(ent_id2name.keys())}')
    print(f'rel id: {min(rel_id2name.keys()), max(rel_id2name.keys())}')
    return {
        'num_ent': num_ent,
        'num_rel': num_rel,
        'ent_id2name': ent_id2name,
        'rel_id2name': rel_id2name,
        'rel_id2inv': rel_id2inv,
        'graphs': graphs
    }

def load_fb15k237_ent_2idname(ent_id2name):
    mid2name_path = 'akgr/metadata/FB15k_mid2name.txt'
    if os.path.exists(mid2name_path) == False:
        print(f'# Error: {mid2name_path} does not exist')
    mid2name = {}
    with open(mid2name_path, 'r', encoding='utf-8') as f:
        rows = csv.reader(f, delimiter='\t')
        for row in rows:
            mid, name = row
            mid2name[mid] = name
    for id, name in ent_id2name.items():
        ent_id2name[id] = mid2name[name]
    # print(ent_id2name)
    return ent_id2name
    # https://huggingface.co/datasets/KGraph/FB15k-237/resolve/main/data/FB15k_mid2name.txt

def load_wn18rr_ent_id2name(ent_id2name):
    # https://stackoverflow.com/questions/8077641/how-to-get-the-wordnet-synset-given-an-offset-id
    import nltk
    nltk.download('wordnet')
    from nltk.corpus import wordnet
    for id, name in ent_id2name.items():
        ent_id2name[id] = wordnet.synset_from_pos_and_offset('n',int(name))
    return ent_id2name

from akgr.kgdata.kgclass import GraphSampler, KG
def load_kg(dataname, reverse_edges_flag=True, id_map_only=False):
    print(f'# loading {dataname}')
    raw_kg_dict = load_kg_common(
        dataname,
        reverse_edges_flag=reverse_edges_flag,
        id_map_only=id_map_only
    )
    if raw_kg_dict == None: return None

    if dataname == 'FB15k-237':
        # tweak mid2name
        raw_kg_dict['ent_id2name'] = load_fb15k237_ent_2idname(raw_kg_dict['ent_id2name'])
    elif dataname == 'WN18RR':
        raw_kg_dict['ent_id2name'] = load_wn18rr_ent_id2name(raw_kg_dict['ent_id2name'])

    return KG(
        num_ent=raw_kg_dict['num_ent'],
        num_rel=raw_kg_dict['num_rel'],
        ent_id2name=raw_kg_dict['ent_id2name'],
        rel_id2name=raw_kg_dict['rel_id2name'],
        rel_id2inv=raw_kg_dict['rel_id2inv'],
        graphs=raw_kg_dict['graphs']
    )


def my_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataname', default='YAGO310')
    args = parser.parse_args()
    return args

def debug():
    args = my_parse_args()
    load_kg(args.dataname)

if __name__ == '__main__':
    debug()