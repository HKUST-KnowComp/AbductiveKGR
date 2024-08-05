#!/usr/bin/python3
import argparse

import os, sys
import json
import pandas as pd

import numpy as np
import torch
from torch.utils.data import DataLoader

# sys.path.append('./utils/')
from akgr.utils.load_util import load_yaml, load_csv, load_sampled_dataset
# from akgr.utils.parsing_util import qry_wordlist_2_actions, qry_wordlist_2_actions_v2

# from akgr.tokenizer import QueryTokenizer, AnswersTokenizer, AnswersQueryTokenizer, ActionTokenizer

from datasets import Dataset
from akgr.utils.parsing_util import qry_shift_indices, ans_shift_indices, qry_str_2_actionstr, list_to_str

import pandas as pd

def pre_pre_processing(
        data: dict, pattern_str_2_id: dict,
        is_act:bool=False):
    # print(data_dict[split])
    df = pd.DataFrame.from_records(data)
    # print("#")
    # print(df)
    source = df['answers'].apply(ans_shift_indices)
    source = source.apply(list_to_str)
    # print(source)
    # By defualy, the target sequence is a query string
    target = df['query'].apply(qry_shift_indices)
    target = target.apply(list_to_str)
    if is_act: # action str
        target = target.apply(qry_str_2_actionstr)
    # print(target)
    pattern_id = df['pattern_str'].apply(lambda x: pattern_str_2_id[x])
    # print(pattern_id)
    # df = pd.DataFrame([source, target, pattern_id])
    # print("#")
    return pd.concat({
        'source': source,
        'target': target,
        'pattern_id': pattern_id}, axis=1)
def new_create_dataset(dataname, scale, answer_size,
        pattern_filtered,
        data_root,
        splits,
        is_act:bool,
        # do_ordering:bool=False,
        # is_shared_ent:bool=False,
        # is_v2:bool=False
        ):

    pattern_str_2_id = dict(zip(pattern_filtered['pattern_str'], pattern_filtered.index))

    data_dict, nentity, nrelation = load_sampled_dataset(
        data_root=data_root,
        dataname=dataname,
        scale=scale,
        answer_size=answer_size,
        splits=splits,
        method='pkl'
    )

    dataset_dict = {}
    for split in splits:
        df = pre_pre_processing(
            data=data_dict[split],
            pattern_str_2_id=pattern_str_2_id,
            is_act=is_act)
        # print(df)
        # exit()
        dataset_dict[split] = Dataset.from_pandas(df, split=split)

    return dataset_dict, nentity, nrelation
def new_create_dataloader(dataset_dict, batch_size:int, drop_last:bool=False, shuffle:bool=True) :
    import warnings
    if drop_last:
        warnings.warn('drop_last is True')
    dataloader_dict = {}
    for split, dataset in dataset_dict.items():
        dataloader_dict[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=4
        )
    return dataloader_dict




if __name__ == '__main__':
    # config_dataloader = load_yaml('akgr/configs/config-dataloader.yml')
    new_dataset_dict, nentity, nrelation = new_create_dataset('DBpedia50', 'debug', 32, is_act=True)
    print(new_dataset_dict['train'][:10])