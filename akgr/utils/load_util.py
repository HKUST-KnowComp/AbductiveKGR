import yaml
def load_yaml(filename: str):
    with open(filename, 'r') as f:
        obj = yaml.safe_load(f)
    return obj

import pandas as pd
def load_jsonl(data_path):
    """
    return a list of dict
    """
    data_dict = pd.read_json(data_path,
        orient='records', lines=True).to_dict(orient='records')
    return data_dict

def load_csv(data_path, **kwargs):
    df = pd.read_csv(data_path, **kwargs)
    return df

def load_and_filter_query_patterns(
        file_name,
        max_dep=3, exclu='', column='original', with_depth=False):
    """
    Terminology:
    - pattern id: the number of the pattern
    - pattern str: the parentheses expression
    - pattern abbr: the abbreviation

    Input:
    - file_name = name of the csv file
    - max_dep = maximum depth
    - exclu = string consists of 'u', 'n', 'i', 'p', seperated by '|'. operations to exclude
    - column = default 'original'

    Output:
    - pattern_filtered
    """
    pattern_table = pd.read_csv(file_name, index_col='id')#.reset_index(drop = True)
    pattern_filterd = pattern_table[[column, 'pattern_abbr', 'original_depth']]

    pattern_filterd = pattern_filterd.rename({column: 'pattern_str'}, axis='columns')

    pattern_filterd = pattern_filterd.loc[pattern_filterd.original_depth <= max_dep]
    # exclude_pattern = '|'.join(excep)
    if exclu is not None:
        pattern_filterd = pattern_filterd.loc[~pattern_filterd.pattern_str.str.contains(exclu, case=False)]
    # pattern_filtered = {}
    # for i in range(pattern_table.shape[0]):
    #     is_selected = True

    #     dep = pattern_table.original_depth[i]
    #     is_selected = dep <= max_dep

    #     if column == 'original':
    #         pattern_str = pattern_table.original[i]
    #     for ch in excep:
    #         if ch in pattern_str:
    #             is_selected = False
    #     if not is_selected: continue
    #     pattern_id = int(pattern_table.formula_id[i][-4:])
    #     if 'Abbreviation' in pattern_table.columns:
    #         abbr = pattern_table.Abbreviation[i]
    #     else:
    #         abbr = ''
    #     if with_depth == True:
    #         pattern_filtered[pattern_id] = (pattern_str, abbr, dep)
    #     else:
    #         pattern_filtered[pattern_id] = (pattern_str, abbr)

    return pattern_filterd

import os
import pickle

def load_sampled_dataset(data_root, dataname, scale, answer_size,
                         splits=['train', 'valid', 'test'],
                         method='jsonl'):
    """
    Output: data_dict ["train"/"valid"/"test"]
    """
    data_dict = {}
    for split in splits:
        data_path = os.path.join(data_root, dataname, \
            f'{dataname}-{scale}-{answer_size}-{split}-a2q.{method}')
        if method == 'jsonl':
            data_dict[split] = load_jsonl(data_path)
        elif method == 'pkl':
            with open(data_path, 'rb') as f:
                data_dict[split] = pickle.load(f)

    stats_path = os.path.join(data_root, dataname, f'stats.txt')
    with open(stats_path) as f:
        lines = f.readlines()
        nentity = int(lines[0].split('\t')[-1])
        nrelation = int(lines[1].split('\t')[-1])
    # data_dict['test'] = [
    #     # {"answers":[3454,6345,3018,20909,19824,2802,9397,5144,16510,23708,23678],"query":["(","i","(","n","(","p","(",-549,")","(","e","(",12994,")",")",")",")","(","p","(",-547,")","(","e","(",2618,")",")",")",")"],"pattern_str":"(i,(n,(p,(e))),(p,(e)))"},
    #     # {"answers":[8645,6511,11929,9818,21918,21471],"query":["(","p","(",-195,")","(","u","(","p","(",-269,")","(","e","(",9116,")",")",")","(","p","(",-194,")","(","e","(",9818,")",")",")",")",")"],"pattern_str":"(p,(u,(p,(e)),(p,(e))))"},
    #     {"answers":[18369,22403,23044,19272,5898,1932,6160,1553,5653,15063,16672,12579,1060,14438,1511,23273,15917,15918,1069,15533,15924,13495,15929,24314,24317,7615],"query":["(","i","(","i","(","n","(","p","(",-659,")","(","e","(",18646,")",")",")",")","(","p","(",-659,")","(","e","(",7020,")",")",")",")","(","p","(",-659,")","(","e","(",7020,")",")",")",")"],"pattern_str":"(i,(i,(n,(p,(e))),(p,(e))),(p,(e)))"},
    #     # {"answers":[9280,4355,12101,11334,23182,8914,1108,8024,10908,9954,23590,11688,7275,23212,2732,9135,17584,9140],"query":["(","i","(","n","(","p","(",-202,")","(","p","(",0,")","(","e","(",1949,")",")",")",")",")","(","p","(",-567,")","(","e","(",8134,")",")",")",")"],"pattern_str":"(i,(n,(p,(p,(e)))),(p,(e)))"},
    #     {"answers":[17410,13187,3781,4581,4968,15085,23121,1845,6870,19801,15068,20029,6527],"query":["(","p","(",-119,")","(","e","(",2916,")",")",")"],"pattern_str":"(p,(e))"}
    # ]
    return data_dict, nentity, nrelation
def jsonl_2_pickle(data_root, dataname, scale, answer_size,
                    splits=['train', 'valid', 'test']):
    data_dict = {}
    for split in splits:
        jsonl_path = os.path.join(data_root, dataname, \
            f'{dataname}-{scale}-{answer_size}-{split}-a2q.jsonl')
        data = load_jsonl(jsonl_path)
        pickle_path = jsonl_path.removesuffix('.jsonl') + '.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)

import torch
import torch.nn as nn
import transformers
def load_model(path, contents:str, epoch,
               return_huggingface_model:bool,
               model=None, optimizer=None, scheduler=None):
    """
    contents:
        - "state_dicts": weights only, need to instantiate first
        - "model": full model, load directly
    """
    # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
    # https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
    if contents == 'state_dicts':
        print(f'# Loading checkpoint (state_dicts) {path}')
        if model == None or optimizer == None or scheduler == None:
            print('# Error: need to instantiate and pass model, optimizer, scheduler')
            exit()
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    elif contents == 'model':
        print(f'# Loading checkpoint (model) {path}')
        if model is not None or optimizer is not None or scheduler is not None:
            print('# Error: cannot pass model, optimizer, scheduler with contents="model"')
            exit()
        checkpoint = torch.load(path)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']
    else:
        print(f'# Error: contents "{contents}" not supported')
        exit()
    last_epoch = checkpoint['epoch']
    if 'loss_log' in checkpoint.keys():
        loss_log = checkpoint['loss_log']
    else:
        loss_log = {'train': {}, 'valid': {}}
    if return_huggingface_model and not isinstance(model, transformers.PreTrainedModel):
        print('Yes, returning .transformer')
        model = model.transformer
    return model, optimizer, scheduler, last_epoch, loss_log

import pathlib
def save_model(path, contents:str,
               model, optimizer=None, scheduler=None, epoch=None, loss_log=None):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    if contents == 'state_dicts':
        print(f'# Saving checkpoint (state_dicts) {path}')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'loss_log': loss_log
        }, path)
    elif contents == 'model':
        print(f'# Saving checkpoint (model) {path}')
        torch.save({
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'epoch': epoch,
            'loss_log': loss_log
        }, path)
    else:
        print(f'# Error: contents "{contents}" not supported')
        exit()

if __name__ == '__main__':
    print(yaml.dump(load_yaml('akgr/configs/config-dataloader.yml')))