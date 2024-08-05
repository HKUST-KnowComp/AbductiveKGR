import os, sys, argparse
from tqdm import tqdm

# data loading and saving
import yaml
import json
import pandas as pd

# sys.path.append('../utils/')
from akgr.utils.load_util import load_yaml, load_csv, load_and_filter_query_patterns
from akgr.kgdata import load_kg

# multiprocessing
import time, math, random
from functools import partial
from multiprocessing import Pool

# Global variable
graph_samplers = None
def init_workers(init_value):
    global graph_samplers
    graph_samplers = init_value

def sample_good_query_given_pattern(mode, max_answers_size, pattern_str):
    """
    graphs['test'] = train + valid + test
    """
    answers_from = {}

    while True:
        sampled_query = graph_samplers[mode].sample_valid_query_given_pattern(pattern_str)

        answers_from['train'] = graph_samplers['train'].search_answers_to_query(sampled_query)
        if mode in ['valid', 'test']:
            answers_from['valid'] = graph_samplers['valid'].search_answers_to_query(sampled_query)

        if mode == 'test':
            answers_from['test'] = graph_samplers['test'].search_answers_to_query(sampled_query)

        if len(answers_from[mode]) > max_answers_size:
            continue

        if judge(answers_from, mode):
            break

    return sampled_query, answers_from, pattern_str

def judge(answers_from, mode):
    """
    If the answers are good or not. Return True if it is good, and False ow.
    """
    if mode == 'train':
        return len(answers_from['train']) > 0
    elif mode == 'valid':
        return len(answers_from['train']) > 0 and len(answers_from['valid']) > 0 \
            and len(answers_from['train']) != len(answers_from['valid'])
    elif mode == 'test':
        return len(answers_from['train']) > 0 and len(answers_from['valid']) > 0 \
            and len(answers_from['test']) > 0 and len(answers_from['test']) != \
                len(answers_from['valid'])
    else:
        return False

def append_aq(answers_queries: list, mode: str, answers_from: dict, query: str, max_answers_size: int, query_type: str):
    def subsample():
        answers = set(random.sample(answers_from[mode], max_answers_size))
        sampled_answers_from = {}
        for split in ['train', 'valid', 'test']:
            sampled_answers_from[split] = list(answers.intersection(answers_from[split]))
            if split == mode:
                break
        return sampled_answers_from
    if len(answers_from[mode]) > max_answers_size:
        while True:
            sampled_answers_from = subsample()
            if judge(sampled_answers_from, mode):
                break
        answers = sampled_answers_from[mode]
    else:
        answers = answers_from[mode]
    answers_queries.append({'answers': answers, 'query': query, 'pattern_str': query_type})

def write_output(answers_queries, dataname, mode, args, id2ent=None):
    df = pd.DataFrame.from_records(answers_queries)
    # random shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    def func_str2int(ids):
        return [int(id) for id in ids]
    df['answers'] = df['answers'].apply(func_str2int)
    output_prefix = f'{dataname}-{args.scale}-{args.max_answer_size}-{mode}'
    path = os.path.join(args.data_root, f'{dataname}/', f'{output_prefix}-a2q.jsonl')
    df.to_json(path, orient='records', lines=True)

    def func_id2ent(ids):
        return [id2ent[id] for id in ids]
    if id2ent != None:
        df['answers'] = df['answers'].apply(func_id2ent)
        os.makedirs(dataname, exist_ok=True)
        df.to_json(f'{dataname}/{output_prefix}-a2q-ent.jsonl', orient='records', lines=True)

def my_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-sampling', default="akgr/configs/config-sampling.yml")
    parser.add_argument('-s', '--scale', help='n-queries setting, e.g., debug/tiny/same')
    parser.add_argument('-a', '--max-answer-size', type=int, default=32)
    parser.add_argument('-p', '--nproc', type=int, default=1, help='num proc')
    parser.add_argument('--data_root', default='./sampled_data/')
    args = parser.parse_args()
    return args

def main():
    args = my_parse_args()
    config_sampling = load_yaml(args.config_sampling)
    print(yaml.dump(config_sampling[args.scale]))

    pattern_filtered = load_and_filter_query_patterns(
        file_name=config_sampling['pattern_table_file'],
        max_dep=2, exclu=None, column='original')

    pattern_filtered.to_csv('akgr/metadata/pattern_filtered.csv', index=True)

    # with open('pattern_filtered.yaml', 'w') as f:
    #     yaml.dump(pattern_filtered, f)

    os.makedirs(args.data_root, exist_ok=True)
    # print(json.dumps(pattern_filtered, indent=4))
    scaling_factor = config_sampling[args.scale]['scale']
    for dataname in config_sampling[args.scale]['datasets']:
        kg = load_kg(dataname)
        num_ent = kg.num_ent
        num_rel = kg.num_rel
        graph_samplers = kg.graph_samplers
        num_train_edges = kg.num_train_edges

        print(f'# Sampling from {dataname} dataset, num_samples_perpattern:')
        num_samples_perpattern = {
            'train': num_train_edges // scaling_factor,
            'valid': (num_train_edges // scaling_factor) // 8,
            'test': (num_train_edges // scaling_factor) // 8
        }
        print(num_samples_perpattern)

        os.makedirs(os.path.join(args.data_root, dataname), exist_ok=True)
        stats_path = os.path.join(args.data_root, f'{dataname}/stats.txt')
        with open(stats_path, 'w') as f:
            f.write(f'nentity\t{num_ent}\n')
            f.write(f'nrelation\t{num_rel}\n')

        patterns_pool = {'train': [], 'valid': [], 'test': []}
        for _, pattern_str in pattern_filtered['pattern_str'].items():
            for split in ['train', 'valid', 'test']:
                num_samples = num_samples_perpattern[split]
                if ("n" in pattern_str) and (split == 'train'):
                    num_samples = num_samples
                patterns_pool[split].extend([pattern_str] * num_samples)
            # if "n" in pattern_str:
            #     patterns_pool['train'].extend([pattern_str] * n_queries['train'] // 10)
            # else:
            #     patterns_pool['train'].extend([pattern_str] * n_queries['train'])
            # patterns_pool['valid'].extend([pattern_str] * n_queries['valid'])
            # patterns_pool['test'].extend([pattern_str] * n_queries['test'])

        def sample_mode(mode):
            """
            Input: patterns list and mode name
            Output: Write to data files
            """
            print(f"Sampling {mode} queries")
            answers_queries = []

            if args.nproc == 1:
                #global graph_samplers
                init_workers(graph_samplers)
                for pattern_str in tqdm(patterns_pool[mode]):
                    func = partial(sample_good_query_given_pattern, mode, args.max_answer_size)
                    sampled_query, answers_from, query_type = func(pattern_str)
                    append_aq(answers_queries, mode, answers_from, sampled_query, args.max_answer_size, query_type)
            else:
                with tqdm(total=len(patterns_pool[mode])) as pbar:
                    with Pool(processes=args.nproc, initializer=init_workers, initargs=(graph_samplers,)) as pool:
                        func = partial(sample_good_query_given_pattern, mode, args.max_answer_size)
                        for sampled_query, answers_from, query_type in pool.imap_unordered(func, patterns_pool[mode]):
                            append_aq(answers_queries, mode, answers_from, sampled_query, args.max_answer_size, query_type)
                            pbar.update()

            write_output(answers_queries, dataname, mode, args)

        sample_mode(mode='train')
        # print('Warning: Not sampling train')
        sample_mode(mode='valid')
        sample_mode(mode='test')
        # print('Warning: Not sampling test')


if __name__ == '__main__':
    main()
    # e.g., python sample_parallel.py -s debug -p 4 -m 10