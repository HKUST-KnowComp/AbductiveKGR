import os, sys, argparse

sys.path.append('./utils/')
from load_util import load_yaml, load_and_filter_query_patterns
import yaml

def count_anchor_nodes(pattern: str) -> int:
    return pattern.count('e')

def my_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--modelname')
    parser.add_argument('-e', '--epoch', type=int, default=None)
    parser.add_argument('-d', '--dataname', default='DBpedia50')
    parser.add_argument('--scale', default='server')
    parser.add_argument('-a', '--max-answer-size', type=int, default=32)
    parser.add_argument('--search_scale', default=None)
    parser.add_argument('--result_root')
    args = parser.parse_args()
    return args
def main():
    """
    Given the scores.yaml of some epoch of <model_name> on <dataname>.
    Report:
    - Average scores (Smatch, Precision, Recall, F1) over all testing data.
    - Average scores over data with 2 anchors and depth (1, 2, 3)
    - Average scores over data with 3 anchors and depth (1, 2, 3)
    """
    args = my_parse_args()

    pattern_filtered_dep = load_and_filter_query_patterns(
        file_name='./data/test_generated_formula_anchor_node=3.csv',
        max_dep=3, exclu='u|n', column='original',
        with_depth=True)
    pattern_str_2_id = dict(zip(pattern_filtered['pattern_str'], pattern_filtered['id']))

    if args.modelname in ['search']:
        scores_file_path = os.path.join(
            args.result_root,
            f'{args.dataname}-{args.scale}-{args.max_answer_size}-scores({args.search_scale}-search-test).yaml')
    else:
        scores_file_path = os.path.join(
            args.result_root,
            f'{args.dataname}-{args.scale}-{args.max_answer_size}-{args.epoch}-scores(test).yaml')
    print(f'# Processing {scores_file_path}')
    scores_stat = load_yaml(scores_file_path)

    result_dict = {}
    for pattern, scores in scores_stat.items():
        if pattern == '#0' or pattern == 'all': continue
        anchor = count_anchor_nodes(pattern)
        id = pattern_str_2_id[pattern]
        dep = pattern_filtered_dep.loc[pattern_filtered_dep['id'] == id, 'original_depth']
        result_dict[id] = {'dep':dep, 'anchor':anchor, 'scores':scores}
    # print(result_dict)

    # Counting
    def translate_stat_index(metric, stat):
        if metric in ['f1', 'prec', 'rec']: return f'ans_{metric}_{stat}'
        else: return f'{metric}_{stat}'
    def count_by_criteria(dep: list, anchor: list):
        scores_report = {}
        for metric in ['f1', 'prec', 'rec', 'smatch']:
            scores_report[metric] = {}
            for stat in ['sum', 'cnt']:
                index = translate_stat_index(metric, stat)
                scores_report[metric][stat] = 0
                for result in result_dict.values():
                    if (not result['dep'] in dep) or (not result['anchor'] in anchor): continue
                    scores_report[metric][stat] += result['scores'][index]
            scores_report[metric]['ave'] = scores_report[metric]['sum'] / scores_report[metric]['cnt']
        # print(dep, anchor, scores_report)
        return scores_report

    scores_report = {}
    for anchor in range(1, 4, 1): # should be (2, 4, 1)
        for dep in range(1, 4, 1):
            scores_report[f'dep={dep} # anchors<={anchor}'] = count_by_criteria(
                dep=[dep],
                anchor=list(range(1, anchor+1, 1))
            )
    scores_report['all'] = count_by_criteria(
                dep=list(range(1, 4, 1)),
                anchor=list(range(1, anchor+1, 1))
            )
    for metric in ['smatch', 'prec', 'rec', 'f1']:
        for typ, scores in scores_report.items():
            print(f'{metric}\t{typ}\t{scores[metric]["ave"]}\t{scores[metric]["cnt"]}')
if __name__ == '__main__':
    main()