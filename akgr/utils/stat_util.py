def initialize_scores_stat(pattern_filtered):
    # initialization
    scores_stat = {} # type: {scores_sum, scores_cnt, scores_ave}
    for _, pattern in pattern_filtered.iterrows():
        scores_stat[pattern['pattern_str'] + pattern['pattern_abbr']] = {}
    scores_stat['all'] = {}
    scores_stat['#0'] = 0
    return scores_stat
import pandas as pd
def count_scores_by_pattern(scores, pattern_id, pattern_filtered, scores_stat): # single score or score batch
    def update_sum_cnt_ave(dic, entry, value):
        if not f'{entry}_sum' in dic:
            dic[f'{entry}_sum'] = dic[f'{entry}_cnt'] = dic[f'{entry}_ave'] = 0
        dic[f'{entry}_sum'] += value
        dic[f'{entry}_cnt'] += 1
        dic[f'{entry}_ave'] = dic[f'{entry}_sum'] / dic[f'{entry}_cnt']

    if type(scores) is not list:
        scores = [scores]
        pattern_id = [pattern_id]

    for i, score in enumerate(scores):
        pattern_str = pattern_filtered.loc[pattern_id[i].item(), 'pattern_str']
        pattern_abbr = pattern_filtered.loc[pattern_id[i].item(), 'pattern_abbr']
        pattern_strabbr = pattern_str + pattern_abbr
        for key, value in score.items():
            update_sum_cnt_ave(scores_stat[pattern_strabbr], key, value)
            update_sum_cnt_ave(scores_stat['all'], key, value)
        # count 0
        if abs(score['smatch']) <= 0.001:
            scores_stat['#0'] += 1
import torch
def stat_scores_by_pattern(scores:list, pattern_id:list, pattern_filtered):
    def get_pattern_format(id: int):
        return str(id) + pattern_filtered.loc[id, 'pattern_str'] + pattern_filtered.loc[id, 'pattern_abbr']
    score_df = pd.DataFrame.from_records(scores)
    if torch.is_tensor(pattern_id):
        score_df['pattern'] = pattern_id
    else:
        score_df['pattern'] = torch.tensor(pattern_id)
    score_df['pattern'] = score_df['pattern'].apply(get_pattern_format)
    # print(score_df)
    output_df = score_df.groupby('pattern').agg(['count', 'mean', 'std'])
    all_df = score_df.drop(['pattern'], axis=1).agg(['count', 'mean', 'std']).T.stack().to_frame().T
    all_df.index.name = 'pattern'
    all_df.rename(index={0:'all'}, inplace=True)
    # print(all_df)
    output_df = pd.concat([output_df, all_df])
    # print(output_df)
    # print(len(scores))
    return output_df
def compute_f1_rec_prec(p, pp, tp):
    if pp == 0:
        prec = 0
    else:
        prec = tp / pp
    rec = tp / p
    f1 = 2 * tp / (p + pp)
    return (f1, rec, prec)