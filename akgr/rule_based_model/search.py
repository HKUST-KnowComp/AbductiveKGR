import argparse, os, sys

from tqdm import tqdm
import random

import yaml

# sys.path.append('./utils/')
# from akgr.utils.parsing_util import qry_str_2_qry_nestedlist,\
#     pattern_struct_2_edge_dict, edge_dict_2_dfs_ordering
from akgr.utils.load_util import load_yaml
from akgr.kgdata import load_kg
from akgr.utils.stat_util import count_scores_by_pattern, initialize_scores_stat,\
    compute_f1_rec_prec

# from akgr.dataloader import create_dataloader

# import networkx as nx

# import itertools

# import akgr.evaluation as evaluation
# from akgr.evaluation import scoring_input_wordlist

# global var
args = None
# graph_samplers = None
# num_ent = None
# num_rel = None
query_pool = None
qry_tokenizer = None
ans_tokenizer = None
pattern_filtered = None
# def recur_search_bf_given_answers(answers, dfs_ordering, split):
#     deg, *sub_orderings = dfs_ordering

#     # Find all predecessor 'p' such that
#     #   there is a fixed relation 'rel' satisfying: the edge (p, rel, ans) exists
#     #   for every 'precrecf1' in answers
#     # That is to find common (pred, key) pairs
#     common_pred_key = {}
#     for i, ans in enumerate(answers):
#         in_edges = graph_samplers[split].in_edges(ans) # (u, v, k) tuples
#         pred_key = {(e[0], e[2]) for e in in_edges}
#         if i == 0: common_pred_key = pred_key
#         else: common_pred_key = common_pred_key.intersection(pred_key)

#     # Compute answers of every (pred, key)
#     answers_pred_key = []
#     for pred, key in common_pred_key:
#         pass
#     # specify relations
#     print(common_pred_key)

# def search_bf_given_answers(answers, pattern, split='test'): # test_only????????
#     pattern_struct = qry_str_2_qry_nestedlist(pattern)
#     edge_dict = pattern_struct_2_edge_dict(pattern_struct)
#     dfs_ordering = edge_dict_2_dfs_ordering(edge_dict, 1)
#     print(edge_dict)
#     print(dfs_ordering)
    # recur_search_bf_given_answers_pattern(answers, pattern_struct)

# Wrong:
# def last_operation_pruning(split, answers):
#     common_key = set()
#     for i, ans in enumerate(answers):
#         in_edges = graph_samplers[split].in_edges(ans) # (u, v, k) tuples
#         key = {e[2] for e in in_edges}
#         if i == 0: common_key = key
#         else: common_key = common_key.intersection(key)
#     feasible_ent = set()
#     for i, ans in enumerate(answers):
#         in_edges = graph_samplers[split].in_edges(ans) # (u, v, k) tuples
#         ent = {e[0] for e in in_edges if e[2] in common_key}
#         feasible_ent.update(ent)
#     return tuple(common_key), tuple(feasible_ent)

# def first_operation_pruning(split, ent_choices):
#     rel_candidates_list = []
#     for ent in ent_choices:
#         out_edges = graph_samplers[split].out_edges(ent) # (u, v, k) tuples
#         key = {e[2] for e in out_edges}
#         rel_candidates_list.append(tuple(key))
#     return tuple(rel_candidates_list)

# def answers_pruning(max_dep, split, answers):
#     ent_reachable = {i:[] for i in range(max_dep+1)}

#     # Wrong:
#     # rel_reachable = {i:[] for i in range(max_dep+1)}
#     # common_key, last_feasible_ent = last_operation_pruning(split, answers)
#     # print('last feasible ent num')
#     # print(len(last_feasible_ent))

#     dist = {}
#     for a in answers:
#         dist[a] = dict(nx.single_target_shortest_path_length(graph_samplers[split].graph, target=a, cutoff=max_dep))
#     for node in graph_samplers[split].graph:
#         all_dist = [dist[a][node] for a in answers if node in dist[a]]
#         min_dist = min(all_dist) if all_dist else 100000000
#         for i in range(max_dep+1):
#             if i >= min_dist: ent_reachable[i].append(node)
#     for i in range(max_dep+1):
#         ent_reachable[i] = tuple(sorted(set(ent_reachable[i])))
#         print('ent reachable with distance at most', i)
#         print(len(ent_reachable[i]))
#         if (i == 1):
#             print(ent_reachable[i])

#     # print('out edges of 12966:')
#     # print(graph_samplers[split].out_edges(12966))
#     # print('out edges of 1531:')
#     # print(graph_samplers[split].out_edges(1531))
#     return ent_reachable

# def enumerate_query_given_pattern(num_ent, num_rel, split: str, answers: list, pattern: str, ent_reachable: dict):
#     num_ent_needed = pattern.count('e')
#     num_rel_needed = pattern.count('p')
#     # num_ent = num_rel = 3


#     all_p_positions = [i for i, ch in enumerate(pattern) if ch == 'p']
#     all_e_positions = [i for i, ch in enumerate(pattern) if ch == 'e']
#     print('all_p_positions')
#     print(all_p_positions)

#     first_p_indices = [i for i, pos in enumerate(all_p_positions) if pattern[pos] == 'p' and pattern[pos+3] == 'e']
#     print('first_p_indices')
#     print(first_p_indices)

#     depth = [0] * len(pattern)
#     # last_p_indices = []
#     for i, pos in enumerate(all_p_positions):
#         # last_p_indices.append(i)
#         # mark the whole sub-pattern enclosed by this "p"
#         sum_paren = 0
#         for now in range(pos-1, len(pattern)):
#             depth[now] += 1
#             if pattern[now] == '(': sum_paren += 1
#             elif pattern[now] == ')': sum_paren -= 1
#             if sum_paren == 0: break
#     # print(pattern)
#     # print('depth')
#     # print(''.join([str(s) for s in depth]))

#     ent_lists = []
#     for i, pos in enumerate(all_e_positions):
#         ent_lists.append(ent_reachable[depth[pos]])

#     # last_rel_candidates = last_operation_pruning(num_rel, 'test', answers, pattern)
#     for ent_choices in itertools.product(*ent_lists):
#         rel_lists = [list(range(num_rel))] * num_rel_needed

#         # for i in last_p_indices:
#         #     rel_lists[i] = last_rel_candidates

#         first_rel_candidates_tuple = first_operation_pruning(split, ent_choices)
#         for i, rels in zip(first_p_indices, first_rel_candidates_tuple):
#             rel_lists[i] = rels

#         # print(rel_lists)
#         # print(rel_lists)
#         for rel_choices in itertools.product(*rel_lists):
#             ent_cnt = 0; rel_cnt = 0; query = ''
#             for ch in pattern:
#                 query += ch
#                 if ch == 'p':
#                     query += f',({rel_choices[rel_cnt]})' # (p,(...)) -> (p,(rel),(...))
#                     rel_cnt += 1
#                 elif ch == 'e':
#                     query += f',({ent_choices[ent_cnt]})' # (e) -> (e,(ent))
#                     ent_cnt += 1
#             yield(query)

# def update_query_pool(typ: str, qry):
#     if type(qry) is str:
#         query_pool[typ].append(qry)
#     elif type(qry) is list:
#         query_pool[typ].extend(qry)

# def common_into_rel(split: str, targets: list):
#     common_rel = set()
#     all_rel = set()
#     pre_ent_by_rel = {}
#     for i, tgt in enumerate(targets):
#         in_edges = graph_samplers[split].in_edges(tgt) # (u, v, k) tuples
#         rel = set()
#         for u, v, k in in_edges:
#             if not k in pre_ent_by_rel: pre_ent_by_rel[k] = set() # WRONG? set(u)
#             else: pre_ent_by_rel[k].add(u)
#             rel.add(k)
#         if i == 0: common_rel = rel
#         else: common_rel = common_rel.intersection(rel)
#         all_rel.update(rel)
#     for rel in all_rel:
#         if not rel in common_rel:
#             pre_ent_by_rel.pop(rel)
#         else:
#             pre_ent_by_rel[rel] = tuple(pre_ent_by_rel[rel])
#     return tuple(common_rel), pre_ent_by_rel

# def union_into_rel(split: str, targets: list):
#     union_rel = set()
#     pre_ent_by_rel = {}
#     for i, tgt in enumerate(targets):
#         in_edges = graph_samplers[split].in_edges(tgt) # (u, v, k) tuples
#         for u, v, k in in_edges:
#             if not k in pre_ent_by_rel: pre_ent_by_rel[k] = set() # WRONG set(u)
#             else: pre_ent_by_rel[k].add(u)
#             union_rel.add(k)
#     for rel in pre_ent_by_rel.keys():
#         pre_ent_by_rel[rel] = tuple(pre_ent_by_rel[rel])
#     return tuple(union_rel), pre_ent_by_rel

# def backward_dfs_p(search_split: str, answers: list, dep: int, max_dep: int, rels):
#     """
#     answers: entities that reachable through rels so far
#     """
#     if dep > 0:
#         for ent in answers:
#             qry = f'(e,({ent}))'
#             for rel in reversed(rels):
#                 qry = f'(p,({rel}),{qry})'
#             update_query_pool(f'{dep}p', qry)
#     if dep == max_dep:
#         return None
#     if dep == 0:
#         feasible_rel, pre_ent_by_rel = common_into_rel(search_split, answers)
#     else:
#         feasible_rel, pre_ent_by_rel = union_into_rel(search_split, answers)
#     # if dep == 1:
#     #     print(feasible_rel)
#     #     print(len(feasible_rel))
#     #     answers_pruning(1, search_split, answers)
#     for rel in feasible_rel:
#         pre_ent = pre_ent_by_rel[rel]
#         backward_dfs_p(search_split, pre_ent, dep+1, max_dep, rels+[rel])


# def backward_search_p(num_ent: int, split: str, answers: list, pattern: str):
#     max_dep = int(pattern[0])
#     ent_candidates = list(range(num_ent))
#     start_node = None
#     dist = {}
#     for a in answers:
#         dist[a] = dict(nx.single_target_shortest_path_length(graph_samplers[split].graph, target=a, cutoff=max_dep))
#         candidates_from_a = [n for n, d in dist[a] if d <= max_dep]
#         if len(candidates_from_a) < ent_candidates:
#             ent_candidates = candidates_from_a
#             start_node = a

#     # !!!! common into rel
#     def backward_dfs(now, dep, cutoff):
#         in_edges = graph_samplers[split].in_edges(now)
#         for u, v, k in in_edges:
#             pass

# def backward_search_i(search_split: str, answers: list, max_num_i: int):
#     feasible_rel, pre_ent_by_rel = common_into_rel(search_split, answers)
#     for num_i in range(2, max_num_i+1):
#         # cnt = 0
#         for rel_choice in itertools.combinations(feasible_rel, num_i):
#             # print(rel_choice)
#             ent_lists = []
#             for rel in rel_choice:
#                 ent_lists.append(pre_ent_by_rel[rel])
#             for ent_choice in itertools.product(*ent_lists):
#                 sub_qry_list = []
#                 for i, (rel, ent) in enumerate(zip(rel_choice, ent_choice), start=1):
#                     now_qry = f'(p,({rel}),(e,({ent})))'
#                     sub_qry_list.append(now_qry)
#                     if i == 1: qry = now_qry
#                     else: qry = f'(i,{qry},{now_qry})'
#                 update_query_pool(f'{num_i}i', qry)
#             # cnt += 1


# def common_predecessors(split: str, tails: list):
#     """
#     return {x : there is some fixed rel such that rel(x, y)==True for all y in targets}
#     meanwhile, sort the set in ascending order of outdegree
#     """
#     # (h, r): c means that (h, r) is incident on c entities in tails
#     incidence_head_rel = {}

#     for i, tail in enumerate(tails):
#         # print(f'tail:{tail}')
#         in_edges = graph_samplers[split].in_edges(tail) # (h, t, r) tuples
#         # print(in_edges)
#         head_rel_pairs = {(h, r) for h, t, r in in_edges}
#         for pair in head_rel_pairs:
#             # initialize
#             if not pair in incidence_head_rel: incidence_head_rel[pair] = 0
#             incidence_head_rel[pair] += 1

#     # sort pairs by incidence in descending order
#     # print('before sorting', incidence_head_rel)
#     incidence_head_rel = dict(reversed(sorted(incidence_head_rel.items(), key=lambda x: x[1])))
#     # print('after sorting', incidence_head_rel)

#     # get the outdegree of pairs
#     odeg_head_rel = {(h, r): graph_samplers[split].out_degree_by_key(h, r) for (h, r) in incidence_head_rel}

#     # create dict of the form (h, r): (incidence, outdegree)
#     incidence_odeg_head_rel = {}
#     for (h, r), incidence in incidence_head_rel.items():
#         # incidence = incidence_head_rel[(h, r)]
#         odeg = odeg_head_rel[(h, r)]
#         incidence_odeg_head_rel[(h, r)] = (incidence, odeg)

#     return incidence_odeg_head_rel

# def greedy_1p(split, answers) -> list:
#     incidence_odeg_head_rel = common_predecessors(split, answers)
#     # print(incidence_odeg_head_rel)

#     # compute f1, rec, prec based on the <split> dataset
#     f1_rec_prec_head_rel = {}
#     p = len(answers)
#     for (h, r), (incidence, odeg) in incidence_odeg_head_rel.items():
#         tp = incidence
#         pp = odeg
#         f1_rec_prec_head_rel[(h, r)] = compute_f1_rec_prec(p, pp, tp)
#         # print(f'({h}, {r}), inc {incidence}, odeg {odeg}, f1 rec prec:\t', round(f1, 4), round(rec, 4), round(prec, 4))
#     # print('before sorting', f1_rec_prec_head_rel)

#     # sort the score dict in descending order based on keys: (1) f1 (2) rec (3) prec
#     f1_rec_prec_head_rel = dict(reversed(sorted(f1_rec_prec_head_rel.items(), key=lambda x: x[1])))
#     if len(f1_rec_prec_head_rel) != 0:
#         h, r = list(f1_rec_prec_head_rel.keys())[0]
#     else:
#         h = random.randint(0, num_ent-1)
#         r = random.randint(0, num_rel-1)
#     # print(f'greedy on {split}: ({h}, {r}): f1, prec, rec {f1_rec_prec_head_rel[(h, r)]}')
#     # return f'(p,({r}),(e,({h})))'
#     return ['(', 'p', '(', -r, ')', '(', 'e', '(', h, ')', ')', ')']

def brute_force_1p(graph_samplers, split, label_ans, label_qry, return_ans:bool=False):
    # print('label ans')
    # print(label_ans)
    # print('label qry')
    # print(label_qry)
    # Conver labels format
    from akgr.utils.parsing_util import qry_unshift_indices, ans_unshift_indices, qry_str_2_wordlist, qry_actionstr_2_actionlist
    # print(qry_actionstr_2_actionlist(label_ans))
    label_ans_unshifted = ans_unshift_indices(qry_actionstr_2_actionlist(label_ans))

    # print('label ans list unshifted')
    # print(label_ans_unshifted)

    # Find all possible in-edges
    all_in_edges = set()
    for i, tail in enumerate(label_ans_unshifted):
        # print(f'tail:{tail}')
        in_edges = graph_samplers[split].in_edges(tail) # (h, t, r) tuples
        all_in_edges = all_in_edges.union(in_edges)
    # print('all edges')
    # print(all_in_edges)
    # Find 1p with the highest jaccard score in <split> graph
    from akgr.evaluation import scoring_input_wordlist, scoring_input_act_batch
    from akgr.utils.parsing_util import qry_shift_indices, ans_shift_indices, qry_str_2_actionstr, list_to_str

    best_j = 0
    best_s = 0
    best_score_split = {'smatch': 0, 'jaccard': 0}
    best_qry_act = ''
    best_qry_list = ''
    for (h, t, r) in all_in_edges:
        # Convert prediction format
        pred_qry_list = ['(', 'p', '(', -r, ')', '(', 'e', '(', h, ')', ')', ')']
        # print('pred qry list')
        # print(pred_qry_list)
        pred_qry_list_shifted = qry_shift_indices(pred_qry_list)
        # print('pred qry list shifted')
        # print(pred_qry_list_shifted)
        pred_act = qry_str_2_actionstr(list_to_str(pred_qry_list_shifted))
        score_now_split = scoring_input_act_batch(
            pred_word_batch=[pred_act],
            label_word_batch=[label_qry],
            ans_word_batch=[label_ans],
            scoring_method=['smatch', 'jaccard', 'precrecf1'],
            graph_samplers=graph_samplers,
            searching_split=split,
        )[0]
        # print(score_now_split)
        if (score_now_split['jaccard'] > best_j) or (score_now_split['jaccard'] == best_j and score_now_split['smatch'] > best_s):
            best_score_split = score_now_split
            best_j = score_now_split['jaccard']
            best_s = score_now_split['smatch']
            best_qry_act = pred_act
            best_qry_list = pred_qry_list
    output = scoring_input_act_batch(
        pred_word_batch=[best_qry_act],
        label_word_batch=[label_qry],
        ans_word_batch=[label_ans],
        scoring_method=['smatch', 'jaccard', 'precrecf1'],
        graph_samplers=graph_samplers,
        searching_split='test',
        return_ans=return_ans
    )
    score_now_test = output[0] if not return_ans else output[0][0]

    # print('label ans')
    # print(label_ans_unshifted)
    # print('best qry')
    # print(best_qry_act)
    # print(f'pred ans (split={split})')
    # print(sorted(graph_samplers[split].search_answers_to_query(best_qry_list)))
    # print(best_score_split)
    # print(f'pred ans (test)')
    # print(sorted(graph_samplers['test'].search_answers_to_query(best_qry_list)))
    # print(score_now_test)
    if return_ans:
        ans = output[1][0]
        return best_qry_act, best_score_split, score_now_test, ans
    else:
        return best_qry_act, best_score_split, score_now_test

# def greedy_i_sort(split, answers):
#     incidence_odeg_head_rel = common_predecessors(split, answers)
#     # print(incidence_odeg_head_rel)

#     # compute f1, rec, prec based on the <split> dataset
#     f1_rec_prec_head_rel = {}
#     p = len(answers)
#     for (h, r), (incidence, odeg) in incidence_odeg_head_rel.items():
#         tp = incidence
#         pp = odeg
#         f1_rec_prec_head_rel[(h, r)] = compute_f1_rec_prec(p, pp, tp)
#         # print(f'({h}, {r}), inc {incidence}, odeg {odeg}, f1 rec prec:\t', round(f1, 4), round(rec, 4), round(prec, 4))
#     # print('before sorting', f1_rec_prec_head_rel)

#     # sort the score dict in descending order based on keys: (1) f1 (2) rec (3) prec
#     f1_rec_prec_head_rel = dict(reversed(sorted(f1_rec_prec_head_rel.items(), key=lambda x: x[1])))
#     # h, r = list(f1_rec_prec_head_rel.keys())[0]
#     PP = set()
#     P = set(answers)
#     qry_str = ''
#     best_score = (0, 0, 0)
#     best_qry_str = ''
#     for i, (h, r) in enumerate(list(f1_rec_prec_head_rel.keys())[:3], start=1):
#         out_edges_key = graph_samplers[split].out_edges_by_key(h, r)
#         now_tails = set([t for _, t, _ in out_edges_key]) # h, t, r
#         if i == 1: PP = now_tails
#         else: PP = PP.intersection(now_tails)
#         TP = P.intersection(PP)
#         # print(P)
#         # print(PP)
#         # print(TP)
#         tp = len(TP); p = len(P); pp = len(PP)
#         if tp == 0: break
#         current_score = compute_f1_rec_prec(p, pp, tp)
#         now_qry_str = f'(p,({r}),(e,({h})))'
#         if i == 1:
#             qry_str = now_qry_str
#         else:
#             qry_str = f'(i,{now_qry_str},{qry_str})'
#         # for debug only:
#         # print(qry_str)
#         # real_score = scoring_input_wordlist(
#         #     pred_str=qry_str, label_str=None,
#         #     ans_word_batch=answers,
#         #     scoring_method=['precrecf1'], graph_samplers=graph_samplers)
#         # print('computed score:', current_score)
#         # print('real score:', real_score)
#         if current_score > best_score:
#             best_score = current_score
#             best_qry_str = qry_str
#     return best_qry_str

# def greedy_i_interactive(split, answers):
#     # Upper-case letters for set, lower-case letters for size
#     PP = set() # predicted positive (PP) set
#     # TP =
#     for _ in range(3):
#         incidence_odeg_head_rel = common_predecessors(split, answers)

#         # compute f1, rec, prec based on the <split> dataset
#         f1_rec_prec_head_rel = {}
#         P = len(answers)
#         for (h, r), (incidence, odeg) in incidence_odeg_head_rel.items():
#             tp = incidence
#             pp = odeg
#             f1_rec_prec_head_rel[(h, r)] = compute_f1_rec_prec(p, pp, tp)

#         # sort the score dict in descending order based on keys: (1) f1 (2) rec (3) prec
#         f1_rec_prec_head_rel = dict(reversed(sorted(f1_rec_prec_head_rel.items(), key=lambda x: x[1])))
#         if len(f1_rec_prec_head_rel) != 0:
#             h, r = list(f1_rec_prec_head_rel.keys())[0]
#         else:
#             h = random.randint(0, num_ent-1)
#             r = random.randint(0, num_rel-1)
#         return f'(p,({r}),(e,({h})))'

def search_and_test(graph_sampler, search_split, tgt_pattern, ans_wordlist, label_qry_wordlist, heuristics, return_ans:bool=False):
    # print('# Start backward_dfs_p')
    # backward_dfs_p(search_split=search_split, answers=ans_wordlist, dep=0, max_dep=3, rels=[])
    # print('# Start backward_dfs_i')
    # backward_search_i(search_split=search_split, answers=ans_wordlist, max_num_i=2)
    # if heuristics == 'greedy_1p':
    #     pred_qry_wordlist, score_now_split = greedy_1p(search_split, ans_wordlist)
    # elif heuristics == 'greedy_i_sort':
    #     print(f'Heuristics {heuristics} not supported')
    #     exit()
    #     pred_qry_wordlist = greedy_i_sort(search_split, ans_wordlist)
    if heuristics == 'brute_force_1p':
        return brute_force_1p(
            graph_sampler, search_split, ans_wordlist, label_qry_wordlist,
            return_ans=return_ans)
    else:
        import warnings
        warnings.warn(f'Heuristics {heuristics} not supported')
        exit()
    # if args.vs:
    #     print(pred_qry_wordlist)
    # score = scoring_input_wordlist(
    #     pred_qry_wordlist=pred_qry_wordlist,
    #     label_qry_wordlist=label_qry_wordlist,
    #     ans_wordlist=ans_wordlist,
    #     do_correction=False,
    #     scoring_method=['smatch', 'precrecf1'],
    #     graph_samplers=graph_samplers,
    #     searching_split='test',
    #     verbose=args.vs)
    # return pred_qry_wordlist, score_now_split, score_now_test

def test_loop(args, graph_samplers, dataloader, search_split):
    from akgr.utils.stat_util import stat_scores_by_pattern
    # initialization
    # scores_stat = initialize_scores_stat(pattern_filtered)

    scores_all = []
    pattern_id_all = []

    niter = len(dataloader)
    for iter, sample in (pbar := tqdm(enumerate(dataloader, start=1), total=niter)):
        ans_wordlist_batch = sample['source']
        label_qry_wordlist_batch = sample['target']
        pattern_id_batch = sample['pattern_id']

        # label_qry_wordlist_batch = qry_tokenizer.batched_detokenize(tgt)
        # ans_wordlist_batch = ans_tokenizer.batched_detokenize(src)
        # print(tgt_str_bacth)
        # print(src_batch)

        for ans, qry, pattern_id in zip(ans_wordlist_batch, label_qry_wordlist_batch, pattern_id_batch):
            pred_qry_act, score_now_split, score_now_test = search_and_test(graph_samplers,
                search_split=search_split, tgt_pattern=pattern_id,
                ans_wordlist=ans, label_qry_wordlist=qry,
                heuristics=args.modelname)
            scores_all.append(score_now_test)
            pattern_id_all.append(pattern_id)
            score_df = stat_scores_by_pattern(scores_all, pattern_id_all, pattern_filtered)
        # exit()
        # break

        scores_path = os.path.join(args.result_root, args.modelname,\
            f'{args.dataname}-{args.scale}-{args.max_answer_size}-{args.modelname}-scores({search_split}).csv')
        score_df.to_csv(scores_path)

def my_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-dataloader', default='akgr/configs/config-dataloader.yml')
    parser.add_argument('--config-batchsize', default='akgr/configs/config-batchsize.yml')
    parser.add_argument('--overwrite_batchsize', type=int, default=0)

    parser.add_argument('--data_root', default='./sampled_data/')
    parser.add_argument('-d', '--dataname', default='FB15k-237')
    parser.add_argument('--scale', default='full')
    parser.add_argument('-a', '--max-answer-size', type=int, default=32)

    parser.add_argument('--result_root', default='./results/')
    parser.add_argument('--modelname', '--heuristics')
    parser.add_argument('--vs', action='store_true', help='verbose flag for smatch result')
    args = parser.parse_args()
    return args

def main():
    import pandas as pd
    from akgr.dataloader import new_create_dataloader, new_create_dataset
    from akgr.tokenizer import create_tokenizer

    global args
    args = my_parse_args()

    if not os.path.exists(os.path.join(args.result_root, args.modelname)):
        os.makedirs(os.path.join(args.result_root, args.modelname))

    config_dataloader = load_yaml(args.config_dataloader)
    offset = config_dataloader['offset']
    special_tokens = config_dataloader['special_tokens']

    global qry_tokenizer, ans_tokenizer

    global pattern_filtered
    pattern_filtered_path = 'akgr/metadata/pattern_filtered.csv'
    pattern_filtered = pd.read_csv(pattern_filtered_path, index_col='id')

    config_batchsize = load_yaml(args.config_batchsize)
    batch_size = config_batchsize['searching'][args.dataname]
    if args.overwrite_batchsize != 0:
        batch_size = args.overwrite_batchsize
    print(f'batch_size:{batch_size}\n')

    splits = ['train', 'valid', 'test']
    dataset_dict, nentity, nrelation = new_create_dataset(
        dataname=args.dataname,
        scale=args.scale,
        answer_size=args.max_answer_size,
        pattern_filtered=pattern_filtered,
        data_root=args.data_root,
        splits=splits,
        is_act=True,
    )
    dataloader_dict = new_create_dataloader(
        dataset_dict=dataset_dict,
        batch_size=batch_size,
        drop_last=False
    )

    # dataloader_dict = \
    #     create_dataloader(args.dataname, args.scale, args.max_answer_size,
    #                       config_dataloader, batch_size,
    #                       data_root=args.data_root, splits=['train', 'valid', 'test'],
    #                       is_ansqry=False,
    #                       is_action_based=False)
    tokenizer, ntoken = create_tokenizer(
        special_tokens=special_tokens,
        offset=offset,
        nentity=nentity,
        nrelation=nrelation,
        is_gpt=False,
    )
    # dataloader = dataloader_dict['dataloader']
    # pattern_filtered = dataloader_dict['pattern_filtered']
    # ans_tokenizer = dataloader_dict['ans_tokenizer']
    # qry_tokenizer = dataloader_dict['qry_tokenizer']

    # pattern_str = '(i,(p,(i,(p,(e)),(p,(e)))),(p,(p,(e))))'
    # pattern_str = '(i,(i,(p,(e)),(p,(p,(p,(e))))),(p,(e)))'
    # {"answers":[3585,5827,3082,3562,3563,6829,3180,3564,3580,12915,6810,3579,8700,3583],
    # "query":"(i,(p,(18),(i,(p,(8),(e,(3562))),(p,(385),(e,(5859))))),(p,(9),(p,(19),(e,(3580)))))",
    # "type":"(i,(p,(i,(p,(e)),(p,(e)))),(p,(p,(e))))"}
    # (i,(p,(8),(e,(3562))),(p,(385),(e,(5859))))
    # global graph_samplers
    kg = load_kg(args.dataname)
    graph_samplers = kg.graph_samplers
    # global num_ent
    # global num_rel
    # num_ent = kg.num_ent
    # num_rel = kg.num_rel
    # evaluation.init_global()
    # evaluation.graph_samplers = kg['graph_samplers']
    # evaluation.num_ent = kg['num_ent']

    # global query_pool
    # print('# Searching for testing dataset on testing graph\n')
    # test_loop(args, dataloader_dict['test'], 'test')
    print('# Searching for testing dataset on training graph\n')
    test_loop(args, graph_samplers, dataloader_dict['test'], 'train')


if __name__ == '__main__':
    main()