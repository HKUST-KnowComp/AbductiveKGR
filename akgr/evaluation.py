import json
import yaml
import networkx as nx


import os, sys

# sys.path.append('./utils/')
from akgr.utils.stat_util import compute_f1_rec_prec
from akgr.utils.parsing_util import qry_wordlist_2_graph, qry_actions_2_graph_wordlist, qry_actionlist_2_wordlist_v2, qry_actionstr_2_wordlist

# sys.path.append(os.path.abspath('./smatch/'))
from smatch import get_best_match as smatch_get_best_match
from smatch import print_alignment as smatch_print_alignment
from smatch import match_triple_dict
from smatch import compute_f as smatch_compute_f


def syntax_correct(input_token_word: list, verbose:bool=False):
    token_word = input_token_word[:]

    if len(token_word) == 0:
        if verbose:
            print('# Warning: correcting empty token_word, return None')
        return None

    corrected_tokens = []
    paren_cnt = 0

    opt = ['p', 'i', 'u', 'n']
    first_opt_pos = len(token_word)
    for o in opt:
        if not o in token_word: continue
        first_opt_pos = min(first_opt_pos, token_word.index(o))
    token_word = token_word[max(first_opt_pos-1, 0):]
    if len(token_word) == 0:
        if verbose:
            print('# Warning: token_word contains no operator, return None')
        return None

    if token_word[0] != '(':
        token_word = ['('] + token_word

    def safe_get_item(lst: list, pos: int):
        if pos >= len(lst): return None
        else: return lst[pos]

    stack = []
    i = 0
    while i < len(token_word):
        tkn = token_word[i]
        if tkn == '(':
            corrected_tokens.append(tkn)
            paren_cnt += 1
            # suff must be ent/rel/opt
            while i+1 < len(token_word) and safe_get_item(token_word, i+1) in ['(', ')']:
                i += 1
        elif tkn == 'e': # e, (, ent, )
            if corrected_tokens[-1] != '(':
                corrected_tokens.append('(')
                paren_cnt += 1
            corrected_tokens.append(tkn)
            stack.append(1)

            if safe_get_item(token_word, i+1) == '(': i += 1
            corrected_tokens.append('(')
            paren_cnt += 1

            if isinstance(safe_get_item(token_word, i+1), int):
                i += 1
                corrected_tokens.append(token_word[i])
            else:
                if verbose:
                    print('# Warning: no ent in "e", return None')
                return None

            if safe_get_item(token_word, i+1) == ')': i += 1
            corrected_tokens.append(')')
            paren_cnt -= 1

            while len(stack) > 0:
                stack[-1] -= 1
                if stack[-1] == 0:
                    stack.pop()
                    if safe_get_item(token_word, i+1) == ')': i += 1
                    corrected_tokens.append(')')
                    paren_cnt -= 1
                else: break
        elif tkn == 'p': # p, (, rel, ), (, sub ,)
            if corrected_tokens[-1] != '(':
                corrected_tokens.append('(')
                paren_cnt += 1
            corrected_tokens.append(tkn)
            stack.append(1)

            if safe_get_item(token_word, i+1) == '(': i += 1
            corrected_tokens.append('(')
            paren_cnt += 1

            if isinstance(safe_get_item(token_word, i+1), int):
                i += 1
                corrected_tokens.append(token_word[i])
            else:
                if verbose:
                    print('# Warning: no rel in "p", return None')
                return None

            if safe_get_item(token_word, i+1) == ')': i += 1
            corrected_tokens.append(')')
            paren_cnt -= 1
        elif tkn == 'i': # i, (, sub, ), (, sub, )
            if corrected_tokens[-1] != '(':
                corrected_tokens.append('(')
                paren_cnt += 1
            corrected_tokens.append(tkn)
            stack.append(2)
        elif tkn == 'u': # i ( sub ) ( sub )
            if corrected_tokens[-1] != '(':
                corrected_tokens.append('(')
                paren_cnt += 1
            corrected_tokens.append(tkn)
            stack.append(2)
        elif tkn == 'n': # n ( sub )
            if corrected_tokens[-1] != '(':
                corrected_tokens.append('(')
                paren_cnt += 1
            corrected_tokens.append(tkn)
            stack.append(1)
        elif tkn == ')':
            if paren_cnt > 0:
                corrected_tokens.append(')')
                paren_cnt -= 1
        i += 1
    while paren_cnt > 0:
        corrected_tokens.append(')')
        paren_cnt -= 1
    return corrected_tokens

def get_smatch_score(pred_graph: dict, tgt_graph: dict, verbose:bool=False):
    """
    relation: key, var1 -> var2
    attribute: key, var1 -> fixed2
    instance: key='instance', var1 -> fixed2
        instances var name should be reindexed from 0

    return: smatch score (F-score)
    """
    def is_var(u):
        return u < 0
    def extract_IRA(graph: dict, pref: str):
        R = []; A = []; I = []
        mapping_var_reid = {}
        for i, v in enumerate(graph['var']):
            mapping_var_reid[v] = i
            I.append(('instance', f'{pref}{i}', 'var'))
        for h, r, t in graph['edges']:
            if not is_var(h): # u is a fixed node
                A.append((str(r), f'{pref}{mapping_var_reid[t]}', str(h)))
            else: # both are variable
                R.append((str(r), f'{pref}{mapping_var_reid[t]}', f'{pref}{mapping_var_reid[h]}'))
        # print(I, R, A)
        total_edges = len(I) + len(R) + len(A)
        return I, R, A, mapping_var_reid, total_edges

    pref_pred = 'p'
    I_pred, R_pred, A_pred, mapping_var_reid_pred, num_edges_pred = extract_IRA(pred_graph, pref=pref_pred)
    pref_tgt = 't'
    I_tgt, R_tgt, A_tgt, mapping_var_reid_tgt, num_edges_tgt = extract_IRA(tgt_graph, pref=pref_tgt)

    # 1: test, 2: gold
    smatch_mapping, smatch_num = smatch_get_best_match(
        instance1=I_pred, attribute1=A_pred, relation1=R_pred,
        instance2=I_tgt, attribute2=A_tgt, relation2=R_tgt,
        prefix1='p', prefix2='t', doinstance=True, doattribute=True, dorelation=True)

    smatch_score = smatch_compute_f(match_num=smatch_num,
        test_num=num_edges_pred,
        gold_num=num_edges_tgt)
    if verbose:
        print('#' * 20)
        print(pred_graph)
        print('I:', I_pred); print('R:', R_pred); print('A:', A_pred)
        print(tgt_graph)
        print('I:', I_tgt); print('R:', R_tgt); print('A:', A_tgt)
        print('# smatch_result')
        print(smatch_mapping, smatch_num)
        print('# alignment', smatch_print_alignment(smatch_mapping, I_pred, I_tgt))
        print('# score')
        print(smatch_score)
    # important !
    match_triple_dict.clear()
    return smatch_score[-1]

from akgr.utils.parsing_util import qry_unshift_indices, ans_unshift_indices, qry_str_2_wordlist, qry_actionstr_2_actionlist
def get_ans_score(
    pred_qry_wordlist: list, label_ans,
    scoring_method:str,
    graph_samplers, searching_split,
    return_ans:bool=False,
    verbose:bool=False):
    # if label_ans == None:
    #     label_ans = graph_samplers['test'].search_answers_to_query(label_qry_wordlist)
    # print('pred qry')
    # print(pred_qry_wordlist)
    pred_qry_unshifted = qry_unshift_indices(pred_qry_wordlist)
    # print('unshifted')
    # print(pred_qry_unshifted)
    pred_ans_unshifted = graph_samplers[searching_split].search_answers_to_query(pred_qry_unshifted)
    # print('pred ans unshifted')
    # print(pred_ans_unshifted)
    label_ans_unshifted = ans_unshift_indices(label_ans)

    if verbose:
        print('label_ans')
        print(label_ans_unshifted)
        print('pred_ans')
        print(pred_ans_unshifted)
        print('common_ans')
        print(list(set(label_ans_unshifted) & set(pred_ans_unshifted)))
    TP = len(list(set(label_ans_unshifted) & set(pred_ans_unshifted)))
    I = TP
    score_dict = {}
    if 'precrecf1' in scoring_method:
        PP = len(pred_ans_unshifted)
        # print(PP)
        P = len(label_ans_unshifted)

        f1, rec, prec = compute_f1_rec_prec(p=P, pp=PP, tp=TP)
        if verbose:
            print('f1 rec prec', f1, rec, prec)
        score_dict.update({'f1':f1, 'rec':rec, 'prec':prec})
    if 'jaccard' in scoring_method:
        U = len(list(set(label_ans_unshifted).union(set(pred_ans_unshifted))))
        score_dict['jaccard'] = I / U
    if return_ans:
        return score_dict, pred_ans_unshifted
    else:
        return score_dict

def scoring_input_wordlist(
        pred_qry_wordlist: list, label_qry_wordlist=None, ans_wordlist=None,
        scoring_method: list=['smatch', 'jaccard'],
        do_correction: bool=False,
        graph_samplers=None,
        searching_split=None,
        return_ans:bool=False,
        verbose:bool=False):
    if ('jaccard' in scoring_method or 'f1precrec' in scoring_method) and searching_split == None:
        raise Exception('searching_split must be provided answer-based scoring method is specified')
    # label_ans has been detokenized
    # original_pred_qry_wordlist = qry_str_2_original_qry_str(pred_qry_wordlist)
    # if label_qry_wordlist is not None:
    #     original_label_qry_wordlist = qry_str_2_original_qry_str(label_qry_wordlist)
    score_dict_now = {}
    ans = None
    if 'precrecf1' in scoring_method: score_dict_now.update({'f1':0, 'rec':0, 'prec':0})
    if 'smatch' in scoring_method: score_dict_now['smatch'] = 0
    if 'jaccard' in scoring_method: score_dict_now['jaccard'] = 0
    if 'count0' in scoring_method: score_dict_now['count0'] = 1
    pred_word_corrected = pred_qry_wordlist
    if do_correction:
        pred_word_corrected = syntax_correct(pred_qry_wordlist, verbose=verbose)
        if verbose:
            print('pred corrected')
            print(pred_word_corrected)
        # 1st validation
        if pred_word_corrected == None:
            (score_dict_now if not return_ans else (score_dict_now, [])) # todo
    label_graph = qry_wordlist_2_graph(label_qry_wordlist)
    pred_graph = qry_wordlist_2_graph(pred_word_corrected)
    # 2nd validation
    if pred_graph == None:
        if verbose:
            print('# token_2_edges None')
        return (score_dict_now if not return_ans else (score_dict_now, []))# todo
    if 'count0' in scoring_method: score_dict_now['count0'] = 0
    if 'smatch' in scoring_method:
        # pred_graph = qry_wordlist_2_graph(pred_qry_wordlist)
        # label_graph = qry_wordlist_2_graph(label_qry_wordlist)
        score = get_smatch_score(pred_graph, label_graph, verbose=verbose)
        score_dict_now['smatch'] = score
    if 'precrecf1' in scoring_method or 'jaccard' in scoring_method:
        output = get_ans_score(
            pred_qry_wordlist=pred_word_corrected,
            scoring_method=scoring_method,
            label_ans=ans_wordlist,
            graph_samplers=graph_samplers,
            searching_split=searching_split,
            return_ans=return_ans,
            verbose=verbose)
        score_dict = output if not return_ans else output[0]
        score_dict_now.update(score_dict)
        if return_ans:
            ans = output[1]
        # print('')
    if return_ans:
        return score_dict_now, ans
    else:
        return score_dict_now

def scoring_input_wordlist_batch(
        pred_word_batch: list, label_word_batch: list, ans_word_batch: list,
        scoring_method: list,
        do_correction: bool=True,
        graph_samplers=None,
        searching_split=None,
        return_failures:bool=False,
        return_ans:bool=False,
        verbose:bool=False):
    scores = []
    ans = []
    failures = []
    score_dict_zero = {}
    if 'precrecf1' in scoring_method: score_dict_zero.update({'f1':0, 'rec':0, 'prec':0})
    if 'smatch' in scoring_method: score_dict_zero['smatch'] = 0
    if 'jaccard' in scoring_method: score_dict_zero['jaccard'] = 0
    if 'count0' in scoring_method: score_dict_zero['count0'] = 1
    for i, (pred_qry, label_qry, label_ans) in \
        enumerate(zip(pred_word_batch, label_word_batch, ans_word_batch)):
        pred_qry_wordlist = qry_str_2_wordlist(pred_qry) if isinstance(pred_qry, str) else pred_qry
        label_qry_wordlist = qry_str_2_wordlist(label_qry) if isinstance(label_qry, str) else label_qry
        label_ans_wordlist = qry_str_2_wordlist(label_ans) if isinstance(label_ans, str) else label_ans
        if verbose:
            print('label')
            print(label_qry)
            print('pred')
            print(pred_qry)

        output = scoring_input_wordlist(
            pred_qry_wordlist=pred_qry_wordlist,
            label_qry_wordlist=label_qry_wordlist,
            ans_wordlist=label_ans_wordlist,
            scoring_method=scoring_method,
            do_correction=do_correction,
            graph_samplers=graph_samplers,
            searching_split=searching_split,
            return_ans=return_ans,
            verbose=verbose,
        )
        score_dict_now = output if not return_ans else score_dict_now[0]
        scores.append(score_dict_now)
        if return_ans:
            ans.append(output[1])
        if return_failures and score_dict_now == score_dict_zero:
            failures.append(i)
    # print(scores)
    # print('-'*50)
    return scores if not return_failures else (scores, failures)

def scoring_input_act_batch(
        pred_word_batch, label_word_batch, ans_word_batch,
        scoring_method: list,
        do_correction: bool=False,
        graph_samplers=None,
        searching_split=None,
        return_failures:bool=False,
        return_ans:bool=False,
        verbose:bool=False):
    scores = []
    ans = []
    failures = []
    score_dict_zero = {}
    if 'precrecf1' in scoring_method: score_dict_zero.update({'f1':0, 'rec':0, 'prec':0})
    if 'smatch' in scoring_method: score_dict_zero['smatch'] = 0
    if 'jaccard' in scoring_method: score_dict_zero['jaccard'] = 0
    if 'count0' in scoring_method: score_dict_zero['count0'] = 1
    for i, (pred_act, label_act, label_ans) in \
        enumerate(zip(pred_word_batch, label_word_batch, ans_word_batch)):

        if verbose:
            print('label')
            print(pred_act)
            print('pred')
            print(label_act)

        try:
            # if is_v2 == False:
                # _, pred_qry_wordlist = qry_actions_2_graph_wordlist(pred_act)
            # else:
            qry_action_2_wordlist_fn = qry_actionstr_2_wordlist if isinstance(pred_act, str) else qry_actionlist_2_wordlist_v2
            pred_qry_wordlist = qry_action_2_wordlist_fn(pred_act)
        except:
            if return_failures: failures.append(i)
            if return_ans: ans.append([])
            scores.append(score_dict_zero)
            continue
        # print('label')
        # print(label_act)
        # print('pred qry (from act)')
        # print(pred_qry_wordlist)
        # # if is_v2 == False:
        #     _, label_qry_wordlist = qry_actions_2_graph_wordlist(label_act)
        # else:
        qry_action_2_wordlist_fn = qry_actionstr_2_wordlist if isinstance(pred_act, str) else qry_actionlist_2_wordlist_v2
        label_qry_wordlist = qry_action_2_wordlist_fn(label_act)
        if isinstance(label_ans, str):
            label_ans_wordlist = qry_actionstr_2_actionlist(label_ans)
        else:
            label_ans_wordlist = label_ans
        output = scoring_input_wordlist(
            pred_qry_wordlist=pred_qry_wordlist,
            label_qry_wordlist=label_qry_wordlist,
            ans_wordlist=label_ans_wordlist,
            scoring_method=scoring_method,
            do_correction=do_correction,
            graph_samplers=graph_samplers,
            searching_split=searching_split,
            return_ans=return_ans,
            verbose=verbose,
        )
        # print(output)
        score_dict_now = output if not return_ans else output[0]
        scores.append(score_dict_now)
        if return_ans:
            ans.append(output[1])
        if return_failures and score_dict_now == score_dict_zero:
            failures.append(i)
    if return_ans:
        return scores, ans
    else:
        return scores if not return_failures else (scores, failures)