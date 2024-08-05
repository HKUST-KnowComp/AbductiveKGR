import pandas as pd
import torch

from akgr.abduction_model.main import ppo_suffix_name

def my_parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    # Configurations
    # parser.add_argument('--mode', default='testing')
    parser.add_argument('--config-dataloader', default='akgr/configs/config-dataloader.yml')
    parser.add_argument('--config-train', default='akgr/configs/config-train.yml')
    parser.add_argument('--config-model', default='akgr/configs/config-model.yml')
    parser.add_argument('--config-batchsize', default='akgr/configs/config-batchsize.yml')
    parser.add_argument('--config_case_study', default='akgr/configs/config-case-study.yaml')
    parser.add_argument('--overwrite_batchsize', type=int, default=0)

    # Data
    parser.add_argument('--data_root', default='./sampled_data/')
    parser.add_argument('-d', '--dataname', default='FB15k-237')
    parser.add_argument('--scale', default='debug')
    parser.add_argument('-a', '--max-answer-size', type=int, default=32)

    parser.add_argument('--vs', action='store_true', help='verbose flag for smatch result')
    parser.add_argument('--do_correction', action='store_true', help='verbose flag for smatch result')

    # Checkpoint
    parser.add_argument('--checkpoint_root', default='./ckpt/')
    parser.add_argument('--ppo_resume_epoch', default=0)

    parser.add_argument('--case_study_root', default='./case_study/')
    args = parser.parse_args()
    return args

def main():
    args = my_parse_args()

    from akgr.utils.load_util import load_yaml

    # Data representation
    global config_dataloader
    config_dataloader = load_yaml(args.config_dataloader)
    global offset, special_tokens
    offset = config_dataloader['offset']
    special_tokens = config_dataloader['special_tokens']
    print(f'config_dataloader:\n{config_dataloader}\n')

    global pattern_filtered
    pattern_filtered_path = 'akgr/metadata/pattern_filtered.csv'
    pattern_filtered = pd.read_csv(pattern_filtered_path, index_col='id')

    from akgr.dataloader import new_create_dataloader, new_create_dataset
    splits = ['test']
    print('Creating dataset & dataloader')
    global nentity, nrelation
    dataset_dict, nentity, nrelation = new_create_dataset(
        dataname=args.dataname,
        scale=args.scale,
        answer_size=args.max_answer_size,
        pattern_filtered=pattern_filtered,
        data_root=args.data_root,
        splits=splits,
        is_act=True
    )
    nrows = dataset_dict['test'].shape[0]
    import random
    sample_indices = random.sample(range(nrows), 64)
    dataset_dict['test'] = dataset_dict['test'].select(sample_indices)
    batchsize = 1

    dataloader_dict = new_create_dataloader(
        dataset_dict=dataset_dict,
        batch_size=batchsize,
        drop_last=False, #or (args.mode == 'testing' and args.accelerate)
        shuffle=False,
    )

    # Graphs (for evaluation)
    from akgr.kgdata import load_kg
    print('Loading graph')
    kg = load_kg(args.dataname)
    graph_samplers = kg.graph_samplers

    case_study_all = dict()
    case_study_all['sample_indices'] = sample_indices

    # ================== model-independent part above ===================
    config_case_study = load_yaml(args.config_case_study)
    from argparse import Namespace

    device = 'cuda' if torch.cuda.is_available() else 'cpu'



    for case_study in config_case_study:
        model_name = case_study['modelname']
        case_study_args = Namespace(
            mode='testing', modelname=model_name,
            dataname=args.dataname, data_root=args.data_root, scale=args.scale, max_answer_size=args.max_answer_size,
            checkpoint_root=args.checkpoint_root, **case_study[args.dataname])
        print(f'# Case study with {model_name}')
        print(case_study_args)
        case_study_all_now = []

        is_ml_model = 'T5' in model_name or 'GPT2' in model_name

        if is_ml_model:
            # Model information
            is_gpt=('GPT2' in model_name)
            is_act=('act' in model_name)
            tgt_len = config_dataloader['act_len'] + 1 if is_act else config_dataloader['qry_len'] + 1
            src_len = config_dataloader['ans_len'] + 1
            print(f'model_name:{model_name}\n')

            # Tokenizer
            print('Creating tokenizer')
            from akgr.tokenizer import create_tokenizer
            tokenizer, ntoken = create_tokenizer(
                special_tokens=special_tokens,
                offset=offset,
                nentity=nentity,
                nrelation=nrelation,
                is_gpt=is_gpt
            )

            common_generation_kwargs = {
                "min_length": -1,
                "top_k": 0.0, # recommended by trl
                "top_p": 1.0,
                'max_length':tgt_len + src_len * (is_gpt == True),
                'pad_token_id':tokenizer.pad_token_id, # same for both tokenizers
                'bos_token_id':tokenizer.bos_token_id,
                'eos_token_id':tokenizer.eos_token_id,
                'do_sample':True,
                # 'prefix_allowed_tokens_fn':prefix_allowed_tokens_fn
            }

            # Model
            # config_model = load_yaml(args.config_model)
            from akgr.abduction_model.main import load_model_by_mode, test_loop
            model = load_model_by_mode(
                args=case_study_args, device=device, model_name=model_name, is_gpt=is_gpt,
                config_model=None, ntoken=None, config_train=None)

        # Case study
        from tqdm import tqdm
        from akgr.tokenizer import new_extract_sample_to_device
        from akgr.abduction_model.main import mask_source
        from akgr.rule_based_model.search import search_and_test
        from akgr.evaluation import scoring_input_act_batch
        from akgr.utils.parsing_util import qry_tokenizer_2_kg_act, ans_tokenizer_2_kg, map_idlist_2_idnamelist
        dataloader = dataloader_dict['test']
        niter = len(dataloader)
        for iter, sample in (pbar := tqdm(enumerate(dataloader, start=1), total=niter)):
            # if iter > 10: break
            if is_ml_model:
                source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask = \
                    new_extract_sample_to_device(device, sample, tokenizer, is_gpt, src_len, tgt_len, True)
                pred = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **common_generation_kwargs)
                if is_gpt: mask_source(device, source_attention_mask, pred, tokenizer)
                pred_decoded = tokenizer.batch_decode(pred, skip_special_tokens=True)
                scoring_fn = scoring_input_act_batch
                score_now, pred_answers_kg = scoring_fn(
                    pred_word_batch=pred_decoded,
                    label_word_batch=target,
                    ans_word_batch=source,
                    scoring_method=['smatch', 'precrecf1', 'jaccard'],
                    do_correction=args.do_correction,
                    graph_samplers=graph_samplers,
                    searching_split='test',
                    return_failures=False,
                    return_ans=True,
                    verbose=args.vs)
            else:
                source = sample['source']
                target = sample['target']
                pattern_id = sample['pattern_id']
                pred_decoded, _, score_now, pred_answers_kg = search_and_test(graph_samplers,
                    search_split='train',
                    tgt_pattern=pattern_id[0],
                    ans_wordlist=source[0],
                    label_qry_wordlist=target[0],
                    heuristics=model_name,
                    return_ans=True)


            # print('>' * 20)
            # print('source')
            answers_kg = ans_tokenizer_2_kg(source[0])
            answers_kg_idnamelist = map_idlist_2_idnamelist(kg, answers_kg)
            # print(source, answers_kg, answers_kg_idnamelist)
            # print('target')
            target_query_kg = qry_tokenizer_2_kg_act(target[0])
            target_query_kg_idnamelist = map_idlist_2_idnamelist(kg, target_query_kg)
            # print(target, target_query_kg, target_query_kg_idnamelist)
            # print('pred')
            # print(pred_decoded)
            pred_query_kg = qry_tokenizer_2_kg_act(pred_decoded[0] if is_ml_model else pred_decoded)
            pred_query_kg_idnamelist = map_idlist_2_idnamelist(kg, pred_query_kg)
            # print(pred_decoded, pred_query_kg, pred_query_kg_idnamelist)
            if is_ml_model:
                pred_answers_kg = pred_answers_kg[0]
            pred_answers_kg = sorted(pred_answers_kg)

            pred_answers_kg_idnamelist = map_idlist_2_idnamelist(kg, pred_answers_kg)
            # print(score_now)
            # print('<' * 20)

            case_study_dict = dict()
            case_study_dict['answers'] = answers_kg_idnamelist
            case_study_dict['target_query'] = ''.join(target_query_kg_idnamelist)
            case_study_dict['predicted_query'] = ''.join(pred_query_kg_idnamelist)
            case_study_dict['predicted_answers'] = pred_answers_kg_idnamelist
            case_study_dict['score'] = score_now
            case_study_all_now.append(case_study_dict)

        case_study_all[f'{model_name}, {case_study[args.dataname]["resume_epoch"]}, {case_study[args.dataname]["ppo_resume_epoch"]}'] = case_study_all_now
        if is_ml_model:
            del model
    case_study_df = pd.DataFrame.from_dict(case_study_all)
    import os
    case_study_path = os.path.join(args.case_study_root, args.dataname, 'case_study.json')
    os.makedirs(os.path.dirname(case_study_path), exist_ok=True)
    case_study_df.to_json(case_study_path, indent=2, force_ascii=False)

if __name__ == '__main__':
    main()