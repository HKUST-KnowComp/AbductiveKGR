import os, sys, argparse, warnings

import json
import yaml
import pandas as pd

import torch
from tqdm import tqdm

import random

# dataloader
from akgr.dataloader import new_create_dataloader, new_create_dataset
from akgr.tokenizer import create_tokenizer, new_extract_sample_to_device

# transformer (huggingface)
from akgr.abduction_model.transformer import create_transformer

# utils
from akgr.utils.stat_util import stat_scores_by_pattern#, initialize_scores_stat
from akgr.utils.load_util import load_yaml, load_model, save_model, load_and_filter_query_patterns
from akgr.kgdata import load_kg
import pandas as pd

# evaluation
from akgr.evaluation import scoring_input_wordlist_batch, scoring_input_act_batch
from akgr.utils.parsing_util import qry_actionprefix_get_branching, is_strint

import wandb
from accelerate import Accelerator

raw_dataset = None

# global var
device = None
pattern_filtered = None

nentity = None
nrelation = None
offset = None
special_tokens = None
# rel_2_allowed_headent = None
# tailent_2_allowed_rel = None

def train_loop(dataloader, model, tokenizer, optimizer, scheduler, model_name,
               is_gpt, is_act, src_len, tgt_len, accelerator):
    # https://pytorch.org/docs/stable/optim.html
    model.train()
    niter = len(dataloader)
    total_loss = 0

    for iter, sample in (pbar := tqdm(enumerate(dataloader), total=niter)):
        # a list of tensors
        source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask = \
            new_extract_sample_to_device(device, sample, tokenizer, is_gpt, src_len, tgt_len, False)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # print(outputs)
        logits, loss = outputs.logits, outputs.loss
        # print(loss)
        pred_argmax = logits.argmax(2)

        _loss = loss.detach().cpu().numpy()

        pbar.set_description(f'loss: {_loss}')
        total_loss += _loss

        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if not ('-5' in model_name or '-01' in model_name):
            scheduler.step()
    return total_loss / niter

def mask_source(device, source_attention_mask, pred, tokenizer):
    # print('source mask')
    # print(source_attention_mask[:3, :15])
    B = pred.shape[0]
    diff = pred.shape[-1] - source_attention_mask.shape[-1]
    prefix_mask = torch.cat([
        source_attention_mask,
        torch.zeros((B, diff), dtype=torch.bool, device=device)], dim=1).to(device)
    # print('prefix mask')
    # print(prefix_mask[:3, :15])
    pred[prefix_mask == 1] = tokenizer.pad_token_id

def valid_loop(args, dataloader, model, tokenizer, graph_samplers,
               is_gpt, is_act, src_len, tgt_len, verbose):
    model.eval()
    niter = len(dataloader)
    total_loss = 0

    # initialization
    # scores_stat = initialize_scores_stat(pattern_filtered)
    scores_all = []
    pattern_id_all = []


    with torch.no_grad():
        for iter, sample in (pbar := tqdm(enumerate(dataloader, start=1), total=niter)):
            source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask = \
                new_extract_sample_to_device(device, sample, tokenizer, is_gpt, src_len, tgt_len, False)

            # print('src, tgt shapes:', src.shape, tgt.shape)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # print(outputs)
            logits, loss = outputs.logits, outputs.loss
            pred_argmax = logits.argmax(2)

            if is_gpt: mask_source(device, source_attention_mask, pred_argmax, tokenizer)
            pred_decoded = tokenizer.batch_decode(pred_argmax, skip_special_tokens=True)

            scoring_fn = scoring_input_act_batch if is_act else scoring_input_wordlist_batch
            scores = scoring_fn(
                pred_word_batch=pred_decoded,
                label_word_batch=target,
                ans_word_batch=source,
                scoring_method=['smatch'],
                do_correction=args.do_correction,
                graph_samplers=graph_samplers,
                verbose=args.vs)
            scores_all.extend(scores)
            pattern_id_all.extend(pattern_id)
            score_df = stat_scores_by_pattern(scores, pattern_id, pattern_filtered)

            _loss = loss.detach().cpu().numpy()
            total_loss += _loss

            pbar.set_description(f'loss ave: {total_loss/iter}, s: {round(score_df.loc["all", ("smatch","mean")], 4)}')
        # scores_ave = scores_sum / scores_cnt
    return total_loss / niter, score_df

def fit(args, nepoch, dataloader, model, tokenizer, optimizer, scheduler, graph_samplers,
        model_name, is_gpt, is_act, src_len, tgt_len,
        last_epoch, loss_log, verbose, accelerator):
    if not accelerator is None:
        model, optimizer, dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )
    result_path = os.path.join(args.result_root, args.modelname,
        f'{args.dataname}-{args.scale}-{args.max_answer_size}_results.txt')

    for epoch in range(last_epoch+1, nepoch+1): # epoch starts from 1
        print('lr:', scheduler.get_last_lr())
        loss_train = train_loop(
            dataloader['train'],
            model,
            tokenizer,
            optimizer, scheduler,
            model_name, is_gpt, is_act, src_len, tgt_len,
            accelerator)
        loss_valid, score_df = valid_loop(args,
            dataloader['valid'], model,
            tokenizer,
            graph_samplers,
            is_gpt, is_act, src_len, tgt_len,
            verbose)
        if ('-5' in model_name or '-01' in model_name):
            scheduler.step()
        # exit()
        loss_log['train'][epoch] = loss_train
        loss_log['valid'][epoch] = loss_valid

        msg = f'epoch: {epoch}, train loss: {loss_train}, '\
            + f'valid loss: {loss_valid}, s (valid): {score_df.loc["all",("smatch","mean")]}'
        print(msg)
        with open(result_path, 'a') as result_file:
            result_file.write(msg + '\n')

        # Saving checkpoint
        if epoch % args.save_frequency == 0 \
            or epoch == nepoch \
            or (len(loss_log['valid'].values()) > 0 and loss_valid <= min(loss_log['valid'].values())):
            ckpt_path = os.path.join(args.checkpoint_root, args.modelname,\
                f'{args.dataname}-{args.scale}-{args.max_answer_size}-{epoch}.pth')
            save_model(ckpt_path, 'model', model, optimizer, scheduler, epoch, loss_log)

        if epoch % args.save_frequency == 0 or epoch == nepoch:
            scores_path = os.path.join(args.result_root, args.modelname,\
                f'{args.dataname}-{args.scale}-{args.max_answer_size}-{epoch}-scores.csv')
            score_df.to_csv(scores_path)

        print('=' * 50)

class Prefix_allowed_tokens_fn:
    def __init__(self, offset, nentity, nrelation, special_tokens, tokenizer):
        self.offset = offset
        self.nentity = nentity
        self.nrelation = nrelation
        self.special_tokens = special_tokens
        # self.bos_token_id = tokenizer.bos_token_id
        # self.sep_token_id = tokenizer.sep_token_id
        # self.eos_token_id = tokenizer.eos_token_id
        # self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        self.iun_ids = tokenizer.convert_tokens_to_ids(['i', 'u', 'n'])
    def get_gathered_tokens(self) -> list:
        return list(range(self.offset + self.nentity + self.nrelation))
    def get_non_special_tokens(self) -> list:
        return self.iun_ids + list(range(self.offset, self.offset + self.nentity + self.nrelation))
    # def get_p_allowed_tokens(offset:int, nentity:int, nrelation:int, rel: int) -> list:
    #     # return list(range(offset, offset + nentity))
    #     return [15,16,17]\
    #         + ([offset + h for h in rel_2_allowed_headent[rel]] if rel in rel_2_allowed_headent \
    #             else list(range(offset, offset + nentity)))\
    #         + list(range(offset + nentity, offset + nentity + nrelation))
    def get_iun_allowed_tokens(self) -> list:
        # ANY \ e
        return self.iun_ids\
                + list(range(self.offset + self.nentity, self.offset + self.nentity + self.nrelation))
    def __cgathered__(self, batch_id: int, input_ids: torch.LongTensor) -> list:
        if input_ids.shape[-1] <= 1:
            return self.get_gathered_tokens()
        is_gpt = (not input_ids[1] in self.get_iun_allowed_tokens())
        prefix_ids = list(input_ids)
        if is_gpt:
            if self.tokenizer.sep_token_id in prefix_ids:
                sep_pos = prefix_ids.index(self.tokenizer.sep_token_id)
                prefix_ids = prefix_ids[sep_pos:]
            else: # Query part does not appear
                return self.get_gathered_tokens()
        # print('prefix')
        # print(prefix_ids)
        last_action = prefix_ids[-1]

        if last_action in [self.tokenizer.bos_token_id, self.tokenizer.sep_token_id]:
            return self.get_non_special_tokens()
        elif last_action in self.iun_ids: # ANY \ e
            return self.get_iun_allowed_tokens()
        # elif operator == 'e':
        elif last_action >= offset and last_action < offset + nentity:
            # omit the first 'START' token
            actionstr_prefix = self.tokenizer.decode(prefix_ids, skip_special_tokens=True)
            branching = qry_actionprefix_get_branching(action_prefix=actionstr_prefix)
            if branching == 'EMPTY': # Query graph is complete, must return EOS
                return [self.tokenizer.eos_token_id]
            else: # i / u
                return self.get_iun_allowed_tokens()
        # elif operator == 'p': # ANY with constraints
        elif last_action >= offset + nentity:
            return self.get_non_special_tokens()
        else: # eos, pad, etc.
            return [self.tokenizer.pad_token_id]
def constrained_inference(args, model, input_ids, attention_mask, max_length,
              bos_token_id, eos_token_id, pad_token_id, tokenizer,
              is_gpt, is_constrained):
    """
    Reference:
    https://github.com/huggingface/transformers/blob/31ec2cb2badfbdd4c1ac9c6c9b8a74e974984206/src/transformers/generation_utils.py#L1622
    """
    # num_beams = 4
    if is_constrained:
        prefix_allowed_tokens_fn = Prefix_allowed_tokens_fn(offset=offset, nentity=nentity, nrelation=nrelation, special_tokens=special_tokens, tokenizer=tokenizer)
    else:
        prefix_allowed_tokens_fn = None
    # if is_constrained:
    #     prefix_constrained_logits_preprocessor = LogitsProcessorList([
    #         myPrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn=prefix_allowed_tokens_fn),
    #     ])
    # else:
    #     prefix_constrained_logits_preprocessor = None
    # input_len = input_ids.shape[-1]
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        min_length=-1,
        top_p=1.0,
        top_k=args.test_top_k,
        do_sample=True,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        # logits_processor=prefix_constrained_logits_preprocessor
    )
    return output

def test_loop(args, dataloader, model, tokenizer, graph_samplers, searching_split, resume_epoch,
            is_gpt, is_act, src_len, tgt_len,
            accelerator,
            score_file_suffix='test'):
    score_file_suffix = f'test|{args.test_proportion}x{args.test_split}_topk{args.test_top_k}_{args.constrained}_{args.test_count0}'
    if args.ppo_resume_epoch != 0:
        score_file_suffix += f'|{ppo_suffix_name(args, args.ppo_resume_epoch)}'

    # print(len(dataloader))
    if not accelerator is None:
        model, dataloader = accelerator.prepare(
            model, dataloader
        )
    # print(len(dataloader))
    model.eval()
    niter = len(dataloader)
    # total_loss = 0

    # initialization
    # scores_stat = initialize_scores_stat(pattern_filtered)
    scores_all = []
    pattern_id_all = []
    failures = []

    # print('# tgt_len', tgt_len)

    import torch.distributed as dist
    with torch.no_grad():
        for iter, sample in (pbar := tqdm(enumerate(dataloader, start=1),
                                          total=niter, disable=(accelerator is not None) and (not accelerator.is_local_main_process))):
            # gathered_sample = accelerator.gather_for_metrics(sample) if accelerator is not None else sample
            source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask = \
                new_extract_sample_to_device(device, sample, tokenizer, is_gpt, src_len, tgt_len, True)

            pred = constrained_inference(args,
                model if accelerator is None else accelerator.unwrap_model(model),
                input_ids, attention_mask,
                max_length=tgt_len + src_len * (is_gpt == True),
                bos_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                tokenizer=tokenizer,
                is_gpt=is_gpt,
                is_constrained=is_act and args.constrained)

            # print(input_ids.shape)

            # print('pred')
            # print(pred[:10])

            if is_gpt: mask_source(device, source_attention_mask, pred, tokenizer)
            pred_decoded = tokenizer.batch_decode(pred, skip_special_tokens=True)

            # print('source')
            # print(source[:10])
            # print('target (label)')
            # print(target[:10])
            # print('input_ids')
            # print(input_ids)
            # print('pred_de')
            # print(pred_decoded[:10])

            scoring_fn = scoring_input_act_batch if is_act else scoring_input_wordlist_batch
            scores, failures_batch_id = scoring_fn(
                pred_word_batch=pred_decoded,
                label_word_batch=target,
                ans_word_batch=source,
                scoring_method=['smatch', 'precrecf1', 'jaccard'] + ['count0'] * (args.test_count0 == True),
                do_correction=args.do_correction,
                graph_samplers=graph_samplers,
                searching_split=searching_split,
                return_failures=True,
                verbose=args.vs)
            # print(scores)
            if accelerator is not None:
                gathered_scores = [None] * accelerator.num_processes
                dist.all_gather_object(gathered_scores, scores)
                gathered_scores = [s for l in gathered_scores for s in l ]
                gathered_pattern_id = accelerator.gather(pattern_id)
            else:
                gathered_scores = scores
                gathered_pattern_id = pattern_id

            if (accelerator is None) or (accelerator.is_main_process):
                scores_all.extend(gathered_scores)
                pattern_id_all.extend(gathered_pattern_id)
                score_df = stat_scores_by_pattern(scores_all, pattern_id_all, pattern_filtered)
                # print(score_df)
                pbar.set_description(f's: {round(score_df.loc["all",("smatch","mean")], 4)}, j: {round(score_df.loc["all",("jaccard","mean")], 4)}')
                scores_path = os.path.join(args.result_root, args.modelname,\
                    f'{args.dataname}-{args.scale}-{args.max_answer_size}-{resume_epoch}-scores({score_file_suffix}).csv')

                score_df.to_csv(scores_path)

    return score_df

from trl import (PPOTrainer, PPOConfig,
    AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead,
    create_reference_model)
def ppo_suffix_name(args, iter):
    name = f'ppo_{args.ppo_lr}'\
            + f'_{args.ppo_smatch_factor}'\
            + f'_{args.ppo_init_kl_coef}'\
            + f'_{args.ppo_cliprange}'\
            + f'_{args.ppo_minibatch}'\
            + f'_{args.ppo_horizon}'\
            + (f'_{args.ppo_epochs}' if args.ppo_epochs != 4 else '')\
            + (f'_{args.ppo_share_embed_layer}' if args.ppo_share_embed_layer else '')\
            + ('_nodecay' if args.ppo_lr_no_decay else '')\
            + ('_peft' if args.ppo_use_peft else '')\
            + (f'_s{args.ppo_search_split}' if args.ppo_search_split != 'train' else '')\
            + f'x{args.ppo_proportion}'\
            + f'-{iter}'
    return name
def reward_fn(args, score:dict):
    return score['smatch'] * args.ppo_smatch_factor + score['jaccard'] * (1 - args.ppo_smatch_factor)
def ppo_train_loop(args, model, ppo_trainer, tokenizer, graph_samplers, is_gpt, is_act, src_len, tgt_len):
    print('Start PPO Tuning')
    # scores_all = []
    # pattern_id_all = []

    is_constrained = is_act and args.constrained
    if is_constrained:
        prefix_allowed_tokens_fn = Prefix_allowed_tokens_fn(offset=offset, nentity=nentity, nrelation=nrelation, special_tokens=special_tokens, tokenizer=tokenizer)
    else:
        prefix_allowed_tokens_fn = None
    generation_kwargs = {
        "min_length": -1,
        "top_k": args.ppo_top_k, # recommended by trl
        "top_p": 1.0,
        'max_length':tgt_len + src_len * (is_gpt == True),
        'pad_token_id':tokenizer.pad_token_id,
        'bos_token_id':tokenizer.bos_token_id,
        'eos_token_id':tokenizer.eos_token_id,
        'do_sample':True,
        'prefix_allowed_tokens_fn':prefix_allowed_tokens_fn
    }

    # Optimization
    niter = len(ppo_trainer.dataloader)
    print(f'niter: {niter}')

    if args.ppo_resume_epoch != 0:
        print(f'Resume optimization from {args.ppo_resume_epoch} + 1')
    for iter, batch in (pbar := tqdm(enumerate(ppo_trainer.dataloader, start=1),
                                     total=niter, disable=not ppo_trainer.accelerator.is_local_main_process)):
        if iter <= args.ppo_resume_epoch:
            continue
        source, target, pattern_id, input_ids = batch['source'], batch['target'], batch['pattern_id'], batch['input_ids']
        # pred_tgt = None
        # while pred_tgt is None:
        try:
            pred_tgt = ppo_trainer.generate(
                input_ids,
                return_prompt=False,
                **generation_kwargs,
            )
        except RuntimeError:
            warnings.warn('Used try-except to skip the "torch.multinomial `inf`, `nan` or element < 0" error')
            continue
        # if is_gpt: mask_source(device, source_attention_mask, pred_tgt, tokenizer)
        pred_tgt_decoded = tokenizer.batch_decode(pred_tgt, skip_special_tokens=True)
        # print('label')
        # print(target[:10])
        # print('pred')
        # print(pred_tgt_decoded[:10])
        # exit()

        scoring_fn = scoring_input_act_batch if is_act else scoring_input_wordlist_batch
        scores, failures_batch_id = scoring_fn(
            pred_word_batch=pred_tgt_decoded,
            label_word_batch=target,
            ans_word_batch=source,
            scoring_method=['smatch', 'jaccard'],
            do_correction=args.do_correction,
            graph_samplers=graph_samplers,
            searching_split=args.ppo_search_split,
            return_failures=True,
            verbose=args.vs)

        rewards = [torch.tensor(reward_fn(args, score), dtype=torch.float, device=ppo_trainer.current_device) for score in scores]

        # print(rewards)
        # print(input_ids)
        # print(pred_tgt)
        try:
            ppo_stats = ppo_trainer.step(
                queries=input_ids,
                responses=pred_tgt,
                scores=rewards)
        except IndexError:
            warnings.warn('Used try-except to escape the "IndexError: -1 ..." error')
            continue
        ppo_trainer.log_stats(ppo_stats, {'query': source, 'response': pred_tgt_decoded}, rewards)
        rewards_mean = ppo_stats['ppo/mean_scores']

        # scores_all.extend(scores)
        # pattern_id_all.extend(pattern_id)
        # score_df = stat_scores_by_pattern(scores_all, pattern_id_all, pattern_filtered)

        pbar.set_description((#f's: {round(score_df.loc["all",("smatch","mean")], 4)}, '
                              #f'j: {round(score_df.loc["all",("jaccard","mean")], 4)}, '
                              f'r: {round(rewards_mean.item(), 6)}, '
                              f'lr: {ppo_trainer.lr_scheduler.get_last_lr()}'))

        if iter % args.save_frequency == 0 or iter == niter:
            ckpt_dir = os.path.join(args.checkpoint_root, args.modelname,\
                f'{args.dataname}-{args.scale}-{args.max_answer_size}-{ppo_suffix_name(args, iter)}')
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
    ckpt_dir = os.path.join(args.checkpoint_root, args.modelname,\
        f'{args.dataname}-{args.scale}-{args.max_answer_size}-{ppo_suffix_name(args, "final")}')
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
def ppo_collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

def optimize(args, dataset, model, tokenizer, graph_samplers, batch_size,
             is_gpt, is_act, src_len, tgt_len):
    print('PPO Setting Up')
    # Prepare dataset
    dataset_path = os.path.join(
        args.data_root, args.dataname,
        f'{args.dataname}-{args.scale}-{args.max_answer_size}-train-a2q-{args.ppo_proportion}')
    if os.path.exists(dataset_path):
        from datasets import load_from_disk
        dataset = load_from_disk(dataset_path)
    else:
        def tokenize_fn(example):
            # example['input_ids'] = tokenizer(example['source'])['inpud_ids']
            example.update(tokenizer(example['source']))
            return example
        dataset = dataset.map(tokenize_fn)
        dataset.set_format(type="torch")
        dataset.save_to_disk(dataset_path)
    # print('dataset size', dataset.shape)
    # exit()
    # Create
    num_shared_layers = None
    pattern = None
    if args.ppo_share_embed_layer:
        num_shared_layers = -1 # arbitrary, not used
        pattern = 'transformer.wte.weight' if is_gpt else 'shared.weight'
    ref_model = create_reference_model(
        model=model,
        num_shared_layers=num_shared_layers,
        pattern=pattern)
    ppo_config = PPOConfig(
        seed=0, # default
        model_name=args.modelname,
        task_name=ppo_suffix_name(args, 0),
        ppo_epochs=args.ppo_epochs,
        learning_rate=args.ppo_lr,
        init_kl_coef=args.ppo_init_kl_coef,
        cliprange=args.ppo_cliprange,
        gamma=0.99,
        horizon=args.ppo_horizon,
        batch_size=batch_size,
        mini_batch_size=args.ppo_minibatch,
        remove_unused_columns=False, # Important. By default, it removes unrecognized columns if hf dataset is passed
        ratio_threshold=10.00,
        use_score_scaling=True,
        use_score_norm=True,
        early_stopping=True,
        optimize_cuda_cache=False, # setting it to True may cause the program to freeze
        log_with='wandb',
        )
    print(ppo_config)
    niter = dataset.shape[0] // batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_config.learning_rate)
    print(f'warm up steps: {int(niter * 0.1)}/{niter}')
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=int(niter * 0.1))
    if args.ppo_lr_no_decay:
        scheduler = scheduler_warmup
    else:
        scheduler_decay = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=niter, eta_min=args.ppo_lr * 0.1)
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler_warmup, scheduler_decay])
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model, ref_model=ref_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        dataset=dataset,
        data_collator=ppo_collator
        )
    # ppo_trainer.dataloader = dataloader

    ppo_train_loop(args=args, model=model, ppo_trainer=ppo_trainer, tokenizer=tokenizer, graph_samplers=graph_samplers,
                   is_gpt=is_gpt, is_act=is_act, src_len=src_len, tgt_len=tgt_len)
    return model

def load_model_by_mode(args, device, model_name, is_gpt, config_model=None, ntoken=None, config_train=None):
    if args.mode in ['training', 'testing', 'optimizing'] and args.resume_epoch != 0:
        resume_path = os.path.join(args.checkpoint_root, args.modelname,\
            f'{args.dataname}-{args.scale}-{args.max_answer_size}-{args.resume_epoch}.pth')
        print(f'Loading model: {resume_path}')
        model, optimizer, scheduler, last_epoch, loss_log = \
            load_model(resume_path, 'model', args.resume_epoch, return_huggingface_model=True)
        model.to(device)
        # Overwrite model name
        model.model_name = model_name

    if args.mode == 'training' and args.resume_epoch == 0:
        print('Creating model')
        model = create_transformer(
            ntoken=ntoken,
            special_tokens=special_tokens,
            model_name=model_name,
            config_model=config_model
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(),
            lr=float(config_train["lr"]))
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
            start_factor=0.1, total_iters=config_train["warm_up"])
        last_epoch=0
        loss_log = {'train': {}, 'valid': {}}

    trl_model_class = AutoModelForCausalLMWithValueHead if is_gpt else AutoModelForSeq2SeqLMWithValueHead

    if args.mode in ['optimizing', 'testing'] and args.ppo_resume_epoch != 0: # Load TRL model wrapper directly
        resume_dir = os.path.join(args.checkpoint_root, args.modelname,\
            f'{args.dataname}-{args.scale}-{args.max_answer_size}-{ppo_suffix_name(args, args.ppo_resume_epoch)}')
        print(f'Loading model: {resume_dir}')
        if args.ppo_use_peft:
            from peft import PeftModel, PeftConfig
            # peft_config = PeftConfig.from_pretrained(resume_dir)
            model = PeftModel.from_pretrained(model, model_id=resume_dir)
        else:
            model = trl_model_class.from_pretrained(resume_dir)
        model.to(device)

    if args.mode == 'optimizing' and args.ppo_resume_epoch == 0: # Convert to TRL wrapper.
        if args.ppo_use_peft:
            from peft import get_peft_model, LoraConfig
            lora_config = LoraConfig(
                r=4 if is_gpt else 8, # 4 to 16 for GPT-2 as suggested in the LoRA Paper
                lora_alpha=32 if is_gpt else 8,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM" if is_gpt else "SEQ_2_SEQ_LM",
            )
            model = get_peft_model(model, peft_config=lora_config)
        model = trl_model_class(pretrained_model=model)
        model.is_peft_model = args.ppo_use_peft
        print('is peft?:', model.is_peft_model)
        # model.is_peft_model = False

    print('model.config:')
    print(model.config)

    if args.mode == 'training': return model, optimizer, scheduler, last_epoch, loss_log
    else: return model

def my_parse_args():
    parser = argparse.ArgumentParser()

    # Configurations
    parser.add_argument('--modelname')
    parser.add_argument('--config-dataloader', default='akgr/configs/config-dataloader.yml')
    parser.add_argument('--config-train', default='akgr/configs/config-train.yml')
    parser.add_argument('--config-model', default='akgr/configs/config-model.yml')
    parser.add_argument('--config-batchsize', default='akgr/configs/config-batchsize.yml')
    parser.add_argument('--overwrite_batchsize', type=int, default=0)

    # Data
    parser.add_argument('--data_root', default='./sampling/')
    parser.add_argument('-d', '--dataname', default='FB15k-237')
    parser.add_argument('--scale', default='debug')
    parser.add_argument('-a', '--max-answer-size', type=int, default=32)

    # Checkpoint
    parser.add_argument('--checkpoint_root', default='./ckpt/')
    parser.add_argument('-r', '--resume_epoch', type=int, default=0)

    parser.add_argument('--vs', action='store_true', help='verbose flag for smatch result')
    parser.add_argument('--do_correction', action='store_true', help='verbose flag for smatch result')

    # Testing
    parser.add_argument('--test_proportion', type=float, default=1)
    parser.add_argument('--test_split', default='test')
    parser.add_argument('--test_top_k', type=int, default=0)
    parser.add_argument('--test_count0', action='store_true')
    parser.add_argument('--result_root', default='./results/')

    parser.add_argument('--save_frequency', type=int, default=1)

    # PPO
    parser.add_argument('--ppo_resume_epoch', default=0)
    parser.add_argument('--ppo_proportion', type=float, default=1)
    parser.add_argument('--ppo_smatch_factor', type=float, default=0)
    parser.add_argument('--ppo_init_kl_coef', type=float, default=0.2)
    parser.add_argument('--ppo_cliprange', type=float, default=0.2)
    parser.add_argument('--ppo_minibatch', type=int, default=1)
    parser.add_argument('--ppo_horizon', type=int, default=10000)
    parser.add_argument('--ppo_lr', type=float)
    parser.add_argument('--ppo_epochs', type=int, default=4)
    parser.add_argument('--ppo_search_split', default='train')
    parser.add_argument('--ppo_share_embed_layer', action='store_true')
    parser.add_argument('--ppo_lr_no_decay', action='store_true')
    parser.add_argument('--ppo_use_peft', action='store_true')
    parser.add_argument('--ppo_top_k', default=0.0)

    parser.add_argument('--mode')
    parser.add_argument('--accelerate', action='store_true')
    parser.add_argument('--constrained', action='store_true')

    # parser.add_argument('--wandb_run_id', default=None)

    args = parser.parse_args()
    return args

def main():
    args = my_parse_args()
    print(f'# Running main.py in {args.mode} mode with:')
    print(f'args:\n{args}\n')

    if not os.path.exists(os.path.join(args.result_root, args.modelname)):
        os.makedirs(os.path.join(args.result_root, args.modelname))

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

    # Graphs (for evaluation)
    print('Loading graph')
    kg = load_kg(args.dataname)
    graph_samplers = kg.graph_samplers

    # Device
    global device
    if args.accelerate and args.mode != 'optimizing':
        accelerator = Accelerator()
        device = accelerator.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'DEVICE: {device}')

    # Model information
    model_name = args.modelname
    is_gpt=('GPT2' in model_name)
    is_act=('act' in model_name)
    tgt_len = config_dataloader['act_len'] + 1 if is_act else config_dataloader['qry_len'] + 1
    src_len = config_dataloader['ans_len'] + 1
    print(f'model_name:{model_name}\n')

    # Batch size
    config_batchsize = load_yaml(args.config_batchsize)
    batch_size = config_batchsize[model_name][args.dataname]
    if args.overwrite_batchsize != 0:
        batch_size = args.overwrite_batchsize
    print(f'batch_size:{batch_size}\n')

    print('=' * 50)

    # Dataset
    if args.mode == 'training':  splits = ['train', 'valid']
    elif args.mode == 'testing': splits = [args.test_split]
    elif args.mode == 'optimizing':
        splits = ['train']
        if args.ppo_search_split != 'train': splits.append(args.ppo_search_split)
    elif args.mode == 'load-save-test': splits = ['train', 'test']

    print('Creating dataset & dataloader')
    global nentity, nrelation
    dataset_dict, nentity, nrelation = new_create_dataset(
        dataname=args.dataname,
        scale=args.scale,
        answer_size=args.max_answer_size,
        pattern_filtered=pattern_filtered,
        data_root=args.data_root,
        splits=splits,
        is_act=is_act
    )
    if args.mode == 'testing' and args.test_proportion < 1:
        nrows = dataset_dict[args.test_split].shape[0]
        dataset_dict[args.test_split] = dataset_dict[args.test_split].select(random.sample(range(nrows), int(nrows * args.test_proportion)))
    if args.mode == 'optimizing' and args.ppo_proportion < 1:
        nrows = dataset_dict['train'].shape[0]
        dataset_dict['train'] = dataset_dict['train'].select(random.sample(range(nrows), int(nrows * args.ppo_proportion)))
    dataloader_dict = new_create_dataloader(
        dataset_dict=dataset_dict,
        batch_size=batch_size,
        drop_last=(args.mode == 'optimizing') #or (args.mode == 'testing' and args.accelerate)
    )

    # Tokenizer
    print('Creating tokenizer')
    tokenizer, ntoken = create_tokenizer(
        special_tokens=special_tokens,
        offset=offset,
        nentity=nentity,
        nrelation=nrelation,
           is_gpt=is_gpt
    )

    # Model
    config_model = load_yaml(args.config_model)
    config_train = load_yaml(args.config_train)
    if model_name in config_train:
        config_train = config_train[model_name]
    else:
        config_train = config_train['default']
        warnings.warn(f'No training configuration specified for {model_name}')
    print(f'config_train:\n{config_train}')

    if args.mode == 'training':
        model, optimizer, scheduler, last_epoch, loss_log = load_model_by_mode(
            args=args, device=device, model_name=model_name, is_gpt=is_gpt,
            config_model=config_model, ntoken=ntoken, config_train=config_train)
    else:
        model = load_model_by_mode(
            args=args, device=device, model_name=model_name, is_gpt=is_gpt,
            config_model=config_model, ntoken=ntoken, config_train=config_train)
    # print('is_encoder_decoder', model.is_encoder_decoder)

    if args.mode == 'training':
        # https://huggingface.co/docs/transformers/training#train-in-native-pytorch
        nepoch = config_train['nepoch']
        fit(args, nepoch, dataloader_dict, model,
            tokenizer, optimizer, scheduler, graph_samplers,
            model_name, is_gpt, is_act, src_len, tgt_len,
            last_epoch, loss_log,
            args.vs,
            accelerator=accelerator if args.accelerate else None)
    elif args.mode == 'testing':
        # preprocess_allowed_rel_ent_map(graph_samplers)
        test_loop(
            args=args,
            dataloader=dataloader_dict[args.test_split],
            model=model,
            tokenizer=tokenizer,
            graph_samplers=graph_samplers,
            searching_split=args.test_split,
            resume_epoch=args.resume_epoch,
            is_gpt=is_gpt, is_act=is_act,
            src_len=src_len, tgt_len=tgt_len,
            accelerator=accelerator if args.accelerate else None)
    elif args.mode == 'optimizing':
        model = optimize(
            args=args,
            dataset=dataset_dict['train'],
            model=model,
            tokenizer=tokenizer,
            graph_samplers=graph_samplers,
            batch_size=batch_size,
            is_gpt=is_gpt, is_act=is_act,
            src_len=src_len, tgt_len=tgt_len
        )

if __name__ == '__main__':
    main()