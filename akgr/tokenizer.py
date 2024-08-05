import sys, json

# sys.path.append('./utils/')
from akgr.utils.load_util import load_yaml


from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast, T5TokenizerFast, GPT2TokenizerFast
def get_vocab(special_tokens, offset, nentity, nrelation):
    vocab = {}
    vocab.update(special_tokens)
    for i in range(1, nentity+1): # [offset, offset + nentity - 1]
        vocab[str(i)] = offset + i - 1
    for i in range(1, nrelation+1): # [offset + nentity, offset + nentity + nrelation - 1]
        vocab[str(-i)] = offset + nentity + i - 1
    # vocab["-1"] = offset
    return vocab, offset + nentity + nrelation
def create_tokenizer(
        special_tokens: dict, offset: int,
        nentity: int, nrelation: int,
        is_gpt: bool):
    pre_tokenizer = WhitespaceSplit()
    vocab, vocab_size = get_vocab(special_tokens, offset=offset, nentity=nentity, nrelation=nrelation)
    model = WordLevel(vocab, unk_token='UNK')
    if not is_gpt:
        post_processor = TemplateProcessing(
            single='$0 END',
            # pair='$A START $B END',
            special_tokens=[('END', special_tokens['END'])]
        )
    else:
        post_processor = TemplateProcessing(
            single='$0 SEP',
            pair='$A SEP $B END',
            special_tokens=[('SEP', special_tokens['SEP']), ('END', special_tokens['END'])]
        )
    tokenizer = Tokenizer(model=model)

    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.post_processor = post_processor
    # Just to let the tokenizer know about special tokens
    tokenizer.add_special_tokens(['START', 'END', 'PAD', 'UNK', 'SEP'])
    import io
    from contextlib import redirect_stdout
    trap = io.StringIO()
    with redirect_stdout(trap):
        TokenizerFast = GPT2TokenizerFast if is_gpt else T5TokenizerFast
        tokenizer = TokenizerFast(
            tokenizer_object=tokenizer,
            bos_token='START',
            eos_token='END',
            pad_token='PAD',
            unk_token='UNK',
            sep_token='SEP',
            ) # default padding side
    return tokenizer, vocab_size

import torch
def new_extract_sample_to_device(device,
        sample, tokenizer, is_gpt:bool,
        src_len, tgt_len, is_gen:bool):
    source = sample['source']
    target = sample['target']
    pattern_id = sample['pattern_id']
    if not is_gpt:
        source_tokenized = tokenizer(
            source,
            padding='max_length',
            max_length=src_len,
            return_tensors="pt").to(device)
        input_ids = source_tokenized.input_ids
        attention_mask = source_tokenized.attention_mask
        # special treatment for T5: ignore end in source
        attention_mask[input_ids == tokenizer.eos_token_id] = 0

        labels = tokenizer(
            target,
            padding='max_length',
            max_length=tgt_len,
            return_tensors="pt").input_ids.to(device)
    else:
        source_target_tokenized = tokenizer(
            source, target,
            padding='longest',
            # max_length=src_len+tgt_len,
            return_tensors="pt").to(device)
        # labels is the source SEP target END, ...
        labels = torch.clone(source_target_tokenized.input_ids)
        # ... with the source part's loss ignored
        source_tokenized = tokenizer(
            source,
            padding='max_length',
            max_length=labels.shape[-1],
            return_tensors="pt").to(device)
        labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id

        if is_gen == False: # (train/valid) input = source SEP target END, default padding side
            input_ids = source_target_tokenized.input_ids
            attention_mask = source_target_tokenized.attention_mask
        else: # (test/optimize) input = source SEP, left padding (align the last tokens to the right)
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = 'left'
            source_tokenized = tokenizer(
                source,
                padding='longest',
                max_length=src_len,
                return_tensors="pt").to(device)
            tokenizer.padding_side = original_padding_side
            input_ids = source_tokenized.input_ids
            attention_mask = source_tokenized.attention_mask

        # labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id

    labels[labels == tokenizer.pad_token_id] = -100
    source_attention_mask = source_tokenized.attention_mask

    return source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask


def debug():
    config_dataloader = load_yaml('akgr/configs/config-dataloader.yml')
    offset = config_dataloader['offset']
    special_tokens = config_dataloader['special_tokens']
    tokenizer, _ = create_tokenizer(special_tokens, offset, nentity=200000, nrelation=2000, is_gpt=True)
    sample1 = {'answers': [1, 2, 3, 4], "query": ["(","i","(","n","(","p","(",-1,")","(","p","(",0,")","(","e","(",0,")",")",")",")",")","(","p","(",-567,")","(","e","(",24623,")",")",")",")"], "pattern_str":"(i,(n,(p,(p,(e)))),(p,(e)))"}
    sample2 = {'answers': [1, 2], "query": ["(", "p", "(", -1, ")", "(", "e", "(", 0, ")", ")", ")"], "pattern_str": "(p,(e))"}
    from akgr.utils.parsing_util import qry_shift_indices, ans_shift_indices, qry_str_2_actionstr
    def list_to_str(l: list) -> str:
        # print('before', l)
        # print('after', ' '.join([str(x) if isinstance(x, int) else x for x in l]))
        return ' '.join([str(x) if isinstance(x, int) else x for x in l])
    sample = {}
    sample['source'] = [list_to_str(ans_shift_indices(sample1['answers'])), list_to_str(ans_shift_indices(sample2['answers']))]
    sample['target'] = [qry_str_2_actionstr(list_to_str(qry_shift_indices(sample1['query']))), qry_str_2_actionstr(list_to_str(qry_shift_indices(sample2['query'])))]
    sample['pattern_id'] = [1, 2]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    source, target, pattern_id, input_ids, attention_mask, labels = \
        new_extract_sample_to_device(device, sample, tokenizer, is_gpt=True, src_len=33, tgt_len=66, is_gen=True)
    # input_ids = tokenizer(sample['source'], padding='max_length', max_length=33, return_tensors="pt")
    print('source')
    print(source)
    print('target')
    print(target)
    print('input_ids')
    print(input_ids)
    print('attention_mask')
    print(attention_mask)
    print('labels')
    print(labels)
    labels[labels == -100] = 0
    print(tokenizer.batch_decode(labels, skip_special_tokens=True))


    source, target, pattern_id, input_ids, attention_mask, labels = \
        new_extract_sample_to_device(device, sample, tokenizer, is_gpt=True, src_len=33, tgt_len=33, is_gen=False)
    # input_ids = tokenizer(sample['source'], padding='max_length', max_length=33, return_tensors="pt")
    print('source')
    print(source)
    print('target')
    print(target)
    print('input_ids')
    print(input_ids)
    print('attention_mask')
    print(attention_mask)
    print('labels')
    print(labels)
    labels[labels == -100] = 0
    print(tokenizer.batch_decode(labels, skip_special_tokens=True))

if __name__ == '__main__':
    debug()