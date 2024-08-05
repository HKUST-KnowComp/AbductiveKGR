from transformers import T5Config, T5ForConditionalGeneration
from transformers import GPT2Config, GPT2LMHeadModel
from akgr.abduction_model.t5 import myT5
import torch
from trl import (PPOTrainer, PPOConfig,
    AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead,
    create_reference_model)
config = T5Config.from_pretrained(
    "t5-small",
    vocab_size=23333,
    num_layers=3
)
transformer = myT5(config)
transformer = transformer.to_bettertransformer()
model = AutoModelForSeq2SeqLMWithValueHead(transformer)

for name, param in model.named_parameters():
    print(name, param.size())

print('=' * 30)

config = GPT2Config.from_pretrained(
    'gpt2',
    vocab_size=23333,
    num_layers=6,
)
transformer = GPT2LMHeadModel(config)
from optimum.bettertransformer import BetterTransformer
transformer = BetterTransformer.transform(transformer)
model = AutoModelForCausalLMWithValueHead(transformer)
total = 0
for name, param in model.named_parameters():
    siz = torch.tensor(param.size()).tolist()
    print(name, siz)
    if len(siz) == 2:
        total += siz[0] * siz[1]
    else:
        total += siz[0]
print(total)

print('=' * 30)

print('abcde'.format(layer=-1))
