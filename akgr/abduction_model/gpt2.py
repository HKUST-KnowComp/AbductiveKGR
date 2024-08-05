from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union

class invalidGPT2PositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        # self.offset = 2
        self.embedding_dim = embedding_dim
    def forward(self, input_ids: torch.Tensor):
        """`input_ids' shape is expected to be [bsz x seqlen]."""

        bsz, seq_len = input_ids.shape[:2]
        return torch.zeros((bsz, seq_len, self.embedding_dim),
                           dtype=torch.long, device=input_ids.device)

class myGPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = invalidGPT2PositionalEmbedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

class myGPT2LM(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = myGPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()