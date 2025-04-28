import os
import time
import math
import tiktoken
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import inspect
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from dataclasses import dataclass

from typing import Tuple, Optional

"""
Rotary Positional Embeddings (RoPE)
"""

def precompute_freqs_cis(dim: int, end: int, theta: float=10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Validate input dimensions
    assert xq.shape[-1] == xk.shape[-1], "Query and Key must have same embedding dimension"
    assert xq.shape[-1] % 2 == 0, "Embedding dimension must be even"

    # Get sequence lengths
    q_len = xq.shape[1]
    k_len = xk.shape[1]

    # Use appropriate parts of freqs_cis for each sequence
    q_freqs = freqs_cis[:q_len]
    k_freqs = freqs_cis[:k_len]

    # Apply rotary embeddings separately
    # split last dimension to [xq.shape[:-1] / 2, 2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Reshape freqs for each
    q_freqs = reshape_for_broadcast(q_freqs, xq_)
    k_freqs = reshape_for_broadcast(k_freqs, xk_)

    # Works for both [bs, seq_len, n_heads * head_dim] and [bs, seq_len, n_heads, head_dim]
    xq_out = torch.view_as_real(xq_ * q_freqs).flatten(xq.ndim - 1)
    xk_out = torch.view_as_real(xk_ * k_freqs).flatten(xk.ndim - 1)

    return xq_out.type_as(xq), xk_out.type_as(xk)

"""
Multi-Head Latent Attention
"""

class MultiHeadLatentAttention(nn.Module):
    """
    Multi Head Latent Attention
    Introduced by DeepSeek
    """
    def __init__(self, config) -> None:
        super().__init__()
        self.seq_len = config.seq_len   # sequence length [seq_len]
        self.d_model = config.d_model   # input dimension [d]
        self.d_embed = config.d_embed   # embedding dimension [d_h * n_h]
        self.n_heads = config.n_heads   # number of attn heads [d_h]
        self.d_rotate = config.d_rotate # decoupled rope dimension [d_r]
        self.d_head = config.d_embed // config.n_heads  # dimension per head [d_h]

        # compression dimension [d_c, d_c_]
        self.d_c = config.d_c
        self.d_c_ = config.d_c_

        assert config.d_embed % config.d_head == 0, "d_embed should be a multiple of d_head"

        # Down-Projection Layers
        self.dkv_proj = nn.Linear(config.d_model, config.d_c) # W_DKV, [d_model, d_c]
        self.dq_proj = nn.Linear(config.d_model, config.d_c_) # W_Q, [d_model, d_c_]

        # Up-Projection Layers
        self.uk_proj = nn.Linear(config.d_c, config.d_embed) # W_UK, [d_c, d_embed]
        self.uv_proj = nn.Linear(config.d_c, config.d_embed) # W_UV, [d_c, d_embed]
        self.uq_proj = nn.Linear(config.d_c_, config.d_embed) # W_UQ, [d_c_, d_embed]

        # Rotation Layers
        self.qr_proj = nn.Linear(config.d_c_, config.d_rotate * config.n_heads) # W_QR, [d_c_, d_rotate * n_heads]
        self.kr_proj = nn.Linear(config.d_model, config.d_rotate)   # W_KR, [d_model, d_rotate]

        # Output Layers
        self.o_proj = nn.Linear(config.d_embed, config.d_model) # W_O, [d_embed, d_model]
        self.o_proj.NANOGPT_SCALE_INIT = 1

        self.register_buffer("freqs_cis", precompute_freqs_cis(self.d_rotate, self.seq_len*2), persistent=False)
        self.scalar = float(1.0 / math.sqrt(self.d_head + self.d_rotate))

        # KV Cache
        self.cache_kv = None
        self.cache_kr = None
        self.position = 0

    def reset_cache(self) -> None:
        """
        For resetting the KV Cache
        """
        self.cache_kv = None
        self.cache_kr = None
        self.position = 0

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None, use_cache=False) -> torch.Tensor:
        # Input Shape: [bs, seq_len, d_model]
        bs, seq_len, _ = x.shape

        c_q = self.dq_proj(x)     # [d_model, d_c_] x [bs, seq_len, d_model] -> [bs, seq_len, d_c_]

        if use_cache:
            if self.cache_kv is None:
                self.cache_kv = torch.zeros((bs, seq_len, self.d_c), device=x.device)
                self.cache_kr = torch.zeros((bs, seq_len, self.d_rotate), device=x.device)

            current_kv = self.dkv_proj(x)
            current_kr = self.kr_proj(x)

            self.cache_kv[:, self.position:self.position + seq_len] = current_kv
            self.cache_kr[:, self.position:self.position + seq_len] = current_kr

            self.position += seq_len

            c_kv = self.cache_kv[:, :self.position]
            k_r = self.cache_kr[:, :self.position]

        else:
            c_kv = self.dkv_proj(x)   # [d_model, d_c] x [bs, seq_len, d_model] -> [bs, seq_len, d_c]
            k_r = self.kr_proj(x)

        q_c = self.uq_proj(c_q)   # [d_c_, d_head] x [bs, seq_len, d_c_] -> [bs, seq_len, d_head]
        k_c = self.uk_proj(c_kv)
        v_c = self.uv_proj(c_kv)

        q_r = self.qr_proj(c_q)
        q_r = q_r.view(bs, seq_len, self.n_heads, self.d_rotate)
        k_r = k_r.unsqueeze(2).expand(-1, -1, self.n_heads, -1)
        q_r, k_r = apply_rotary_emb(q_r, k_r, self.freqs_cis)

        q_c = q_c.view(bs, seq_len, self.n_heads, self.d_head)
        k_c = k_c.view(bs, seq_len, self.n_heads, self.d_head)
        v_c = v_c.view(bs, seq_len, self.n_heads, self.d_head)

        q_t = torch.cat([q_c, q_r], dim=-1)
        k_t = torch.cat([k_c, k_r], dim=-1)

        q_t = q_t.transpose(1, 2)
        k_t = k_t.transpose(1, 2)
        v_c = v_c.transpose(1, 2)

        attn = torch.matmul(q_t, k_t.transpose(-1, -2))
        attn = attn * self.scalar

        if mask is not None:
            mask = mask.masked_fill(mask==0, -torch.inf)
            attn = attn + mask

        attn_score = F.softmax(attn, dim=-1)
        attn_output = torch.matmul(attn_score, v_c)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bs, seq_len, self.n_heads*self.d_head)

        output = self.o_proj(attn_output)

        return output

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.d_embed, 4 * config.d_embed)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.d_embed, config.d_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadLatentAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None, use_cache=False):
        x = x + self.attn(self.ln1(x), mask=mask, use_cache=use_cache)
        x = x + self.mlp(self.ln2(x))
        x = self.dropout(x)
        return x

class NanoGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token Embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model, device=config.device)
        # Transformer Layers
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        # Final LayerNorm
        self.ln_f = nn.LayerNorm(config.d_model)
        # Output Projection
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Tying weights
        self.tok_emb.weight = self.lm_head.weight
        # Init Weights
        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layers)**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def reset_cache(self):
        for block in self.blocks:
            block.attn.reset_cache()

    def forward(self, idx, targets=None, mask=None, use_cache=False):
        bs, seq_len = idx.size()
        idx = idx.to(self.config.device)

        # Token Embeddings
        x = self.tok_emb(idx) # [bs, seq_len, d_model]
        # Create causal mask if not provided
        if mask is None and seq_len > 1:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=self.config.device)).view(1, seq_len, seq_len)

        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask, use_cache=use_cache)
        # Apply final layernorm
        x = self.ln_f(x)
        # Get logits
        logits = self.lm_head(x) # [bs, seq_len, vocab_size]
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        self.reset_cache()

        for _ in range(max_new_tokens):
            # Get only last token for efficiency (kv cache)
            idx_cond = idx[:, -1:]
            # Forward pass
            logits = self(idx_cond, use_cache=True)
            # Take the logits at the final step
            logits = logits[:, -1, :] # [bs, vocab_size]
            # Apply temperature
            logits = logits / temperature

            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Apply softmax to convert to probs
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

def create_masks(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.view(1, seq_len, seq_len)

@dataclass
class Config:
    bs = 8                           # batch size
    seq_len = 1024                   # sequence length
    d_model = 768                     # input dimension [d]
    d_embed = 768                    # embedding dimension [d_h * n_h]
    n_heads = 12                      # number of attention heads [n_h]
    d_head = d_embed // n_heads      # dimension per head [d_h]

    # Keeping both query compression dimension
    # and key-value compression dimension same
    d_c = 64                         # compression dimension [d_c]
    d_c_ = 64                        # compression dimension [d_c]
    d_rotate = 32
    n_layers = 12
    dropout = 0.1
    vocab_size = 50304

    learning_rate = 3e-4
    weight_decay = 0.1
    grad_clip = 1.0
    max_epochs = 10
    steps_per_epoch = 100
    valid_steps = 10
    checkpoint_path = "model_checkpoints/NanoGPT_MLA.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "/data/tinystories"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 16 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high')

config = Config()
model = NanoGPT(config)
model.to(config.device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 5 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 5 == 0) or last_step):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Once upon a time,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()
