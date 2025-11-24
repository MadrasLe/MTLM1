#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training of a LLaMA-style Transformer (pure, non-MoE) featuring:
- RMSNorm
- Multi-head Attention with RoPE (Rotary Positional Embeddings)
- SwiGLU MLP
- Direct loading of pre-tokenized data
- Mixed precision (bf16/amp)
- Gradient accumulation, clipping, and warmup+cosine scheduler
- CORRECT WEIGHT DECAY GROUPS
- CORRECTED GRADSCALER AND AUTOCAST LOGIC
- GUARANTEED WEIGHT INITIALIZATION
- CHECKPOINTING AND RESUMING
- FLASH-ATTENTION WITH FORCED DTYPE TO BF16
"""

import os
import math
import time
import argparse
import glob
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Dependency on `datasets` is required to load .arrow files
try:
    from datasets import Dataset as HFDataset
    from datasets import concatenate_datasets
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("⚠️  The `datasets` library was not found. Install it with: pip install datasets")


# ------------------------------------------------------------
# OPTIONAL OPTIMIZATIONS
# ------------------------------------------------------------
try:
    from flash_attn import flash_attn_func as _flash_attn
    HAS_FLASH = True
    print("✅ Flash Attention 2 AVAILABLE")
except Exception:
    HAS_FLASH = False
    print("⚠️  Flash Attention 2 not available")


# ------------------------------------------------------------
# RoPE (Rotary Positional Embeddings)
# ------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H, D = x.shape
        cos = self.cos[:S].unsqueeze(0).unsqueeze(2)
        sin = self.sin[:S].unsqueeze(0).unsqueeze(2)
        x = x.view(B, S, H, D // 2, 2)
        x_even, x_odd = x[..., 0], x[..., 1]
        y_even = x_even * cos - x_odd * sin
        y_odd  = x_even * sin + x_odd * cos
        return torch.stack((y_even, y_odd), dim=-1).reshape(B, S, H, D)

# ------------------------------------------------------------
# RMSNorm (LLaMA style)
# ------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x

# ------------------------------------------------------------
# Attention (SDPA/Flash) with RoPE
# ------------------------------------------------------------
class LLaMAAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, rope: RotaryEmbedding):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = rope
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, self.n_heads, self.head_dim)
        k = k.view(B, S, self.n_heads, self.head_dim)
        v = v.view(B, S, self.n_heads, self.head_dim)
        q = self.rope(q)
        k = self.rope(k)

        if HAS_FLASH and self.training:
            # ==========================================================
            # ================ FINAL CORRECTION, FORCED ================
            # ==========================================================
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
            # ==========================================================
            # ==========================================================
            out = _flash_attn(q, k, v, dropout_p=self.dropout, causal=True, return_attn_probs=False)
            out = out.contiguous().view(B, S, D)
        else:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
            out = out.transpose(1, 2).contiguous().view(B, S, D)

        return self.o_proj(out)

# ------------------------------------------------------------
# SwiGLU MLP (LLaMA style)
# ------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, multiple_of: int = 256, hidden_multiple: float = 5/3):
        super().__init__()
        hidden = int(d_model * hidden_multiple)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(self.act(self.w1(x)) * self.w2(x))

# ------------------------------------------------------------
# Transformer Block (LLaMA-like)
# ------------------------------------------------------------
class LLaMABlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, rope: RotaryEmbedding):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = LLaMAAttention(d_model, n_heads, dropout, rope)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# ------------------------------------------------------------
# Main Model
# ------------------------------------------------------------
class LLaMAModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 1024, n_heads: int = 16, n_layers: int = 24,
                 max_seq_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        rope = RotaryEmbedding(d_model // n_heads, max_seq_len)
        self.layers = nn.ModuleList([LLaMABlock(d_model, n_heads, dropout, rope) for _ in range(n_layers)])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.criterion = nn.CrossEntropyLoss()
        
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.drop(self.tok_emb(input_ids))
        for blk in self.layers:
            x = blk(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = self.criterion(logits.view(-1, self.vocab_size), labels.view(-1))
        return logits, loss

# ------------------------------------------------------------
# Data Loading Utilities
# ------------------------------------------------------------

def load_tokenizer(name_or_path: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

class PretokenizedArrowDataset(Dataset):
    def __init__(self, dataset_folder):
        if not DATASETS_AVAILABLE:
            raise ImportError("The `datasets` library is required. Install with: pip install datasets")
        
        arrow_files = glob.glob(os.path.join(dataset_folder, "*.arrow"))
        if not arrow_files:
            raise ValueError(f"No .arrow files found in {dataset_folder}")
        
        datasets = [HFDataset.from_file(file) for file in arrow_files]
        self.dataset = concatenate_datasets(datasets)
        print(f"✅ Pre-sliced dataset loaded: {len(self.dataset):,} samples from {len(arrow_files)} .arrow files")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {'input_ids': torch.tensor(self.dataset[idx]['input_ids'], dtype=torch.long)}

# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------
@dataclass
class TrainConfig:
    tokenizer: str = "mistralai/Mistral-7B-v0.1"
    out_dir: str = "./ckpts-llama-min"
    seq_len: int = 1024
    batch_size: int = 8
    grad_accum: int = 8
    epochs: int = 1
    d_model: int = 512
    n_heads: int = 16
    n_layers: int = 18
    dropout: float = 0.05
    lr: float = 7e-4
    weight_decay: float = 0.01
    warmup_steps: int = 300
    max_steps: int = 0
    bf16: bool = True
    log_interval: int = 25
    ckpt_interval: int = 500
    resume: Optional[str] = None

def wd_groups(model: nn.Module, weight_decay: float) -> List[Dict]:
    """Separates model parameters into two groups: one with and one without weight decay."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() > 1 and not n.endswith(".bias") and "norm" not in n.lower():
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def print_model_info(model: LLaMAModel, cfg: TrainConfig, ds_len: int):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*60)
    print(" " * 20 + "📊 MODEL INFO")
    print("="*60)
    print(f" 🏛️   Architecture:")
    print(f" │  ├─ Style: ................... LLaMA (Dense)")
    print(f" │  ├─ Model Dimension (d_model): {model.d_model}")
    print(f" │  ├─ Number of Layers (n_layers): {model.n_layers}")
    print(f" │  ├─ Number of Heads (n_heads): . {model.n_heads}")
    print(f" │  └─ Vocabulary Size: ........... {model.vocab_size:,}")
    print(f" ----------------------------------------------------------")
    print(f" 🧮   Parameters:")
    print(f" │  ├─ Total: .................... {total_params/1e6:.2f}M ({total_params:,})")
    print(f" │  └─ Trainable: ............... {trainable_params/1e6:.2f}M ({trainable_params:,})")
    print(f" ----------------------------------------------------------")
    print(f" 🚀   Training Config:")
    print(f" │  ├─ Sequence Length: ........... {cfg.seq_len}")
    print(f" │  ├─ Total Samples: ............. {ds_len:,}")
    print(f" │  ├─ Batch Size per GPU: ........ {cfg.batch_size}")
    print(f" │  ├─ Gradient Accumulation: ..... {cfg.grad_accum}")
    print(f" │  └─ Effective Batch Size: ...... {cfg.batch_size * cfg.grad_accum}")
    print("="*60)

def save_checkpoint(model: nn.Module, opt: torch.optim.Optimizer, step: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"global_step{step}")
    state = { "model": model.state_dict(), "optimizer": opt.state_dict(), "step": step }
    torch.save(state, path)
    print(f"💾 Checkpoint saved at: {path}")

def load_checkpoint(model: nn.Module, opt: torch.optim.Optimizer, path: str) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if opt is not None and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    step = int(ckpt.get("step", 0))
    print(f"📂 Resumed from: {path} (step={step})")
    return step

def cosine_with_warmup(optimizer, warmup, total, base_lr, min_lr):
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(step):
        if step < warmup: return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return min_lr / base_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (1 - min_lr / base_lr)
    return LambdaLR(optimizer, lr_lambda)

def train(cfg: TrainConfig):
    DATASET_PATH = "/content/dataset/train"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = load_tokenizer(cfg.tokenizer)
    
    ds = PretokenizedArrowDataset(DATASET_PATH)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    model = LLaMAModel(vocab_size=tok.vocab_size, d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers, max_seq_len=cfg.seq_len, dropout=cfg.dropout).to(device)

    print_model_info(model, cfg, len(ds))

    param_groups = wd_groups(model, cfg.weight_decay)
    opt = torch.optim.AdamW(param_groups, lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8)

    total_steps = cfg.max_steps if cfg.max_steps > 0 else cfg.epochs * max(1, len(dl) // cfg.grad_accum)
    min_lr = cfg.lr * 0.1
    scheduler = cosine_with_warmup(opt, cfg.warmup_steps, total_steps, cfg.lr, min_lr)
    
    use_fp16 = torch.cuda.is_available() and not cfg.bf16
    amp_dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    
    step = 0
    micro_step = 0

    if cfg.resume is not None and os.path.exists(cfg.resume):
        step = load_checkpoint(model, opt, cfg.resume)

    model.train()
    t0 = time.time()
    tokens_seen = 0
    running_loss = []

    print(f"\n🚀 STARTING TRAINING for {total_steps} steps...")

    for epoch in range(cfg.epochs if cfg.max_steps == 0 else 10_000_000):
        progress_bar = tqdm(dl, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for batch in progress_bar:
            full_sequence = batch['input_ids'].to(device)
            inputs = full_sequence[:, :-1].contiguous()
            labels = full_sequence[:, 1:].contiguous()
            
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                _, loss = model(inputs, labels)
            
            loss = loss / cfg.grad_accum
            
            scaler.scale(loss).backward()
            
            micro_step += 1
            tokens_seen += inputs.numel()

            if micro_step % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                scheduler.step()
                
                step += 1
                running_loss.append(loss.item() * cfg.grad_accum)

                if step % cfg.log_interval == 0 and step > 0:
                    avg_loss = sum(running_loss[-cfg.log_interval:]) / min(len(running_loss), cfg.log_interval)
                    elapsed = time.time() - t0
                    tps = tokens_seen / max(1e-9, elapsed)
                    
                    log_msg = (f"[Step {step}/{total_steps}] Loss={avg_loss:.3f} | LR={opt.param_groups[0]['lr']:.2e} | TPS={tps:,.0f} | GPU Mem={torch.cuda.memory_allocated()/1e9:.2f}GB")
                    progress_bar.set_postfix_str(f"Loss={avg_loss:.3f}, TPS={tps:,.0f}")
                    tqdm.write(log_msg)

                if step > 0 and step % cfg.ckpt_interval == 0:
                    save_checkpoint(model, opt, step, cfg.out_dir)

            if total_steps and step >= total_steps: break
        if total_steps and step >= total_steps: break

    save_checkpoint(model, opt, step, cfg.out_dir)
    elapsed = time.time() - t0
    print("\n✅ Training finished!")
    print(f"⏱️ Total time: {elapsed/3600:.2f}h | Tokens/s (avg): {tokens_seen/max(1e-9, elapsed):,.0f}")

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument('--tokenizer', type=str, default='mistralai/Mistral-7B-v0.1')
    p.add_argument('--out_dir', type=str, default='./ckpts-llama-min-arrow')
    p.add_argument('--seq_len', type=int, default=1024)
    p.add_argument('--batch_size', type=int, default=12)
    p.add_argument('--grad_accum', type=int, default=8)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--d_model', type=int, default=768)
    p.add_argument('--n_heads', type=int, default=12)
    p.add_argument('--n_layers', type=int, default=14)
    p.add_argument('--dropout', type=float, default=0.05)
    p.add_argument('--lr', type=float, default=8e-4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_steps', type=int, default=300)
    p.add_argument('--max_steps', type=int, default=0)
    p.add_argument('--bf16', action='store_true')
    p.add_argument('--no-bf16', dest='bf16', action='store_false')
    p.set_defaults(bf16=True)
    p.add_argument('--log_interval', type=int, default=25)
    p.add_argument('--ckpt_interval', type=int, default=500)
    p.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume from")
    args = p.parse_args()
    return TrainConfig(**vars(args))

if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
