# -*- coding: utf-8 -*-
# JAX/Flax LLM Inference Script
# Features: Orbax Checkpoint Loading & Token-aware Streaming Generation

import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import orbax.checkpoint as ocp
from transformers import AutoTokenizer

# --- Configuration ---
# Configuration loaded from environment variables with defaults matching the training script.
CFG = {
    "workdir": os.getenv("WORKDIR", "./llm_jax_output"),
    "tokenizer_path": os.getenv("TOKENIZER_PATH", "mistralai/Mistral-7B-v0.1"),
    "max_seq_len": int(os.getenv("MAX_SEQ_LEN", 1024)),
    "model_dim": int(os.getenv("MODEL_DIM", 768)),
    "num_layers": int(os.getenv("NUM_LAYERS", 18)),
    "num_heads": int(os.getenv("NUM_HEADS", 8)),
    "mlp_ratio": int(os.getenv("MLP_RATIO", 4)),
    "dtype": jnp.bfloat16
}
CFG["mlp_dim"] = int(CFG["model_dim"] * CFG["mlp_ratio"] * 2 / 3)

# --- Model Architecture ---
# Must match the training architecture exactly.

def make_causal_mask(x):
    idx = jnp.arange(x.shape[1])
    mask = idx[:, None] >= idx[None, :]
    return jnp.expand_dims(mask, axis=(0, 1))

class RMSNorm(nn.Module):
    dtype: jnp.dtype = jnp.bfloat16
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],), jnp.float32)
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        return (x * jax.lax.rsqrt(var + self.eps) * scale).astype(self.dtype)

class TransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    dropout: float
    mlp_dim: int
    n_layers: int
    dtype: jnp.dtype = jnp.bfloat16
    
    @nn.compact
    def __call__(self, x, mask, deterministic):
        norm_x = RMSNorm(dtype=self.dtype)(x)
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads, dtype=jnp.float32, deterministic=deterministic
        )(norm_x, mask=mask) 
        x = x + nn.Dropout(self.dropout)(h, deterministic=deterministic)
        
        h = nn.Dense(self.mlp_dim, dtype=self.dtype)(RMSNorm(dtype=self.dtype)(x))
        h = nn.gelu(h)
        h = nn.Dense(self.d_model, dtype=self.dtype)(h)
        return x + nn.Dropout(self.dropout)(h, deterministic=deterministic)

RemattedBlock = nn.remat(TransformerBlock, static_argnums=(3,))

class TransformerLM(nn.Module):
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    max_len: int
    dropout: float
    mlp_dim: int
    dtype: jnp.dtype = jnp.bfloat16
    
    def setup(self):
        self.tok_emb = nn.Embed(self.vocab_size, self.d_model, dtype=self.dtype)
        self.pos_emb = self.param("pos_emb", nn.initializers.normal(stddev=0.02), 
                                  (self.max_len, self.d_model))
        self.drop = nn.Dropout(self.dropout)
        self.norm_f = RMSNorm(dtype=self.dtype)
        self.head = nn.Dense(self.vocab_size, use_bias=False, dtype=jnp.float32)
        self.blocks = [
            RemattedBlock(self.d_model, self.n_heads, self.dropout, self.mlp_dim, self.n_layers, dtype=self.dtype, name=f'block_{i}') 
            for i in range(self.n_layers)
        ]
    
    def __call__(self, x, deterministic=True):
        mask = make_causal_mask(x)
        h = self.tok_emb(x) + self.pos_emb[:x.shape[1]].astype(self.dtype)
        h = self.drop(h, deterministic=deterministic)
        for blk in self.blocks: 
            h = blk(h, mask, deterministic)
        return self.head(self.norm_f(h))

# --- Initialization & Loading ---

def load_tokenizer():
    print("[INFO] Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(CFG["tokenizer_path"], use_fast=True)
        print(f"[INFO] Loaded local tokenizer: {CFG['tokenizer_path']}")
    except Exception as e:
        print(f"[WARNING] Local load failed ({e}). Downloading default Mistral...")
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
    return tokenizer

def load_checkpoint(vocab_size):
    print("[INFO] Initializing model structure...")
    model = TransformerLM(
        vocab_size, CFG["model_dim"], CFG["num_layers"], CFG["num_heads"], 
        CFG["max_seq_len"], 0.0, CFG["mlp_dim"], CFG["dtype"]
    )
    
    ckpt_dir = os.path.join(CFG["workdir"], "ckpts")
    mngr = ocp.CheckpointManager(ckpt_dir, options=ocp.CheckpointManagerOptions())
    
    latest_step = mngr.latest_step()
    if latest_step is None:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        
    print(f"[INFO] Restoring checkpoint from step {latest_step}...")
    # Restore raw data first to inspect structure
    raw_restored = mngr.restore(latest_step, args=ocp.args.StandardRestore(item=None))
    
    # Robust parameter extraction logic
    if 'params' in raw_restored:
        params = raw_restored['params']
    elif 'item' in raw_restored and 'params' in raw_restored['item']:
        params = raw_restored['item']['params']
    else:
        # Fallback: assume the root object is the params dict
        params = raw_restored.get('params', raw_restored)
        
    print("[INFO] Model parameters loaded successfully.")
    return model, params

# --- Inference Logic ---

def generate_text(model, params, tokenizer, prompt, max_new_tokens=100, temperature=0.7, seed=None):
    """
    Generates text using the loaded model.
    Uses a decoding strategy that preserves correct spacing for sentencepiece tokenizers.
    """
    print(f"\n[PROMPT] {prompt}")
    print("[OUTPUT] ", end="", flush=True)
    
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = jnp.array([tokens], dtype=jnp.int32)
    
    # Track the full decoded text to handle spacing correctly
    text_so_far = tokenizer.decode(tokens, skip_special_tokens=True)
    print(text_so_far, end="", flush=True)
    
    # Random seed initialization
    if seed is None:
        seed = np.random.randint(0, 1e9)
    rng_key = jax.random.PRNGKey(seed)

    for _ in range(max_new_tokens):
        # Forward pass
        logits = model.apply({"params": params}, input_ids, deterministic=True)
        next_token_logits = logits[0, -1, :] / temperature
        
        # Sampling
        rng_key, subkey = jax.random.split(rng_key)
        next_token = jax.random.categorical(subkey, next_token_logits)
        
        # Update Input IDs
        input_ids = jnp.concatenate([input_ids, next_token[None, None]], axis=1)
        
        # Context Window Management
        if input_ids.shape[1] > CFG["max_seq_len"]:
            input_ids = input_ids[:, -CFG["max_seq_len"]:]
            
        # --- Spacing Fix Strategy ---
        # Decode the entire sequence and print only the new suffix.
        # This allows the tokenizer to correctly handle spaces that depend on context.
        all_tokens = np.array(input_ids[0])
        new_text = tokenizer.decode(all_tokens, skip_special_tokens=True)
        
        diff = new_text[len(text_so_far):]
        print(diff, end="", flush=True)
        text_so_far = new_text
        
        if next_token == tokenizer.eos_token_id: 
            break
            
    print("\n" + "-"*50)

# --- Main Execution ---

if __name__ == "__main__":
    # Setup
    tokenizer = load_tokenizer()
    model, params = load_checkpoint(vocab_size=tokenizer.vocab_size)
    
    # Test Prompts
    test_prompts = [
        "Science is important because",
        "The history of Brazil is",
        "Artificial Intelligence is"
    ]
    
    for prompt in test_prompts:
        generate_text(model, params, tokenizer, prompt)
