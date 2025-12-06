# -*- coding: utf-8 -*-
# JAX/Flax Causal LLM Training Script
# Supports TPU/GPU with NamedSharding and Gradient Checkpointing.

import os
import sys
import subprocess
import math
import time
import functools
import random
import glob
import numpy as np
from pathlib import Path

# --- Dependency Management ---
def install_dependencies():
    """Installs required packages for JAX TPU environment."""
    packages = [
        "jax[tpu]==0.7.1 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html",
        "flax==0.12.0",
        "optax==0.2.2",
        "orbax-checkpoint==0.5.10",
        "transformers==4.44.2",
        "tokenizers==0.19.1",
        "datasets>=2.20.0",
        "accelerate>=0.33.0"
    ]
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "hf-xet"])
        cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir"] + packages
        # Splitting the -f argument correctly for subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", 
                               "-f", "https://storage.googleapis.com/jax-releases/libtpu_releases.html",
                               "jax[tpu]==0.7.1"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir"] + packages[1:])
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Dependency installation failed: {e}")
        sys.exit(1)

# Uncomment the line below if dependencies need to be installed at runtime
# install_dependencies()

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU') # Prevent TF from allocating GPU memory

import jax
import jax.numpy as jnp
from jax import random as jr, config
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import flax.linen as nn
from flax.training import train_state as ts
import optax
import orbax.checkpoint as ocp
from transformers import AutoTokenizer

# --- Configuration ---
# Configuration is loaded from environment variables with sensible defaults.
CFG = {
    "workdir": os.getenv("WORKDIR", "./llm_jax_output"),
    "dataset_path": os.getenv("DATASET_PATH", "./data/dataset.txt"),
    "tokenizer_path": os.getenv("TOKENIZER_PATH", "mistralai/Mistral-7B-v0.1"),
    "max_seq_len": int(os.getenv("MAX_SEQ_LEN", 1024)),
    "model_dim": int(os.getenv("MODEL_DIM", 768)),
    "num_layers": int(os.getenv("NUM_LAYERS", 18)),
    "num_heads": int(os.getenv("NUM_HEADS", 8)),
    "mlp_ratio": int(os.getenv("MLP_RATIO", 4)),
    "dropout": float(os.getenv("DROPOUT", 0.1)),
    "lr": float(os.getenv("LR", 4e-4)),
    "warmup_steps": int(os.getenv("WARMUP_STEPS", 200)),
    "weight_decay": float(os.getenv("WEIGHT_DECAY", 0.01)),
    "grad_clip": float(os.getenv("GRAD_CLIP", 1.0)),
    "global_batch_size": int(os.getenv("GLOBAL_BATCH_SIZE", 256)),
    "total_train_steps": int(os.getenv("TOTAL_STEPS", 0)) if os.getenv("TOTAL_STEPS") else None,
    "eval_every": int(os.getenv("EVAL_EVERY", 1000)),
    "save_every": int(os.getenv("SAVE_EVERY", 500)),
    "ckpt_keep": int(os.getenv("CKPT_KEEP", 3)),
    "seed": int(os.getenv("SEED", 42)),
    "prefetch_buffer": tf.data.AUTOTUNE,
}
CFG["mlp_dim"] = int(CFG["model_dim"] * CFG["mlp_ratio"] * 2 / 3)

# Setup directories
Path(CFG["workdir"]).mkdir(parents=True, exist_ok=True)
Path(os.path.join(CFG["workdir"], "ckpts")).mkdir(parents=True, exist_ok=True)

# JAX Setup
config.update("jax_default_matmul_precision", "bfloat16")
os.environ["JAX_PLATFORMS"] = "tpu,cpu"

num_devices = jax.device_count()
local_batch_size = CFG["global_batch_size"] // num_devices
mesh = Mesh(mesh_utils.create_device_mesh((num_devices,)), axis_names=("data",))
data_sharding = NamedSharding(mesh, P("data", None))

print(f"[INFO] JAX Version: {jax.__version__}")
print(f"[INFO] Devices: {num_devices} | Global Batch: {CFG['global_batch_size']} | Local Batch: {local_batch_size}")

# --- Tokenizer ---
try:
    tokenizer = AutoTokenizer.from_pretrained(CFG["tokenizer_path"], use_fast=True)
    print(f"[INFO] Tokenizer loaded: {CFG['tokenizer_path']}")
except Exception as e:
    print(f"[WARNING] Failed to load local tokenizer. Fallback to default. Error: {e}")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)

if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.pad_token_id

# Pad vocabulary to be divisible by number of devices for efficient sharding
vocab_size = tokenizer.vocab_size
if pad_needed := (-vocab_size) % num_devices:
    tokenizer.add_tokens([f"<pad_{i}>" for i in range(pad_needed)])
    vocab_size = tokenizer.vocab_size
print(f"[INFO] Final Vocabulary Size: {vocab_size}")

# --- Data Pipeline ---
def get_txt_files(data_path):
    """Recursively finds text files in the given directory."""
    path_obj = Path(data_path)
    if path_obj.is_file():
        return [str(path_obj)]
    
    files = glob.glob(f"{data_path}/**/*.txt", recursive=True)
    if not files:
        files = glob.glob(f"{data_path}/*.txt")
    if not files:
        raise FileNotFoundError(f"No text files found in {data_path}")
    return sorted(files)

def create_dataset_pipeline(files, seq_len, batch_size, split_mode=None, shuffle=True):
    """Generates a tf.data.Dataset from text files."""
    def generator():
        if shuffle and split_mode == 'train':
            random.shuffle(files)
        
        token_buffer = []
        for file_path in files:
            file_size = os.path.getsize(file_path)
            split_byte = int(file_size * 0.9)
            
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                if split_mode == 'val':
                    f.seek(split_byte)
                    f.readline() # Skip potentially incomplete line
                
                current_byte = f.tell()
                
                for line in f:
                    if split_mode == 'train':
                        current_byte += len(line.encode('utf-8'))
                        if current_byte > split_byte:
                            break 
                    
                    if not line.strip():
                        continue
                    
                    tokens = tokenizer.encode(line, add_special_tokens=False) + [tokenizer.eos_token_id]
                    token_buffer.extend(tokens)
                    
                    while len(token_buffer) >= seq_len:
                        yield np.array(token_buffer[:seq_len], dtype=np.int32)
                        token_buffer = token_buffer[seq_len:]

    ds = tf.data.Dataset.from_generator(
        generator, 
        output_signature=tf.TensorSpec(shape=(seq_len,), dtype=np.int32)
    )
    
    if shuffle and split_mode == 'train':
        ds = ds.shuffle(2000)
    
    return ds.batch(batch_size, drop_remainder=True).prefetch(CFG["prefetch_buffer"])

# Pipeline Initialization
try:
    all_files = get_txt_files(CFG["dataset_path"])
    split_idx = int(len(all_files) * 0.9) if len(all_files) > 1 else 0
    
    if len(all_files) == 1:
        print("[INFO] Single file detected. Using internal 90/10 split.")
        train_ds = create_dataset_pipeline(all_files, CFG["max_seq_len"], CFG["global_batch_size"], split_mode='train', shuffle=True)
        val_ds = create_dataset_pipeline(all_files, CFG["max_seq_len"], CFG["global_batch_size"], split_mode='val', shuffle=False)
    else:
        print(f"[INFO] Multi-file dataset: {len(all_files)} files.")
        train_ds = create_dataset_pipeline(all_files[:split_idx], CFG["max_seq_len"], CFG["global_batch_size"], split_mode=None, shuffle=True)
        val_ds = create_dataset_pipeline(all_files[split_idx:], CFG["max_seq_len"], CFG["global_batch_size"], split_mode=None, shuffle=False)

    # Step estimation
    if CFG["total_train_steps"] is None:
        total_bytes = sum(os.path.getsize(f) for f in all_files) * 0.9
        est_tokens = total_bytes // 3.5 
        steps_per_epoch = (est_tokens // CFG["max_seq_len"]) // CFG["global_batch_size"]
        CFG["total_train_steps"] = max(100, int(steps_per_epoch * 2))
        print(f"[INFO] Estimated Tokens: {int(est_tokens):,} | Total Steps: {CFG['total_train_steps']:,}")
        
except Exception as e:
    print(f"[ERROR] Pipeline setup failed: {e}")
    sys.exit(1)

# --- Model Architecture ---
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
        h = nn.Dense(self.d_model, dtype=self.dtype, 
                     kernel_init=nn.initializers.normal(stddev=0.02/math.sqrt(2*self.n_layers)))(h)
        return x + nn.Dropout(self.dropout)(h, deterministic=deterministic)

# Apply Gradient Checkpointing (Rematerialization) to save memory
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
        self.tok_emb = nn.Embed(self.vocab_size, self.d_model, dtype=self.dtype, 
                                embedding_init=nn.initializers.normal(stddev=0.02))
        self.pos_emb = self.param("pos_emb", nn.initializers.normal(stddev=0.02), 
                                  (self.max_len, self.d_model))
        self.drop = nn.Dropout(self.dropout)
        self.norm_f = RMSNorm(dtype=self.dtype)
        self.head = nn.Dense(self.vocab_size, use_bias=False, dtype=jnp.float32, 
                             param_dtype=jnp.float32, kernel_init=nn.initializers.normal(stddev=0.02))
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

# --- Training State & Sharding ---
def create_train_state(rng, model):
    print("[INFO] Initializing model parameters...")
    dummy_input = jnp.zeros((local_batch_size, CFG["max_seq_len"]), dtype=jnp.int32)
    variables = model.init({"params": rng, "dropout": rng}, dummy_input, deterministic=False)
    
    # Define sharding for parameters
    def shard_fn(p): 
        return NamedSharding(mesh, P("data")) if p.ndim == 1 else NamedSharding(mesh, P("data", None))
    
    params = jax.tree_util.tree_map(lambda p, s=shard_fn: jax.device_put(p, s(p)), variables["params"])
    
    # Optimizer schedule
    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0, 
        peak_value=CFG["lr"], 
        warmup_steps=CFG["warmup_steps"], 
        decay_steps=CFG["total_train_steps"], 
        end_value=CFG["lr"] * 0.1
    )
    
    tx = optax.chain(
        optax.clip_by_global_norm(CFG["grad_clip"]), 
        optax.adamw(learning_rate=sched, weight_decay=CFG["weight_decay"])
    )
    
    return ts.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Initialize Model and State
model = TransformerLM(
    vocab_size, CFG["model_dim"], CFG["num_layers"], CFG["num_heads"], 
    CFG["max_seq_len"], CFG["dropout"], CFG["mlp_dim"], CFG["compute_dtype"]
)

state = create_train_state(jr.PRNGKey(CFG["seed"]), model)
ckpt_manager = ocp.CheckpointManager(
    os.path.join(CFG["workdir"], "ckpts"), 
    options=ocp.CheckpointManagerOptions(max_to_keep=CFG["ckpt_keep"], create=True)
)

# Restore checkpoint if available
start_step = 0
if latest_step := ckpt_manager.latest_step():
    state = ckpt_manager.restore(latest_step, args=ocp.args.StandardRestore(state))
    start_step = latest_step
    print(f"[INFO] Resuming from checkpoint: {latest_step}")

# --- Execution Loops ---
train_iter = iter(train_ds.as_numpy_iterator())
val_iter = iter(val_ds.as_numpy_iterator())

@functools.partial(jax.jit)
def train_step(state, batch, rng):
    """Performs a single training step with global loss calculation."""
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch, deterministic=False, rngs={"dropout": rng})
        
        logits_shift = logits[:, :-1]
        labels_shift = batch[:, 1:]
        
        loss_per_token = optax.softmax_cross_entropy_with_integer_labels(logits_shift, labels_shift)
        mask = (labels_shift != pad_token_id).astype(jnp.float32)
        
        # Loss aggregation handles sharding automatically via NamedSharding
        loss = (loss_per_token * mask).sum() / (mask.sum() + 1e-6)
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, optax.global_norm(grads)

@functools.partial(jax.jit)
def eval_step(state, batch):
    """Performs a single evaluation step."""
    logits = state.apply_fn({"params": state.params}, batch, deterministic=True)
    labels_shift = batch[:, 1:]
    mask = (labels_shift != pad_token_id).astype(jnp.float32)
    loss_per_token = optax.softmax_cross_entropy_with_integer_labels(logits[:, :-1], labels_shift)
    
    return (loss_per_token * mask).sum() / (mask.sum() + 1e-6)

# Main Loop
print(f"\n[INFO] Starting training loop | Steps: {start_step} to {CFG['total_train_steps']}")
t0 = time.perf_counter()

for step in range(start_step, CFG["total_train_steps"]):
    try:
        batch_np = next(train_iter)
    except StopIteration:
        train_iter = iter(train_ds.as_numpy_iterator())
        batch_np = next(train_iter)
    
    batch = jax.device_put(jnp.asarray(batch_np), data_sharding)
    rng = jr.fold_in(jr.PRNGKey(CFG["seed"]), step)
    
    state, loss, grad_norm = train_step(state, batch, rng)
    
    # Logging
    if (step + 1) % 20 == 0:
        t1 = time.perf_counter()
        tps = (20 * CFG["global_batch_size"] * CFG["max_seq_len"]) / (t1 - t0)
        print(f"Step {step+1}/{CFG['total_train_steps']} | Loss: {loss:.4f} | GradNorm: {grad_norm:.2f} | TPS: {tps:,.0f}")
        t0 = t1

    # Validation
    if (step + 1) % CFG["eval_every"] == 0:
        val_losses = []
        for _ in range(20):
            try:
                val_batch = jax.device_put(jnp.asarray(next(val_iter)), data_sharding)
                val_losses.append(eval_step(state, val_batch))
            except StopIteration:
                val_iter = iter(val_ds.as_numpy_iterator())
                break
        
        if val_losses:
            avg_val = np.mean(val_losses)
            print(f"[VAL] Step {step+1} | Loss: {avg_val:.4f} | Delta: {avg_val - loss:.4f}")

    # Checkpointing
    if (step + 1) % CFG["save_every"] == 0 or (step + 1) == CFG["total_train_steps"]:
        ckpt_manager.save(step + 1, args=ocp.args.StandardSave(jax.device_get(state)))
        print(f"[SAVE] Checkpoint saved at step {step+1}")

print("[INFO] Training completed successfully.")
