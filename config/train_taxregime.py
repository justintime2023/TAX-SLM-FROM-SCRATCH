import os

# I/O
out_dir = 'out-taxregime'
eval_interval = 250
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# Data
dataset = 'taxregime'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 256

# Model - Small architecture for Mac
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0
bias = False

# AdamW optimizer
learning_rate = 6e-4
max_iters = 20000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate decay
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 20000
min_lr = 6e-5

# System
device = 'mps'  # For Mac M1/M2/M3
dtype = 'float32'
compile = False  # Don't compile on Mac
