#!/usr/bin/env python3
"""
diagnose_checkpoint.py
Prints all checkpoint keys, shapes, and stats to understand the exact model structure.
"""
import argparse
import torch
import numpy as np

# Set up argument parsing so the path isn't hardcoded
parser = argparse.ArgumentParser(description="Diagnose PyTorch checkpoint structure.")
parser.add_argument(
    "--ckpt", 
    type=str, 
    required=True, 
    help="Path to the model checkpoint file (e.g., path/to/checkpoint.pth)"
)
cli_args = parser.parse_args()

CKPT = cli_args.ckpt

# Load the checkpoint
state = torch.load(CKPT, map_location='cpu', weights_only=False)

print("=" * 70)
print("TOP-LEVEL KEYS IN CHECKPOINT:")
print(list(state.keys()))

print("\n" + "=" * 70)
print("ARGS stored in checkpoint:")
args = state.get('args', None)
if args is not None:
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
else:
    print("  No args found.")

print("\n" + "=" * 70)
model_state = state['model']
print(f"TOTAL KEYS IN model state_dict: {len(model_state)}\n")

print("ALL KEYS + SHAPES:")
for k, v in model_state.items():
    print(f"  {k:55s}  {str(list(v.shape)):30s}  dtype={v.dtype}")

# Check if there's a head
print("\n" + "=" * 70)
print("HEAD / CLASSIFIER KEYS:")
head_keys = [k for k in model_state if 'head' in k or 'classifier' in k or 'fc_norm' in k]
for k in head_keys:
    v = model_state[k]
    print(f"  {k:55s}  shape={list(v.shape)}")
    if v.numel() < 20:
        print(f"    values: {v.numpy()}")

print("\n" + "=" * 70)
print("EMBEDDING / NORM KEYS:")
for k in model_state:
    if 'norm' in k and 'blocks' not in k:
        v = model_state[k]
        print(f"  {k:55s}  shape={list(v.shape)}")

print("\n" + "=" * 70)
print("HEAD WEIGHT STATS (to check if it was actually trained):")
for k in head_keys:
    v = model_state[k].float().numpy()
    print(f"  {k}: mean={v.mean():.6f}  std={v.std():.6f}  "
          f"min={v.min():.6f}  max={v.max():.6f}")

# Check global_pool vs cls_token usage
print("\n" + "=" * 70)
print("ARCHITECTURE CLUES:")
has_fc_norm   = any('fc_norm' in k for k in model_state)
has_norm      = any(k == 'norm.weight' or k == 'norm.bias' for k in model_state)
has_pos_drop  = any('pos_drop' in k for k in model_state)
print(f"  has fc_norm  (global_pool=True)  : {has_fc_norm}")
print(f"  has norm     (global_pool=False) : {has_norm}")
print(f"  has pos_drop                     : {has_pos_drop}")
print(f"  num transformer blocks           : "
      f"{max(int(k.split('.')[1]) for k in model_state if k.startswith('blocks.')) + 1}")
print(f"  embed_dim (from pos_embed)       : {model_state['pos_embed'].shape[-1]}")
print(f"  num_patches+1 (from pos_embed)   : {model_state['pos_embed'].shape[1]}")
print(f"  → patch_size inferred            : "
      f"{int((model_state['pos_embed'].shape[1] - 1) ** 0.5 * 16 / 14)}?  "
      f"(num_patches={model_state['pos_embed'].shape[1]-1})")

if has_fc_norm:
    print("\n  → Model uses GLOBAL AVERAGE POOL (fc_norm present, no norm)")
    print("    forward: x = x[:,1:,:].mean(dim=1) → fc_norm → head")
else:
    print("\n  → Model uses CLS TOKEN (norm present, no fc_norm)")
    print("    forward: x = norm(x)[:,0] → head")

print("\n" + "=" * 70)
print("DONE. Share this output to determine the correct test architecture.")

# python diagnose_checkpoint.py --ckpt path/to/your/checkpoint-min_val_loss.pth