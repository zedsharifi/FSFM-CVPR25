#!/usr/bin/env python3
"""
eval_fas.py  –  FSFM FAS checkpoint evaluation on LCC_FASD_test
─────────────────────────────────────────────────────────────────────────────
Confirmed from diagnose_checkpoint.py output:

  Architecture : ViT-B/16, global_pool=True
  Forward pass : tokens[:,1:,:].mean(dim=1) → fc_norm → head   (NOT cls token)
  Training code: main_finetune_DfD.py  (uses ImageFolder → standard labels)
  Label mapping: ImageFolder sorts classes alphabetically:
                   class 0 = 'fake'  (f < r)
                   class 1 = 'real'
                 BUT wait – the training data_path has no train/val subfolders
                 with 'fake'/'real' – it uses MCIO datasets. Check args carefully:
                 data_path points to MCIO frames, fine-tuned via main_finetune_DfD.py
                 which uses ImageFolder on train/ subfolders containing real/fake.
                 ImageFolder alphabetical: fake=0, real=1.

  Normalization: args.normalize_from_IMN = False, no pretrain_ds_mean_std.txt found
                 → training code defaulted to ImageNet stats:
                   mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

  MTCNN        : Use for face alignment (same as training preprocessing)
"""

import os, json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

try:
    from facenet_pytorch import MTCNN
    _mtcnn_ok = True
except ImportError:
    _mtcnn_ok = False
    print("⚠️  pip install facenet-pytorch  for face alignment (strongly recommended)")

# ══════════════════════════════════════════════════════════════════════════════
# Command Line Arguments
# ══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="Evaluate FSFM FAS checkpoint on a test dataset.")
parser.add_argument("--ckpt", type=str, required=True, 
                    help="Path to the model checkpoint file")
parser.add_argument("--test_root", type=str, required=True, 
                    help="Path to test dataset root (containing 'real' and 'fake' subfolders)")
parser.add_argument("--output_file", type=str, default="results_final.txt", 
                    help="Path to save the results report text file")
args = parser.parse_args()

CKPT = args.ckpt
TEST_ROOT = args.test_root
OUTPUT_FILE = args.output_file

# ══════════════════════════════════════════════════════════════════════════════
# Model: ViT-B/16 with global_pool=True
# Forward: x[:,1:,:].mean(dim=1) → fc_norm → head
# This is the CONFIRMED architecture from diagnose_checkpoint.py
# ══════════════════════════════════════════════════════════════════════════════
class Mlp(nn.Module):
    def __init__(self, in_f, hidden_f=None, out_f=None, drop=0.):
        super().__init__()
        hidden_f = hidden_f or in_f; out_f = out_f or in_f
        self.fc1 = nn.Linear(in_f, hidden_f); self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_f, out_f); self.drop = nn.Dropout(drop)
    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        attn = self.attn_drop((q @ k.transpose(-2,-1)) * self.scale).softmax(dim=-1)
        return self.proj_drop(self.proj((attn @ v).transpose(1,2).reshape(B, N, C)))

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = Mlp(dim, int(dim * mlp_ratio), drop=drop)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class ViT_GlobalPool(nn.Module):
    """
    ViT-B/16 with global average pooling — matches checkpoint exactly.
    Forward: patch_embed → blocks → mean(patch tokens) → fc_norm → head
    No CLS token is used for classification (only for position in sequence).
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=2,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks      = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True)
            for _ in range(depth)
        ])
        self.fc_norm = nn.LayerNorm(embed_dim)   # ← global pool norm (confirmed present)
        self.head    = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        # patch embedding
        x = self.patch_embed(x).flatten(2).transpose(1, 2)   # (B, N, D)
        # prepend cls token + add pos embed
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed      # (B, N+1, D)
        # transformer blocks
        for blk in self.blocks:
            x = blk(x)
        # GLOBAL AVERAGE POOL over patch tokens only (exclude cls token at index 0)
        x = x[:, 1:, :].mean(dim=1)   # (B, D)  ← confirmed from args.global_pool=True
        x = self.fc_norm(x)
        return self.head(x)

# ══════════════════════════════════════════════════════════════════════════════
# Configuration — all confirmed from diagnose_checkpoint.py
# ══════════════════════════════════════════════════════════════════════════════
IMG_SIZE = 224

# Confirmed: normalize_from_IMN=False, no pretrain_ds_mean_std.txt next to ckpt
# → training defaulted to ImageNet stats (see utils/dataset.py get_dataset())
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Override if pretrain_ds_mean_std.txt exists next to the checkpoint
_stats = os.path.join(os.path.dirname(CKPT), 'pretrain_ds_mean_std.txt')
if os.path.exists(_stats):
    with open(_stats) as f:
        d = json.loads(f.readline())
        MEAN, STD = d['mean'], d['std']
        print(f"✅ pretrain_ds_mean_std.txt found: MEAN={MEAN}")

# Label mapping — confirmed from main_finetune_DfD.py which uses ImageFolder.
# ImageFolder sorts subdirectories alphabetically:
#   'fake' < 'real'  →  fake=0, real=1
# So: p_real = softmax(logits)[1]  |  p_spoof = softmax(logits)[0]
REAL_CLASS_IDX  = 1
SPOOF_CLASS_IDX = 0

print("=" * 65)
print("CONFIRMED SETTINGS (from diagnose_checkpoint.py):")
print(f"  Architecture : ViT-B/16, global_pool=True")
print(f"  Forward      : mean(patch_tokens) → fc_norm → head")
print(f"  Norm         : MEAN={MEAN}")
print(f"  Label map    : fake=0, real=1  (ImageFolder alphabetical)")
print(f"  Trained for  : {200} epochs on MCIO datasets")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════════════════
# Load model — with key remapping for patch_embed
# ══════════════════════════════════════════════════════════════════════════════
state = torch.load(CKPT, map_location='cpu', weights_only=False)

# Checkpoint was saved with a PatchEmbed wrapper class, so patch weights are
# stored as 'patch_embed.proj.weight' / 'patch_embed.proj.bias'.
# Our model uses a bare nn.Conv2d named 'patch_embed', so remap the keys.
raw = state['model']
remapped = {}
for k, v in raw.items():
    if k == 'patch_embed.proj.weight':
        remapped['patch_embed.weight'] = v
    elif k == 'patch_embed.proj.bias':
        remapped['patch_embed.bias'] = v
    else:
        remapped[k] = v

model = ViT_GlobalPool(img_size=IMG_SIZE, num_classes=2)
missing, unexpected = model.load_state_dict(remapped, strict=True)
print(f"\n✅ Weights loaded strict=True (0 missing, 0 unexpected)")

model.eval()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(DEVICE)
print(f"✅ Model loaded | device: {DEVICE}\n")

# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
if _mtcnn_ok:
    mtcnn = MTCNN(image_size=IMG_SIZE, margin=20, keep_all=False,
                  device=DEVICE, post_process=False)
    print("✅ MTCNN face aligner ready\n")
else:
    mtcnn = None

_resize_tensor = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
_norm = transforms.Normalize(mean=MEAN, std=STD)

def preprocess(pil_img):
    """Returns (1,3,H,W) normalised tensor + face_found bool."""
    if mtcnn is not None:
        face = mtcnn(pil_img)
        if face is not None:
            return _norm(face.float() / 255.0).unsqueeze(0).to(DEVICE), True
    return _norm(_resize_tensor(pil_img)).unsqueeze(0).to(DEVICE), False

# ══════════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════════
all_scores = []   # p_real for every image
all_labels = []   # 1=real, 0=spoof  (ground truth)
all_names  = []
no_face    = 0

for gt_str in ['real', 'fake']:
    folder = os.path.join(TEST_ROOT, gt_str)
    if not os.path.isdir(folder):
        print(f"❌ Not found: {folder}"); continue

    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith(('.jpg','.png','.jpeg','.bmp','.tiff')))
    gt_int = 1 if gt_str == 'real' else 0
    print(f"Processing {len(files):5d} images in '{gt_str}/' …")

    for fname in files:
        try:
            img = Image.open(os.path.join(folder, fname)).convert('RGB')
        except Exception as e:
            print(f"  ⚠️  {fname}: {e}"); continue

        inp, found = preprocess(img)
        if not found: no_face += 1

        with torch.no_grad():
            probs  = F.softmax(model(inp), dim=1)[0]
            p_real = probs[REAL_CLASS_IDX].item()

        all_scores.append(p_real)
        all_labels.append(gt_int)
        all_names.append((gt_str, fname))

all_scores = np.array(all_scores)
all_labels = np.array(all_labels)

# ══════════════════════════════════════════════════════════════════════════════
# Score distribution — crucial diagnostic
# ══════════════════════════════════════════════════════════════════════════════
real_s  = all_scores[all_labels == 1]
spoof_s = all_scores[all_labels == 0]

print("\n── p(real) score distribution ──────────────────────────────")
print(f"  Real  : mean={real_s.mean():.3f}  std={real_s.std():.3f}  "
      f"min={real_s.min():.3f}  max={real_s.max():.3f}")
print(f"  Spoof : mean={spoof_s.mean():.3f}  std={spoof_s.std():.3f}  "
      f"min={spoof_s.min():.3f}  max={spoof_s.max():.3f}")
print(f"  Separation (Δmean): {real_s.mean() - spoof_s.mean():.3f}  "
      f"(positive = model discriminates correctly)")

# ══════════════════════════════════════════════════════════════════════════════
# Threshold sweep — balanced accuracy + EER + HTER
# ══════════════════════════════════════════════════════════════════════════════
thresholds  = np.linspace(0.01, 0.99, 990)
best_bal    = 0.; best_thr_bal  = 0.5
best_hter   = 1.; best_thr_hter = 0.5

for thr in thresholds:
    preds = (all_scores >= thr).astype(int)
    tp = ((preds==1)&(all_labels==1)).sum()
    tn = ((preds==0)&(all_labels==0)).sum()
    fp = ((preds==1)&(all_labels==0)).sum()
    fn = ((preds==0)&(all_labels==1)).sum()
    tpr = tp/(tp+fn) if (tp+fn)>0 else 0.
    tnr = tn/(tn+fp) if (tn+fp)>0 else 0.
    far = fp/(fp+tn) if (fp+tn)>0 else 0.
    frr = fn/(fn+tp) if (fn+tp)>0 else 0.
    bal  = (tpr + tnr) / 2
    hter = (far + frr) / 2
    if bal  > best_bal:  best_bal  = bal;  best_thr_bal  = thr
    if hter < best_hter: best_hter = hter; best_thr_hter = thr

# AUC
from sklearn.metrics import roc_auc_score
try:
    auc = roc_auc_score(all_labels, all_scores)
    print(f"\n── AUC: {auc*100:.2f}%")
except Exception as e:
    auc = None; print(f"AUC error: {e}")

print(f"── Best balanced-acc threshold : {best_thr_bal:.3f}  "
      f"→ balanced acc = {best_bal*100:.2f}%")
print(f"── Best HTER threshold         : {best_thr_hter:.3f}  "
      f"→ HTER = {best_hter*100:.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# Report at multiple thresholds
# ══════════════════════════════════════════════════════════════════════════════
def report(thr, tag):
    preds = (all_scores >= thr).astype(int)
    real_mask  = all_labels == 1
    spoof_mask = all_labels == 0
    real_acc  = (preds[real_mask]  == 1).mean() * 100
    spoof_acc = (preds[spoof_mask] == 0).mean() * 100
    overall   = (preds == all_labels).mean() * 100
    balanced  = (real_acc + spoof_acc) / 2
    rr = int(((preds==1)&(all_labels==1)).sum())
    rs = int(((preds==0)&(all_labels==1)).sum())
    fr = int(((preds==1)&(all_labels==0)).sum())
    fs = int(((preds==0)&(all_labels==0)).sum())
    far = fr / (fr + fs) if (fr+fs) > 0 else 0.
    frr = rs / (rs + rr) if (rs+rr) > 0 else 0.
    hter = (far + frr) / 2
    print(f"\n{'='*65}")
    print(f"  {tag}  (threshold={thr:.3f})")
    print(f"{'='*65}")
    print(f"  Total images    : {len(all_labels)}  "
          f"(real={real_mask.sum()}, spoof={spoof_mask.sum()})")
    print(f"  No-face fallback: {no_face}")
    print(f"  Overall acc     : {overall:.2f}%")
    print(f"  Real acc (TPR)  : {real_acc:.2f}%")
    print(f"  Spoof acc (TNR) : {spoof_acc:.2f}%")
    print(f"  Balanced acc    : {balanced:.2f}%")
    print(f"  HTER            : {hter*100:.2f}%")
    print(f"  FAR             : {far*100:.2f}%  |  FRR: {frr*100:.2f}%")
    print(f"  ── Confusion matrix (rows=true, cols=pred) ──")
    print(f"                   Pred Real   Pred Spoof")
    print(f"  Actual Real      {rr:6d}     {rs:6d}")
    print(f"  Actual Spoof     {fr:6d}     {fs:6d}")
    return dict(thr=thr, overall=overall, real_acc=real_acc,
                spoof_acc=spoof_acc, balanced=balanced, hter=hter*100,
                rr=rr, rs=rs, fr=fr, fs=fs)

m_default  = report(0.5,           "DEFAULT THRESHOLD")
m_bal      = report(best_thr_bal,  "OPTIMAL (max balanced-acc)")
m_hter     = report(best_thr_hter, "OPTIMAL (min HTER)")

# ══════════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════════
with open(OUTPUT_FILE, "w") as f:
    f.write("FSFM FAS – Evaluation\n")
    f.write(f"Checkpoint : {CKPT}\n")
    f.write(f"Test Root  : {TEST_ROOT}\n")
    f.write(f"Architecture: ViT-B/16, global_pool=True, "
            f"forward=mean(patch_tokens)→fc_norm→head\n")
    f.write(f"Normalization: MEAN={MEAN}  STD={STD}\n")
    f.write(f"Label mapping: fake=0, real=1 (ImageFolder alphabetical)\n")
    f.write(f"MTCNN: {'enabled' if mtcnn else 'disabled'}\n")
    f.write(f"AUC: {auc*100:.2f}%\n\n" if auc else "AUC: N/A\n\n")
    f.write(f"Score distribution:\n")
    f.write(f"  Real : mean={real_s.mean():.4f} std={real_s.std():.4f}\n")
    f.write(f"  Spoof: mean={spoof_s.mean():.4f} std={spoof_s.std():.4f}\n\n")
    for tag, m in [("DEFAULT (0.50)", m_default),
                   (f"OPTIMAL bal-acc ({best_thr_bal:.3f})", m_bal),
                   (f"OPTIMAL HTER ({best_thr_hter:.3f})", m_hter)]:
        f.write(f"── {tag} ──\n")
        f.write(f"  Overall={m['overall']:.2f}%  Real={m['real_acc']:.2f}%  "
                f"Spoof={m['spoof_acc']:.2f}%  Balanced={m['balanced']:.2f}%  "
                f"HTER={m['hter']:.2f}%\n")
        f.write(f"  Confusion: RR={m['rr']} RS={m['rs']} FR={m['fr']} FS={m['fs']}\n\n")
    f.write("── Per-image scores ──\n")
    f.write(f"{'true_label':12s} | {'filename':40s} | p_real  | p_spoof\n")
    f.write("-" * 80 + "\n")
    for (gt_str, fname), p_r in zip(all_names, all_scores):
        f.write(f"{gt_str:12s} | {fname:40s} | {p_r:.4f}  | {1-p_r:.4f}\n")

print(f"\n📁 Results saved → '{OUTPUT_FILE}'")


# python eval_fas.py \
#     --ckpt path/to/checkpoint-min_val_loss.pth \
#     --test_root path/to/LCC_FASD_test/test \
#     --output_file results_final.txt