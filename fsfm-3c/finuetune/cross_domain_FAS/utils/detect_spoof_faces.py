#!/usr/bin/env python3
"""
detect_spoof_faces.py  -  Run FSFM FAS on all .jpg files
                            and save each image with overlaid prediction.
"""

import os, sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path

try:
    from facenet_pytorch import MTCNN
    _mtcnn_ok = True
except ImportError:
    _mtcnn_ok = False
    print("⚠️  pip install facenet-pytorch  for face alignment (strongly recommended)")

# ────────────────── Command Line Arguments ───────────────────────────────────
parser = argparse.ArgumentParser(description="Detect spoof faces in a directory of images.")
parser.add_argument("--ckpt", type=str, required=True, 
                    help="Path to the model checkpoint file")
parser.add_argument("--input_dir", type=str, required=True, 
                    help="Directory containing input .jpg images")
parser.add_argument("--output_dir", type=str, required=True, 
                    help="Directory to save annotated images and predictions.csv")
args = parser.parse_args()

CKPT = args.ckpt
INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir

# ────────────────── Model definition ─────────────────────────────────────────
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
        self.fc_norm = nn.LayerNorm(embed_dim)
        self.head    = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = x[:, 1:, :].mean(dim=1)   # global pool (not CLS)
        x = self.fc_norm(x)
        return self.head(x)

# ────────────────── Configuration ───────────────────────────────────────────
IMG_SIZE   = 224
THRESHOLD  = 0.5         # decision threshold (p_real > THR → Real)
REAL_CLASS_IDX = 1       # from your label mapping (ImageFolder: fake=0, real=1)

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Override if pretrain_ds_mean_std.txt exists next to the checkpoint
import json
_stats = os.path.join(os.path.dirname(CKPT), 'pretrain_ds_mean_std.txt')
if os.path.exists(_stats):
    with open(_stats) as f:
        d = json.loads(f.readline())
        MEAN, STD = d['mean'], d['std']
        print(f"✅ pretrain_ds_mean_std.txt found: MEAN={MEAN}")

print(f"Threshold = {THRESHOLD}  |  Image size = {IMG_SIZE}")
print(f"Normalization: MEAN={MEAN}  STD={STD}")

# ────────────────── Load model ──────────────────────────────────────────────
state = torch.load(CKPT, map_location='cpu', weights_only=False)
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
model.load_state_dict(remapped, strict=True)
model.eval()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(DEVICE)
print(f"✅ Model loaded | device: {DEVICE}\n")

# ────────────────── Preprocessing ────────────────────────────────────────────
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
    """Return (1,3,H,W) normalised tensor + face_found bool."""
    if mtcnn is not None:
        face = mtcnn(pil_img)
        if face is not None:
            return _norm(face.float() / 255.0).unsqueeze(0).to(DEVICE), True
    # fallback: simple resize + norm
    return _norm(_resize_tensor(pil_img)).unsqueeze(0).to(DEVICE), False

# ────────────────── Overlay helper ────────────────────────────────────────────
def draw_prediction(pil_img, label, confidence):
    """Render 'REAL (0.95)' or 'FAKE (0.12)' at bottom of image."""
    draw = ImageDraw.Draw(pil_img)
    # use a default font – you can specify a .ttf file for better appearance
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()

    text = f"{label} ({confidence:.2f})"
    # get text size so we can center it
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    img_w, img_h = pil_img.size

    # Position: bottom centre, with a small margin
    x = (img_w - w) // 2
    y = img_h - h - 20

    # Draw a semi-transparent black rectangle behind the text for readability
    rect_margin = 10
    draw.rectangle([x - rect_margin, y - rect_margin,
                    x + w + rect_margin, y + h + rect_margin],
                   fill=(0, 0, 0, 180))
    # Draw white text
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return pil_img

# ────────────────── Main loop: walk through input_dir ───────────────────────
all_files = []
for root, dirs, files in os.walk(INPUT_DIR):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg')):   # only jpg as requested
            all_files.append(os.path.join(root, f))

print(f"Found {len(all_files)} JPG image(s) in {INPUT_DIR}\n")
if len(all_files) == 0:
    sys.exit("No images found. Exiting.")

os.makedirs(OUTPUT_DIR, exist_ok=True)

results = []   # list of (relative_path, p_real, p_spoof, predicted_label)

for idx, filepath in enumerate(all_files, 1):
    rel_path = os.path.relpath(filepath, INPUT_DIR)
    out_path = os.path.join(OUTPUT_DIR, rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    try:
        img = Image.open(filepath).convert('RGB')
    except Exception as e:
        print(f"[{idx:4d}/{len(all_files)}] ⚠️  {rel_path} : open error ({e})")
        continue

    inp, face_found = preprocess(img)
    with torch.no_grad():
        logits = model(inp)
        probs  = F.softmax(logits, dim=1)[0]   # [p_fake, p_real]
        p_real = probs[REAL_CLASS_IDX].item()
        p_fake = probs[1 - REAL_CLASS_IDX].item()

    label = "REAL" if p_real >= THRESHOLD else "FAKE"
    confidence = p_real if label == "REAL" else p_fake

    # Optional: print progress every 100 images
    if idx % 100 == 0 or idx == 1:
        print(f"[{idx:4d}/{len(all_files)}] {rel_path} → {label} (p_real={p_real:.3f})"
              f"{'' if face_found else ' (no face)'}")

    # Draw the prediction on the original (not resized) image
    annotated_img = draw_prediction(img.copy(), label, confidence)
    annotated_img.save(out_path)

    results.append([rel_path, p_real, p_fake, label, face_found])

# ────────────────── Save summary CSV ─────────────────────────────────────────
csv_path = os.path.join(OUTPUT_DIR, "predictions.csv")
with open(csv_path, "w") as f:
    f.write("relative_path,p_real,p_fake,prediction,face_found\n")
    for rel_path, p_real, p_fake, label, face_found in results:
        f.write(f"{rel_path},{p_real:.4f},{p_fake:.4f},{label},{face_found}\n")

print(f"\n✅ Done. Processed {len(results)}/{len(all_files)} images.")
print(f"   Output folder   : {OUTPUT_DIR}")
print(f"   Predictions CSV : {csv_path}")


# python detect_spoof_faces.py \
#     --ckpt path/to/checkpoint-min_val_loss.pth \
#     --input_dir path/to/video_frames \
#     --output_dir path/to/video_frames_with_predictions