"""Demo interface for pharmacy-label damage detection and reconstruction.

The application uses a Keras-trained U-Net for mask prediction and a conditional
diffusion model for reconstruction. The U-Net weights are converted in a small
CPU-only helper process before inference so TensorFlow and PyTorch do not need
to initialize their GPU libraries in the same process.

Example:
    python ie643_demo.py --unet unet_group_safe_best.keras --recon best_ema.pth

Optional arguments include --data-root, --split-csv, and --share.
"""
import os
import argparse
import csv
import json
import math
import random
import subprocess
import sys
import tempfile
import time
import traceback
from functools import wraps
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# NOTE: TensorFlow is never imported in this process. Loading TF and PyTorch CUDA/cuDNN libraries together
# crashed the Colab process. The Keras U-Net weights are read once by a CPU-only helper subprocess and loaded
# into an identical PyTorch U-Net, which is checked numerically against Keras at startup.
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity
from skimage.color import rgb2lab, lab2rgb, deltaE_ciede2000

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

cv2.setNumThreads(2)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


# 1. U-Net detector (architecture from U_Net_Group_Safe notebook, weights exported from the .keras file)
KERAS_EXPORT_SCRIPT = r"""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf
from tensorflow import keras
def dice_coef(y_true, y_pred):
    y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1]); y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    i = tf.reduce_sum(y_true * y_pred)
    return (2.0 * i + 1e-7) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-7)
def dice_loss(a, b): return 1.0 - dice_coef(a, b)
def combined_loss(a, b): return keras.losses.binary_crossentropy(a, b) + dice_loss(a, b)
def iou(y_true, y_pred):
    y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1]); y_pred = tf.cast(tf.reshape(y_pred, [-1]) > 0.5, tf.float32)
    i = tf.reduce_sum(y_true * y_pred)
    return (i + 1e-7) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - i + 1e-7)
path, out = sys.argv[1], sys.argv[2]
m = keras.models.load_model(path, compile=False, custom_objects=dict(dice_coef=dice_coef, dice_loss=dice_loss,
                            combined_loss=combined_loss, iou=iou))
arrays, layers = {}, []
for L in m.layers:
    kind = type(L).__name__
    if kind in ("Conv2D", "Conv2DTranspose", "BatchNormalization"):
        ws = L.get_weights()
        for j, w in enumerate(ws):
            arrays[f"w{len(layers)}_{j}"] = w
        layers.append({"kind": kind, "name": L.name, "n": len(ws),
                       "eps": float(getattr(L, "epsilon", 0.0)), "act": str(getattr(getattr(L, "activation", None), "__name__", ""))})
size = int(m.input_shape[1] or 256)
rng = np.random.default_rng(0)
x = rng.random((1, size, size, 3)).astype(np.float32)
arrays["probe_x"] = x
arrays["probe_y"] = m(x, training=False).numpy()
np.savez(out, **arrays)
json.dump({"size": size, "layers": layers, "params": int(m.count_params())}, open(out + ".json", "w"))
print("exported", len(layers), "layers")
"""


class KConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout, eps=1e-3)
        self.c2, self.b2 = nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout, eps=1e-3)

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))))


class KerasUNetTorch(nn.Module):
    """Exact PyTorch replica of unet() in the U-Net notebook (filters 32-64-128-256-512)."""
    def __init__(self):
        super().__init__()
        f = [32, 64, 128, 256, 512]
        self.enc = nn.ModuleList([KConvBlock(3, f[0])] + [KConvBlock(f[i - 1], f[i]) for i in range(1, 5)])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(f[i], f[i - 1], 2, stride=2) for i in range(4, 0, -1)])
        self.dec = nn.ModuleList([KConvBlock(f[i - 1] * 2, f[i - 1]) for i in range(4, 0, -1)])
        self.out = nn.Conv2d(f[0], 1, 1)

    def forward(self, x):                      # x: (B,3,H,W) in [0,1]
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < 4:
                skips.append(x); x = F.max_pool2d(x, 2)
        for up, blk in zip(self.ups, self.dec):
            x = blk(torch.cat([up(x), skips.pop()], dim=1))
        return torch.sigmoid(self.out(x))

    def keras_order(self):
        # modules in the order Keras lists Conv2D / BatchNormalization / Conv2DTranspose layers
        mods = []
        for b in self.enc:
            mods += [b.c1, b.b1, b.c2, b.b2]
        for up, b in zip(self.ups, self.dec):
            mods += [up, b.c1, b.b1, b.c2, b.b2]
        return mods + [self.out]


def _assign(mod, kind, ws):
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
    if kind == "Conv2D":
        assert isinstance(mod, nn.Conv2d), f"expected Conv2d, got {type(mod).__name__}"
        mod.weight.data.copy_(t(ws[0].transpose(3, 2, 0, 1))); mod.bias.data.copy_(t(ws[1]))
    elif kind == "Conv2DTranspose":           # keras (kh,kw,out,in) -> torch (in,out,kh,kw)
        assert isinstance(mod, nn.ConvTranspose2d), f"expected ConvTranspose2d, got {type(mod).__name__}"
        mod.weight.data.copy_(t(ws[0].transpose(3, 2, 0, 1))); mod.bias.data.copy_(t(ws[1]))
    elif kind == "BatchNormalization":
        assert isinstance(mod, nn.BatchNorm2d), f"expected BatchNorm2d, got {type(mod).__name__}"
        g, b, mu, var = ws
        mod.weight.data.copy_(t(g)); mod.bias.data.copy_(t(b))
        mod.running_mean.data.copy_(t(mu)); mod.running_var.data.copy_(t(var))


class Detector:
    def __init__(self, path):
        t0 = time.time()
        path = str(path)
        cache = Path(tempfile.gettempdir()) / f"unet_export_{abs(hash((path, os.path.getmtime(path))))}.npz"
        if not cache.exists() or not Path(str(cache) + ".json").exists():
            print("[U-Net] exporting Keras weights in a CPU-only helper process ...", flush=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", TF_CPP_MIN_LOG_LEVEL="3")
            p = subprocess.run([sys.executable, "-c", KERAS_EXPORT_SCRIPT, path, str(cache)],
                               env=env, capture_output=True, text=True)
            if p.returncode != 0:
                raise RuntimeError(f"Keras export failed:\n{p.stdout[-2000:]}\n{p.stderr[-4000:]}")
        meta = json.load(open(str(cache) + ".json"))
        arr = np.load(cache)
        self.size = meta["size"]
        self.net = KerasUNetTorch()
        mods = self.net.keras_order()
        if len(mods) != len(meta["layers"]):
            raise RuntimeError(f"U-Net layer count mismatch: keras {len(meta['layers'])} vs torch {len(mods)}")
        for i, (mod, L) in enumerate(zip(mods, meta["layers"])):
            _assign(mod, L["kind"], [arr[f"w{i}_{j}"] for j in range(L["n"])])
        self.net.to(DEVICE).eval()
        # parity check against the Keras prediction on a fixed random input
        with torch.no_grad():
            y = self.net(torch.from_numpy(arr["probe_x"]).permute(0, 3, 1, 2).to(DEVICE)).cpu().numpy()[0, 0]
        diff = float(np.abs(y - arr["probe_y"][0, :, :, 0]).max())
        self.parity = diff
        status = "OK" if diff < 1e-3 else "WARNING: outputs differ from Keras!"
        print(f"[U-Net] loaded {path} | input {self.size}x{self.size} | params {meta['params']:,} | device {DEVICE} | "
              f"max |torch-keras| = {diff:.2e} ({status}) | {time.time()-t0:.1f}s", flush=True)

    @torch.no_grad()
    def predict_prob(self, rgb_u8):
        # same as tf.image.resize(..., 'bilinear') on the uint8 image (half-pixel centres, no antialias), then /255
        x = torch.from_numpy(np.ascontiguousarray(rgb_u8)).permute(2, 0, 1)[None].float().to(DEVICE)
        if x.shape[-1] != self.size or x.shape[-2] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False, antialias=False)
        p = self.net(x / 255.0)[0, 0].float().cpu().numpy()
        return p.astype(np.float32)


# 2. Reconstruction diffusion model (from resume416reconepoch19 notebook, cells 6-7)
def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

def norm(ch):
    return nn.GroupNorm(32 if ch % 32 == 0 else 8, ch)

def zero_module(m):
    for p in m.parameters():
        nn.init.zeros_(p)
    return m

class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim, dropout):
        super().__init__()
        self.n1, self.c1 = norm(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout * 2)
        self.n2, self.drop = norm(cout), nn.Dropout(dropout)
        self.c2 = zero_module(nn.Conv2d(cout, cout, 3, padding=1))
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.t(F.silu(temb))[:, :, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + scale) + shift
        h = self.c2(self.drop(F.silu(h)))
        return h + self.skip(x)

class AttnBlock(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.heads, self.n = heads, norm(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = zero_module(nn.Conv2d(ch, ch, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, self.heads, C // self.heads, H * W).unbind(1)
        q, k, v = (t.transpose(-1, -2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v)
        return x + self.proj(o.transpose(-1, -2).reshape(B, C, H, W))

class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__(); self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__(); self.conv = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))

class RestorationUNet(nn.Module):
    def __init__(self, in_ch=7, out_ch=3, base=64, mults=(1, 2, 3, 4), num_res=2, attn_levels=(3,), dropout=0.1):
        super().__init__()
        self.base, tdim = base, base * 4
        self.temb = nn.Sequential(nn.Linear(base, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.conv_in = nn.Conv2d(in_ch, base, 3, padding=1)
        self.down, chans, ch = nn.ModuleList(), [base], base
        for lvl, m in enumerate(mults):
            for _ in range(num_res):
                blk = nn.ModuleList([ResBlock(ch, base * m, tdim, dropout)]); ch = base * m
                if lvl in attn_levels:
                    blk.append(AttnBlock(ch))
                self.down.append(blk); chans.append(ch)
            if lvl != len(mults) - 1:
                self.down.append(nn.ModuleList([Downsample(ch)])); chans.append(ch)
        self.mid = nn.ModuleList([ResBlock(ch, ch, tdim, dropout), AttnBlock(ch), ResBlock(ch, ch, tdim, dropout)])
        self.up = nn.ModuleList()
        for lvl, m in reversed(list(enumerate(mults))):
            for i in range(num_res + 1):
                blk = nn.ModuleList([ResBlock(ch + chans.pop(), base * m, tdim, dropout)]); ch = base * m
                if lvl in attn_levels:
                    blk.append(AttnBlock(ch))
                if lvl != 0 and i == num_res:
                    blk.append(Upsample(ch))
                self.up.append(blk)
        self.out = nn.Sequential(norm(ch), nn.SiLU(), zero_module(nn.Conv2d(ch, out_ch, 3, padding=1)))
        self.depth = len(mults) - 1

    @staticmethod
    def _run(blk, h, temb):
        for layer in blk:
            h = layer(h, temb) if isinstance(layer, ResBlock) else layer(h)
        return h

    def forward(self, x, t):
        H, W = x.shape[-2:]
        f = 2 ** self.depth
        ph, pw = (f - H % f) % f, (f - W % f) % f
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        temb = self.temb(timestep_embedding(t, self.base))
        h = self.conv_in(x); hs = [h]
        for blk in self.down:
            h = self._run(blk, h, temb); hs.append(h)
        h = self._run(self.mid, h, temb)
        for blk in self.up:
            h = self._run(blk, torch.cat([h, hs.pop()], dim=1), temb)
        return self.out(h)[..., :H, :W]


class Reconstructor:
    def __init__(self, path, timesteps=1000):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck.get("cfg", {}) or {}
        self.image_size = int(cfg.get("image_size", 416))
        self.use_damaged_pixels = bool(cfg.get("use_damaged_pixels", True))
        self.net = RestorationUNet(in_ch=7, base=cfg.get("base_channels", 64),
                                   mults=tuple(cfg.get("channel_mults", (1, 2, 3, 4))),
                                   num_res=cfg.get("num_res_blocks", 2),
                                   attn_levels=tuple(cfg.get("attn_levels", (3,))),
                                   dropout=cfg.get("dropout", 0.1))
        sd = ck["ema"] if "ema" in ck else ck.get("model", ck)       # best_ema.pth or last_checkpoint.pth
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        self.net.load_state_dict(sd)
        self.net.to(DEVICE).eval()
        self.epoch = ck.get("epoch", ck.get("state", {}).get("epoch", "?"))
        # cosine schedule (Diffusion class)
        T, s = int(cfg.get("timesteps", timesteps)), 0.008
        x = torch.linspace(0, T, T + 1, dtype=torch.float64)
        ac = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
        ac = ac / ac[0]
        betas = (1 - ac[1:] / ac[:-1]).clamp(max=0.999)
        self.T = T
        self.alpha_bar = torch.cumprod(1 - betas, 0).float().to(DEVICE)
        n = sum(p.numel() for p in self.net.parameters())
        t0 = time.time()
        z = np.zeros((self.image_size, self.image_size, 3), np.float32)
        mk = np.zeros((self.image_size, self.image_size), np.float32); mk[:16, :16] = 1
        self.sample(z, mk, steps=2)                                            # warm-up (cuDNN init)
        print(f"[Recon] loaded {path} | epoch {self.epoch} | {self.image_size}px | params {n/1e6:.2f}M | "
              f"device {DEVICE} | warm-up {time.time()-t0:.1f}s", flush=True)

    @torch.no_grad()
    def sample(self, damaged01, mask01, steps=25, seed=0):
        """damaged01: (H,W,3) float [0,1] at image_size; mask01: (H,W) {0,1}. Returns raw prediction (H,W,3) [0,1]."""
        d = torch.from_numpy(np.ascontiguousarray(damaged01)).permute(2, 0, 1)[None].float().to(DEVICE) * 2 - 1
        m = torch.from_numpy(np.ascontiguousarray(mask01.astype(np.float32)))[None, None].to(DEVICE)
        cond = torch.cat([d if self.use_damaged_pixels else d * (1 - m), m], dim=1)
        gen = torch.Generator(device=DEVICE).manual_seed(int(seed))
        x = torch.randn(d.shape, generator=gen, device=DEVICE)
        ts = torch.linspace(self.T - 1, 0, int(steps), device=DEVICE).round().long()
        for i, t in enumerate(ts):
            tb = torch.full((1,), int(t), device=DEVICE, dtype=torch.long)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                v = self.net(torch.cat([x, cond], dim=1), tb).float()
            a = self.alpha_bar[t]
            x0 = (a.sqrt() * x - (1 - a).sqrt() * v).clamp(-1, 1)
            eps = (x - a.sqrt() * x0) / (1 - a).sqrt()
            if i < len(ts) - 1:
                a_next = self.alpha_bar[ts[i + 1]]
                x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
            else:
                x = x0
        raw = (x.clamp(-1, 1) + 1) / 2
        return raw[0].permute(1, 2, 0).cpu().float().numpy()


# 3. Harmonisers, baseline and metrics (cells 8, 9, 13)
def _ring(mask, width):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    return (cv2.dilate(mask.astype(np.uint8), k) > 0) & ~mask

def composite(pred, known, mask):
    m = mask[..., None].astype(np.float32)
    return pred * m + known * (1 - m)

def offset_field_correct(pred, known, mask, ring_width=12, sigma=18.0, fallback_k=0.02):
    ring = _ring(mask, ring_width)
    if ring.sum() < 20:
        return composite(pred, known, mask)
    r = ring.astype(np.float32)
    diff = (known - pred) * r[..., None]
    num = cv2.GaussianBlur(diff, (0, 0), sigma)
    den = cv2.GaussianBlur(r, (0, 0), sigma)[..., None]
    g = diff.sum(axis=(0, 1)) / max(r.sum(), 1.0)
    return composite(np.clip(pred + (num + fallback_k * g) / (den + fallback_k), 0, 1), known, mask)

def lab_stats_transfer(pred, known, mask, ring_width=12):
    ring = _ring(mask, ring_width)
    if ring.sum() < 20:
        return composite(pred, known, mask)
    lp_r, lk_r = rgb2lab(pred[ring][None])[0], rgb2lab(known[ring][None])[0]
    lp_m = rgb2lab(pred[mask][None])[0]
    mu_p, sd_p = lp_r.mean(0), lp_r.std(0) + 1e-3
    mu_k, sd_k = lk_r.mean(0), lk_r.std(0) + 1e-3
    new = (lp_m - mu_p) * np.clip(sd_k / sd_p, 0.5, 2.0) + mu_k
    out = known.copy()
    out[mask] = np.clip(lab2rgb(new[None])[0], 0, 1)
    return out

def poisson_blend(pred, known, mask, pad=16):
    if mask.sum() < 10:
        return composite(pred, known, mask)
    to8 = lambda a: (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    src = cv2.copyMakeBorder(to8(pred), pad, pad, pad, pad, cv2.BORDER_REFLECT)
    dst = cv2.copyMakeBorder(to8(known), pad, pad, pad, pad, cv2.BORDER_REFLECT)
    m = cv2.copyMakeBorder(mask.astype(np.uint8) * 255, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    x, y, w, h = cv2.boundingRect(m)
    try:
        out = cv2.seamlessClone(src, dst, m, (x + w // 2, y + h // 2), cv2.NORMAL_CLONE)
    except cv2.error:
        return composite(pred, known, mask)
    return composite(out[pad:-pad, pad:-pad].astype(np.float32) / 255.0, known, mask)

def offset_then_poisson(pred, known, mask):
    ring = _ring(mask, 12)
    if ring.sum() < 20:
        return poisson_blend(pred, known, mask)
    r = ring.astype(np.float32)
    diff = (known - pred) * r[..., None]
    num = cv2.GaussianBlur(diff, (0, 0), 18.0)
    den = cv2.GaussianBlur(r, (0, 0), 18.0)[..., None]
    g = diff.sum(axis=(0, 1)) / max(r.sum(), 1.0)
    return poisson_blend(np.clip(pred + (num + 0.02 * g) / (den + 0.02), 0, 1), known, mask)

HARMONIZERS = {
    "lab_stats": lab_stats_transfer,          # selected on validation in the notebook
    "raw_composite": composite,
    "offset_field": offset_field_correct,
    "poisson": poisson_blend,
    "offset+poisson": offset_then_poisson,
}

def telea_baseline(known, mask):
    img8 = (np.clip(known, 0, 1) * 255 + 0.5).astype(np.uint8)
    return cv2.inpaint(img8, mask.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA).astype(np.float32) / 255.0

def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 100.0 if mse == 0 else 10 * math.log10(1.0 / mse)

def image_metrics(pred, clean, mask):
    m = mask.astype(bool)
    out = {"PSNR": psnr(pred, clean),
           "SSIM": structural_similarity(pred, clean, channel_axis=2, data_range=1.0),
           "PixAcc%": float(np.mean(np.abs(pred - clean) < 0.15) * 100)}
    if m.sum() > 0:
        p, c = pred[m], clean[m]
        out["mPSNR"] = psnr(p, c)
        out["mPixAcc%"] = float(np.mean(np.abs(p - c) < 0.15) * 100)
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        if (y1 - y0) >= 7 and (x1 - x0) >= 7:
            out["mSSIM(bbox)"] = structural_similarity(pred[y0:y1, x0:x1], clean[y0:y1, x0:x1],
                                                        channel_axis=2, data_range=1.0)
        out["mΔE2000"] = float(np.mean(deltaE_ciede2000(rgb2lab(p[None]), rgb2lab(c[None]))))
    return out

def seg_metrics(pred_bin, gt_bin):
    p, g = pred_bin.astype(bool), gt_bin.astype(bool)
    inter, ps, gs = (p & g).sum(), p.sum(), g.sum()
    dice = (2 * inter + 1e-7) / (ps + gs + 1e-7)
    iou = (inter + 1e-7) / (ps + gs - inter + 1e-7)
    tp, fp, fn = inter, (p & ~g).sum(), (~p & g).sum()
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {"Dice": float(dice), "IoU": float(iou), "Precision": float(prec), "Recall": float(rec)}

METRIC_ORDER = ["mPSNR", "mSSIM(bbox)", "mPixAcc%", "mΔE2000", "PSNR", "SSIM", "PixAcc%"]


# 4. Image helpers
def to_rgb_u8(img):
    if img is None:
        return None
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img * (255 if img.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img

def resize_rgb01(rgb_u8, size):
    # same as read_rgb() in the reconstruction notebook (INTER_AREA when size differs)
    if rgb_u8.shape[0] != size or rgb_u8.shape[1] != size:
        rgb_u8 = cv2.resize(rgb_u8, (size, size), interpolation=cv2.INTER_AREA)
    return rgb_u8.astype(np.float32) / 255.0

def mask_from_upload(img, size):
    a = np.asarray(img)
    if a.ndim == 3:
        a = a[..., :3].max(-1) if a.shape[-1] >= 3 else a[..., 0]
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8) if a.max() <= 1.0 else a.astype(np.uint8)
    if a.shape[0] != size or a.shape[1] != size:
        a = cv2.resize(a, (size, size), interpolation=cv2.INTER_NEAREST)
    return a > 127

def dilate(mask, px):
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0

def u8(a):
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)

def overlay(rgb01, mask, gt=None):
    out = rgb01.copy()
    red = np.array([1.0, 0.15, 0.15], np.float32)
    out[mask] = 0.55 * out[mask] + 0.45 * red
    o8 = u8(out)
    if gt is not None:  # green outline = ground-truth mask
        cnts, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(o8, cnts, -1, (0, 200, 0), 1)
    return o8

def heatmap(prob):
    return cv2.cvtColor(cv2.applyColorMap(u8(prob), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)

def zoom_figure(panels, mask, pad=12):
    ys, xs = np.nonzero(mask)
    if len(ys):
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, mask.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, mask.shape[1])
    else:
        y0, y1, x0, x1 = 0, mask.shape[0], 0, mask.shape[1]
    fig, ax = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.6 * max((y1 - y0) / max(x1 - x0, 1), 0.35) + 0.6))
    ax = np.atleast_1d(ax)
    for a, (title, im) in zip(ax, panels):
        a.imshow(np.clip(im[y0:y1, x0:x1], 0, 1)); a.set_title(title, fontsize=10); a.axis("off")
    plt.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


# 5. Dataset access (split CSV from the reconstruction notebook, or folder scan)
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def load_samples(data_root, split_csv, split):
    root = Path(data_root)
    if split_csv and Path(split_csv).is_file():
        rows = []
        with open(split_csv, newline="") as f:
            for r in csv.DictReader(f):
                if split == "all" or r["split"] == split:
                    rows.append({"damaged": r["damaged_file"], "clean": r["clean_file"], "mask": r["mask_file"]})
        return rows, f"{len(rows)} '{split}' samples from {Path(split_csv).name}"
    # fallback: build_source_groups() from the notebook
    clean_map = {Path(f).stem.split("_")[0]: f for f in os.listdir(root / "clean") if Path(f).suffix.lower() in IMG_EXTS}
    mask_map = {Path(f).stem: f for f in os.listdir(root / "masks") if Path(f).suffix.lower() in IMG_EXTS}
    rows = []
    for f in sorted(os.listdir(root / "damaged")):
        st = Path(f).stem
        if Path(f).suffix.lower() in IMG_EXTS and st in mask_map:
            rows.append({"damaged": f, "clean": clean_map.get(st.split("_")[0]), "mask": mask_map[st]})
    return rows, f"{len(rows)} samples from folder scan (no split CSV, so this is NOT restricted to the test split)"

def read_rgb_u8(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def read_mask_u8(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(f"Could not read {path}")
    return m


# 6. Core pipeline
DET: Detector = None
REC: Reconstructor = None
ARGS = None

def run_pipeline(damaged_u8, clean_u8=None, gt_mask_img=None, mask_source="U-Net prediction",
                 threshold=0.5, dilate_px=0, steps=25, seed=0, harm="lab_stats", run_recon=True):
    S = REC.image_size
    known = resize_rgb01(damaged_u8, S)
    clean = resize_rgb01(clean_u8, S) if clean_u8 is not None else None
    gt_mask = mask_from_upload(gt_mask_img, S) if gt_mask_img is not None else None
    res = {"known": known, "clean": clean, "gt_mask": gt_mask, "times": {}}

    # ---- detection ----
    t0 = time.time()
    print(f"[run] input {damaged_u8.shape} -> U-Net ...", flush=True)
    prob_s = DET.predict_prob(damaged_u8)
    print(f"[run] U-Net done in {time.time()-t0:.2f}s", flush=True)
    res["times"]["U-Net"] = time.time() - t0
    prob = cv2.resize(prob_s, (S, S), interpolation=cv2.INTER_LINEAR)
    pred_mask = prob > threshold
    res["prob"], res["pred_mask"] = prob, pred_mask
    if gt_mask_img is not None:
        # detection metrics at the U-Net's own resolution (as in the training notebook: nearest resize, >0)
        gt_s = mask_from_upload_any(gt_mask_img, DET.size)
        res["seg"] = seg_metrics(prob_s > threshold, gt_s)

    if mask_source.startswith("Ground") and gt_mask is not None:
        mask = gt_mask.copy()
    else:
        mask = pred_mask.copy()
    mask = dilate(mask, int(dilate_px))
    res["mask"] = mask
    if not run_recon:
        return res

    # ---- reconstruction ----
    if mask.sum() == 0:
        res["raw"] = known.copy()
        res["times"]["Diffusion"] = 0.0
    else:
        t0 = time.time()
        print(f"[run] diffusion DDIM-{steps} on {DEVICE}, mask {100*mask.mean():.1f}% ...", flush=True)
        res["raw"] = REC.sample(known, mask, steps=steps, seed=seed)
        print(f"[run] diffusion done in {time.time()-t0:.2f}s", flush=True)
        res["times"]["Diffusion"] = time.time() - t0
    t0 = time.time()
    res["harm"] = HARMONIZERS[harm](res["raw"], known, mask) if mask.any() else known.copy()
    res["times"]["Harmoniser"] = time.time() - t0
    res["telea"] = telea_baseline(known, mask)
    res["composite"] = composite(res["raw"], known, mask)
    return res

def mask_from_upload_any(img, size):
    # U-Net convention: any positive pixel is foreground, nearest resize
    a = np.asarray(img)
    if a.ndim == 3:
        a = a[..., :3].max(-1)
    a = cv2.resize(a.astype(np.uint8) if a.dtype != np.uint8 else a, (size, size), interpolation=cv2.INTER_NEAREST)
    return a > 0


# 7. Gradio callbacks
def show_errors(fn):
    # print the full traceback in the Colab cell AND show the message in the browser
    @wraps(fn)
    def inner(*a, **k):
        try:
            return fn(*a, **k)
        except gr.Error:
            raise
        except Exception as e:
            traceback.print_exc(); sys.stderr.flush()
            raise gr.Error(f"{type(e).__name__}: {e}")
    return inner


@show_errors
def ui_single(damaged, clean, gt_mask, mask_source, threshold, dilate_px, steps, seed, harm):
    if damaged is None:
        raise gr.Error("Upload a damaged label image first.")
    damaged, clean = to_rgb_u8(damaged), to_rgb_u8(clean)
    if mask_source.startswith("Ground") and gt_mask is None:
        raise gr.Error("'Ground-truth mask' selected but no mask uploaded.")
    r = run_pipeline(damaged, clean, gt_mask, mask_source, threshold, dilate_px, int(steps), int(seed), harm)
    known, mask = r["known"], r["mask"]

    # metrics table
    rows = []
    eval_mask = r["gt_mask"] if r["gt_mask"] is not None else mask   # score inside the GT mask when we have it
    if r["clean"] is not None:
        for name, img in [("Damaged input (no fix)", known), ("OpenCV Telea", r["telea"]),
                          ("Model raw composite", r["composite"]), (f"Model + {harm}", r["harm"])]:
            m = image_metrics(img, r["clean"], eval_mask)
            rows.append({"Method": name, **{k: round(float(m[k]), 4) for k in METRIC_ORDER if k in m}})
    if rows:
        rows[0]["Method"] += "  [scored in " + ("GT mask]" if r["gt_mask"] is not None else "used mask]")
    rec_df = pd.DataFrame(rows) if rows else pd.DataFrame({"Info": ["Upload the clean ground-truth image to get PSNR / SSIM / ΔE metrics."]})
    if "seg" in r:
        seg_df = pd.DataFrame([{"Mask": f"U-Net @ thr {threshold:.2f}", **{k: round(float(v), 4) for k, v in r["seg"].items()},
                                "Pred. damaged %": round(100 * r["pred_mask"].mean(), 2),
                                "GT damaged %": round(100 * r["gt_mask"].mean(), 2)}])
    else:
        seg_df = pd.DataFrame([{"Mask": f"U-Net @ thr {threshold:.2f}", "Pred. damaged %": round(100 * r["pred_mask"].mean(), 2),
                                "Info": "Upload the GT mask to get Dice / IoU"}])

    panels = [("Damaged", known), ("OpenCV Telea", r["telea"]), ("Model raw composite", r["composite"]),
              (f"Model + {harm}", r["harm"])]
    if r["clean"] is not None:
        panels.append(("Ground truth", r["clean"]))
    zoom = zoom_figure(panels, mask)

    t = r["times"]
    info = (f"**Mask used:** {'ground truth' if mask_source.startswith('Ground') else 'U-Net prediction'}"
            f" ({100*mask.mean():.2f}% of pixels{', dilated ' + str(int(dilate_px)) + ' px' if dilate_px else ''})  ·  "
            f"**Timing:** U-Net {t['U-Net']*1000:.0f} ms · diffusion DDIM-{int(steps)} {t['Diffusion']:.2f} s · "
            f"harmoniser {t['Harmoniser']*1000:.0f} ms  ·  device `{DEVICE}`")
    if mask.sum() == 0:
        info += "\n\n⚠️ The mask is empty, so nothing was reconstructed. Try a lower threshold."

    out_path = Path(tempfile.gettempdir()) / f"reconstructed_{int(time.time())}.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(u8(r["harm"]), cv2.COLOR_RGB2BGR))
    print(f"[run] finished: mask {100*mask.mean():.2f}% | diffusion {t['Diffusion']:.2f}s", flush=True)
    return (heatmap(r["prob"]), u8(mask.astype(np.float32)), overlay(known, mask, r["gt_mask"]),
            u8(r["harm"]), u8(r["raw"]), zoom, seg_df, rec_df, info, str(out_path))


@show_errors
def ui_load_sample(data_root, split_csv, split, index):
    if not data_root or not Path(data_root).is_dir():
        raise gr.Error(f"Dataset folder not found: {data_root}")
    rows, msg = load_samples(data_root, split_csv, split)
    if not rows:
        raise gr.Error("No samples found.")
    idx = random.randrange(len(rows)) if index is None or int(index) < 0 else int(index) % len(rows)
    s, root = rows[idx], Path(data_root)
    damaged = read_rgb_u8(root / "damaged" / s["damaged"])
    clean = read_rgb_u8(root / "clean" / s["clean"]) if s.get("clean") else None
    mask = read_mask_u8(root / "masks" / s["mask"])
    return damaged, clean, mask, f"{msg} · loaded **#{idx}**: `{s['damaged']}`", idx


@show_errors
def ui_batch(data_root, split_csv, split, n_images, steps, harm, threshold, dilate_px, use_gt_too,
             progress=gr.Progress()):
    if not data_root or not Path(data_root).is_dir():
        raise gr.Error(f"Dataset folder not found: {data_root}")
    rows, msg = load_samples(data_root, split_csv, split)
    rows = [r for r in rows if r.get("clean")]
    if not rows:
        raise gr.Error("No samples with a clean image found.")
    n = min(int(n_images), len(rows))
    idx = np.linspace(0, len(rows) - 1, n).round().astype(int).tolist()   # evenly spaced, like the notebook
    root = Path(data_root)
    per_image, t0 = [], time.time()
    for j, i in enumerate(progress.tqdm(idx, desc="Evaluating")):
        s = rows[i]
        damaged = read_rgb_u8(root / "damaged" / s["damaged"])
        clean = read_rgb_u8(root / "clean" / s["clean"])
        gtm = read_mask_u8(root / "masks" / s["mask"])
        r = run_pipeline(damaged, clean, gtm, "U-Net prediction", threshold, dilate_px, int(steps), 0, harm)
        base = {"file": s["damaged"], **{f"det_{k}": v for k, v in r["seg"].items()}}
        cl, known, gt = r["clean"], r["known"], r["gt_mask"]
        # baselines and model are scored inside the GROUND-TRUTH mask, so all methods are compared on the same pixels
        for name, img in [("damaged_input", known), ("opencv_telea(GT mask)", telea_baseline(known, gt)),
                          (f"model+{harm} (U-Net mask)", r["harm"])]:
            per_image.append({**base, "method": name, **image_metrics(img, cl, gt)})
        if use_gt_too:
            raw = REC.sample(known, dilate(gt, int(dilate_px)), steps=int(steps), seed=0) if gt.any() else known
            img = HARMONIZERS[harm](raw, known, gt) if gt.any() else known
            per_image.append({**base, "method": f"model+{harm} (GT mask)", **image_metrics(img, cl, gt)})
    df = pd.DataFrame(per_image)
    metric_cols = [c for c in METRIC_ORDER if c in df.columns]
    summary = df.groupby("method", sort=False)[metric_cols].mean().round(4).reset_index()
    det = df.drop_duplicates("file")[["det_Dice", "det_IoU", "det_Precision", "det_Recall"]].mean().round(4)
    det_df = pd.DataFrame([{"Images": n, "Dice": det["det_Dice"], "IoU": det["det_IoU"],
                            "Precision": det["det_Precision"], "Recall": det["det_Recall"]}])
    out_csv = Path(tempfile.gettempdir()) / f"ie643_batch_{split}_{n}_{int(time.time())}.csv"
    df.to_csv(out_csv, index=False)
    info = (f"{msg}. Evaluated {n} evenly spaced images in {(time.time()-t0)/60:.1f} min (DDIM-{int(steps)}, seed 0). "
            f"Reconstruction metrics are computed inside the ground-truth damage mask.")
    return det_df, summary, info, str(out_csv)


# 8. UI
def build_ui():
    harm_choices = list(HARMONIZERS)
    with gr.Blocks(title="IE643 · Damaged Label Restoration") as demo:
        gr.Markdown(
            "# IE643 · Damaged drug-label detection and reconstruction\n"
            f"**Detector:** U-Net (Keras, {DET.size}×{DET.size}) &nbsp;→&nbsp; "
            f"**Reconstructor:** conditional diffusion, v-prediction (PyTorch, {REC.image_size}×{REC.image_size}, "
            f"EMA epoch {REC.epoch}) &nbsp;→&nbsp; **colour harmoniser** &nbsp;·&nbsp; running on `{DEVICE}`")

        with gr.Tab("Single image"):
            with gr.Row():
                with gr.Column(scale=1):
                    in_damaged = gr.Image(label="Damaged label (required)", type="numpy", height=260)
                    with gr.Row():
                        in_clean = gr.Image(label="Clean ground truth (optional → metrics)", type="numpy", height=160)
                        in_gtmask = gr.Image(label="GT mask (optional → Dice/IoU)", type="numpy",
                                             image_mode="L", height=160)
                    with gr.Accordion("Load a sample from the dataset", open=bool(ARGS.data_root)):
                        ds_root = gr.Textbox(label="Dataset root (contains clean/ damaged/ masks/)", value=ARGS.data_root or "")
                        ds_csv = gr.Textbox(label="Split CSV (dataset_split_v2.csv), optional", value=ARGS.split_csv or "")
                        with gr.Row():
                            ds_split = gr.Dropdown(["test", "val", "train", "all"], value="test", label="Split")
                            ds_index = gr.Number(value=-1, precision=0, label="Index (-1 = random)")
                        btn_load = gr.Button("Load sample")
                        ds_info = gr.Markdown()
                    with gr.Accordion("Settings", open=False):
                        mask_source = gr.Radio(["U-Net prediction", "Ground-truth mask"], value="U-Net prediction",
                                               label="Mask fed to the reconstructor")
                        threshold = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="U-Net threshold")
                        dilate_px = gr.Slider(0, 10, value=0, step=1, label="Dilate mask (px)")
                        steps = gr.Slider(5, 100, value=25, step=1, label="DDIM steps (notebook used 25)")
                        seed = gr.Number(value=0, precision=0, label="Sampling seed")
                        harm = gr.Dropdown(harm_choices, value="lab_stats", label="Harmoniser")
                    btn_run = gr.Button("Run detection + reconstruction", variant="primary")
                with gr.Column(scale=2):
                    info = gr.Markdown()
                    with gr.Row():
                        o_prob = gr.Image(label="U-Net damage probability", height=240)
                        o_mask = gr.Image(label="Mask used", height=240)
                        o_ovl = gr.Image(label="Overlay (red = mask, green = GT outline)", height=240)
                    with gr.Row():
                        o_harm = gr.Image(label="Reconstructed (harmonised)", height=300)
                        o_raw = gr.Image(label="Raw diffusion output (full frame)", height=300)
                    o_zoom = gr.Image(label="Zoom on damaged region")
                    o_seg = gr.Dataframe(label="Detection")
                    o_rec = gr.Dataframe(label="Reconstruction metrics (m* = inside mask; ΔE lower is better)")
                    o_file = gr.File(label="Download reconstructed PNG")
            btn_run.click(ui_single, [in_damaged, in_clean, in_gtmask, mask_source, threshold, dilate_px, steps, seed, harm],
                          [o_prob, o_mask, o_ovl, o_harm, o_raw, o_zoom, o_seg, o_rec, info, o_file])
            btn_load.click(ui_load_sample, [ds_root, ds_csv, ds_split, ds_index],
                           [in_damaged, in_clean, in_gtmask, ds_info, ds_index])

        with gr.Tab("Batch evaluation"):
            gr.Markdown("Runs the full pipeline over evenly spaced images of a split and averages the metrics. "
                        "On a T4 each image takes about 4 s at DDIM-25 (double that with the GT-mask run enabled); "
                        "on CPU it is much slower.")
            with gr.Row():
                b_root = gr.Textbox(label="Dataset root", value=ARGS.data_root or "")
                b_csv = gr.Textbox(label="Split CSV", value=ARGS.split_csv or "")
            with gr.Row():
                b_split = gr.Dropdown(["test", "val", "all"], value="test", label="Split")
                b_n = gr.Slider(1, 500, value=20, step=1, label="Number of images")
                b_steps = gr.Slider(5, 100, value=25, step=1, label="DDIM steps")
                b_harm = gr.Dropdown(harm_choices, value="lab_stats", label="Harmoniser")
            with gr.Row():
                b_thr = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="U-Net threshold")
                b_dil = gr.Slider(0, 10, value=0, step=1, label="Dilate mask (px)")
                b_gt = gr.Checkbox(value=True, label="Also reconstruct with the GT mask (isolates detector errors)")
            b_run = gr.Button("Run batch evaluation", variant="primary")
            b_info = gr.Markdown()
            b_det = gr.Dataframe(label="Detection (U-Net vs GT mask, at U-Net resolution)")
            b_sum = gr.Dataframe(label="Reconstruction (mean over images)")
            b_file = gr.File(label="Per-image CSV")
            b_run.click(ui_batch, [b_root, b_csv, b_split, b_n, b_steps, b_harm, b_thr, b_dil, b_gt],
                        [b_det, b_sum, b_info, b_file])
    return demo


def main():
    global DET, REC, ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", required=True, help="path to unet_group_safe_best.keras")
    ap.add_argument("--recon", required=True, help="path to best_ema.pth (or last_checkpoint.pth)")
    ap.add_argument("--data-root", default="", help="optional dataset folder with clean/ damaged/ masks/")
    ap.add_argument("--split-csv", default="", help="optional dataset_split_v2.csv")
    ap.add_argument("--share", action="store_true", help="public gradio.live link (needed in Colab/Kaggle)")
    ap.add_argument("--port", type=int, default=7860)
    ARGS = ap.parse_args()
    DET = Detector(ARGS.unet)
    REC = Reconstructor(ARGS.recon)
    build_ui().queue().launch(share=ARGS.share, server_name="0.0.0.0", server_port=ARGS.port)


if __name__ == "__main__":
    main()
