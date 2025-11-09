# 0) ローカル実行用パス設定（プロジェクト構成に合わせる）
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
work_dir = root_dir
out_dir = os.path.join(root_dir, 'data', 'output')
os.makedirs(out_dir, exist_ok=True)

# 1) Imports
import math
import random
import time
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# dynamo無効化
try:
    import torch._dynamo
    torch._dynamo.disable()
except Exception:
    pass

# 2) 乱数・デバイス
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)
if device.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# 3) nn.* 利用の簡易チェック
def enforce_nn_restriction(model: nn.Module):
    allowed = {'Module', 'Sequential'}
    for m in model.modules():
        mod = m.__class__.__module__
        name = m.__class__.__name__
        if not mod.startswith('torch.nn'):
            continue
        if name not in allowed:
            raise RuntimeError(f"不許可の nn.* を検出: {mod}.{name}")

# 4) データ読み込み（data/input 配下）
data_dir = os.path.join(work_dir, 'data', 'input')
x_train_u8 = np.load(os.path.join(data_dir, 'x_train.npy'))
x_test_u8  = np.load(os.path.join(data_dir, 'x_test.npy'))
# t_train or y_train どちらでも対応
y_path1 = os.path.join(data_dir, 't_train.npy')
y_path2 = os.path.join(data_dir, 'y_train.npy')
t_train = np.load(y_path1) if os.path.exists(y_path1) else np.load(y_path2)

# 形状・型の正規化（28x28, float32 in [0,1]）
def to_img01(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[1:] == (28, 28):
        return arr.astype(np.float32) / 255.0
    elif arr.ndim == 2 and arr.shape[1] == 28 * 28:
        return arr.reshape(-1, 28, 28).astype(np.float32) / 255.0
    else:
        raise ValueError(f"想定外の形状: {arr.shape}")

x_train_01 = to_img01(x_train_u8)
x_test_01  = to_img01(x_test_u8)
t_train    = t_train.astype('int64').reshape(-1)
assert t_train.min() >= 0 and t_train.max() <= 9, "ラベルは0..9の整数である必要があります"

# 5) Dataset / Aug
class TrainImgDataset(Dataset):
    def __init__(self, x28, y, augment=True):
        self.x28 = x28
        self.y = y
        self.augment = augment
        self.aug_scale = 1.0  # 0..1：Aug強度と確率のスケール

    def __len__(self):
        return len(self.y)

    def augment_img(self, img28: np.ndarray) -> np.ndarray:
        s = float(self.aug_scale)  # 0..1
        if s <= 0.0:
            return img28

        im = Image.fromarray((img28 * 255.0).astype(np.uint8))
        rot   = np.random.uniform(-8 * s, 8 * s)
        trans = (np.random.uniform(-1.5 * s, 1.5 * s), np.random.uniform(-1.5 * s, 1.5 * s))
        scale = np.random.uniform(1.0 - 0.06 * s, 1.0 + 0.06 * s)
        shear = np.random.uniform(-6 * s, 6 * s)
        cx, cy = 14, 14
        th = math.radians(rot)
        sh = math.radians(shear)
        a = scale * math.cos(th)
        b = scale * (-math.sin(th + sh))
        c = scale * math.sin(th)
        d = scale * math.cos(th + sh)
        tx, ty = trans
        M = (a, b, -a * cx - b * cy + cx + tx, c, d, -c * cx - d * cy + cy + ty)
        im = im.transform((28, 28), Image.AFFINE, M, resample=Image.BILINEAR, fillcolor=0)
        arr = np.asarray(im).astype(np.float32) / 255.0

        # 水平反転：25% * s
        if np.random.rand() < 0.25 * s:
            arr = np.fliplr(arr).copy()

        # cutout：0.5 * s の確率で適用
        if np.random.rand() < 0.5 * s:
            box = np.random.randint(5, 8)
            y = np.random.randint(0, 28)
            x = np.random.randint(0, 28)
            y0 = max(0, y - box // 2)
            y1 = min(28, y0 + box)
            x0 = max(0, x - box // 2)
            x1 = min(28, x0 + box)
            arr[y0:y1, x0:x1] = 0.0
        return arr

    def __getitem__(self, i):
        img = self.x28[i]
        if self.augment:
            img = self.augment_img(img)
        return torch.from_numpy(img).unsqueeze(0), torch.tensor(self.y[i], dtype=torch.long)

class ValImgDataset(Dataset):
    def __init__(self, x28, y):
        self.x28 = x28
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.x28[i]).unsqueeze(0), torch.tensor(self.y[i], dtype=torch.long)

class TestImgDataset(Dataset):
    def __init__(self, x28):
        self.x28 = x28

    def __len__(self):
        return self.x28.shape[0]

    def __getitem__(self, i):
        return torch.from_numpy(self.x28[i]).unsqueeze(0)

# split（D: 末尾10%をval）
VAL_FRACTION = 0.10
N = len(x_train_01)
idx = np.arange(N)
rng = np.random.default_rng(seed)
rng.shuffle(idx)
val_size = int(N * VAL_FRACTION)
val_idx = idx[:val_size]
trn_idx = idx[val_size:]
train_ds = TrainImgDataset(x_train_01[trn_idx], t_train[trn_idx], augment=True)
val_ds   = ValImgDataset  (x_train_01[val_idx],   t_train[val_idx])
test_ds  = TestImgDataset (x_test_01)

# DataLoader（D: eval=4096固定）
BATCH_SIZE_TRAIN = 2048 if device.type == 'cuda' else 256
BATCH_SIZE_EVAL  = 4096 if device.type == 'cuda' else 512
num_workers = 2
pin = (device.type == 'cuda')
train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True,
    drop_last=True, pin_memory=pin, num_workers=num_workers,
    persistent_workers=(num_workers > 0), prefetch_factor=2
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE_EVAL, shuffle=False,
    drop_last=False, pin_memory=pin, num_workers=num_workers,
    persistent_workers=(num_workers > 0), prefetch_factor=2
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE_EVAL, shuffle=False,
    drop_last=False, pin_memory=pin, num_workers=num_workers,
    persistent_workers=(num_workers > 0), prefetch_factor=2
)

# 6) HOG 3スケール + 生画素（HOGはFP32のまま）
class TorchHOGMulti(nn.Module):
    """
    出力次元（固定）: 784 + 1152 + 2304 + 4056 = 8296
    前計算: 各cellサイズ用の (H,W)->cellIndex マップをbuffer化
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.feat_dim = 784 + 1152 + 2304 + 4056
        self.register_buffer('feat_mean', torch.zeros(self.feat_dim))
        self.register_buffer('feat_std',  torch.ones(self.feat_dim))
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer('Kx', kx.view(1, 1, 1, 1, 3, 3))
        self.register_buffer('Ky', ky.view(1, 1, 1, 1, 3, 3))
        # 事前計算: 28x28前提のセルマップ（cell=4,3,2）
        self._cell_maps = {}
        for cell in (4, 3, 2):
            H = W = 28
            Hc, Wc = H // cell, W // cell
            yy = torch.arange(H).view(H, 1).expand(H, W)
            xx = torch.arange(W).view(1, W).expand(H, W)
            cy = torch.div(yy, cell, rounding_mode='floor').clamp(0, Hc - 1)
            cx = torch.div(xx, cell, rounding_mode='floor').clamp(0, Wc - 1)
            cell_id = (cy * Wc + cx).to(torch.long)  # (H,W) -> [0..Hc*Wc-1]
            self.register_buffer(f'cell_id_{cell}', cell_id)
            self._cell_maps[cell] = (Hc, Wc, getattr(self, f'cell_id_{cell}'))

    def _unfold3x3(self, x):
        B, C, H, W = x.shape
        pad = torch.zeros((B, C, H + 2, W + 2), device=x.device, dtype=x.dtype)
        pad[:, :, 1:-1, 1:-1] = x
        sN, sC, sH, sW = pad.stride()
        return pad.as_strided(size=(B, C, H, W, 3, 3), stride=(sN, sC, sH, sW, sH, sW))

    def _hog_generic(self, gx, gy, bins, cell):
        mag = torch.sqrt(gx * gx + gy * gy + self.eps)
        ang = torch.atan2(gy, gx) * (180.0 / math.pi)
        ang = (ang + 180.0) % 180.0
        B, _, H, W = mag.shape
        Hc, Wc, cell_id = self._cell_maps[cell]
        bin_w = 180.0 / bins
        b0 = torch.div(ang, bin_w, rounding_mode='floor').clamp(0, bins - 1).to(torch.long)
        frac = (ang - b0 * bin_w) / bin_w
        b1 = (b0 + 1) % bins
        base = (torch.arange(B, device=mag.device).view(B, 1, 1, 1) * (Hc * Wc)).expand(B, 1, H, W)
        dest = (base + cell_id.to(mag.device)).reshape(-1)
        m = mag.reshape(-1)
        f = frac.reshape(-1)
        b0f = b0.reshape(-1)
        b1f = b1.reshape(-1)
        hist_1d = torch.zeros(B * Hc * Wc * bins, device=mag.device, dtype=mag.dtype)
        idx0 = dest * bins + b0f
        hist_1d.index_add_(0, idx0, m * (1.0 - f))
        idx1 = dest * bins + b1f
        hist_1d.index_add_(0, idx1, m * f)
        hist = hist_1d.view(B, Hc, Wc, bins)
        b1 = hist[:, :-1, :-1, :]
        b2 = hist[:, :-1, 1:, :]
        b3 = hist[:, 1:, :-1, :]
        b4 = hist[:, 1:, 1:, :]
        blocks = torch.stack([b1, b2, b3, b4], dim=-2).reshape(B, Hc - 1, Wc - 1, 4 * bins)
        norm = torch.sqrt((blocks * blocks).sum(dim=-1, keepdim=True) + 1e-6)
        blocks = torch.clamp(blocks / norm, max=0.2)
        norm2 = torch.sqrt((blocks * blocks).sum(dim=-1, keepdim=True) + 1e-6)
        blocks = blocks / norm2
        return blocks.reshape(B, -1)

    def forward(self, x):
        x = x.float()
        win = self._unfold3x3(x)
        gx = (win * self.Kx).sum(dim=(-1, -2)) * (1.0 / 4.0)
        gy = (win * self.Ky).sum(dim=(-1, -2)) * (1.0 / 4.0)
        hog8_c4 = self._hog_generic(gx, gy, bins=8, cell=4)   # 1152
        hog9_c3 = self._hog_generic(gx, gy, bins=9, cell=3)   # 2304
        hog6_c2 = self._hog_generic(gx, gy, bins=6, cell=2)   # 4056
        flat = x.reshape(x.size(0), -1)                       # 784
        feat = torch.cat([flat, hog8_c4, hog9_c3, hog6_c2], dim=1)  # →8296
        feat = (feat - self.feat_mean) / self.feat_std
        return feat

    @torch.no_grad()
    def estimate_and_set_norm(self, loader, on_device='cpu'):
        tmp_dev = torch.device(on_device)
        self.to(tmp_dev)
        self.eval()
        s = torch.zeros(self.feat_dim, device=tmp_dev, dtype=torch.float64)
        ss = torch.zeros(self.feat_dim, device=tmp_dev, dtype=torch.float64)
        cnt = 0
        for xb, _ in loader:
            xb = xb.to(tmp_dev, non_blocking=False)
            feat = self.forward(xb).to(torch.float64)
            s += feat.sum(dim=0)
            ss += (feat * feat).sum(dim=0)
            cnt += feat.size(0)
        mean = s / max(1, cnt)
        var = ss / max(1, cnt) - mean * mean
        std = torch.sqrt(torch.clamp(var, min=1e-6))
        self.feat_mean = mean.to(torch.float32)
        self.feat_std = std.to(torch.float32)

# 7) 自作 MLP / 損失など（Dropout増加スケジュール）
class LayerNorm1D(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        x32 = x.float()
        m = x32.mean(dim=1, keepdim=True)
        v = x32.var(dim=1, keepdim=True, unbiased=False)
        y = (x32 - m) / torch.sqrt(v + self.eps)
        y = (y * self.gamma.float() + self.beta.float()).to(x.dtype)
        return y

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if (not self.training) or self.p <= 0.0:
            return x
        mask = (torch.rand_like(x) > self.p).float()
        return (x * mask) / (1.0 - self.p)

def gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * (x ** 3))))

def log_softmax(z):
    return z - torch.logsumexp(z, dim=1, keepdim=True)

# gather版 label smoothing
def ce_label_smoothing(logits, targets, eps=0.06, C=10):
    lp = log_softmax(logits)
    nll_true = -lp.gather(1, targets.unsqueeze(1)).squeeze(1).mean()
    ce_uni = -lp.mean(dim=1).mean()
    return (1.0 - eps) * nll_true + eps * ce_uni

class Dense(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        w = torch.randn(in_dim, out_dim) * math.sqrt(2.0 / in_dim)
        b = torch.zeros(out_dim)
        self.W = nn.Parameter(w)
        self.b = nn.Parameter(b)

    def forward(self, x):
        return x @ self.W + self.b

class Classifier(nn.Module):
    def __init__(self, in_dim, h1=3072, h2=1536, h3=768, out_dim=10,
                 p_init=(0.10, 0.12, 0.15),
                 p_final=(0.32, 0.35, 0.38)):
        super().__init__()
        self.fc1 = Dense(in_dim, h1)
        self.ln1 = LayerNorm1D(h1)
        self.do1 = Dropout(p_init[0])
        self.fc2 = Dense(h1, h2)
        self.ln2 = LayerNorm1D(h2)
        self.do2 = Dropout(p_init[1])
        self.fc3 = Dense(h2, h3)
        self.ln3 = LayerNorm1D(h3)
        self.do3 = Dropout(p_init[2])
        self.fc4 = Dense(h3, out_dim)
        self.p_init = p_init
        self.p_final = p_final

    def set_dropout(self, epoch, max_epoch):
        r = min(1.0, max(0.0, epoch / max(1, max_epoch - 1)))
        p_now = tuple(pi + (pf - pi) * r for pi, pf in zip(self.p_init, self.p_final))
        self.do1.p, self.do2.p, self.do3.p = p_now

    def forward(self, feat):
        x = gelu(self.ln1(self.fc1(feat)))
        x = self.do1(x)
        x = gelu(self.ln2(self.fc2(x)))
        x = self.do2(x)
        x = gelu(self.ln3(self.fc3(x)))
        x = self.do3(x)
        return self.fc4(x)

class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hog = TorchHOGMulti()
        self.mlp = Classifier(in_dim=self.hog.feat_dim)  # 8296に自動合わせ

    def set_dropout(self, e, E):
        self.mlp.set_dropout(e, E)

    def forward(self, x):
        feat = self.hog(x)  # HOGはFP32
        return self.mlp(feat)

model = FullModel()
enforce_nn_restriction(model)

# 8) 特徴の平均・分散推定（CUDAあればGPU）
with torch.no_grad():
    tmp_loader = DataLoader(
        ValImgDataset(x_train_01, t_train),
        batch_size=2048, shuffle=False, pin_memory=False, num_workers=0
    )
    t0 = time.time()
    model.hog.estimate_and_set_norm(tmp_loader, on_device=('cuda' if device.type == 'cuda' else 'cpu'))
    print(f"feature mean/std estimated on {'cuda' if device.type=='cuda' else 'cpu'} in {time.time()-t0:.1f}s; feat_dim={model.hog.feat_dim}")

# モデルをデバイスへ
model = model.to(device)

# 9) Optim/スケジュール/EMA/SWA
base_lr = 2.5e-3
n_epochs = 90
warmup = 6
weight_decay = 6e-4
aug_stop = 32
mixup_stop = 28
mixup_alpha = 0.18
max_grad_norm = 5.0

# Augフェード設定（ep10→26を1→0）
AUG_FADE_START, AUG_FADE_END = 10, 26
def aug_ratio(epoch: int) -> float:
    if epoch < AUG_FADE_START:
        return 1.0
    if epoch >= AUG_FADE_END:
        return 0.0
    t = (epoch - AUG_FADE_START) / max(1, (AUG_FADE_END - AUG_FADE_START))
    return float(1.0 - t)

decay, no_decay = [], []
for n, p in model.named_parameters():
    (decay if n.endswith('W') else no_decay).append(p)
optimizer = optim.AdamW([
    {'params': decay, 'weight_decay': weight_decay, 'lr': base_lr},
    {'params': no_decay, 'weight_decay': 0.0, 'lr': base_lr},
])

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = [p.detach().clone() for p in model.parameters()]

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow, model.parameters()):
            s.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        for p, s in zip(model.parameters(), self.shadow):
            p.data.copy_(s)

ema = EMA(model, decay=0.999)

class SWA:
    def __init__(self, model):
        self.n = 0
        self.shadow = [p.detach().clone().zero_() for p in model.parameters()]

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        for s, p in zip(self.shadow, model.parameters()):
            s.mul_(self.n - 1).add_(p.data).div_(self.n)

    @torch.no_grad()
    def apply_to(self, model):
        for p, s in zip(model.parameters(), self.shadow):
            p.data.copy_(s)

swa = SWA(model)

def set_lr(opt, lr):
    for g in opt.param_groups:
        g['lr'] = lr

def lr_sched(e):
    if e < warmup:
        return base_lr * (e + 1) / warmup
    t = (e - warmup) / max(1, (n_epochs - warmup - 1))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))

def mixup_batch(x, y, alpha):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[perm]
    return x_mix, y, y[perm], lam

# Label Smoothing のスケジュール（0.12 → 0.04）
def ls_eps(epoch):
    return 0.12 - 0.08 * (epoch / max(1, n_epochs - 1))

@torch.no_grad()
def eval_acc(model, loader):
    model.eval()
    ok = 0
    tot = 0
    loss_list = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=('cuda' if device.type == 'cuda' else 'cpu'),
                                dtype=torch.float16, enabled=(device.type == 'cuda')):
            logits = model(xb)
            loss = ce_label_smoothing(logits, yb, eps=0.06, C=10)  # ログ用
        loss_list.append(loss.item())
        ok += (logits.argmax(1) == yb).sum().item()
        tot += yb.size(0)
    return ok / max(1, tot), float(np.mean(loss_list))

best_acc = 0.0
best_state = [p.detach().clone() for p in model.parameters()]
best_tag = "last"  # 'last' / 'ema' / 'swa'

# 10) Train（AMP + 勾配クリップ）
scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

for epoch in range(n_epochs):
    # EMAの減衰を段階的に強める
    ema.decay = 0.9992 if epoch < 60 else 0.9996

    # Augフェード適用
    r = aug_ratio(epoch)
    model.train()
    train_ds.augment = (r > 0.0)
    train_ds.aug_scale = r
    model.set_dropout(epoch, n_epochs)
    set_lr(optimizer, lr_sched(epoch))

    loss_tr = []
    tr_ok = 0
    tr_tot = 0

    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        use_mix = (epoch < mixup_stop)
        if use_mix:
            xb, ya, yb2, lam = mixup_batch(xb, yb, mixup_alpha)

        with torch.amp.autocast(device_type=('cuda' if device.type == 'cuda' else 'cpu'),
                                dtype=torch.float16, enabled=(device.type == 'cuda')):
            logits = model(xb)
            eps_now = ls_eps(epoch)
            if use_mix:
                loss = lam * ce_label_smoothing(logits, ya, eps=eps_now, C=10) + \
                       (1.0 - lam) * ce_label_smoothing(logits, yb2, eps=eps_now, C=10)
            else:
                loss = ce_label_smoothing(logits, yb, eps=eps_now, C=10)

        scaler.scale(loss).backward()
        # unscale → global-norm clip → step
        scaler.unscale_(optimizer)
        total_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_sq += p.grad.data.float().pow(2).sum().item()
        total_norm = math.sqrt(total_sq)
        if total_norm > max_grad_norm and total_norm > 0:
            scale = max_grad_norm / (total_norm + 1e-6)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(scale)

        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        if epoch >= aug_stop:  # 後半のみSWA更新
            swa.update(model)

        loss_tr.append(loss.item())
        with torch.no_grad():
            pred = logits.argmax(1)
            tr_ok += (pred == (ya if use_mix else yb)).sum().item()
            tr_tot += yb.size(0)

    # ---- Validation：last / EMA / SWA
    with torch.no_grad():
        backup = [p.detach().clone() for p in model.parameters()]

    # last
    for p, q in zip(model.parameters(), backup):
        p.data.copy_(q)
    va_acc_last, va_loss_last = eval_acc(model, val_loader)

    # ema
    ema.apply_to(model)
    va_acc_ema, va_loss_ema = eval_acc(model, val_loader)

    # swa（後半のみ）
    has_swa = (epoch >= aug_stop and swa.n > 0)
    if has_swa:
        for p, q in zip(model.parameters(), backup):
            p.data.copy_(q)
        swa.apply_to(model)
        va_acc_swa, va_loss_swa = eval_acc(model, val_loader)
    else:
        va_acc_swa, va_loss_swa = 0.0, 0.0

    # 元に戻す
    with torch.no_grad():
        for p, q in zip(model.parameters(), backup):
            p.data.copy_(q)

    pick_tag = "ema"
    pick_acc = va_acc_ema
    if va_acc_last > pick_acc:
        pick_acc, pick_tag = va_acc_last, "last"
    if has_swa and va_acc_swa > pick_acc:
        pick_acc, pick_tag = va_acc_swa, "swa"

    print(
        f"EPOCH {epoch:02d} | LR {optimizer.param_groups[0]['lr']:.5f} "
        f"| Train [Loss {np.mean(loss_tr):.4f}, Acc {tr_ok / max(1, tr_tot):.4f}] "
        f"| Valid last/ema/swa Acc [{va_acc_last:.4f}/{va_acc_ema:.4f}/{va_acc_swa:.4f}] "
        f"-> pick={pick_tag}({pick_acc:.4f}) | Aug={'ON' if train_ds.augment else 'OFF'} s={r:.2f}"
    )

    if pick_acc > best_acc:
        best_acc = pick_acc
        best_tag = pick_tag
        with torch.no_grad():
            if pick_tag == "ema":
                ema.apply_to(model)
            elif pick_tag == "swa":
                swa.apply_to(model)
            best_state = [p.detach().clone() for p in model.parameters()]
            for p, q in zip(model.parameters(), backup):
                p.data.copy_(q)

# 11) ベスト重みを最終適用
with torch.no_grad():
    for p, q in zip(model.parameters(), best_state):
        p.data.copy_(q)
print(f"[Finalize] Best={best_tag} | ValAcc={best_acc:.4f}")

# 12) Proto補正（任意ON, 学習データ統計を使用）
USE_PROTO_BLEND = True
k_proto = 0.10
t_proto = 6.0
k_gauss = 0.04
t_gauss = 2.0
m_thr = 0.22
temp = 0.6

if USE_PROTO_BLEND:
    print("[Proto] building class centers/variances on train set ...")
    feat_all = []
    lab_all = []
    with torch.no_grad():
        model.eval()
        dl = DataLoader(ValImgDataset(x_train_01, t_train), batch_size=2048, shuffle=False, num_workers=0)
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            f = model.hog(xb).cpu()
            feat_all.append(f)
            lab_all.append(yb)
    F = torch.cat(feat_all, 0)
    Y = torch.cat(lab_all, 0)
    C = 10
    center = torch.stack([F[Y == c].mean(0) for c in range(C)], 0)
    var = torch.stack([F[Y == c].var(0, unbiased=False) + 1e-6 for c in range(C)], 0)

    def blend_logits(base_logits, feat, k_proto=k_proto, t_proto=t_proto,
                     k_gauss=k_gauss, t_gauss=t_gauss, m_thr=m_thr, temp=temp):
        with torch.no_grad():
            conf = torch.softmax(base_logits, dim=1).max(1).values
        d2 = torch.cdist(feat, center, p=2.0) ** 2
        proto = -k_proto * (d2 / t_proto)
        x = feat.unsqueeze(1)
        m = center.unsqueeze(0)
        v = var.unsqueeze(0)
        gauss = -((x - m) ** 2 / (2.0 * v)).sum(dim=2)
        gauss = k_gauss * (gauss / t_gauss)
        bonus = proto + gauss
        gate = torch.clamp((m_thr - conf).unsqueeze(1), min=0.0) / (m_thr + 1e-6)
        return base_logits / (temp + 1e-6) + gate * bonus

# 13) TTA 定義（8種: id/±5°/±7°/±1平行移動/flip）
tta_kinds = 8

def _affine_np(arr, rot=0.0, tx=0.0, ty=0.0):
    im = Image.fromarray((arr * 255.0).astype(np.uint8))
    cx, cy = 14, 14
    th = math.radians(rot)
    a = math.cos(th)
    b = -math.sin(th)
    c = math.sin(th)
    d = math.cos(th)
    M = (a, b, -a * cx - b * cy + cx + tx, c, d, -c * cx - d * cy + cy + ty)
    im = im.transform((28, 28), Image.AFFINE, M, resample=Image.BILINEAR, fillcolor=0)
    return np.asarray(im).astype(np.float32) / 255.0

def tta_single_variants(img28):
    outs = [img28]  # id=0
    outs.append(_affine_np(img28, rot=5))
    outs.append(_affine_np(img28, rot=-5))
    outs.append(_affine_np(img28, rot=7))
    outs.append(_affine_np(img28, rot=-7))
    outs.append(_affine_np(img28, tx=1, ty=1))
    outs.append(_affine_np(img28, tx=-1, ty=-1))
    outs.append(np.fliplr(img28).copy())  # 追加: 水平反転 (idx=7)
    return outs

@torch.no_grad()
def run_tta_logits_batch(model, batch_imgs_np, keep_idx):
    """
    keep_idx: 使用するTTAインデックスの配列 (e.g., [0,1,3,...])
    返り値: (K_keep, N, 10) のロジット
    """
    N = batch_imgs_np.shape[0]
    K = len(keep_idx)
    stacked = np.empty((N * K, 1, 28, 28), dtype=np.float32)
    ptr = 0
    for i in range(N):
        vars_i = tta_single_variants(batch_imgs_np[i])
        for k, j in enumerate(keep_idx):
            stacked[ptr + k, 0] = vars_i[j]
        ptr += K
    xb = torch.from_numpy(stacked).to(device, non_blocking=True)
    with torch.amp.autocast(device_type=('cuda' if device.type == 'cuda' else 'cpu'),
                             dtype=torch.float16, enabled=(device.type == 'cuda')):
        logits = model(xb)  # (N*K, 10)
    logits = logits.reshape(N, K, -1).permute(1, 0, 2).contiguous()  # (K, N, 10)
    return logits.detach().cpu()

# 14) TTA重み学習 + 温度スケーリング + Identityアンカー
print("[TTA] collecting validation logits for p-based weight learning ...")
val_imgs_np = val_ds.x28.astype(np.float32)
val_labels = val_ds.y
C = 10
K_all = tta_kinds

# 平行移動のみ固定にする（必要なら False に）
FORCE_TTA_KEEP = True
FORCED_KEEP = [0, 5, 6]  # id, (+1,+1), (-1,-1)

tta_logits_all = []
with torch.no_grad():
    model.eval()
    step = 2048
    for i in range(0, len(val_imgs_np), step):
        imgs = val_imgs_np[i:i + step]
        log_all = run_tta_logits_batch(model, imgs, list(range(K_all)))  # (K_all, Nb, 10)
        tta_logits_all.append(log_all)
tta_stack_all = torch.cat(tta_logits_all, dim=1)  # (K_all, N, 10)
y_tensor = torch.from_numpy(val_labels).long()

with torch.no_grad():
    id_pred = tta_stack_all[0].argmax(1)
    id_acc = (id_pred == y_tensor).float().mean().item()
    acc_each = []
    for k in range(K_all):
        acc_k = (tta_stack_all[k].argmax(1) == y_tensor).float().mean().item()
        acc_each.append(acc_k)

prune_drop = 0.002
keep = [k for k, a in enumerate(acc_each) if a >= id_acc - prune_drop]
if len(keep) < 2:
    keep = list(range(K_all))
if FORCE_TTA_KEEP:
    keep = FORCED_KEEP

K_keep = len(keep)
id_pos = keep.index(0)
tta_stack = tta_stack_all[keep]  # (K_keep, N, 10)
print(f"[TTA] kept transforms: {keep} (K_keep={K_keep}) | id_pos={id_pos}")

# 温度スケーリング（identity）
id_logits = tta_stack[id_pos]
logT = torch.zeros(1, requires_grad=True)
opt_T = torch.optim.LBFGS([logT], lr=0.1, max_iter=50)

def _nll_T():
    T = torch.exp(logT).clamp(0.5, 5.0)
    lp = (id_logits / T) - torch.logsumexp(id_logits / T, dim=1, keepdim=True)
    nll = (-(lp.gather(1, y_tensor.unsqueeze(1)).squeeze(1))).mean()
    return nll

def _closure():
    opt_T.zero_grad()
    loss = _nll_T()
    loss.backward()
    return loss

opt_T.step(_closure)
T_cal = torch.exp(logT).detach().clamp(0.5, 5.0)
print(f"[TTA] learned temperature T_cal = {float(T_cal):.4f}")

# pベース期待重み（列softmax）
v = torch.zeros(K_keep, C, requires_grad=True)
opt_v = torch.optim.Adam([v], lr=0.05, weight_decay=1e-3)

def ce_from_logits(L, y):
    lp = L - torch.logsumexp(L, dim=1, keepdim=True)
    return (-(lp.gather(1, y.unsqueeze(1)).squeeze(1))).mean()

for it in range(200):
    W = torch.softmax(v, dim=0)
    p_val = torch.softmax(id_logits / T_cal, 1)
    w_val = p_val @ W.T
    Lw = (tta_stack * w_val.T.unsqueeze(2)).sum(0)
    loss = ce_from_logits(Lw, y_tensor)
    opt_v.zero_grad()
    loss.backward()
    opt_v.step()
    if (it + 1) % 50 == 0:
        with torch.no_grad():
            acc = (Lw.argmax(1) == y_tensor).float().mean().item()
            print(f"[TTA] it={it+1:03d} loss={loss.item():.4f} acc={acc:.4f}")

with torch.no_grad():
    W_final = torch.softmax(v, dim=0).detach()
    p_val = torch.softmax(id_logits / T_cal, 1)
    w_val = p_val @ W_final.T
    Lw = (tta_stack * w_val.T.unsqueeze(2)).sum(0)
    acc_final = (Lw.argmax(1) == y_tensor).float().mean().item()
    print(f"[TTA] final acc (p-based mix) = {acc_final:.4f}")

def apply_identity_anchor(w_batch, id_logits_batch, T_cal, id_col, max_boost=0.40):
    with torch.no_grad():
        conf, _ = torch.softmax(id_logits_batch / T_cal, 1).max(1)
        alpha = torch.clamp((conf - 0.60) / 0.25, 0.0, 1.0) * max_boost
        w_id = w_batch[:, id_col]
        w_batch[:, id_col] = w_id * (1.0 - alpha) + alpha
        w_batch = w_batch / (w_batch.sum(1, keepdim=True) + 1e-8)
        return w_batch

# 15) MC-Dropout自動ON/OFF（最適化版TTAに対応）
def enable_test_dropout(model, p_tuple):
    model.train()
    model.mlp.do1.p, model.mlp.do2.p, model.mlp.do3.p = p_tuple

def disable_test_dropout(model):
    model.eval()

@torch.no_grad()
def val_acc_no_mc():
    model.eval()
    ok = 0
    tot = 0
    step = 1024
    for i in range(0, len(val_imgs_np), step):
        imgs = val_imgs_np[i:i + step]
        stack = run_tta_logits_batch(model, imgs, keep)  # (K_keep, Nb, 10)
        id_log = stack[id_pos]
        p = torch.softmax(id_log / T_cal, 1)
        w = p @ W_final.T
        w = apply_identity_anchor(w, id_log, T_cal, id_pos)
        L = (stack * w.T.unsqueeze(2)).sum(0)
        pred = L.argmax(1).numpy()
        y = val_labels[i:i + step]
        ok += (pred == y).sum()
        tot += len(y)
    return ok / max(1, tot)

@torch.no_grad()
def val_acc_with_mc(scale, n_pass=4):
    pf = model.mlp.p_final
    p_tuple = tuple(min(0.5, max(0.0, scale * p)) for p in pf)
    model.eval()
    ok = 0
    tot = 0
    step = 512
    for i in range(0, len(val_imgs_np), step):
        imgs = val_imgs_np[i:i + step]
        base_stack = run_tta_logits_batch(model, imgs, keep)  # (K_keep, Nb, 10)
        id_log = base_stack[id_pos]
        p = torch.softmax(id_log / T_cal, 1)
        w = p @ W_final.T
        w = apply_identity_anchor(w, id_log, T_cal, id_pos)
        base_mix = (base_stack * w.T.unsqueeze(2)).sum(0)
        enable_test_dropout(model, p_tuple)
        mc_accum = None
        for _ in range(n_pass):
            cur_stack = run_tta_logits_batch(model, imgs, keep)
            cur_mix = (cur_stack * w.T.unsqueeze(2)).sum(0)
            mc_accum = cur_mix if mc_accum is None else (mc_accum + cur_mix)
        disable_test_dropout(model)
        mc_mix = mc_accum / float(n_pass)
        logits = 0.5 * base_mix + 0.5 * mc_mix
        pred = logits.argmax(1).numpy()
        y = val_labels[i:i + step]
        ok += (pred == y).sum()
        tot += len(y)
    return ok / max(1, tot)

val_imgs_np = val_ds.x28.astype(np.float32)
val_labels = val_ds.y
val_no_mc = val_acc_no_mc()
print(f"[MC] val_acc without MC = {val_no_mc:.4f}")
scales = [0.2, 0.3, 0.4, 0.5, 0.7]
acc_grid = []
for s in scales:
    a = val_acc_with_mc(s, n_pass=4)
    acc_grid.append(a)
    print(f"[MC] scale={s} -> val_acc={a:.4f}")

best_idx = int(np.argmax(acc_grid))
best_scale = scales[best_idx]
best_mc = acc_grid[best_idx]
if (best_mc - val_no_mc) < 5e-4:
    USE_MC = False
    print(f"[MC] auto-disabled (delta={best_mc - val_no_mc:+.4f} < 0.0005)")
else:
    USE_MC = True
    print(f"[MC] choose scale={best_scale} (gain={best_mc - val_no_mc:+.4f})")

# 16) 推論：pベース期待重み + Identityアンカー + （任意）MC平均 + （任意）プロト補正
model.eval()
t_pred = []
with torch.no_grad():
    if USE_PROTO_BLEND:
        feat_te = []
        for xb in test_loader:
            xb = xb.to(device, non_blocking=True)
            feat_te.append(model.hog(xb).cpu())
        feat_te = torch.cat(feat_te, 0)

    pf = model.mlp.p_final
    p_tuple = tuple(min(0.5, max(0.0, best_scale * p)) for p in pf)

    offset = 0
    for batch in test_loader:
        imgs = batch.squeeze(1).numpy()

        # base（Kまとめて1forward）
        base_stack = run_tta_logits_batch(model, imgs, keep)  # (K_keep, Nb, 10)
        id_log = base_stack[id_pos]
        p_batch = torch.softmax(id_log / T_cal, 1)
        w_batch = p_batch @ W_final.T
        w_batch = apply_identity_anchor(w_batch, id_log, T_cal, id_pos)
        logits_sum = (base_stack * w_batch.T.unsqueeze(2)).sum(0)  # (Nb, 10)

        # MC（任意）
        if USE_MC:
            enable_test_dropout(model, p_tuple)
            mc_accum = None
            for _ in range(4):
                cur_stack = run_tta_logits_batch(model, imgs, keep)
                cur_mix = (cur_stack * w_batch.T.unsqueeze(2)).sum(0)
                mc_accum = cur_mix if mc_accum is None else (mc_accum + cur_mix)
            disable_test_dropout(model)
            logits_sum = 0.5 * logits_sum + 0.5 * (mc_accum / 4.0)

        # プロト補正（任意）
        if USE_PROTO_BLEND:
            n_b = logits_sum.size(0)
            f_batch = feat_te[offset:offset + n_b]
            logits_sum = blend_logits(logits_sum, f_batch)

        t_pred.extend(logits_sum.argmax(1).tolist())
        offset += len(imgs)

# 17) 保存（data/output 配下）
sub = pd.Series(t_pred, name='label')
out_path = os.path.join(out_dir, 'submission.csv')
sub.to_csv(out_path, header=True, index_label='id')
print(f"Saved: {out_path} | Best Val({best_tag}): {best_acc:.4f}")