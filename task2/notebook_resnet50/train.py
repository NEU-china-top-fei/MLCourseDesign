"""Six-class facial expression classification; runs on Kaggle GPU."""
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import sklearn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.models import resnet50, ResNet50_Weights

SEED = 2026
EPOCHS = 24
BATCH = 256
SIZE = 112
CLASS_NAMES = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
OUT = Path('/kaggle/working')
OUT.mkdir(exist_ok=True)
assert torch.cuda.is_available(), 'This notebook requires an enabled Kaggle GPU.'
device = torch.device('cuda')
torch.set_num_threads(2)
torch.backends.cudnn.benchmark = True

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)
print('GPU:', torch.cuda.get_device_name(), flush=True)
print('Versions:', torch.__version__, torchvision.__version__, sklearn.__version__, flush=True)
train_dirs = [p for p in Path('/kaggle/input').rglob('train') if (p / 'Happy').is_dir()]
assert len(train_dirs) == 1, train_dirs
train_dir = train_dirs[0]
test_dir = train_dir.parent / 'test'
sample_candidates = [p for p in Path('/kaggle/input').rglob('submission.csv') if p.parent in train_dir.parents]
assert len(sample_candidates) == 1, sample_candidates
sample = pd.read_csv(sample_candidates[0])
assert list(sample.columns) == ['ID', 'Emotion']
from concurrent.futures import ThreadPoolExecutor
paths, labels = [], []
for label, cls in enumerate(CLASS_NAMES):
    for path in sorted((train_dir / cls).glob('*')):
        if path.suffix.lower() in {'.jpg', '.png', '.jpeg'}:
            paths.append(str(path))
            labels.append(label)
def read_gray(path):
    with Image.open(path) as im:
        a = np.asarray(im.convert('L'), dtype=np.uint8)
    assert a.shape == (48, 48)
    return a
with ThreadPoolExecutor(max_workers=16) as pool:
    pixels = list(pool.map(read_gray, paths))
groups = [a.tobytes() for a in pixels]
y = np.asarray(labels)
# Keep identical training images together to avoid optimistic validation.
train_idx, valid_idx = next(StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=SEED).split(y, y, groups))
assert not (set(np.asarray(groups)[train_idx]) & set(np.asarray(groups)[valid_idx]))
pd.DataFrame({'path': paths, 'label': y, 'validation': np.isin(np.arange(len(y)), valid_idx)}).to_csv(OUT / 'split.csv', index=False)
images = torch.from_numpy(np.stack(pixels)).unsqueeze(1).to(device)
targets = torch.as_tensor(y, dtype=torch.long, device=device)
with ThreadPoolExecutor(max_workers=16) as pool:
    test_pixels = list(pool.map(read_gray, [test_dir / name for name in sample.ID]))
test_images = torch.from_numpy(np.stack(test_pixels)).unsqueeze(1).to(device)
print('Data:', len(y), len(sample), 'validation:', len(valid_idx), 'class counts:', np.bincount(y), flush=True)
mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1)
std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1)

def prepare(x, augment=False, flip=False):
    x = x.float().div_(255)
    n = len(x)
    if augment:
        angle = (torch.rand(n, device=device) * 2 - 1) * (12 * math.pi / 180)
        scale = .88 + .24 * torch.rand(n, device=device)
        mirror = torch.where(torch.rand(n, device=device) < .5, -1., 1.)
        theta = torch.zeros(n, 2, 3, device=device)
        theta[:, 0, 0] = angle.cos() * scale * mirror
        theta[:, 0, 1] = -angle.sin() * scale
        theta[:, 1, 0] = angle.sin() * scale * mirror
        theta[:, 1, 1] = angle.cos() * scale
        theta[:, :, 2] = (torch.rand(n, 2, device=device) * 2 - 1) * .12
        grid = F.affine_grid(theta, (n, 1, SIZE, SIZE), align_corners=False)
        x = F.grid_sample(x, grid, padding_mode='reflection', align_corners=False)
        contrast = .85 + .3 * torch.rand(n, 1, 1, 1, device=device)
        brightness = (torch.rand(n, 1, 1, 1, device=device) * 2 - 1) * .08
        x = ((x - .5) * contrast + .5 + brightness).clamp_(0, 1)
    else:
        x = F.interpolate(x, size=(SIZE, SIZE), mode='bilinear', align_corners=False)
        if flip:
            x = x.flip(-1)
    if augment and random.random() < .2:
        # Small random occlusion; coordinates vary each minibatch.
        h, w = random.randint(12, 32), random.randint(12, 32)
        top, left = random.randrange(SIZE-h), random.randrange(SIZE-w)
        mask = torch.rand(n, 1, 1, 1, device=device) < .5
        region = x[:, :, top:top+h, left:left+w]
        region.copy_(torch.where(mask, .5, region))
    return ((x.expand(-1, 3, -1, -1) - mean) / std).contiguous(memory_format=torch.channels_last)

# 在 Kaggle 挂载前序 Notebook 的预测输出。
old_val_path = next(Path('/kaggle/input').rglob('validation_predictions.npz'))
old_test_path = next(Path('/kaggle/input').rglob('test_predictions.npz'))
old_val = np.load(old_val_path)
old_test = np.load(old_test_path)
assert np.array_equal(old_val['index'], valid_idx)
assert np.array_equal(old_val['labels'], y[valid_idx])
assert np.array_equal(old_test['ids'], sample.ID.to_numpy(dtype=str))
baseline_val_prob = old_val['probabilities']
baseline_test_prob = old_test['ensemble']

seed_everything(SEED + 20)
base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
base.fc = nn.Sequential(nn.Dropout(.4), nn.Linear(2048, 6))
initial_state = copy.deepcopy(base.state_dict())
del base

def make_model():
    model = resnet50(weights=None)
    model.fc = nn.Sequential(nn.Dropout(.4), nn.Linear(2048, 6))
    model.load_state_dict(initial_state)
    return model.to(device=device, memory_format=torch.channels_last)

@torch.inference_mode()
def predict(model, source, indices=None, tta=True):
    model.eval()
    if indices is None:
        indices = np.arange(len(source))
    result = []
    for start in range(0, len(indices), BATCH):
        raw = source[indices[start:start+BATCH]]
        with torch.autocast('cuda', dtype=torch.float16):
            p = model(prepare(raw)).float().softmax(1)
            if tta:
                p = (p + model(prepare(raw, flip=True)).float().softmax(1)) / 2
        result.append(p.cpu().numpy())
    return np.concatenate(result)

history = []

def optimizer_groups(model):
    groups = {}
    for name, param in model.named_parameters():
        if name.startswith('fc.'):
            lr = 6e-4
        elif name.startswith(('conv1.', 'bn1.', 'layer1.')):
            lr = 7e-5
        else:
            lr = 2e-4
        decay = .03 if param.ndim > 1 else 0.
        groups.setdefault((lr, decay), []).append(param)
    return [{'params': params, 'lr': lr, 'weight_decay': decay} for (lr, decay), params in groups.items()]

def train(indices, epochs, stage, validation=None, final_kind='ema'):
    model = make_model()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(optimizer_groups(model))
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss(label_smoothing=.10)
    base_lrs = [group['lr'] for group in optimizer.param_groups]
    steps_per_epoch = math.ceil(len(indices) / BATCH)
    total_steps = epochs * steps_per_epoch
    warmup_steps = 2 * steps_per_epoch
    best, best_epoch, best_kind = -1., epochs, final_kind
    steps = 0
    ema_params, live_params = list(ema.parameters()), list(model.parameters())
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        order = np.random.permutation(indices)
        total_loss = 0.
        for start in range(0, len(order), BATCH):
            idx = order[start:start+BATCH]
            if steps < warmup_steps:
                factor = .05 + .95 * (steps + 1) / warmup_steps
            else:
                phase = (steps - warmup_steps) / max(1, total_steps - warmup_steps)
                factor = .02 + .98 * .5 * (1 + math.cos(math.pi * phase))
            for group, lr in zip(optimizer.param_groups, base_lrs):
                group['lr'] = lr * factor
            x, target = prepare(images[idx], augment=True), targets[idx]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                if np.random.rand() < .35:
                    lam = np.random.beta(.3, .3)
                    perm = torch.randperm(len(x), device=device)
                    logits = model(lam * x + (1 - lam) * x[perm])
                    loss = lam * criterion(logits, target) + (1 - lam) * criterion(logits, target[perm])
                else:
                    loss = criterion(model(x), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            decay = min(.995, (1 + steps) / (10 + steps))
            with torch.no_grad():
                torch._foreach_mul_(ema_params, decay)
                torch._foreach_add_(ema_params, live_params, alpha=1-decay)
                for ema_buffer, live_buffer in zip(ema.buffers(), model.buffers()):
                    ema_buffer.copy_(live_buffer)
            total_loss += loss.item() * len(idx)
        row = {'stage': stage, 'epoch': epoch + 1, 'loss': total_loss / len(order)}
        if validation is not None:
            for kind, candidate in [('raw', model), ('ema', ema)]:
                prob = predict(candidate, images, validation, tta=False)
                score = float(accuracy_score(y[validation], prob.argmax(1)))
                row[f'{kind}_val_accuracy'] = score
                if score > best:
                    best, best_epoch, best_kind = score, epoch + 1, kind
                    torch.save(candidate.state_dict(), OUT / 'resnet50_validation_best.pt')
        row['seconds'] = round(time.time() - t0, 1)
        history.append(row)
        pd.DataFrame(history).to_csv(OUT / 'history.csv', index=False)
        print(json.dumps(row), flush=True)
    if validation is not None:
        model.load_state_dict(torch.load(OUT / 'resnet50_validation_best.pt', map_location=device, weights_only=True))
    elif final_kind == 'ema':
        model.load_state_dict(ema.state_dict())
    if validation is None:
        torch.save(model.state_dict(), OUT / 'resnet50_full.pt')
    return model, best_epoch, best_kind

model, best_epoch, best_kind = train(train_idx, EPOCHS, 'validation', valid_idx)
valid_prob = predict(model, images, valid_idx)
val_score = float(accuracy_score(y[valid_idx], valid_prob.argmax(1)))
print('Validation TTA:', val_score, 'best_epoch:', best_epoch, 'kind:', best_kind, flush=True)
blend_results = []
for old_weight in [0., .15, .3, .45]:
    p = old_weight * baseline_val_prob + (1-old_weight) * valid_prob
    blend_results.append({'old_weight': old_weight, 'accuracy': float(accuracy_score(y[valid_idx], p.argmax(1)))})
best_blend = max(blend_results, key=lambda row: (row['accuracy'], -row['old_weight']))
# Keep the new model alone unless mixing gives a meaningful validation improvement.
old_weight = best_blend['old_weight'] if best_blend['accuracy'] >= val_score + .001 else 0.
chosen_val = old_weight * baseline_val_prob + (1-old_weight) * valid_prob
report = {'architecture': 'resnet50', 'image_size': SIZE, 'validation_accuracy_tta': val_score,
          'baseline_validation_accuracy': float(accuracy_score(y[valid_idx], baseline_val_prob.argmax(1))),
          'blend_validation_accuracy': float(accuracy_score(y[valid_idx], chosen_val.argmax(1))),
          'baseline_weight': old_weight, 'blend_search': blend_results,
          'best_epoch': best_epoch, 'best_kind': best_kind, 'class_names': CLASS_NAMES,
          'classification_report': classification_report(y[valid_idx], chosen_val.argmax(1), target_names=CLASS_NAMES, output_dict=True),
          'confusion_matrix': confusion_matrix(y[valid_idx], chosen_val.argmax(1)).tolist(),
          'validation_note': 'Same grouped holdout as baseline. Blend weights selected on this holdout; not an independent final performance estimate.',
          'gpu': torch.cuda.get_device_name(), 'seed': SEED + 20}
(OUT / 'metrics.json').write_text(json.dumps(report, indent=2))
np.savez_compressed(OUT / 'validation_predictions.npz', index=valid_idx, labels=y[valid_idx], probabilities=valid_prob)
val_test_prob = predict(model, test_images)
sample.assign(Emotion=val_test_prob.argmax(1)).to_csv(OUT / 'submission_validation.csv', index=False)
print('Blend comparison:', json.dumps(blend_results), 'selected old weight:', old_weight, flush=True)
del model
torch.cuda.empty_cache()

seed_everything(SEED + 21)
full_model, _, _ = train(np.arange(len(y)), best_epoch, 'full', final_kind=best_kind)
full_prob = predict(full_model, test_images)
new_prob = .8 * full_prob + .2 * val_test_prob
ensemble = old_weight * baseline_test_prob + (1-old_weight) * new_prob
submission = sample.assign(Emotion=ensemble.argmax(1).astype(int))
assert submission.ID.equals(sample.ID) and submission.ID.is_unique
assert submission.Emotion.between(0, 5).all() and not submission.isna().any().any()
submission.to_csv(OUT / 'submission.csv', index=False)
sample.assign(Emotion=new_prob.argmax(1)).to_csv(OUT / 'submission_resnet50.csv', index=False)
np.savez_compressed(OUT / 'test_predictions.npz', ids=sample.ID.to_numpy(dtype=str), validation_model=val_test_prob, full_model=full_prob, ensemble=ensemble)
print('COMPLETE:', submission.shape, submission.Emotion.value_counts().sort_index().to_dict(), flush=True)
