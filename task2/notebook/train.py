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
from torchvision.models import resnet18, ResNet18_Weights

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
sample_candidates = list(Path('/kaggle/input').rglob('submission.csv'))
assert len(sample_candidates) == 1, sample_candidates
sample = pd.read_csv(sample_candidates[0])
assert list(sample.columns) == ['ID', 'Emotion']
train_dirs = [p for p in Path('/kaggle/input').rglob('train') if (p / 'Happy').is_dir()]
assert len(train_dirs) == 1, train_dirs
train_dir = train_dirs[0]
test_dir = train_dir.parent / 'test'
paths, labels, pixels, groups = [], [], [], []
for label, cls in enumerate(CLASS_NAMES):
    for path in sorted((train_dir / cls).glob('*')):
        if path.suffix.lower() not in {'.jpg', '.png', '.jpeg'}:
            continue
        a = np.asarray(Image.open(path).convert('L'), dtype=np.uint8)
        assert a.shape == (48, 48)
        paths.append(str(path))
        labels.append(label)
        pixels.append(a)
        groups.append(a.tobytes())
y = np.asarray(labels)
# Keep identical training images together to avoid optimistic validation.
train_idx, valid_idx = next(StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=SEED).split(y, y, groups))
assert not (set(np.asarray(groups)[train_idx]) & set(np.asarray(groups)[valid_idx]))
pd.DataFrame({'path': paths, 'label': y, 'validation': np.isin(np.arange(len(y)), valid_idx)}).to_csv(OUT / 'split.csv', index=False)
images = torch.from_numpy(np.stack(pixels)).unsqueeze(1).to(device)
targets = torch.as_tensor(y, dtype=torch.long, device=device)
test_images = torch.from_numpy(np.stack([np.asarray(Image.open(test_dir / name).convert('L'), dtype=np.uint8) for name in sample.ID])).unsqueeze(1).to(device)
print('Data:', len(y), len(sample), 'validation:', len(valid_idx), 'class counts:', np.bincount(y), flush=True)
mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1)
std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1)

def prepare(x, augment=False, flip=False):
    x = x.float().div_(255)
    n = len(x)
    if augment:
        angle = (torch.rand(n, device=device) * 2 - 1) * (15 * math.pi / 180)
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
    return ((x.expand(-1, 3, -1, -1) - mean) / std).contiguous(memory_format=torch.channels_last)


base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
base.fc = nn.Sequential(nn.Dropout(.3), nn.Linear(base.fc.in_features, 6))
initial_state = copy.deepcopy(base.state_dict())
del base

def make_model():
    model = resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(.3), nn.Linear(model.fc.in_features, 6))
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

def train(indices, epochs, stage, validation=None):
    model = make_model()
    head_ids = {id(p) for p in model.fc.parameters()}
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.parameters() if id(p) not in head_ids], 'lr': 3e-4},
        {'params': model.fc.parameters(), 'lr': 1e-3},
    ], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss(label_smoothing=.05)
    best, best_epoch = -1., epochs
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        order = np.random.permutation(indices)
        total_loss = 0.
        for start in range(0, len(order), BATCH):
            idx = order[start:start+BATCH]
            x, target = prepare(images[idx], augment=True), targets[idx]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                if np.random.rand() < .25:
                    lam = np.random.beta(.2, .2)
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
            total_loss += loss.item() * len(idx)
        scheduler.step()
        row = {'stage': stage, 'epoch': epoch + 1, 'loss': total_loss / len(order)}
        if validation is not None:
            prob = predict(model, images, validation, tta=False)
            score = accuracy_score(y[validation], prob.argmax(1))
            row['val_accuracy'] = float(score)
            if score > best:
                best, best_epoch = score, epoch + 1
                torch.save(model.state_dict(), OUT / f'{stage}_best.pt')
        row['seconds'] = round(time.time() - t0, 1)
        history.append(row)
        pd.DataFrame(history).to_csv(OUT / 'history.csv', index=False)
        print(json.dumps(row), flush=True)
    if validation is not None:
        model.load_state_dict(torch.load(OUT / f'{stage}_best.pt', map_location=device, weights_only=True))
    else:
        torch.save(model.state_dict(), OUT / f'{stage}_final.pt')
    return model, best_epoch

model, best_epoch = train(train_idx, EPOCHS, 'validation', valid_idx)
valid_prob = predict(model, images, valid_idx)
val_score = accuracy_score(y[valid_idx], valid_prob.argmax(1))
print('Validation accuracy with horizontal-flip TTA:', val_score, 'best epoch:', best_epoch, flush=True)
np.savez_compressed(OUT / 'validation_predictions.npz', index=valid_idx, labels=y[valid_idx], probabilities=valid_prob)
report = {'validation_accuracy_tta': val_score, 'best_epoch': best_epoch,
          'class_names': CLASS_NAMES, 'seed': SEED,
          'classification_report': classification_report(y[valid_idx], valid_prob.argmax(1), target_names=CLASS_NAMES, output_dict=True),
          'confusion_matrix': confusion_matrix(y[valid_idx], valid_prob.argmax(1)).tolist(),
          'train_rows': len(y), 'test_rows': len(sample), 'gpu': torch.cuda.get_device_name()}
(OUT / 'metrics.json').write_text(json.dumps(report, indent=2))
val_test_prob = predict(model, test_images)
sample.assign(Emotion=val_test_prob.argmax(1)).to_csv(OUT / 'submission_validation.csv', index=False)
del model
torch.cuda.empty_cache()

# Refit on every labeled training image; use the validation-selected epoch budget.
seed_everything(SEED + 1)
full_model, _ = train(np.arange(len(y)), best_epoch, 'full')
full_prob = predict(full_model, test_images)
# The refit sees all labels, while the independently augmented validation model adds diversity.
ensemble = .75 * full_prob + .25 * val_test_prob
submission = sample.assign(Emotion=ensemble.argmax(1).astype(int))
assert submission.ID.equals(sample.ID) and submission.ID.is_unique
assert submission.Emotion.between(0, 5).all() and not submission.isna().any().any()
submission.to_csv(OUT / 'submission.csv', index=False)
sample.assign(Emotion=full_prob.argmax(1)).to_csv(OUT / 'submission_full.csv', index=False)
np.savez_compressed(OUT / 'test_predictions.npz', ids=sample.ID.to_numpy(dtype=str), validation_model=val_test_prob, full_model=full_prob, ensemble=ensemble)
print('COMPLETE:', OUT / 'submission.csv', submission.shape, submission.Emotion.value_counts().sort_index().to_dict(), flush=True)
