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
from torchvision.models import resnet50, ResNet50_Weights, convnext_tiny, ConvNeXt_Tiny_Weights

SEED = 2026
EPOCHS = 20
BATCH = 128
SIZE = 160
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
print('REMOTE_EXECUTION_STARTED: 3-fold FER ensemble; GPU:', torch.cuda.get_device_name(), flush=True)
print('Versions:', torch.__version__, torchvision.__version__, sklearn.__version__, flush=True)
competition_roots = [p for p in [Path('/kaggle/input/competitions/2026task2'), Path('/kaggle/input/2026task2')] if p.exists()]
search_root = competition_roots[0] if competition_roots else Path('/kaggle/input')
train_dirs = [p for p in search_root.rglob('train') if (p / 'Happy').is_dir()]
assert len(train_dirs) == 1, train_dirs
train_dir = train_dirs[0]
test_dir = train_dir.parent / 'test'
sample_candidates = [p / 'submission.csv' for p in train_dir.parents if (p / 'submission.csv').is_file()]
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

from collections import defaultdict
pixel_labels = defaultdict(set)
for h, label in zip(groups, y):
    pixel_labels[h].add(int(label))
test_keys = [a.tobytes() for a in test_pixels]
exact_indices = np.array([i for i, h in enumerate(test_keys) if len(pixel_labels.get(h, set())) == 1], dtype=int)
exact_labels = np.array([next(iter(pixel_labels[test_keys[i]])) for i in exact_indices])

def save_submission(prob, name, correct_exact=False):
    assert prob.shape == (len(sample), 6) and np.isfinite(prob).all()
    labels = prob.argmax(1).astype(int)
    if correct_exact:
        labels[exact_indices] = exact_labels
    result = sample.assign(Emotion=labels)
    assert result.ID.equals(sample.ID) and result.ID.is_unique
    assert result.Emotion.between(0, 5).all() and not result.isna().any().any()
    result.to_csv(OUT / name, index=False)

# Read prior predictions directly from mounted Kaggle notebook outputs.
# Discover by metadata and reject ambiguous matches instead of taking the first file.
prediction_dirs = [p.parent for p in Path('/kaggle/input').rglob('validation_predictions.npz')]
previous = {}
for directory in prediction_dirs:
    metric_file = directory / 'metrics.json'
    if not metric_file.exists():
        continue
    meta = json.loads(metric_file.read_text())
    name = meta.get('architecture', 'resnet18')
    if name not in {'resnet18', 'resnet50', 'convnext_tiny'}:
        continue
    assert name not in previous, f'Duplicate mounted source: {name}'
    val = np.load(directory / 'validation_predictions.npz')
    test = np.load(directory / 'test_predictions.npz')
    assert np.array_equal(val['index'], valid_idx)
    assert np.array_equal(val['labels'], y[valid_idx])
    assert np.array_equal(test['ids'], sample.ID.to_numpy(dtype=str))
    full_weight = .75 if name == 'resnet18' else .8
    previous[name] = {'val': val['probabilities'],
                      'test': full_weight * test['full_model'] + (1-full_weight) * test['validation_model']}
assert set(previous) == {'resnet18', 'resnet50', 'convnext_tiny'}, list(previous)

def acc(prob, labels):
    return float(accuracy_score(labels, prob.argmax(1)))

def nll(prob, labels):
    return float(-np.log(np.clip(prob[np.arange(len(labels)), labels], 1e-8, 1)).mean())

# Small predeclared grid. Public leaderboard scores are not used to tune weights.
old_candidates = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]
old_candidates += [(c, r, 1-c-r) for c in [.4, .6, .8] for r in [.15, .3, .45] if c+r <= 1]
names = ['convnext_tiny', 'resnet50', 'resnet18']
old_search = []
for weights in old_candidates:
    prob = sum(w * previous[name]['val'] for w, name in zip(weights, names))
    old_search.append({'weights': weights, 'accuracy': acc(prob, y[valid_idx]), 'nll': nll(prob, y[valid_idx])})
old_choice = max(old_search, key=lambda item: (item['accuracy'], -item['nll']))
old_val_prob = sum(w * previous[name]['val'] for w, name in zip(old_choice['weights'], names))
old_test_prob = sum(w * previous[name]['test'] for w, name in zip(old_choice['weights'], names))
save_submission(old_test_prob, 'submission_quick_pure.csv')
save_submission(old_test_prob, 'submission_quick.csv', correct_exact=True)
# Always leave a valid output even if a later experiment reaches the time budget.
save_submission(old_test_prob, 'submission_pure.csv')
save_submission(old_test_prob, 'submission.csv', correct_exact=True)
report = {'status': 'prior_model_blend_ready', 'exact_duplicate_audit': audit,
          'prior_model_order': names, 'prior_blend_search': old_search, 'prior_blend_choice': old_choice,
          'user_reported_public_scores': {'resnet18': .702, 'resnet50': .716, 'convnext': .718},
          'score_caveat': 'Validation and OOF scores are used for checkpoint/ensemble selection and are not independent test scores. Exact-match correction has no unbiased grouped-validation accuracy estimate.'}
(OUT / 'metrics.json').write_text(json.dumps(report, indent=2))
print('QUICK_SUBMISSIONS_READY:', old_choice, flush=True)

CONFIGS = [
    {'name': 'convnext_tiny', 'size': 160, 'batch': 128, 'epochs': 20, 'lr': 1.5e-4},
    {'name': 'resnet50', 'size': 144, 'batch': 128, 'epochs': 20, 'lr': 1.5e-4},
]
initial_states = {}

def make_model(config):
    name = config['name']
    first = name not in initial_states
    if name == 'convnext_tiny':
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if first else None,
                              stochastic_depth_prob=.25)
        model.classifier[2] = nn.Sequential(nn.Dropout(.4), nn.Linear(768, 6))
    else:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if first else None)
        model.fc = nn.Sequential(nn.Dropout(.45), nn.Linear(2048, 6))
    if first:
        initial_states[name] = copy.deepcopy(model.state_dict())
    else:
        model.load_state_dict(initial_states[name])
    return model.to(device=device, memory_format=torch.channels_last)

def optimizer_groups(model, config):
    grouped = {}
    for name, param in model.named_parameters():
        if config['name'] == 'convnext_tiny':
            depth = int(name.split('.')[1]) // 2 if name.startswith('features.') else 4
            lr = config['lr'] * .7 ** (3-depth) if depth < 4 else 4e-4
        else:
            lr = 4e-4 if name.startswith('fc.') else config['lr']
            if name.startswith(('conv1.', 'bn1.', 'layer1.')):
                lr *= .4
        wd = .05 if param.ndim > 1 else 0.
        grouped.setdefault((lr, wd), []).append(param)
    return [{'params': p, 'lr': lr, 'weight_decay': wd} for (lr, wd), p in grouped.items()]

@torch.inference_mode()
def predict(model, source, indices=None, tta=False):
    model.eval()
    indices = np.arange(len(source)) if indices is None else indices
    result = []
    for start in range(0, len(indices), BATCH):
        raw = source[indices[start:start+BATCH]]
        with torch.autocast('cuda', dtype=torch.float16):
            prob = model(prepare(raw)).float().softmax(1)
            if tta:
                prob = (prob + model(prepare(raw, flip=True)).float().softmax(1)) / 2
        result.append(prob.cpu().numpy())
    return np.concatenate(result)

history, fold_reports = [], []
started_at = time.monotonic()
TRAIN_BUDGET_SECONDS = 9900  # Leaves time for final inference within the 3h session limit.

def fit_fold(config, fold, train_indices, validation_indices):
    global BATCH, SIZE
    BATCH, SIZE = config['batch'], config['size']
    seed_everything(4200 + fold * 13 + (100 if config['name'] == 'resnet50' else 0))
    # Deduplicate within the training partition, skipping contradictory-label groups.
    seen, clean_train = set(), []
    for i in train_indices:
        h = groups[i]
        if h not in seen and len(pixel_labels[h]) == 1:
            clean_train.append(i)
            seen.add(h)
    clean_train = np.asarray(clean_train)
    assert not (set(np.asarray(groups)[clean_train]) & set(np.asarray(groups)[validation_indices]))
    print('FOLD_START:', config['name'], fold, 'train:', len(clean_train), 'validation:', len(validation_indices), flush=True)
    model = make_model(config)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(optimizer_groups(model, config))
    scaler = torch.amp.GradScaler('cuda')
    loss_fn = nn.CrossEntropyLoss(label_smoothing=.10)
    base_lrs = [g['lr'] for g in optimizer.param_groups]
    steps_per_epoch = math.ceil(len(clean_train) / BATCH)
    warmup = steps_per_epoch * 2
    total_steps = config['epochs'] * steps_per_epoch
    step, best, best_epoch, stale = 0, -1., 0, 0
    checkpoint = OUT / f"{config['name']}_fold{fold}.pt"
    ema_params, live_params = list(ema.parameters()), list(model.parameters())
    for epoch in range(config['epochs']):
        t0 = time.monotonic()
        model.train()
        losses = 0.
        for offset in range(0, len(clean_train), BATCH):
            if offset == 0:
                order = np.random.permutation(clean_train)
            idx = order[offset:offset+BATCH]
            factor = .05 + .95*(step+1)/warmup if step < warmup else .03 + .97*.5*(1+math.cos(math.pi*(step-warmup)/max(1,total_steps-warmup)))
            for param_group, lr in zip(optimizer.param_groups, base_lrs):
                param_group['lr'] = lr * factor
            x, labels = prepare(images[idx], augment=True), targets[idx]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                if random.random() < .45:
                    lam = np.random.beta(.4, .4)
                    perm = torch.randperm(len(x), device=device)
                    logits = model(lam*x + (1-lam)*x[perm])
                    loss = lam*loss_fn(logits, labels) + (1-lam)*loss_fn(logits, labels[perm])
                else:
                    loss = loss_fn(model(x), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            decay = min(.995, (step+1)/(step+10))
            with torch.no_grad():
                torch._foreach_mul_(ema_params, decay)
                torch._foreach_add_(ema_params, live_params, alpha=1-decay)
                for a, b in zip(ema.buffers(), model.buffers()):
                    a.copy_(b)
            losses += loss.item()*len(idx)
        prob = predict(ema, images, validation_indices)
        score = acc(prob, y[validation_indices])
        stale += 1
        if score > best:
            best, best_epoch, stale = score, epoch+1, 0
            torch.save(ema.state_dict(), checkpoint)
        row = {'architecture': config['name'], 'fold': fold, 'epoch': epoch+1,
               'loss': losses/len(clean_train), 'val_accuracy': score, 'seconds': round(time.monotonic()-t0, 1)}
        history.append(row)
        pd.DataFrame(history).to_csv(OUT / 'history.csv', index=False)
        print(json.dumps(row), flush=True)
        if (epoch >= 9 and stale >= 5) or time.monotonic()-started_at > TRAIN_BUDGET_SECONDS:
            break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    val_prob = predict(model, images, validation_indices, tta=True)
    test_prob = predict(model, test_images, tta=True)
    fold_report = {'architecture': config['name'], 'fold': fold, 'best_epoch': best_epoch,
                   'validation_accuracy_tta': acc(val_prob, y[validation_indices]),
                   'train_after_deduplication': len(clean_train)}
    fold_reports.append(fold_report)
    np.savez_compressed(OUT / f"{config['name']}_fold{fold}_predictions.npz",
                        indices=validation_indices, labels=y[validation_indices], validation=val_prob, test=test_prob)
    print('FOLD_COMPLETE:', fold_report, flush=True)
    return val_prob, test_prob

fold_ids = np.full(len(y), -1, dtype=int)
folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=31415).split(y, y, groups))
for fold, (_, vi) in enumerate(folds):
    fold_ids[vi] = fold
assert (fold_ids >= 0).all()
pd.DataFrame({'path': paths, 'label': y, 'pixel_group': groups, 'fold': fold_ids}).to_csv(OUT / 'cv_split.csv', index=False)
oof = {c['name']: np.full((len(y), 6), np.nan, dtype=np.float32) for c in CONFIGS}
test_by_arch = {c['name']: [] for c in CONFIGS}
for config in CONFIGS:
    for fold, (ti, vi) in enumerate(folds):
        if time.monotonic()-started_at > TRAIN_BUDGET_SECONDS:
            break
        vp, tp = fit_fold(config, fold, ti, vi)
        oof[config['name']][vi] = vp
        test_by_arch[config['name']].append(tp)
        torch.cuda.empty_cache()
        report['folds'] = fold_reports
        report['status'] = 'cross_validation_running'
        (OUT / 'metrics.json').write_text(json.dumps(report, indent=2))

complete_archs = [c['name'] for c in CONFIGS if np.isfinite(oof[c['name']]).all()]
report['complete_architectures'] = complete_archs
if complete_archs:
    if len(complete_archs) == 2:
        grid = [.25, .5, .75, 1.]
        cv_search = []
        for weight in grid:
            p = weight*oof['convnext_tiny']+(1-weight)*oof['resnet50']
            cv_search.append({'convnext_weight': weight, 'oof_accuracy': acc(p, y), 'nll': nll(p, y)})
        choice = max(cv_search, key=lambda row: (row['oof_accuracy'], -row['nll']))
        w = choice['convnext_weight']
        cv_prob = w*oof['convnext_tiny']+(1-w)*oof['resnet50']
        cv_test = w*np.mean(test_by_arch['convnext_tiny'], axis=0)+(1-w)*np.mean(test_by_arch['resnet50'], axis=0)
        report['cv_architecture_search'] = cv_search
        report['cv_architecture_choice'] = choice
    else:
        cv_prob = oof[complete_archs[0]]
        cv_test = np.mean(test_by_arch[complete_archs[0]], axis=0)
    # Every CV prediction below is generated without fitting that image's group.
    # Select a coarse mixture with the earlier full-data models on their held-out subset.
    final_search = []
    for old_weight in [0., .25, .5, .75, 1.]:
        p = old_weight*old_val_prob+(1-old_weight)*cv_prob[valid_idx]
        final_search.append({'prior_weight': old_weight, 'holdout_accuracy': acc(p, y[valid_idx]), 'nll': nll(p, y[valid_idx])})
    final_choice = max(final_search, key=lambda row: (row['holdout_accuracy'], -row['nll']))
    old_weight = final_choice['prior_weight']
    final_prob = old_weight*old_test_prob+(1-old_weight)*cv_test
    save_submission(cv_test, 'submission_cv_pure.csv')
    save_submission(cv_test, 'submission_cv.csv', correct_exact=True)
    save_submission(final_prob, 'submission_pure.csv')
    save_submission(final_prob, 'submission.csv', correct_exact=True)
    report.update(oof_accuracy=acc(cv_prob, y), final_blend_search=final_search,
                  final_blend_choice=final_choice,
                  oof_classification_report=classification_report(y, cv_prob.argmax(1), target_names=CLASS_NAMES, output_dict=True),
                  oof_confusion_matrix=confusion_matrix(y, cv_prob.argmax(1)).tolist())
    np.savez_compressed(OUT / 'ensemble_predictions.npz', ids=sample.ID.to_numpy(dtype=str),
                        labels=y, cv_oof=cv_prob, cv_test=cv_test, final_test=final_prob)
report['status'] = 'complete' if len(fold_reports) == 6 else 'complete_with_budget_limit'
report['training_seconds'] = time.monotonic()-started_at
(OUT / 'metrics.json').write_text(json.dumps(report, indent=2))
print('COMPLETE: submission.csv and submission_pure.csv generated; metrics:', json.dumps(report), flush=True)
