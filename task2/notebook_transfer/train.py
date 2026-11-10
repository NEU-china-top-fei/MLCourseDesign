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
import subprocess
import sys
try:
    import timm
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'timm>=1.0.15,<2'])
    import timm

SEED = 2026
EPOCHS = 24
BATCH = 96
SIZE = 192
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
print('REMOTE_EXECUTION_STARTED: expanded-pretraining FER ensemble; GPU:', torch.cuda.get_device_name(), flush=True)
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

def prepare(x, augment=False, flip=False, zoom=1.):
    x = x.float().div_(255)
    n = len(x)
    if augment:
        angle = (torch.rand(n, device=device) * 2 - 1) * (10 * math.pi / 180)
        scale = .88 + .24 * torch.rand(n, device=device)
        mirror = torch.where(torch.rand(n, device=device) < .5, -1., 1.)
        theta = torch.zeros(n, 2, 3, device=device)
        theta[:, 0, 0] = angle.cos() * scale * mirror
        theta[:, 0, 1] = -angle.sin() * scale
        theta[:, 1, 0] = angle.sin() * scale * mirror
        theta[:, 1, 1] = angle.cos() * scale
        theta[:, :, 2] = (torch.rand(n, 2, device=device) * 2 - 1) * .12
        grid = F.affine_grid(theta, (n, 1, SIZE, SIZE), align_corners=False)
        x = F.grid_sample(x, grid, mode='bicubic', padding_mode='reflection', align_corners=False)
        contrast = .85 + .3 * torch.rand(n, 1, 1, 1, device=device)
        brightness = (torch.rand(n, 1, 1, 1, device=device) * 2 - 1) * .08
        x = ((x - .5) * contrast + .5 + brightness).clamp_(0, 1)
    else:
        if zoom == 1.:
            x = F.interpolate(x, size=(SIZE, SIZE), mode='bicubic', align_corners=False)
        else:
            theta = torch.zeros(n, 2, 3, device=device)
            theta[:, 0, 0] = zoom
            theta[:, 1, 1] = zoom
            grid = F.affine_grid(theta, (n, 1, SIZE, SIZE), align_corners=False)
            x = F.grid_sample(x, grid, mode='bicubic', padding_mode='reflection', align_corners=False)
        x = x.clamp_(0, 1)
        if flip:
            x = x.flip(-1)
    if augment and random.random() < .15:
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

# Reconstruct the previous 0.729 solution's holdout probabilities without
# evaluating any full-data checkpoint on labels it was trained on.
cv_files = list(Path('/kaggle/input').rglob('ensemble_predictions.npz'))
assert len(cv_files) == 1, cv_files
cv_file = cv_files[0]
cv_meta = json.loads((cv_file.parent / 'metrics.json').read_text())
cv_arrays = np.load(cv_file)
assert np.array_equal(cv_arrays['ids'], sample.ID.to_numpy(dtype=str))
assert np.array_equal(cv_arrays['labels'], y)
prior_val = sum(w*previous[name]['val'] for w, name in zip(cv_meta['prior_blend_choice']['weights'], cv_meta['prior_model_order']))
w = cv_meta['final_blend_choice']['prior_weight']
baseline_val = w*prior_val+(1-w)*cv_arrays['cv_oof'][valid_idx]
baseline_test = cv_arrays['final_test']
assert abs(acc(baseline_val, y[valid_idx])-cv_meta['final_blend_choice']['holdout_accuracy']) < 1e-6
save_submission(baseline_test, 'submission_baseline.csv', correct_exact=True)
save_submission(baseline_test, 'submission.csv', correct_exact=True)
save_submission(baseline_test, 'submission_pure.csv')

CONFIGS = [
    {'name': 'convnextv2', 'model': 'convnextv2_tiny.fcmae_ft_in22k_in1k', 'size': 192,
     'batch': 96, 'epochs': 24, 'lr': 8e-5, 'head_lr': 4e-4, 'seed': 5201},
    {'name': 'efficientnetv2', 'model': 'tf_efficientnetv2_s.in21k_ft_in1k', 'size': 192,
     'batch': 96, 'epochs': 24, 'lr': 1e-4, 'head_lr': 4e-4, 'seed': 6201},
]
report = {'status': 'baseline_restored', 'user_reported_baseline_public_score': .729,
          'baseline_holdout_accuracy': acc(baseline_val, y[valid_idx]), 'models': [], 'errors': [],
          'configs': CONFIGS, 'exact_duplicate_audit': audit,
          'evaluation_note': 'All model, view and ensemble choices use the same grouped holdout. It is a model-selection set, not an independent test estimate. No full-refit model is evaluated on this holdout.',
          'refit_schedule': 'Same 24-epoch learning-rate horizon as validation, stopped at selected epoch; the cosine schedule is not compressed.'}
history = []
initial_states, data_configs = {}, {}
started_at = time.monotonic()
TRAIN_BUDGET_SECONDS = 9600

def save_report():
    report['elapsed_seconds'] = round(time.monotonic()-started_at, 1)
    (OUT / 'metrics.json').write_text(json.dumps(report, indent=2))

save_report()
print('BASELINE_RESTORED:', report['baseline_holdout_accuracy'], flush=True)

def make_model(config):
    global SIZE, BATCH, mean, std
    name = config['name']
    first = name not in initial_states
    model = timm.create_model(config['model'], pretrained=first, num_classes=6,
                              drop_rate=.35, drop_path_rate=.2)
    if first:
        initial_states[name] = copy.deepcopy(model.state_dict())
        data_configs[name] = timm.data.resolve_model_data_config(model)
    else:
        model.load_state_dict(initial_states[name])
    SIZE, BATCH = config['size'], config['batch']
    data_config = data_configs[name]
    mean = torch.tensor(data_config['mean'], device=device).view(1, 3, 1, 1)
    std = torch.tensor(data_config['std'], device=device).view(1, 3, 1, 1)
    assert torch.isfinite(mean).all() and (std > 0).all()
    return model.to(device=device, memory_format=torch.channels_last)

def optimizer_groups(model, config):
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    grouped = {}
    for name, param in model.named_parameters():
        lr = config['lr']
        if id(param) in head_ids:
            lr = config['head_lr']
        elif config['name'] == 'convnextv2':
            if name.startswith('stem.'):
                lr *= .35
            elif name.startswith('stages.'):
                stage = int(name.split('.')[1])
                lr *= .7**(3-stage)
        elif name.startswith(('conv_stem.', 'bn1.', 'blocks.0.', 'blocks.1.')):
            lr *= .4
        decay = .04 if param.ndim > 1 else 0.
        grouped.setdefault((lr, decay), []).append(param)
    return [{'params': params, 'lr': lr, 'weight_decay': decay} for (lr, decay), params in grouped.items()]

def clean_indices(indices):
    seen, result = set(), []
    for i in indices:
        h = groups[i]
        if h not in seen and len(pixel_labels[h]) == 1:
            seen.add(h)
            result.append(i)
    return np.asarray(result)

@torch.inference_mode()
def predict(model, source, indices=None, mode='flip'):
    model.eval()
    indices = np.arange(len(source)) if indices is None else indices
    views = [(False, 1.)]
    if mode in {'flip', 'cropflip'}:
        views += [(True, 1.)]
    if mode == 'cropflip':
        views += [(False, .90), (True, .90)]
    result = []
    for start in range(0, len(indices), BATCH):
        raw = source[indices[start:start+BATCH]]
        prob = torch.zeros(len(raw), 6, device=device)
        with torch.autocast('cuda', dtype=torch.float16):
            for flip, zoom in views:
                prob += model(prepare(raw, flip=flip, zoom=zoom)).float().softmax(1)
        result.append((prob/len(views)).cpu().numpy())
    return np.concatenate(result)

def train(config, indices, stage, epochs, validation=None):
    seed_everything(config['seed'] + (17 if stage == 'full' else 0))
    indices = clean_indices(indices)
    model = make_model(config)
    # Fail early on architecture, normalization or output-shape incompatibility.
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        smoke = model(prepare(images[indices[:2]]))
    assert smoke.shape == (2, 6) and torch.isfinite(smoke).all()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(optimizer_groups(model, config))
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss(label_smoothing=.08)
    base_lrs = [g['lr'] for g in optimizer.param_groups]
    steps_per_epoch = math.ceil(len(indices)/BATCH)
    # The refit deliberately retains the VALIDATION schedule horizon.
    horizon = config['epochs']*steps_per_epoch
    warmup = 2*steps_per_epoch
    best, best_epoch, best_kind, stale, step, completed = -1., 0, 'ema', 0, 0, 0
    checkpoint = OUT / f"{config['name']}_{stage}.pt"
    ema_params, live_params = list(ema.parameters()), list(model.parameters())
    for epoch in range(epochs):
        if time.monotonic()-started_at > TRAIN_BUDGET_SECONDS and completed:
            break
        t0 = time.monotonic()
        order = np.random.permutation(indices)
        model.train()
        total_loss = 0.
        for start in range(0, len(order), BATCH):
            idx = order[start:start+BATCH]
            factor = .05+.95*(step+1)/warmup if step < warmup else .03+.97*.5*(1+math.cos(math.pi*(step-warmup)/max(1,horizon-warmup)))
            for group, lr in zip(optimizer.param_groups, base_lrs):
                group['lr'] = lr*factor
            x, target = prepare(images[idx], augment=True), targets[idx]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                if random.random() < .30:
                    lam = np.random.beta(.25, .25)
                    perm = torch.randperm(len(x), device=device)
                    logits = model(lam*x+(1-lam)*x[perm])
                    loss = lam*criterion(logits, target)+(1-lam)*criterion(logits, target[perm])
                else:
                    loss = criterion(model(x), target)
            assert torch.isfinite(loss), 'Non-finite training loss'
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
                for ema_buffer, live_buffer in zip(ema.buffers(), model.buffers()):
                    ema_buffer.copy_(live_buffer)
            total_loss += loss.item()*len(idx)
        row = {'model': config['name'], 'stage': stage, 'epoch': epoch+1, 'loss': total_loss/len(indices)}
        stale += 1
        if validation is not None:
            for kind, candidate in [('raw', model), ('ema', ema)]:
                score = acc(predict(candidate, images, validation, mode='plain'), y[validation])
                row[f'{kind}_val_accuracy'] = score
                if score > best:
                    best, best_epoch, best_kind, stale = score, epoch+1, kind, 0
                    torch.save(candidate.state_dict(), checkpoint)
        completed = epoch+1
        row['seconds'] = round(time.monotonic()-t0, 1)
        history.append(row)
        pd.DataFrame(history).to_csv(OUT / 'history.csv', index=False)
        print(json.dumps(row), flush=True)
        if validation is not None and epoch >= 11 and stale >= 6:
            break
    if validation is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    elif config['refit_kind'] == 'ema':
        model.load_state_dict(ema.state_dict())
    if validation is None:
        torch.save(model.state_dict(), checkpoint)
    return model, {'best_epoch': best_epoch, 'best_kind': best_kind, 'completed_epochs': completed,
                   'requested_epochs': epochs, 'training_rows_after_dedup': len(indices)}

predictions = {'baseline': {'val': baseline_val, 'test': baseline_test}}

def update_ensemble():
    keys = list(predictions)
    # Coarse fixed simplex grid; include baseline-only to avoid forced replacement.
    candidates = []
    if len(keys) == 1:
        candidates = [[1.]]
    elif len(keys) == 2:
        candidates = [[w, 1-w] for w in [0., .25, .5, .75, 1.]]
    else:
        candidates = [[a/4, b/4, (4-a-b)/4] for a in range(5) for b in range(5-a)]
    search = []
    for weights in candidates:
        p = sum(w*predictions[k]['val'] for w, k in zip(weights, keys))
        search.append({'weights': weights, 'accuracy': acc(p, y[valid_idx]), 'nll': nll(p, y[valid_idx])})
    best = max(search, key=lambda item: (item['accuracy'], -item['nll']))
    if best['accuracy'] < report['baseline_holdout_accuracy']+.001:
        best = search[-1] if len(keys) == 2 else next(item for item in search if item['weights'][0] == 1.)
    val = sum(w*predictions[k]['val'] for w, k in zip(best['weights'], keys))
    test = sum(w*predictions[k]['test'] for w, k in zip(best['weights'], keys))
    save_submission(test, 'submission.csv', correct_exact=True)
    save_submission(test, 'submission_pure.csv')
    report.update(ensemble_model_order=keys, ensemble_search=search, ensemble_choice=best,
                  classification_report=classification_report(y[valid_idx], val.argmax(1), target_names=CLASS_NAMES, output_dict=True),
                  confusion_matrix=confusion_matrix(y[valid_idx], val.argmax(1)).tolist())
    np.savez_compressed(OUT / 'ensemble_predictions.npz', ids=sample.ID.to_numpy(dtype=str),
                        validation_indices=valid_idx, labels=y[valid_idx], validation=val, test=test)
    save_report()

import gc
import traceback
for config in CONFIGS:
    if time.monotonic()-started_at > TRAIN_BUDGET_SECONDS-300:
        report['errors'].append({'model': config['name'], 'reason': 'Skipped because the training budget is exhausted.'})
        break
    try:
        print('MODEL_START:', config['model'], flush=True)
        model, details = train(config, train_idx, 'validation', config['epochs'], valid_idx)
        val_flip = predict(model, images, valid_idx, mode='flip')
        val_crop = predict(model, images, valid_idx, mode='cropflip')
        mode = 'cropflip' if acc(val_crop, y[valid_idx]) >= acc(val_flip, y[valid_idx])+.001 else 'flip'
        val_prob = val_crop if mode == 'cropflip' else val_flip
        val_test = predict(model, test_images, mode=mode)
        details.update(model=config['name'], tta_mode=mode, flip_accuracy=acc(val_flip, y[valid_idx]),
                       cropflip_accuracy=acc(val_crop, y[valid_idx]), pretrained_data_config=data_configs[config['name']])
        np.savez_compressed(OUT / f"{config['name']}_validation_predictions.npz", index=valid_idx,
                            labels=y[valid_idx], probabilities=val_prob, test=val_test)
        predictions[config['name']] = {'val': val_prob, 'test': val_test}
        report['models'].append(details)
        update_ensemble()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        # Reserve enough time to complete the refit, otherwise retain the valid holdout model.
        times = [row['seconds'] for row in history if row['model'] == config['name']]
        refit_estimate = np.median(times[-5:])*details['best_epoch']*1.20+120
        if time.monotonic()-started_at+refit_estimate < TRAIN_BUDGET_SECONDS:
            config['refit_kind'] = details['best_kind']
            full_model, full_details = train(config, np.arange(len(y)), 'full', details['best_epoch'])
            if full_details['completed_epochs'] == details['best_epoch']:
                full_test = predict(full_model, test_images, mode=mode)
                predictions[config['name']]['test'] = .8*full_test+.2*val_test
                details['refit'] = full_details
                np.savez_compressed(OUT / f"{config['name']}_test_predictions.npz", ids=sample.ID.to_numpy(dtype=str),
                                    validation_model=val_test, full_model=full_test)
            else:
                details['refit'] = {'used': False, 'reason': 'Time-limited partial refit; retained validation checkpoint predictions.'}
            del full_model
            gc.collect()
            torch.cuda.empty_cache()
        else:
            details['refit'] = {'used': False, 'reason': 'Insufficient remaining time for full refit.'}
        save_submission(predictions[config['name']]['test'], f"submission_{config['name']}.csv", correct_exact=True)
        update_ensemble()
        initial_states.pop(config['name'], None)
    except Exception as error:
        report['errors'].append({'model': config['name'], 'type': type(error).__name__, 'reason': str(error)})
        traceback.print_exc()
        save_report()
        gc.collect()
        torch.cuda.empty_cache()
report['status'] = 'complete' if not report['errors'] else 'complete_with_recorded_errors'
save_report()
print('COMPLETE: submission.csv saved. Ensemble:', report.get('ensemble_choice'), flush=True)
