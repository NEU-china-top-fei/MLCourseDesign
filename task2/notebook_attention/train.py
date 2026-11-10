"""Six-class facial expression classification; runs on Kaggle GPU."""
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import sklearn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score
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
print('REMOTE_EXECUTION_STARTED: ConvNeXt V2 residual local-attention comparison; GPU:', torch.cuda.get_device_name(), flush=True)
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

import gc
import traceback

def acc(prob, labels):
    return float(accuracy_score(labels, prob.argmax(1)))

def nll(prob, labels):
    return float(-np.log(np.clip(prob[np.arange(len(labels)), labels], 1e-8, 1)).mean())

source_files = list(Path('/kaggle/input').rglob('convnextv2_validation_predictions.npz'))
assert len(source_files) == 1, source_files
SOURCE = source_files[0].parent
prior_report = json.loads((SOURCE / 'metrics.json').read_text())
prior = np.load(SOURCE / 'ensemble_predictions.npz')
old_model = np.load(source_files[0])
assert np.array_equal(prior['ids'], sample.ID.to_numpy(dtype=str))
assert np.array_equal(prior['validation_indices'], valid_idx)
assert np.array_equal(prior['labels'], y[valid_idx])
assert np.array_equal(old_model['index'], valid_idx)
assert np.array_equal(old_model['labels'], y[valid_idx])
baseline_val, baseline_test = prior['validation'], prior['test']
old_val, old_test = old_model['probabilities'], old_model['test']
old_details = next(m for m in prior_report['models'] if m['model'] == 'convnextv2')
prior_conv_checkpoint = SOURCE / 'convnextv2_validation.pt'
assert prior_conv_checkpoint.exists()
assert old_details['best_epoch'] == 24 and old_details['best_kind'] == 'raw'
full_source = SOURCE / 'convnextv2_test_predictions.npz'
if full_source.exists():
    prior_full = np.load(full_source)
    old_test = .8*prior_full['full_model']+.2*prior_full['validation_model']
conv_weight = prior_report['ensemble_choice']['weights'][prior_report['ensemble_model_order'].index('convnextv2')]
assert 0 < conv_weight < 1
other_val = (baseline_val-conv_weight*old_val)/(1-conv_weight)
other_test = (baseline_test-conv_weight*old_test)/(1-conv_weight)
assert other_val.min() > -1e-5 and other_test.min() > -1e-5
other_val = np.clip(other_val, 0, 1)
other_test = np.clip(other_test, 0, 1)
other_val /= other_val.sum(1, keepdims=True)
other_test /= other_test.sum(1, keepdims=True)

save_submission(baseline_test, 'submission_baseline.csv', correct_exact=True)
save_submission(baseline_test, 'submission.csv', correct_exact=True)
save_submission(baseline_test, 'submission_pure.csv')

# Appended to the existing data/source-loading code by analysis/build_attention.py.
# Research hypothesis: learned local pooling can complement global average pooling.
# 借鉴 DAN 的局部注意力思路，采用轻量残差适配器。
MODEL = 'convnextv2_tiny.fcmae_ft_in22k_in1k'
ADAPT_EPOCHS = 8
TRAIN_SECONDS = 6600
started_at = time.monotonic()
selection_local, audit_local = next(StratifiedGroupKFold(
    n_splits=2, shuffle=True, random_state=8419).split(
        y[valid_idx], y[valid_idx], np.asarray(groups)[valid_idx]))
assert not (set(np.asarray(groups)[valid_idx[selection_local]]) &
            set(np.asarray(groups)[valid_idx[audit_local]]))
seen, clean = set(), []
for i in train_idx:
    if groups[i] not in seen and len(pixel_labels[groups[i]]) == 1:
        seen.add(groups[i])
        clean.append(i)
fit_idx = np.asarray(clean)
REPLACE_FRACTION = .5  # Fixed: replace half of the existing 25% ConvNeXt component.
report = {
    'status': 'prepared', 'baseline_holdout_accuracy': acc(baseline_val,y[valid_idx]),
    'user_reported_public_score': .738, 'arms': [], 'errors': [],
    'selection_rows':len(selection_local), 'audit_rows':len(audit_local),
    'max_epochs_per_arm':ADAPT_EPOCHS, 'new_weight_downloads':False,
    'replacement_fraction':REPLACE_FRACTION,
    'note':'Epoch selection uses one half of the old holdout. The other half is inspected only after training each arm. The old base models were already selected on this holdout, so this is an exploratory confirmation, not a new independent test. No full-data training occurs.',
}
history=[]

def save_report():
    report['elapsed_seconds']=round(time.monotonic()-started_at,1)
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2))

class LocalResidual(nn.Module):
    def __init__(self, enabled):
        super().__init__()
        self.base=timm.create_model(MODEL,pretrained=False,num_classes=6,
                                    drop_rate=.35,drop_path_rate=.2)
        self.base.load_state_dict(torch.load(prior_conv_checkpoint,map_location='cpu',weights_only=True),strict=True)
        self.enabled=enabled
        channels=self.base.num_features
        self.local_norm=nn.LayerNorm(channels)
        self.queries=nn.Linear(channels,2,bias=False)
        self.local_classifier=nn.Linear(2*channels,6)
        nn.init.zeros_(self.local_classifier.weight)
        nn.init.zeros_(self.local_classifier.bias)
        # Adapt only the last stage and heads. Freeze identical early features in both arms.
        for name,p in self.base.named_parameters():
            p.requires_grad_(name.startswith(('stages.3.','head.','norm_pre.')))
        for part in (self.local_norm,self.queries,self.local_classifier):
            part.requires_grad_(enabled)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.base.stem.eval()
            for stage in list(self.base.stages)[:3]:
                stage.eval()
        return self

    def forward(self,x):
        features=self.base.forward_features(x)
        logits=self.base.forward_head(features)
        if not self.enabled:
            return logits
        assert features.ndim==4 and features.shape[1]==self.base.num_features
        tokens=self.local_norm(features.flatten(2).transpose(1,2))
        attention=self.queries(tokens).transpose(1,2).float().softmax(dim=-1).to(tokens.dtype)
        local=torch.bmm(attention,tokens).flatten(1)
        return logits+self.local_classifier(local)

@torch.inference_mode()
def predict(model,source,indices):
    model.eval()
    result=[]
    for start in range(0,len(indices),BATCH):
        raw=source[indices[start:start+BATCH]]
        with torch.autocast('cuda',dtype=torch.float16):
            p=.5*(model(prepare(raw)).float().softmax(1)+
                   model(prepare(raw,flip=True)).float().softmax(1))
        result.append(p.cpu().numpy())
    return np.concatenate(result)

def combined(p,base,old):
    result=base+conv_weight*REPLACE_FRACTION*(p-old)
    assert np.isfinite(result).all() and result.min()>-1e-5
    result=np.clip(result,0,1)
    return result/result.sum(1,keepdims=True)

save_report()
for name,enabled in [('continuation_control',False),('local_attention',True)]:
    if time.monotonic()-started_at>TRAIN_SECONDS-300:
        report['errors'].append({'arm':name,'reason':'Budget exhausted before arm start.'})
        break
    try:
        seed_everything(8419)
        model=LocalResidual(enabled).to(device=device,memory_format=torch.channels_last)
        config=timm.data.resolve_model_data_config(model.base)
        mean=torch.tensor(config['mean'],device=device).view(1,3,1,1)
        std=torch.tensor(config['std'],device=device).view(1,3,1,1)
        model.eval()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            x=prepare(images[train_idx[:2]])
            expected=model.base(x)
            actual=model(x)
            assert actual.shape==(2,6) and torch.isfinite(actual).all()
            torch.testing.assert_close(actual,expected,rtol=1e-3,atol=1e-3)
        initial=predict(model,images,valid_idx)
        # Check the mounted checkpoint and preprocessing reproduce the original probabilities.
        assert np.max(np.abs(initial-old_val))<.015, 'Checkpoint or preprocessing does not match source.'
        selection_labels=y[valid_idx[selection_local]]
        initial_combined=combined(initial,baseline_val,old_val)
        best_key=(acc(initial_combined[selection_local],selection_labels),
                  -nll(initial_combined[selection_local],selection_labels))
        checkpoint=OUT/f'{name}.pt'
        torch.save(model.state_dict(),checkpoint)
        best_epoch=0
        groups_opt=[]
        for parameter_name,p in model.named_parameters():
            if not p.requires_grad:
                continue
            lr=1e-5 if parameter_name.startswith('base.stages.') else 3e-5
            if not parameter_name.startswith('base.'):
                lr=2e-4
            groups_opt.append({'params':[p],'lr':lr,'initial_lr':lr,
                               'weight_decay':0. if p.ndim<2 else .04})
        optimizer=torch.optim.AdamW(groups_opt)
        scaler=torch.amp.GradScaler('cuda')
        criterion=nn.CrossEntropyLoss(label_smoothing=.08)
        steps=math.ceil(len(fit_idx)/BATCH)
        completed=0
        for epoch in range(ADAPT_EPOCHS):
            if time.monotonic()-started_at>TRAIN_SECONDS-180:
                break
            t0=time.monotonic()
            # Match data order, augmentation and initialization seed across control and attention.
            seed_everything(13001+epoch)
            order=np.random.permutation(fit_idx)
            model.train()
            total_loss=0.
            for start in range(0,len(order),BATCH):
                step=epoch*steps+start//BATCH
                factor=.1+.9*(step+1)/steps if step<steps else .1+.9*.5*(1+math.cos(math.pi*(step-steps)/((ADAPT_EPOCHS-1)*steps)))
                for group in optimizer.param_groups:
                    group['lr']=group['initial_lr']*factor
                idx=order[start:start+BATCH]
                x,target=prepare(images[idx],augment=True),targets[idx]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    if random.random()<.3:
                        lam=np.random.beta(.25,.25)
                        perm=torch.randperm(len(x),device=device)
                        logits=model(lam*x+(1-lam)*x[perm])
                        loss=lam*criterion(logits,target)+(1-lam)*criterion(logits,target[perm])
                    else:
                        loss=criterion(model(x),target)
                assert torch.isfinite(loss)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(),5.)
                scaler.step(optimizer)
                scaler.update()
                total_loss+=loss.item()*len(idx)
            # Audit labels are not used for epoch selection.
            selection_prob=predict(model,images,valid_idx[selection_local])
            selection_mix=combined(selection_prob,baseline_val[selection_local],old_val[selection_local])
            key=(acc(selection_mix,selection_labels),-nll(selection_mix,selection_labels))
            if key>best_key:
                best_key=key
                best_epoch=epoch+1
                torch.save(model.state_dict(),checkpoint)
            row={'arm':name,'epoch':epoch+1,'loss':total_loss/len(fit_idx),
                 'selection_ensemble_accuracy':key[0],'seconds':round(time.monotonic()-t0,1)}
            history.append(row)
            pd.DataFrame(history).to_csv(OUT/'history.csv',index=False)
            print(json.dumps(row),flush=True)
            completed=epoch+1
        model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
        vp=predict(model,images,valid_idx)
        tp=predict(model,test_images,np.arange(len(sample)))
        vm=combined(vp,baseline_val,old_val)
        tm=combined(tp,baseline_test,old_test)
        info={'name':name,'enabled_attention':enabled,'best_epoch':best_epoch,'completed_epochs':completed,
              'selection_accuracy':acc(vm[selection_local],y[valid_idx[selection_local]]),
              'audit_accuracy':acc(vm[audit_local],y[valid_idx[audit_local]]),
              'baseline_audit_accuracy':acc(baseline_val[audit_local],y[valid_idx[audit_local]]),
              'holdout_accuracy':acc(vm,y[valid_idx]),'holdout_nll':nll(vm,y[valid_idx])}
        report['arms'].append(info)
        np.savez_compressed(OUT/f'{name}_predictions.npz',validation=vp,test=tp,
                            ensemble_validation=vm,ensemble_test=tm,indices=valid_idx,ids=sample.ID.to_numpy(dtype=str))
        save_submission(tm,f'submission_{name}.csv',correct_exact=True)
        save_report()
        print('ARM_COMPLETE:',json.dumps(info),flush=True)
        del model,optimizer,scaler
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as error:
        report['errors'].append({'arm':name,'type':type(error).__name__,'message':str(error)})
        traceback.print_exc()
        save_report()
        # 基线输出用于异常恢复，实验状态另行记录。
        report['status']='failed'
        report['selected']='baseline_recovery_only'
        save_report()
        (OUT/'RUN_FAILED.txt').write_text(
            f"Training failed in {name}: {type(error).__name__}: {error}\n"
            'Any baseline CSV is unchanged recovery output, not an improved result.\n')
        raise

# Baseline remains default unless an exploratory confirmation clears both gates.
eligible=[a for a in report['arms'] if a['best_epoch']>0 and
          a['holdout_accuracy']>=report['baseline_holdout_accuracy']+.002 and
          a['audit_accuracy']>=a['baseline_audit_accuracy']+.002]
report['selected']='baseline'
if eligible:
    winner=max(eligible,key=lambda a:(a['audit_accuracy'],a['selection_accuracy']))
    p=np.load(OUT/f"{winner['name']}_predictions.npz")['ensemble_test']
    save_submission(p,'submission.csv',correct_exact=True)
    save_submission(p,'submission_pure.csv')
    report['selected']=winner['name']
report['status']='complete' if not report['errors'] else 'complete_with_errors'
report['changed_test_labels_vs_baseline']=int((
    pd.read_csv(OUT/'submission.csv').Emotion.to_numpy()!=
    pd.read_csv(OUT/'submission_baseline.csv').Emotion.to_numpy()).sum())
report['selection_reason']=('Candidate cleared both improvement gates.' if eligible else
                            'No candidate cleared both improvement gates; baseline is unchanged.')
if not any(a['completed_epochs']>0 for a in report['arms']):
    report['status']='failed_no_training'
    save_report()
    raise RuntimeError('No training epoch completed; baseline is only a recovery output.')
save_report()
print('COMPLETE:',report['selected'],flush=True)
