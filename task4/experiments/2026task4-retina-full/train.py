"""Retinal vessel segmentation. All training runs on Kaggle GPU.
No external data or pretrained vessel models are used.
"""
# Source method: https://github.com/orobix/retina-unet/tree/f6ff63bb39409942fd3aaf6907dbce74ca143207
# MIT License
# 
# Copyright (c) 2016 Daniele Cortinovis, Orobix Srl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# 
import os, sys, json, time, random, platform, copy
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

CFG = dict(seed=20261031, folds=4, steps=3000, batch_size=32,
           patch_size=96, learning_rate=0.0002, image_size=512, warm_start=False,
           eval_every=600, channels=[32,64,128], architecture='Retina shallow patch UNet 48-to-96 refinement', preprocessing='minmax-gray CLAHE gamma1.2', dropout=.2)
OUT = Path('/kaggle/working') if Path('/kaggle').exists() else Path('outputs')
OUT.mkdir(exist_ok=True)
START = time.time()
random.seed(CFG['seed']); np.random.seed(CFG['seed']); torch.manual_seed(CFG['seed'])
torch.set_num_threads(2)
assert torch.cuda.is_available(), 'GPU is required; enable GPU in Kaggle Notebook settings.'
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
device = torch.device('cuda')
print('CONFIG', json.dumps(CFG), flush=True)
print('GPU', torch.cuda.get_device_name(0), 'Torch', torch.__version__, flush=True)
candidates=[p.parent for p in Path('/kaggle/input').rglob('sample_submission.csv') if (p.parent/'train/image').is_dir()]
assert len(candidates)==1, candidates
ROOT=candidates[0]
assert (ROOT / 'train/image').is_dir(), ROOT
print('DATA ROOT', ROOT, flush=True)

# Match official segmentation_to_csv.py: 512x512 bilinear, then >0.
def metric_mask(path):
    return np.asarray(Image.open(path).convert('L').resize((512,512), Image.Resampling.BILINEAR)) > 0

# Orobix retina-unet preprocessing, independently expressed.
# Global standardization cancels under the subsequent per-image min/max affine map.
# No train/validation/test population statistics are needed.
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
GAMMA_TABLE=np.asarray([255*(v/255.)**(1/1.2) for v in range(256)],dtype=np.uint8)
def retina_channel(rgb):
    gray=rgb.astype(np.float64)@np.array([.299,.587,.114])
    gray=(255*(gray-gray.min())/max(float(gray.max()-gray.min()),1e-8)).clip(0,255).astype(np.uint8)
    return cv2.LUT(clahe.apply(gray),GAMMA_TABLE)

def load_image(path):
    rgb=np.asarray(Image.open(path).convert('RGB').resize((512,512),Image.Resampling.BILINEAR))
    aux=retina_channel(rgb)
    return aux[:,:,None]

train_paths = sorted((ROOT/'train/image').glob('*.jpg'), key=lambda p:int(p.stem))
test_paths = sorted((ROOT/'test/image').glob('*.jpg'), key=lambda p:int(p.stem))
assert len(train_paths)==20 and len(test_paths)==20
images = [load_image(p) for p in train_paths]
masks = [metric_mask(ROOT/'train/label'/p.name).astype(np.uint8) for p in train_paths]
test_images = [load_image(p) for p in test_paths]
print('Training foreground fractions', [round(float(m.mean()),4) for m in masks], flush=True)

class Patches(Dataset):
    def __init__(self,indices):self.indices=list(map(int,indices))
    def __len__(self):return CFG['steps']*CFG['batch_size']
    def __getitem__(self,index):
        i=random.choice(self.indices);s=CFG['patch_size']
        # Uniform crops include retinal boundary and outside-FOV background.
        y=random.randint(0,512-s);x=random.randint(0,512-s)
        im=images[i][y:y+s,x:x+s].copy();mask=masks[i][y:y+s,x:x+s].copy()
        k=random.randrange(4);im=np.rot90(im,k);mask=np.rot90(mask,k)
        if random.random()<.5:im=im[:,::-1];mask=mask[:,::-1]
        im=im.astype(np.float32)/255.
        if random.random()<.5:im=np.clip(im*random.uniform(.9,1.1)+random.uniform(-.03,.03),0,1)
        return torch.from_numpy(np.ascontiguousarray(im.transpose(2,0,1)*2-1)),torch.from_numpy(mask.copy()[None].astype(np.float32))

class RetinaBlock(nn.Module):
    def __init__(self,a,b):
        super().__init__();self.layers=nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.ReLU(),nn.Dropout(.2),nn.Conv2d(b,b,3,padding=1),nn.ReLU())
    def forward(self,x):return self.layers(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__();self.enc=nn.ModuleList([RetinaBlock(1,32),RetinaBlock(32,64),RetinaBlock(64,128)])
        self.dec=nn.ModuleList([RetinaBlock(192,64),RetinaBlock(96,32)]);self.head=nn.Conv2d(32,1,1)
    def forward(self,x):
        skips=[]
        for i,block in enumerate(self.enc):
            if i:x=F.max_pool2d(x,2)
            x=block(x);skips.append(x)
        for block,skip in zip(self.dec,reversed(skips[:-1])):
            x=F.interpolate(x,size=skip.shape[-2:],mode='nearest');x=block(torch.cat([skip,x],1))
        return self.head(x)

def loss_fn(logits, targets):
    bce=F.binary_cross_entropy_with_logits(logits,targets)
    p=logits.float().sigmoid()
    dice=(2*(p*targets).sum((1,2,3))+1)/(p.sum((1,2,3))+targets.sum((1,2,3))+1)
    return .8*bce+.2*(1-dice.mean())

@torch.inference_mode()
def predict(model,ims,tta=True):
    model.eval(); result=[]
    for im in ims:
        x=torch.from_numpy(np.ascontiguousarray(im.transpose(2,0,1))).float().to(device)[None]/127.5-1
        total=torch.zeros((1,1,512,512),device=device)
        variants=[(k,flip) for k in range(4) for flip in (False,True)] if tta else [(0,False)]
        for k,flip in variants:
            z=torch.rot90(x,k,(-2,-1))
            if flip:z=z.flip(-1)
            with torch.autocast('cuda',dtype=torch.float16): p=model(z).float().sigmoid()
            if flip:p=p.flip(-1)
            total+=torch.rot90(p,-k,(-2,-1))
        result.append((total/len(variants))[0,0].cpu().numpy())
    return np.stack(result)

THRESHOLDS=np.arange(.2,.751,.025)
def evaluate(probs,truth):
    truth=np.asarray(truth,dtype=bool)
    values=[]
    for t in THRESHOLDS:
        pred=probs>t
        inter=(pred&truth).sum((1,2)); den=pred.sum((1,2))+truth.sum((1,2))
        values.append(float(np.mean((2*inter+1e-8)/(den+1e-8))))
    k=int(np.argmax(values))
    return values[k],float(THRESHOLDS[k]),values


# Fixed before any new competition submission: medians of existing four-fold
# best steps, scaled by20/15 to retain approximate image exposures.
FULL_CFG=dict(seeds=[20261102,20261103],stages=[dict(patch_size=48,batch_size=64,steps=8400,learning_rate=.001),dict(patch_size=96,batch_size=32,steps=3200,learning_rate=.0002)],selection='Existing four-fold Retina family; stage48 median6300 and stage96 median2400, exposure-scaled20/15; EMA chosen in advance',train_ids=list(range(1,21)),cv_proxy_source='neuliandanshi/2026task4-retina-refine',cv_proxy_dice=.8386973514292475)
print('FULL TRAINING CONFIG',json.dumps(FULL_CFG),flush=True)
seed_predictions=[];summaries=[]
for seed in FULL_CFG['seeds']:
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=UNet().to(device)
    for stage_number,stage in enumerate(FULL_CFG['stages']):
        CFG.update(stage);stage_seed=seed+100000*stage_number
        random.seed(stage_seed);np.random.seed(stage_seed);torch.manual_seed(stage_seed+20000)
        loader=DataLoader(Patches(range(20)),batch_size=CFG['batch_size'],num_workers=2,pin_memory=True,drop_last=True,persistent_workers=True,generator=torch.Generator().manual_seed(stage_seed+10000))
        optimizer=torch.optim.AdamW(model.parameters(),lr=CFG['learning_rate'],weight_decay=1e-4)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=CFG['steps'],eta_min=3e-5)
        scaler=torch.amp.GradScaler('cuda');ema=copy.deepcopy(model).eval()
        for p in ema.parameters():p.requires_grad_(False)
        stage_start=time.time();running=0.
        for step,(x,y) in enumerate(loader,1):
            model.train();x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss=loss_fn(model(x),y)
            assert torch.isfinite(loss)
            scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),5.);scaler.step(optimizer);scaler.update();schedule.step()
            with torch.no_grad():
                decay=min(.995,(1+step)/(10+step))
                for ep,p in zip(ema.parameters(),model.parameters()):ep.lerp_(p,1-decay)
                for eb,b in zip(ema.buffers(),model.buffers()):eb.copy_(b)
            running+=loss.item()
            if step%500==0 or step==CFG['steps']:
                interval=500 if step%500==0 else step%500
                print('FULL TRAIN',json.dumps(dict(seed=seed,stage=stage_number,step=step,steps=CFG['steps'],loss=running/interval,seconds=time.time()-stage_start)),flush=True);running=0.
            if step>=CFG['steps']:break
        model.load_state_dict(ema.state_dict());summaries.append(dict(seed=seed,stage=stage_number,steps=CFG['steps'],seconds=time.time()-stage_start))
        del loader,optimizer,schedule,scaler,ema;torch.cuda.empty_cache()
    model.eval();p=predict(model,test_images);assert p.shape==(20,512,512) and np.isfinite(p).all();seed_predictions.append(p)
    torch.save(dict(model=model.state_dict(),config=FULL_CFG,seed=seed,train_ids=list(range(1,21))),OUT/f'seed{seed}.pt')
    del model;torch.cuda.empty_cache()
mean=np.mean(seed_predictions,axis=0,dtype=np.float32)
np.savez_compressed(OUT/'test_probabilities.npz',test=mean.astype('float16'),seed_test=np.stack(seed_predictions).astype('float16'),ids=np.arange(1,21))
report=dict(config=FULL_CFG,stages=summaries,seeds=FULL_CFG['seeds'],seconds=time.time()-START,gpu=torch.cuda.get_device_name(0),torch=torch.__version__)
(OUT/'metrics.json').write_text(json.dumps(report,indent=2));print('FULL TRAIN RESULT',json.dumps(report),flush=True)
