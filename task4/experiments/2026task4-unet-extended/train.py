"""Retinal vessel segmentation. All training runs on Kaggle GPU.
No external data or pretrained vessel models are used.
"""
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

CFG = dict(seed=20261031, folds=4, steps=2800, batch_size=16,
           patch_size=192, learning_rate=0.0004, image_size=512, warm_start=True,
           eval_every=700, channels=[24, 48, 96, 192, 256])
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
roots = [p.parent.parent for p in Path('/kaggle/input').rglob('sample_submission.csv')
         if (p.parent / 'train' / 'image').is_dir()]
if roots:
    ROOT = roots[0] / 'segmentation' if (roots[0] / 'segmentation').exists() else roots[0]
else:
    candidates = [p.parent for p in Path('/kaggle/input').rglob('sample_submission.csv')
                  if (p.parent / 'train' / 'image').is_dir()]
    assert candidates, 'Competition data not attached'
    ROOT = candidates[0]
assert (ROOT / 'train/image').is_dir(), ROOT
print('DATA ROOT', ROOT, flush=True)

# Match official segmentation_to_csv.py: 512x512 bilinear, then >0.
def metric_mask(path):
    return np.asarray(Image.open(path).convert('L').resize((512,512), Image.Resampling.BILINEAR)) > 0

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
def load_image(path):
    rgb = np.asarray(Image.open(path).convert('RGB').resize((512,512), Image.Resampling.BILINEAR))
    contrast = clahe.apply(rgb[:,:,1])
    return np.concatenate([rgb, contrast[:,:,None]], axis=2)

train_paths = sorted((ROOT/'train/image').glob('*.jpg'), key=lambda p:int(p.stem))
test_paths = sorted((ROOT/'test/image').glob('*.jpg'), key=lambda p:int(p.stem))
assert len(train_paths)==20 and len(test_paths)==20
images = [load_image(p) for p in train_paths]
masks = [metric_mask(ROOT/'train/label'/p.name).astype(np.uint8) for p in train_paths]
test_images = [load_image(p) for p in test_paths]
print('Training foreground fractions', [round(float(m.mean()),4) for m in masks], flush=True)

class Patches(Dataset):
    def __init__(self, indices): self.indices=list(map(int,indices))
    def __len__(self): return CFG['steps']*CFG['batch_size']
    def __getitem__(self, index):
        i=random.choice(self.indices)
        # Crop slightly larger and rotate before taking a 192px center crop.
        s=CFG['patch_size']; extra=32; z=s+extra
        pad=48
        src=cv2.copyMakeBorder(images[i],pad,pad,pad,pad,cv2.BORDER_REFLECT_101)
        tgt=cv2.copyMakeBorder(masks[i],pad,pad,pad,pad,cv2.BORDER_REFLECT_101)
        y=random.randint(0,512+2*pad-z); x=random.randint(0,512+2*pad-z)
        im=src[y:y+z,x:x+z].copy(); mask=tgt[y:y+z,x:x+z].copy()
        if random.random()<0.7:
            mat=cv2.getRotationMatrix2D((z/2,z/2),random.uniform(-180,180),random.uniform(.85,1.2))
            im=cv2.warpAffine(im,mat,(z,z),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
            mask=cv2.warpAffine(mask,mat,(z,z),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REFLECT_101)
        im=im[extra//2:extra//2+s,extra//2:extra//2+s]
        mask=mask[extra//2:extra//2+s,extra//2:extra//2+s]
        if random.random()<.5: im=im[:,::-1]; mask=mask[:,::-1]
        if random.random()<.5: im=im[::-1]; mask=mask[::-1]
        im=im.astype(np.float32)/255.
        if random.random()<.8:
            im=np.clip(im*random.uniform(.85,1.15)+random.uniform(-.06,.06),0,1)
            im=np.power(im,random.uniform(.8,1.25))
        if random.random()<.25:
            im=np.clip(im+np.random.normal(0,.012,im.shape).astype(np.float32),0,1)
        return torch.from_numpy(np.ascontiguousarray(im.transpose(2,0,1)*2-1)), torch.from_numpy(mask.copy()[None].astype(np.float32))

class Block(nn.Module):
    def __init__(self,a,b):
        super().__init__()
        self.main=nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.ReLU(inplace=True),
                                nn.Conv2d(b,b,3,padding=1,bias=False),nn.BatchNorm2d(b))
        self.skip=nn.Conv2d(a,b,1,bias=False) if a!=b else nn.Identity()
    def forward(self,x): return F.relu(self.main(x)+self.skip(x),inplace=True)

class UNet(nn.Module):
    def __init__(self):
        super().__init__(); c=CFG['channels']
        self.enc=nn.ModuleList([Block(4,c[0])]+[Block(c[i-1],c[i]) for i in range(1,len(c))])
        self.dec=nn.ModuleList([Block(c[i]+c[i-1],c[i-1]) for i in range(len(c)-1,0,-1)])
        self.head=nn.Conv2d(c[0],1,1)
    def forward(self,x):
        skips=[]
        for i,b in enumerate(self.enc):
            if i:x=F.max_pool2d(x,2)
            x=b(x); skips.append(x)
        for b,skip in zip(self.dec,reversed(skips[:-1])):
            x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            x=b(torch.cat([x,skip],1))
        return self.head(x)

def loss_fn(logits, targets):
    bce=F.binary_cross_entropy_with_logits(logits,targets)
    p=logits.float().sigmoid()
    dice=(2*(p*targets).sum((1,2,3))+1)/(p.sum((1,2,3))+targets.sum((1,2,3))+1)
    return .5*bce+.5*(1-dice.mean())

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

order=np.random.RandomState(CFG['seed']).permutation(20)
val_groups=np.array_split(order,CFG['folds'])
fold_split={str(f):[train_paths[i].stem for i in ids] for f,ids in enumerate(val_groups)}
(OUT/'folds.json').write_text(json.dumps(fold_split,indent=2))
oof=np.zeros((20,512,512),np.float32)
test_sum=np.zeros((20,512,512),np.float32)
summary=[]; history=[]
for fold,val_ids in enumerate(val_groups):
    seed=CFG['seed']+fold*97
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train_ids=[i for i in range(20) if i not in val_ids]
    assert not set(train_ids).intersection(val_ids)
    print('FOLD',fold,'train',train_ids,'validation',val_ids.tolist(),flush=True)
    loader=DataLoader(Patches(train_ids),batch_size=CFG['batch_size'],num_workers=2,
                      pin_memory=True,drop_last=True,persistent_workers=True)
    model=UNet().to(device)
    checkpoints=list(Path('/kaggle/input').rglob(f'fold{fold}.pt'))
    assert checkpoints,'Baseline output must be attached'
    previous=torch.load(checkpoints[0],map_location='cpu',weights_only=False)
    assert previous['val_ids']==val_ids.tolist(),'Cross-validation split mismatch'
    model.load_state_dict(previous['model']); del previous
    optimizer=torch.optim.AdamW(model.parameters(),lr=CFG['learning_rate'],weight_decay=1e-4)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=CFG['steps'],eta_min=3e-5)
    scaler=torch.amp.GradScaler('cuda')
    ema=copy.deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    best=-1; running=0.; fold_start=time.time()
    initial=predict(model,[images[i] for i in val_ids])
    initial_score,_,_=evaluate(initial,[masks[i] for i in val_ids])
    print('INITIAL',fold,initial_score,flush=True)
    if CFG.get('warm_start'):
        best=initial_score
        torch.save(dict(model=model.state_dict(),config=CFG,fold=fold,val_ids=val_ids.tolist(),step=0,dice=best),OUT/f'fold{fold}.pt')
    for step,(x,y) in enumerate(loader,1):
        model.train(); x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16): loss=loss_fn(model(x),y)
        assert torch.isfinite(loss),'Nonfinite loss'
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(),5.)
        scaler.step(optimizer); scaler.update(); schedule.step()
        with torch.no_grad():
            decay=min(.995,(1+step)/(10+step))
            for ep,p in zip(ema.parameters(),model.parameters()):ep.lerp_(p,1-decay)
            for eb,b in zip(ema.buffers(),model.buffers()):eb.copy_(b)
        running+=loss.item()
        if step%100==0:
            print(f'fold={fold} step={step}/{CFG["steps"]} loss={running/100:.5f} seconds={time.time()-fold_start:.0f}',flush=True)
            running=0.
        if step%CFG['eval_every']==0 or step==CFG['steps']:
            for variant,eval_model in [('raw',model),('ema',ema)]:
                vp=predict(eval_model,[images[i] for i in val_ids])
                score,threshold,_=evaluate(vp,[masks[i] for i in val_ids])
                row=dict(fold=fold,step=step,variant=variant,dice=score,threshold=threshold)
                history.append(row); print('VALIDATION',json.dumps(row),flush=True)
                if score>best:
                    best=score
                    torch.save(dict(model=eval_model.state_dict(),config=CFG,fold=fold,
                                    val_ids=val_ids.tolist(),step=step,dice=score,variant=variant),OUT/f'fold{fold}.pt')
            (OUT/'history.json').write_text(json.dumps(history,indent=2))
        if step>=CFG['steps']:break
    checkpoint=torch.load(OUT/f'fold{fold}.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'])
    vp=predict(model,[images[i] for i in val_ids]); oof[val_ids]=vp
    tp=predict(model,test_images); test_sum+=tp/CFG['folds']
    np.savez_compressed(OUT/f'fold{fold}_probabilities.npz',val_ids=val_ids,oof=vp.astype(np.float16),test=tp.astype(np.float16))
    summary.append(dict(fold=fold,best_dice=best,best_step=checkpoint['step'],seconds=time.time()-fold_start))
    print('FOLD COMPLETE',json.dumps(summary[-1]),flush=True)
    del loader,optimizer,schedule,model,ema,checkpoint
    torch.cuda.empty_cache()

score,threshold,curve=evaluate(oof,masks)
np.savez_compressed(OUT/'probabilities.npz',oof=oof.astype(np.float16),test=test_sum.astype(np.float16),targets=np.asarray(masks),ids=np.arange(1,21))
report=dict(config=CFG,folds=summary,oof_dice=score,threshold=threshold,
            threshold_curve=[dict(threshold=float(t),dice=s) for t,s in zip(THRESHOLDS,curve)],
            seconds=time.time()-START,torch=torch.__version__,gpu=torch.cuda.get_device_name(0))
(OUT/'metrics.json').write_text(json.dumps(report,indent=2))

# Official format is row-major (C order), 1-based run starts.
def encode_rle(mask):
    pixels=np.asarray(mask,dtype=np.uint8).ravel(order='C')
    padded=np.concatenate([[0],pixels,[0]])
    runs=np.flatnonzero(padded[1:]!=padded[:-1])+1
    runs[1::2]-=runs[::2]
    return ' '.join(map(str,runs))

def decode_rle(value):
    pairs=np.array(list(map(int,value.split())),dtype=int).reshape(-1,2)
    flat=np.zeros(512*512,np.uint8)
    for start,length in pairs: flat[start-1:start-1+length]=1
    return flat.reshape(512,512)

sample=pd.read_csv(ROOT/'sample_submission.csv',dtype={'Id':str}).fillna('')
by_id={p.stem:test_sum[i]>threshold for i,p in enumerate(test_paths)}
assert set(sample.Id)==set(by_id)
mask_dir=OUT/'predicted_masks'; mask_dir.mkdir(exist_ok=True)
encoded=[]
for ident in sample.Id:
    m=by_id[ident]; rle=encode_rle(m)
    assert np.array_equal(decode_rle(rle),m)
    assert 0.01<float(m.mean())<0.4,(ident,float(m.mean()))
    Image.fromarray(m.astype(np.uint8)*255).save(mask_dir/f'{ident}.png')
    encoded.append(rle)
sample['Predicted']=encoded
assert sample.shape==(20,2) and list(sample.columns)==['Id','Predicted'] and sample.Id.is_unique
sample.to_csv(OUT/'submission.csv',index=False)
# Verify original converter's exact operation against our encoding.
for ident,rle in zip(sample.Id,encoded):
    original=metric_mask(mask_dir/f'{ident}.png')
    assert np.array_equal(decode_rle(rle),original)
print('RESULT',json.dumps(report),flush=True)
print('SUBMISSION',OUT/'submission.csv','rows',len(sample),flush=True)
