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

CFG = dict(seed=20261031, folds=4, steps=2667, batch_size=4,
           patch_size=512, learning_rate=0.0008, image_size=512, warm_start=False,
           eval_every=500, channels=[16,32,64,128], architecture='CNN token context UNet', context_kind='local')
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
        im=images[i].copy(); mask=masks[i].copy()
        k=random.randrange(4);im=np.rot90(im,k);mask=np.rot90(mask,k)
        if random.random()<.5:im=im[:,::-1];mask=mask[:,::-1]
        if random.random()<.5:
            mat=cv2.getRotationMatrix2D((256,256),random.uniform(-30,30),random.uniform(.85,1.15))
            mat[:,2]+=np.random.uniform(-24,24,size=2)
            im=cv2.warpAffine(im,mat,(512,512),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            mask=cv2.warpAffine(mask,mat,(512,512),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
        im=im.astype(np.float32)/255.
        if random.random()<.8:
            im=np.clip(im*random.uniform(.85,1.15)+random.uniform(-.06,.06),0,1)
            im=np.power(im,random.uniform(.8,1.25))
        if random.random()<.25:
            im=np.clip(im+np.random.normal(0,.012,im.shape).astype(np.float32),0,1)
        return torch.from_numpy(np.ascontiguousarray(im.transpose(2,0,1)*2-1)), torch.from_numpy(mask.copy()[None].astype(np.float32))

"""Independent implementation of CNN + token-context + U-Net ideas.
Reference: Chen et al., TransUNet, arXiv:2102.04306.
The submitted competition code does not copy the unlicensed user-linked source.
"""
class ResidualConv(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.SiLU(),
                                  nn.Conv2d(b,b,3,padding=1,bias=False),nn.BatchNorm2d(b))
        self.shortcut=nn.Identity() if a==b else nn.Conv2d(a,b,1,bias=False)
    def forward(self,x):return F.silu(self.layers(x)+self.shortcut(x))

class ContextBlock(nn.Module):
    def __init__(self, dim, heads, global_attention):
        super().__init__();self.heads=heads;self.dim=dim;self.global_attention=global_attention
        self.norm1=nn.LayerNorm(dim);self.qkv=nn.Linear(dim,3*dim,bias=False);self.proj=nn.Linear(dim,dim)
        self.norm2=nn.LayerNorm(dim)
        self.ffn=nn.Sequential(nn.Linear(dim,2*dim),nn.GELU(),nn.Dropout(.1),nn.Linear(2*dim,dim),nn.Dropout(.1))
        self.local=nn.Conv2d(dim,dim,3,padding=1,groups=dim,bias=False) if not global_attention else None
    def forward(self,x,h,w):
        b,n,d=x.shape;v=self.norm1(x)
        q,k,value=self.qkv(v).view(b,n,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        if self.global_attention:
            mixed=F.scaled_dot_product_attention(q,k,value,dropout_p=0.)
            mixed=mixed.transpose(1,2).reshape(b,n,d)
        else:
            mixed=(q.sigmoid()*k.tanh()*value).transpose(1,2).reshape(b,n,d)
            mixed=self.local(mixed.transpose(1,2).reshape(b,d,h,w)).flatten(2).transpose(1,2)
        x=x+self.proj(mixed)
        return x+self.ffn(self.norm2(x))

class UNet(nn.Module):
    def __init__(self):
        super().__init__();c=[16,32,64,128];d=128
        self.encoder=nn.ModuleList([ResidualConv(4,c[0])]+[ResidualConv(c[i-1],c[i]) for i in range(1,4)])
        self.tokenize=nn.Conv2d(c[-1],d,2,stride=2)
        self.position=nn.Parameter(torch.empty(1,d,32,32));nn.init.trunc_normal_(self.position,std=.02)
        use_attention=CFG['context_kind']=='attention'
        self.context=nn.ModuleList([ContextBlock(d,4,use_attention) for _ in range(2)])
        self.context_out=nn.Conv2d(d,c[-1],3,padding=1)
        self.context_weight=nn.Parameter(torch.tensor(-2.0))
        self.decoder=nn.ModuleList([ResidualConv(c[i]+c[i-1],c[i-1]) for i in range(3,0,-1)])
        self.head=nn.Conv2d(c[0],1,1)
    def forward(self,x):
        skips=[]
        for i,block in enumerate(self.encoder):
            if i:x=F.max_pool2d(x,2)
            x=block(x);skips.append(x)
        t=self.tokenize(x);h,w=t.shape[-2:]
        position=F.interpolate(self.position,size=(h,w),mode='bilinear',align_corners=False)
        tokens=(t+position).flatten(2).transpose(1,2)
        for block in self.context:tokens=block(tokens,h,w)
        t=tokens.transpose(1,2).reshape(x.shape[0],-1,h,w)
        # Restore the actual CNN bottleneck grid before using any skip connection.
        context=F.interpolate(self.context_out(t),size=x.shape[-2:],mode='bilinear',align_corners=False)
        x=x+self.context_weight.sigmoid()*context
        for block,skip in zip(self.decoder,reversed(skips[:-1])):
            x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            x=block(torch.cat([x,skip],dim=1))
        return self.head(x)

def loss_fn(logits, targets):
    bce=F.binary_cross_entropy_with_logits(logits,targets)
    p=logits.float().sigmoid()
    dice=(2*(p*targets).sum((1,2,3))+1)/(p.sum((1,2,3))+targets.sum((1,2,3))+1)
    return .6*bce+.4*(1-dice.mean())

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


FULL_CFG=dict(seeds=[20261110,20261111],steps=2667,schedule_steps=3333,learning_rate=.0008,batch_size=4,variant='raw',train_ids=list(range(1,21)),selection='Original CV selected raw in all4 folds; median best step2000 of2500, scaled20/15 to stop2667 on cosine horizon3333',cv_family_dice=.8390947133745239)
print('CONTEXT FULL CONFIG',json.dumps(FULL_CFG),flush=True)
seed_predictions=[];summary=[]
for seed in FULL_CFG['seeds']:
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);model=UNet().to(device)
    loader=DataLoader(Patches(range(20)),batch_size=4,num_workers=2,pin_memory=True,drop_last=True,persistent_workers=True,generator=torch.Generator().manual_seed(seed+10000))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0008,weight_decay=1e-4);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=3333,eta_min=3e-5);scaler=torch.amp.GradScaler('cuda');torch.manual_seed(seed+20000)
    running=0.;seed_start=time.time();print('CONTEXT FULL SEED',seed,'PARAMETERS',sum(p.numel() for p in model.parameters()),flush=True)
    for step,(x,y) in enumerate(loader,1):
        model.train();x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):loss=loss_fn(model(x),y)
        assert torch.isfinite(loss)
        scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),5.);scaler.step(optimizer);scaler.update();schedule.step();running+=loss.item()
        if step%200==0 or step==2667:
            count=200 if step%200==0 else step%200
            print('CONTEXT FULL TRAIN',json.dumps(dict(seed=seed,step=step,steps=2667,loss=running/count,lr=schedule.get_last_lr()[0],seconds=time.time()-seed_start)),flush=True);running=0.
        if step>=2667:break
    model.eval();p=predict(model,test_images);assert p.shape==(20,512,512) and np.isfinite(p).all();seed_predictions.append(p)
    torch.save(dict(model=model.state_dict(),config=CFG,full_config=FULL_CFG,seed=seed,train_ids=list(range(1,21)),step=2667),OUT/f'seed{seed}.pt');summary.append(dict(seed=seed,steps=2667,seconds=time.time()-seed_start,final_lr=schedule.get_last_lr()[0]))
    del loader,optimizer,schedule,scaler,model;torch.cuda.empty_cache()
p=np.mean(seed_predictions,axis=0,dtype=np.float32);np.savez_compressed(OUT/'test_probabilities.npz',test=p.astype('float16'),seed_test=np.stack(seed_predictions).astype('float16'),ids=np.arange(1,21))
report=dict(config=FULL_CFG,seeds=summary,seconds=time.time()-START,gpu=torch.cuda.get_device_name(0),torch=torch.__version__)
(OUT/'metrics.json').write_text(json.dumps(report,indent=2));print('CONTEXT FULL RESULT',json.dumps(report),flush=True)
