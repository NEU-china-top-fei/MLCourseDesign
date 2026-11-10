'实验计算与数据检查。'
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

CFG = dict(seed=20261031, folds=4, steps=667, batch_size=2,
           patch_size=512, learning_rate=0.00005, image_size=512, warm_start=True,
           eval_every=200, channels=[64,128,256,512], architecture='FIVES-pretrained 32M TransUNet', alignment_mode='legacy', pretrained_model='rexinshiminxiaozhou/blood_vessel_segmentation/transformers/default/1')
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
assert len(candidates)==1,candidates
ROOT=candidates[0]
assert (ROOT / 'train/image').is_dir(), ROOT
print('DATA ROOT', ROOT, flush=True)

# Match official segmentation_to_csv.py: 512x512 bilinear, then >0.
def metric_mask(path):
    return np.asarray(Image.open(path).convert('L').resize((512,512), Image.Resampling.BILINEAR)) > 0

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
def load_image(path):
    rgb = np.asarray(Image.open(path).convert('RGB').resize((512,512), Image.Resampling.BILINEAR))
    return rgb

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

"""Independent tensor-level implementation for the MIT-licensed public weights.
Model: rexinshiminxiaozhou/blood_vessel_segmentation/transformers/default/1.
Architecture concept: TransUNet (Chen et al., arXiv:2102.04306).
"""
class ConvPair(nn.Module):
    def __init__(self, a, b, middle=None):
        super().__init__();m=b if middle is None else middle
        self.layers=nn.Sequential(nn.Conv2d(a,m,3,padding=1,bias=False),nn.BatchNorm2d(m),nn.ReLU(inplace=True),
                                  nn.Conv2d(m,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.ReLU(inplace=True))
    def forward(self,x):return self.layers(x)

class GlobalMix(nn.Module):
    def __init__(self,dim=512,heads=8):
        super().__init__();self.heads=heads
        self.norm_attention=nn.LayerNorm(dim);self.qkv=nn.Linear(dim,3*dim,bias=False)
        self.attention_projection=nn.Linear(dim,dim);self.norm_mlp=nn.LayerNorm(dim)
        self.feedforward=nn.Sequential(nn.Linear(dim,2048),nn.GELU(),nn.Dropout(0.),nn.Linear(2048,dim),nn.Dropout(0.))
    def forward(self,x):
        b,n,d=x.shape
        q,k,v=self.qkv(self.norm_attention(x)).view(b,n,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        z=F.scaled_dot_product_attention(q,k,v,dropout_p=0.).transpose(1,2).reshape(b,n,d)
        x=x+self.attention_projection(z)
        return x+self.feedforward(self.norm_mlp(x))

class FivesTransUNet(nn.Module):
    def __init__(self,image_size=512,align_mode='legacy'):
        super().__init__();self.align_mode=align_mode;self.image_size=image_size
        c=[64,128,256,512]
        self.encoder=nn.ModuleList([ConvPair(3,64)]+[ConvPair(c[i-1],c[i]) for i in range(1,4)])
        self.patch_projection=nn.Conv2d(512,512,2,stride=2)
        self.position=nn.Parameter(torch.empty(1,(image_size//16)**2+1,512));nn.init.normal_(self.position,std=.02)
        self.class_token=nn.Parameter(torch.empty(1,1,512));nn.init.normal_(self.class_token,std=.02)
        self.transformer=nn.ModuleList([GlobalMix() for _ in range(6)])
        self.context_conv=nn.Conv2d(512,512,3,padding=1)
        self.decoder=nn.ModuleList([ConvPair(c[i]+c[i-1],c[i-1],(c[i]+c[i-1])//2) for i in range(3,0,-1)])
        self.head=nn.Conv2d(64,1,1)
    def forward(self,x):
        skip=[]
        for i,block in enumerate(self.encoder):
            if i:x=F.max_pool2d(x,2)
            x=block(x);skip.append(x)
        feature=self.patch_projection(x);h,w=feature.shape[-2:];b=feature.shape[0]
        tokens=torch.cat([self.class_token.expand(b,-1,-1),feature.flatten(2).transpose(1,2)],dim=1)
        if self.position.shape[1]!=tokens.shape[1]:
            old=int((self.position.shape[1]-1)**.5)
            grid=self.position[:,1:].transpose(1,2).reshape(1,512,old,old)
            grid=F.interpolate(grid,size=(h,w),mode='bilinear',align_corners=False).flatten(2).transpose(1,2)
            position=torch.cat([self.position[:,:1],grid],dim=1)
        else:position=self.position
        tokens=tokens+position
        for layer in self.transformer:tokens=layer(tokens)
        context=self.context_conv(tokens[:,1:].transpose(1,2).reshape(b,512,h,w))
        if self.align_mode=='aligned':
            context=F.interpolate(context,size=skip[-1].shape[-2:],mode='bilinear',align_corners=True)
        x=context
        for block,s in zip(self.decoder,reversed(skip[:-1])):
            x=F.interpolate(x,scale_factor=2.,mode='bilinear',align_corners=True)
            dy=s.shape[-2]-x.shape[-2];dx=s.shape[-1]-x.shape[-1]
            x=F.pad(x,[dx//2,dx-dx//2,dy//2,dy-dy//2])
            x=block(torch.cat([s,x],dim=1))
        return self.head(x)

    def import_public_state(self,state):
        mapped={};used=set();prefix=['inc','down1.1','down2.1','down3.1']
        fields={'norm_attention':'norm1','qkv':'attn.qkv','attention_projection':'attn.proj','norm_mlp':'norm2','feedforward':'mlp'}
        for name,target in self.state_dict().items():
            if name=='position':source='patch_embed.pos_embedding'
            elif name=='class_token':source='patch_embed.cls_token'
            elif name.startswith('encoder.'):
                parts=name.split('.');source=prefix[int(parts[1])]+'.double_conv.'+'.'.join(parts[3:])
            elif name.startswith('decoder.'):
                parts=name.split('.');source=f'up{int(parts[1])+1}.conv.double_conv.'+'.'.join(parts[3:])
            elif name.startswith('patch_projection.'):source='patch_embed.proj.0.'+name.split('.',1)[1]
            elif name.startswith('context_conv.'):source='transformer_conv.'+name.split('.',1)[1]
            elif name.startswith('head.'):source='outc.conv.'+name.split('.',1)[1]
            elif name.startswith('transformer.'):
                parts=name.split('.');source='transformer_encoders.'+parts[1]+'.'+fields[parts[2]]+'.'+'.'.join(parts[3:])
            else:raise ValueError(name)
            assert source in state,(name,source)
            assert state[source].shape==target.shape,(name,source,state[source].shape,target.shape)
            mapped[name]=state[source];used.add(source)
        assert used==set(state),'Unmapped external checkpoint tensors'
        self.load_state_dict(mapped,strict=True)
        return len(mapped)

class UNet(FivesTransUNet):
    def __init__(self):
        super().__init__(image_size=512,align_mode=CFG['alignment_mode'])

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

THRESHOLDS=np.arange(.1,.851,.025)
def evaluate(probs,truth):
    truth=np.asarray(truth,dtype=bool)
    values=[]
    for t in THRESHOLDS:
        pred=probs>t
        inter=(pred&truth).sum((1,2)); den=pred.sum((1,2))+truth.sum((1,2))
        values.append(float(np.mean((2*inter+1e-8)/(den+1e-8))))
    k=int(np.argmax(values))
    return values[k],float(THRESHOLDS[k]),values


FULL_CFG={'name': 'FIVES family full20 fixed refit', 'kind': 'existing_selected_family_refit', 'priority': 2, 'seeds': [20261130, 20261131], 'train_ids': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 'steps': 667, 'schedule_steps': 1067, 'batch_size': 2, 'alignment': 'legacy', 'pretrained_model': 'rexinshiminxiaozhou/blood_vessel_segmentation/transformers/default/1', 'encoder_transformer_lr': 1e-05, 'decoder_lr': 5e-05, 'weight_decay': 0.0001, 'cosine_eta_min': 1e-06, 'ema_decay': 'min(.995,(1+step)/(10+step)); copy buffers as original', 'seed_output': 'fixed mean of rawD4 and EMA D4, then mean across2 seeds', 'family_weight': 0.125, 'threshold': 0.4500000000000001, 'blend_policy': 'replace only oldFIVES leaf; retain current full20 context and Retina; preserve context-stage and FIVES-stage float16 boundaries'}

weight_paths=list(Path('/kaggle/input').rglob('best_model.pth'));assert len(weight_paths)==1,weight_paths;weight_path=weight_paths[0]
PRETRAINED=torch.load(weight_path,map_location='cpu',weights_only=True)
print('FIVES FULL CONFIG',json.dumps(FULL_CFG),flush=True);seed_predictions=[];raw_predictions=[];ema_predictions=[];summary=[]
for seed in FULL_CFG['seeds']:
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);model=UNet().to(device);assert model.import_public_state(PRETRAINED)==158
    for module in model.modules():
        if isinstance(module,nn.BatchNorm2d):module.momentum=.01
    
    slow=[];fast=[]
    for name,parameter in model.named_parameters():
        (slow if name.startswith(('encoder.','patch_projection.','transformer.')) or name in ['position','class_token'] else fast).append(parameter)
    optimizer=torch.optim.AdamW([dict(params=slow,lr=1e-5),dict(params=fast,lr=5e-5)],weight_decay=1e-4);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=1067,eta_min=1e-6);scaler=torch.amp.GradScaler('cuda');ema=copy.deepcopy(model).eval()
    for parameter in ema.parameters():parameter.requires_grad_(False)
    torch.manual_seed(seed+20000);loader=DataLoader(Patches(range(20)),batch_size=2,num_workers=2,pin_memory=True,drop_last=True,persistent_workers=True,generator=torch.Generator().manual_seed(seed+10000));running=0.;seed_start=time.time()
    print('FIVES FULL SEED',seed,flush=True)
    for step,(x,y) in enumerate(loader,1):
        model.train();x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):loss=loss_fn(model(x),y)
        assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),5.);scaler.step(optimizer);scaler.update();schedule.step()
        with torch.no_grad():
            decay=min(.995,(1+step)/(10+step))
            for ep,parameter in zip(ema.parameters(),model.parameters()):ep.lerp_(parameter,1-decay)
            for eb,buffer in zip(ema.buffers(),model.buffers()):eb.copy_(buffer)
        running+=loss.item()
        if step%100==0 or step==667:
            count=100 if step%100==0 else step%100
            print('FIVES FULL TRAIN',json.dumps(dict(seed=seed,step=step,steps=667,loss=running/count,lr=schedule.get_last_lr(),seconds=time.time()-seed_start)),flush=True);running=0.
            assert time.time()-START<1080,'Predeclared full20 resource cap'
        if step>=667:break
    model.eval();raw=predict(model,test_images);ep=predict(ema,test_images);assert np.isfinite(raw).all() and np.isfinite(ep).all();p=(raw+ep)/2;seed_predictions.append(p);raw_predictions.append(raw);ema_predictions.append(ep)
    ckpath=OUT/f'seed{seed}.pt';torch.save(dict(raw_model=model.state_dict(),ema_model=ema.state_dict(),config=FULL_CFG,seed=seed,train_ids=list(range(1,21)),step=667),ckpath)
    summary.append(dict(seed=seed,steps=667,seconds=time.time()-seed_start,final_lr=schedule.get_last_lr()));print('FIVES SEED COMPLETE',json.dumps(summary[-1]),flush=True)
    del model,ema,loader,optimizer,schedule,scaler;torch.cuda.empty_cache()
p=np.mean(seed_predictions,axis=0,dtype=np.float32);np.savez_compressed(OUT/'test_probabilities.npz',test=p.astype('float16'),seed_test=np.stack(seed_predictions).astype('float16'),raw_test=np.stack(raw_predictions).astype('float16'),ema_test=np.stack(ema_predictions).astype('float16'),ids=np.arange(1,21))
report=dict(config=FULL_CFG,seeds=summary,seconds=time.time()-START,gpu=torch.cuda.get_device_name(0),torch=torch.__version__)
(OUT/'metrics.json').write_text(json.dumps(report,indent=2));print('FIVES FULL RESULT',json.dumps(report),flush=True)
