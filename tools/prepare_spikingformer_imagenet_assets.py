#!/usr/bin/env python3
"""从官方 Spikingformer CPU(torch backend) 导出一个 ImageNet 样例的黄金轨迹。"""
from __future__ import annotations
import argparse, csv, sys
from functools import partial
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

def fold(conv, bn):
    w=conv.weight.detach().cpu(); scale=bn.weight.detach().cpu()/torch.sqrt(bn.running_var.detach().cpu()+bn.eps)
    conv_bias = torch.zeros_like(bn.running_mean) if conv.bias is None else conv.bias.detach().cpu()
    b=(conv_bias-bn.running_mean.detach().cpu())*scale+bn.bias.detach().cpu()
    return (w*scale.reshape(-1,*([1]*(w.ndim-1)))).numpy(),b.numpy()

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--image',type=Path,required=True);p.add_argument('--npz',type=Path,required=True);p.add_argument('--input-csv',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.repo/'imagenet')); import model
    m=model.vit_snn(img_size_h=224,img_size_w=224,patch_size=16,embed_dims=768,num_heads=8,mlp_ratios=4,in_channels=3,num_classes=1000,qkv_bias=False,norm_layer=partial(torch.nn.LayerNorm,eps=1e-6),depths=8,sr_ratios=1,T=4).eval()
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)['state_dict'];m.load_state_dict({k.removeprefix('module.'):v for k,v in ck.items()},strict=True)
    trace={}; hooks=[]
    def hook(name): return lambda _,__,out: trace.__setitem__(name,out.detach().cpu().numpy())
    for name,module in m.named_modules():
        # LIF/BN 给出每个脉冲与分支投影，Transformer block 给出两次模拟残差后的
        # multi-valued activation；均以官方 forward 的逐 timestep 输出为准。
        if name.endswith(('_lif','_bn')) or name == 'head' or name.startswith('block.') and name.count('.') == 1:
            hooks.append(module.register_forward_hook(hook(name)))
    x=v2.Compose([v2.Resize(224,antialias=True),v2.CenterCrop(224),v2.ToImage(),v2.ToDtype(torch.float32,scale=True),v2.Normalize([.485,.456,.406],[.229,.224,.225])])(Image.open(a.image).convert('RGB')).unsqueeze(0)
    with torch.no_grad(): logits=m(x)
    for h in hooks:h.remove()
    arrays={'input_image':x.numpy(),'logits':logits.numpy(),'prediction':np.array([logits.argmax().item()],np.int64)}
    arrays.update({f'trace__{k}':v for k,v in trace.items()})
    for name,module in m.named_modules():
        if isinstance(module,(torch.nn.Conv1d,torch.nn.Conv2d,torch.nn.Linear)):
            arrays[f'weight__{name}']=module.weight.detach().cpu().numpy()
            if module.bias is not None: arrays[f'bias__{name}']=module.bias.detach().cpu().numpy()
    for name,module in m.named_modules():
        if name.endswith('_conv'):
            bn=dict(m.named_modules()).get(name[:-5]+'_bn')
            if isinstance(bn,(torch.nn.BatchNorm1d,torch.nn.BatchNorm2d)):
                arrays[f'fold_weight__{name}'],arrays[f'fold_bias__{name}']=fold(module,bn)
    spike=trace['patch_embed.proj1_lif']
    if set(np.unique(spike).tolist())-{0.,1.}:raise ValueError('boundary 不是 binary spike')
    a.input_csv.parent.mkdir(parents=True,exist_ok=True)
    with a.input_csv.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['generated_time','current_time','spike_id','timestep','layer_id','src_neuron','src_pe','src_router','dst_pe','dst_router','value'])
        for sid,(t,c,y,x0) in enumerate(np.argwhere(spike[:,0]!=0)): w.writerow([0,0,sid,int(t)+1,'input',((int(y)*spike.shape[3]+int(x0))*spike.shape[1]+int(c)),0,0,0,0,1])
    a.npz.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.npz,**arrays)
    print({'prediction':int(arrays['prediction'][0]),'boundary_shape':list(spike.shape),'boundary_spikes':int(np.count_nonzero(spike))})
if __name__=='__main__':main()
