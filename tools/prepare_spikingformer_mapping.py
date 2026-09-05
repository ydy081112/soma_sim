#!/usr/bin/env python3
"""由官方 Spikingformer 导出物生成 SOMA 的全图 spatial/SSA/local-state mapping。"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

def plan(h,w,k=1,s=1,p=0):
    patterns={}; ids=[]; bases=[]
    for y in range(h):
      for x in range(w):
        q=[]
        for ky in range(k):
          for kx in range(k):
            oy=y+p-ky; ox=x+p-kx
            if oy%s==0 and ox%s==0 and 0<=oy//s<(h+2*p-k)//s+1 and 0<=ox//s<(w+2*p-k)//s+1:q.append(((oy//s)*((w+2*p-k)//s+1)+ox//s,ky*k+kx))
        b=min((v[0] for v in q),default=0); sig=tuple((a-b,z) for a,z in q)
        if sig not in patterns:patterns[sig]=len(patterns)
        ids.append(patterns[sig]);bases.append(b)
    ptr=[0]; off=[]; wo=[]
    for sig in patterns:
      off += [a for a,_ in sig]; wo += [z for _,z in sig]; ptr.append(len(off))
    return [np.array(ids,np.int32),np.array(bases,np.int32),np.array(ptr,np.int32),np.array(off,np.int32),np.array(wo,np.int64)]

def item(lines,d):
    first=True
    for k,v in d.items():
      if isinstance(v,bool):v=str(v).lower()
      if isinstance(v,list):v='['+', '.join(map(str,v))+']'
      lines.append(('    - ' if first else '      ')+f'{k}: {v}');first=False

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--npz',type=Path,required=True);p.add_argument('--mapping',type=Path,required=True);p.add_argument('--max-neurons',type=int,default=1024);a=p.parse_args()
 src=np.load(a.source); out={}; layers=[]; conns=[]; routes=[]; core=0; maxn=a.max_neurons
 def add_layer(name,c,h,w,model='lif',threshold=1.,prefix=None):
  nonlocal core
  n=c*h*w; cnt=(n+maxn-1)//maxn; layers.append(dict(id=name,op='conv2d',pe=core//4,core=core%4,router=core//4,neurons=n,source_neurons=n,output_h=h,output_w=w,output_channels=c,aggregate_core_count=cnt,weight_prefix=prefix or name,neuron_model=model,leak=.5 if model=='lif' else 1,input_scale=.5 if model=='lif' else 1,threshold=threshold,reset='hard'));core+=cnt
 def add_input(name,c,h,w): layers.append(dict(id=name,op='input',pe=0,core=0,router=0,neurons=c*h*w,output_h=h,output_w=w,output_channels=c,virtual_input=True,direct_input=True))
 def spatial(source,target,weight,h,w,k=1,s=1,pad=0,channelwise=False):
  pref=target+'__spatial'; wt=np.asarray(weight,np.float32)
  if wt.ndim==4: wt=wt.transpose(1,2,3,0)
  elif wt.ndim==3:
   raw=wt[:,:,0];wt=raw.T.reshape(raw.shape[1],1,1,raw.shape[0])
  out[pref+'_weight']=wt.reshape(-1); x=plan(h,w,k,s,pad)
  for suf,v in zip(('_plan_pattern_id','_plan_dst_base','_pattern_ptr','_pattern_dst_offset','_pattern_weight_offset'),x):out[pref+suf]=v
  conns.append(dict(from_=source,to=target,type='spatial',hardware_type='spatial',weight_prefix=pref,channelwise=channelwise))
 def local(source,target):
  pref=target+'__'+source;out[pref+'_weight']=np.array([1],np.float32);conns.append(dict(from_=source,to=target,type='local_state_buffer',hardware_type='identity',weight_prefix=pref))
 def folded(name):return src['fold_weight__'+name],src['fold_bias__'+name]
 # tokenizer: post-boundary input -> maxpool/LIF -> ConvBN(local Cx state) chains.
 add_input('input',96,224,224); add_layer('pool1',96,112,112,'lif');
 out['pool1_bias']=np.zeros(96,np.float32); spatial('input','pool1',np.ones(9,np.float32),224,224,3,2,1,True)
 prev='pool1'; h=112
 for i,(conv,c) in enumerate((('patch_embed.proj1_conv',192),('patch_embed.proj2_conv',384),('patch_embed.proj3_conv',768),('patch_embed.proj4_conv',768)),1):
  an=f'token_conv{i}';add_layer(an,c,h,h,'multi_valued_state');w,b=folded(conv);out[an+'_bias']=b;spatial(prev,an,w,h,h,3,1,1)
  if i<4:
   lif=f'token_lif{i+1}';add_layer(lif,c,h,h,'lif');local(an,lif);prev=lif;h//=2
   pool=f'pool{i+1}';add_layer(pool,c,h,h,'lif');out[pool+'_bias']=np.zeros(c,np.float32);spatial(prev,pool,np.ones(9,np.float32),h*2,h*2,3,2,1,True);prev=pool
  else: prev=an
 # transformer blocks: all names are generated from topology, never inspected by C++ runtime.
 for bidx in range(8):
  base=f'b{bidx}'; proj=f'{base}_proj_lif';add_layer(proj,768,14,14,'lif');local(prev,proj)
  qkv=[]
  for kind in 'qkv':
   an=f'{base}_{kind}_pre';sp=f'{base}_{kind}';add_layer(an,768,14,14,'multi_valued_state');w,bb=folded(f'block.{bidx}.attn.{kind}_conv');out[an+'_bias']=bb;spatial(proj,an,w,14,14);add_layer(sp,768,14,14,'lif');local(an,sp);qkv.append(sp)
  att=f'{base}_attn';add_layer(att,768,14,14,'lif',.5);layers[-1].update(operator_type='timestep_spike_attention',attention_heads=8,attention_rows=196,attention_reduction=96,attention_output_layout='row_head_column',attention_scale=.125,attention_kind='qkv')
  for name,op in zip(qkv,('q','k','v')):conns.append(dict(from_=name,to=att,type='attention_operand',hardware_type='dense',weight_prefix=att,operand=op,operand_layout='row_head_reduction'))
  pa=f'{base}_attn_proj';add_layer(pa,768,14,14,'multi_valued_state');w,bb=folded(f'block.{bidx}.attn.proj_conv');out[pa+'_bias']=bb;spatial(att,pa,w,14,14)
  r1=f'{base}_res1';add_layer(r1,768,14,14,'multi_valued_state');local(prev,r1);local(pa,r1)
  m1=f'{base}_mlp1_lif';add_layer(m1,768,14,14,'lif');local(r1,m1)
  ma=f'{base}_mlp1';add_layer(ma,3072,14,14,'multi_valued_state');w,bb=folded(f'block.{bidx}.mlp.mlp1_conv');out[ma+'_bias']=bb;spatial(m1,ma,w,14,14)
  m2=f'{base}_mlp2_lif';add_layer(m2,3072,14,14,'lif');local(ma,m2)
  mb=f'{base}_mlp2';add_layer(mb,768,14,14,'multi_valued_state');w,bb=folded(f'block.{bidx}.mlp.mlp2_conv');out[mb+'_bias']=bb;spatial(m2,mb,w,14,14)
  prev=f'{base}_res2';add_layer(prev,768,14,14,'multi_valued_state');local(r1,prev);local(mb,prev)
 # YAML and weights: connection parser expects 'from', not Python keyword spelling.
 # 4 个 tokenizer stage 与每个 Transformer block 的 LIF pipeline 都使用
 # timestep buffer；flush 按 graph depth 留足传播/静默 state transition 帧。
 lines=['mapping:','  model: spikingformer_imagenet_official','  flush_timesteps: 100','  stream_input_records: true','  layers:']
 for x in layers:item(lines,x)
 lines.append('  connections:')
 for x in conns:
  x={'from':x.pop('from_'),**x};item(lines,x);routes.append((x['from'],x['to']))
 lines.append('  routes:')
 starts={x['id']:x['pe'] for x in layers}
 for f,t in routes:item(lines,{'from':f,'to':t,'routers':[starts[f],starts[t]]})
 a.mapping.parent.mkdir(parents=True,exist_ok=True);a.mapping.write_text('\n'.join(lines)+'\n')
 a.npz.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.npz,**out)
 print({'layers':len(layers),'connections':len(conns),'cores':core})
if __name__=='__main__':main()
