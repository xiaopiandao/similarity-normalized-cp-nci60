"""Train a transparent Mamba-inspired selective state-space profile model."""
from __future__ import annotations
import argparse, copy, json, os, random
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    from scripts.run_phase1a_baselines import Phase1Data, point_metrics
except ModuleNotFoundError:
    from run_phase1a_baselines import Phase1Data, point_metrics

THRESHOLDS=np.asarray([4.1,5.0,6.0,7.0,8.0],dtype=np.float32)

def cell_features(root:Path, cell_lines:list[str], n_components:int=32)->tuple[np.ndarray,np.ndarray,dict]:
    out=root/'data'/'features'/'cellminer'; f=out/f'rna_pca_{n_components}.npz'
    if f.exists():
        z=np.load(f,allow_pickle=True); saved=list(z['cell_lines'].astype(str))
        if saved==cell_lines: return z['features'].astype(np.float32),z['type_ids'].astype(np.int64),json.loads(str(z['metadata']))
    rna=pd.read_csv(root/'data'/'processed'/'cellminer'/'omics_rna.csv.gz',usecols=cell_lines)
    x=rna[cell_lines].apply(pd.to_numeric,errors='coerce').fillna(0).to_numpy(dtype=np.float32).T
    x=StandardScaler().fit_transform(x)
    pca=PCA(n_components=n_components,svd_solver='full',random_state=20260721)
    features=pca.fit_transform(x).astype(np.float32)
    types=[name.split(':',1)[0] for name in cell_lines]; type_map={v:i for i,v in enumerate(sorted(set(types)))}
    type_ids=np.asarray([type_map[v] for v in types],dtype=np.int64)
    meta={'source':'omics_rna.csv.gz','components':n_components,'explained_variance_ratio_sum':float(pca.explained_variance_ratio_.sum()),'type_map':type_map,'labels_used':False}
    np.savez_compressed(f,features=features,type_ids=type_ids,cell_lines=np.asarray(cell_lines),metadata=json.dumps(meta))
    return features,type_ids,meta

def set_seed(seed:int)->None:
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.use_deterministic_algorithms(True,warn_only=True)

def activity(y:np.ndarray)->np.ndarray:
    return np.digitize(y,THRESHOLDS,right=False).astype(np.int64)

def build_model(cell_dim:int,n_types:int):
    import torch
    from torch import nn
    class Block(nn.Module):
        def __init__(self,d:int):
            super().__init__(); self.norm=nn.LayerNorm(d); self.inp=nn.Linear(d,3*d); self.conv=nn.Conv1d(d,d,3,padding=1,groups=d); self.out=nn.Linear(d,d); self.ff=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,2*d),nn.GELU(),nn.Dropout(.1),nn.Linear(2*d,d))
        def forward(self,x):
            z=self.norm(x); value,decay,gate=self.inp(z).chunk(3,-1); value=self.conv(value.transpose(1,2)).transpose(1,2); h=torch.zeros_like(value[:,0]); ys=[]
            for t in range(value.shape[1]):
                a=torch.sigmoid(decay[:,t]); h=a*h+(1-a)*torch.tanh(value[:,t]); ys.append(torch.sigmoid(gate[:,t])*h)
            return x+self.out(torch.stack(ys,1))+self.ff(x)
    class Model(nn.Module):
        def __init__(self):
            super().__init__(); d=192; self.token=nn.Linear(64,d); self.blocks=nn.ModuleList([Block(d) for _ in range(4)]); self.chem=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,128),nn.GELU())
            self.cell=nn.Sequential(nn.Linear(cell_dim,96),nn.GELU(),nn.Linear(96,128)); self.cell_id=nn.Embedding(60,128); self.type_id=nn.Embedding(n_types,128)
            self.fuse=nn.Sequential(nn.Linear(384,192),nn.GELU(),nn.Dropout(.15),nn.Linear(192,96),nn.GELU())
            self.reg=nn.Linear(96,1); self.cls=nn.Linear(96,6)
        def forward(self,x,cell_x,type_ids):
            b=x.shape[0]; x=self.token(x.reshape(b,32,64));
            for block in self.blocks: x=block(x)
            chem=self.chem(x.mean(1)); idx=torch.arange(60,device=x.device); cell=self.cell(cell_x)+self.cell_id(idx)+self.type_id(type_ids)
            c=chem[:,None,:].expand(-1,60,-1); ce=cell[None].expand(b,-1,-1); f=self.fuse(torch.cat([c,ce,c*ce],-1)); return self.reg(f).squeeze(-1),self.cls(f)
    return Model()

def main()->None:
    p=argparse.ArgumentParser(); p.add_argument('--root',default='.'); p.add_argument('--family',default='leader_cluster'); p.add_argument('--seed',type=int,default=1); p.add_argument('--epochs',type=int,default=60); p.add_argument('--patience',type=int,default=10); p.add_argument('--batch-size',type=int,default=256); a=p.parse_args()
    import torch
    from torch import nn
    root=Path(a.root); set_seed(20000+a.seed); data=Phase1Data(root); split=root/'data'/'splits'/'phase1a'/f'{a.family}_seed_{a.seed}'; out=root/'results'/'phase2'/f'{a.family}_seed_{a.seed}'; out.mkdir(parents=True,exist_ok=True)
    loaded={k:data.load_part(split,k) for k in ('fit','source_cal','valid','test')}; _,xf,yf=loaded['fit']; mean=np.nanmean(yf,0).astype(np.float32); scale=np.nanstd(yf,0).astype(np.float32); scale[scale<1e-6]=1
    cell_x,type_ids,cell_meta=cell_features(root,data.cell_lines); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=build_model(cell_x.shape[1],int(type_ids.max()+1)).to(device); cell_t=torch.from_numpy(cell_x).to(device); type_t=torch.from_numpy(type_ids).to(device)
    y_norm=(yf-mean)/scale; mask=np.isfinite(yf); y_class=activity(np.nan_to_num(yf,nan=4.5)); counts=np.bincount(y_class[mask],minlength=6); weights=(counts.sum()/np.maximum(counts,1))**.5; weights=weights/weights.mean()
    ds=torch.utils.data.TensorDataset(torch.from_numpy(xf.toarray()).float(),torch.from_numpy(np.nan_to_num(y_norm)).float(),torch.from_numpy(mask).bool(),torch.from_numpy(y_class)); loader=torch.utils.data.DataLoader(ds,batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(20000+a.seed),pin_memory=device.type=='cuda')
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-5); ce=nn.CrossEntropyLoss(weight=torch.from_numpy(weights).float().to(device),reduction='none'); mt=torch.from_numpy(mean).to(device); st=torch.from_numpy(scale).to(device); vx=torch.from_numpy(loaded['valid'][1].toarray()).float().to(device); vy=loaded['valid'][2]
    best=float('inf'); best_state=None; stale=0; hist=[]
    for epoch in range(1,a.epochs+1):
        model.train(); loss_sum=0.; n=0
        for xb,yb,mb,cb in loader:
            xb,yb,mb,cb=[v.to(device,non_blocking=True) for v in (xb,yb,mb,cb)]; opt.zero_grad(set_to_none=True); pred,logits=model(xb,cell_t,type_t); mse=((pred-yb)[mb]**2).mean(); c_loss=ce(logits.reshape(-1,6),cb.reshape(-1)).reshape_as(mb)[mb].mean(); loss=mse+.20*c_loss; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); loss_sum+=float(loss)*int(mb.sum()); n+=int(mb.sum())
        model.eval();
        with torch.no_grad(): vp=(model(vx,cell_t,type_t)[0]*st+mt).cpu().numpy()
        rmse=point_metrics(vy,vp)[0]['rmse_macro']; hist.append({'epoch':epoch,'train_loss':loss_sum/n,'valid_rmse_macro':rmse}); print(f'epoch={epoch:03d} loss={loss_sum/n:.6f} valid_rmse={rmse:.6f}',flush=True)
        if rmse<best-1e-5: best=rmse; best_state=copy.deepcopy(model.state_dict()); stale=0
        else:
            stale+=1
            if stale>=a.patience: break
    model.load_state_dict(best_state); model.eval(); payload={'cell_lines':np.asarray(data.cell_lines,dtype='U')}; report={'model':'Mamba-inspired selective state-space multi-task profile model','family':a.family,'seed':a.seed,'device':str(device),'best_valid_rmse_macro':best,'epochs_completed':len(hist),'cell_features':cell_meta,'parts':{}}
    with torch.no_grad():
        for part in ('source_cal','valid','test'):
            ids,x,y=loaded[part]; pred=(model(torch.from_numpy(x.toarray()).float().to(device),cell_t,type_t)[0]*st+mt).cpu().numpy(); metrics,per=point_metrics(y,pred); report['parts'][part]=metrics; per.insert(0,'cell_line',data.cell_lines); per.to_csv(out/f'ssm_{part}_per_cell.csv',index=False); payload[f'nsc_{part}']=ids; payload[f'pred_{part}']=pred.astype(np.float32)
    np.savez_compressed(out/'ssm_profile_predictions.npz',**payload); pd.DataFrame(hist).to_csv(out/'ssm_history.csv',index=False); torch.save({'state_dict':best_state,'mean':mean,'scale':scale},out/'ssm_checkpoint.pt'); (out/'ssm_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__': main()

