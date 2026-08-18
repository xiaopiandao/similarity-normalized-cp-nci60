"""Compare local calibration baselines and evaluate high-confidence screening."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
try:
 from scripts.evaluate_phase1a_conformal import EvaluationData,finite_sample_quantile,interval_metrics,nearest_fit_similarity,similarity_bin_metrics
 from scripts.evaluate_phase1b_similarity import fit_similarity_scale,similarity_normalized_intervals,standard_intervals
except ModuleNotFoundError:
 from evaluate_phase1a_conformal import EvaluationData,finite_sample_quantile,interval_metrics,nearest_fit_similarity,similarity_bin_metrics
 from evaluate_phase1b_similarity import fit_similarity_scale,similarity_normalized_intervals,standard_intervals

BINS=np.asarray([0,.4,.5,.6,.7,.8,1.000001]); K=512; ALPHA=.1
def sim(out,label,data,fit,ids):
 p=out/f'nearest_fit_tanimoto_{label}.csv'
 if p.exists(): return pd.read_csv(p).nearest_fit_tanimoto.to_numpy(float)
 v=nearest_fit_similarity(data.x[data.rows(fit)],data.x[data.rows(ids)]); pd.DataFrame({'nsc':ids,'nearest_fit_tanimoto':v}).to_csv(p,index=False); return v
def mondrian(cal_y,cal_p,cal_s,test_p,test_s):
 res=np.abs(cal_y-cal_p); base=np.asarray([finite_sample_quantile(res[:,j],ALPHA) for j in range(60)]); lo=np.empty_like(test_p); hi=np.empty_like(test_p); fallback=0
 cg=np.digitize(cal_s,BINS[1:-1]); tg=np.digitize(test_s,BINS[1:-1])
 for g in range(6):
  ix=np.where(tg==g)[0]; ci=np.where(cg==g)[0]
  if len(ci)<100: q=base; fallback+=len(ix)
  else: q=np.asarray([finite_sample_quantile(res[ci,j],ALPHA) for j in range(60)])
  lo[ix]=test_p[ix]-q; hi[ix]=test_p[ix]+q
 return lo,hi,fallback
def local_knn(cal_y,cal_p,cal_x,test_p,test_x):
 # fixed K=512; test-specific calibration neighbors, no response-based tuning.
 cc=np.asarray(cal_x.sum(1)).ravel(); tc=np.asarray(test_x.sum(1)).ravel(); res=np.abs(cal_y-cal_p); lo=np.empty_like(test_p); hi=np.empty_like(test_p)
 for st in range(0,len(test_p),32):
  en=min(st+32,len(test_p)); inter=(test_x[st:en]@cal_x.T).toarray(); uni=tc[st:en,None]+cc[None]-inter; s=np.divide(inter,uni,out=np.zeros_like(inter),where=uni>0); ix=np.argpartition(s,-K,axis=1)[:,-K:]
  for b,row in enumerate(ix):
   scores=res[row]
   q=np.asarray([finite_sample_quantile(scores[:,j],ALPHA) for j in range(60)])
   lo[st+b]=test_p[st+b]-q; hi[st+b]=test_p[st+b]+q
 return lo,hi
def screening(truth,pred,lower,method,simv):
 # Cell-level strong activity: confirmed only when lower confidence bound > 6.
 obs=np.isfinite(truth); selected=(lower>6)&obs; potent=(truth>=6)&obs; n=int(selected.sum()); tp=int((selected&potent).sum())
 # molecule-level broad spectrum criterion: >=3 selected/true potent cell lines.
 sm=selected.sum(1)>=3; tm=potent.sum(1)>=3
 cell_total=int(potent.sum()); molecule_total=int(tm.sum())
 low_cell=selected[simv<.4]; low_molecule=sm[simv<.4]
 return {
  'method':method,'level':'cell','selected':n,'precision':tp/n if n else np.nan,
  'recall':tp/cell_total if cell_total else np.nan,'false_discovery_rate':1-tp/n if n else np.nan,
  'low_similarity_precision':((selected&potent)[simv<.4].sum()/low_cell.sum()) if low_cell.sum() else np.nan,
 },{
  'method':method,'level':'molecule_ge3','selected':int(sm.sum()),
  'precision':float((sm&tm).sum()/sm.sum()) if sm.sum() else np.nan,
  'recall':float((sm&tm).sum()/molecule_total) if molecule_total else np.nan,
  'false_discovery_rate':float(1-(sm&tm).sum()/sm.sum()) if sm.sum() else np.nan,
  'low_similarity_precision':float((sm&tm)[simv<.4].sum()/low_molecule.sum()) if low_molecule.sum() else np.nan,
 }
def one(root,data,fam,seed):
 d=root/'data'/'splits'/'phase1a'/f'{fam}_seed_{seed}'; src=root/'results'/'phase1a'/f'{fam}_seed_{seed}'; out=root/'results'/'phase3'/f'{fam}_seed_{seed}'; out.mkdir(parents=True,exist_ok=True); z=np.load(src/'mlp_predictions.npz')
 ids={x:z[f'nsc_{x}'].astype('int64') for x in ('source_cal','valid','test')}; pred={x:z[f'pred_{x}'].astype(float) for x in ids}; y={x:data.truth(ids[x]) for x in ids}; fit=pd.read_csv(d/'fit_nsc.csv').nsc.to_numpy('int64')
 ss={x:sim(out,x,data,fit,ids[x]) for x in ids}; l1,h1=standard_intervals(y['source_cal'],pred['source_cal'],pred['test'],ALPHA); l3,h3,fb=mondrian(y['source_cal'],pred['source_cal'],ss['source_cal'],pred['test'],ss['test']); l4,h4=local_knn(y['source_cal'],pred['source_cal'],data.x[data.rows(ids['source_cal'])],pred['test'],data.x[data.rows(ids['test'])]); smodel,_,diag=fit_similarity_scale(ss['valid'],y['valid'],pred['valid']); l5,h5,_,_=similarity_normalized_intervals(y['source_cal'],pred['source_cal'],ss['source_cal'],ss['test'],smodel,pred['test'],ALPHA)
 rows=[]; bins=[]; screens=[]
 for name,lo,hi in [('M1_global',l1,h1),('M3_mondrian',l3,h3),('M4_knn512',l4,h4),('M5_similarity_normalized',l5,h5)]:
  rows.append({'family':fam,'seed':seed,'method':name,**interval_metrics(y['test'],lo,hi,'marginal')}); bins += [{'family':fam,'seed':seed,**r} for r in similarity_bin_metrics(ss['test'],y['test'],lo,hi,name)]; screens += [dict(family=fam,seed=seed,**r) for r in screening(y['test'],pred['test'],lo,name,ss['test'])]
 # point threshold as a screening comparator.
 screens += [dict(family=fam,seed=seed,**r) for r in screening(y['test'],pred['test'],pred['test'], 'Point_prediction_threshold',ss['test'])]
 (out/'detail.json').write_text(json.dumps({'mondrian_test_fallback_compounds':fb,'scale':diag,'knn_k':K},indent=2),encoding='utf-8')
 return rows,bins,screens
def main():
 a=argparse.ArgumentParser();a.add_argument('--root',default='.');a.add_argument('--seeds',default='1,2,3,4,5');args=a.parse_args(); root=Path(args.root);data=EvaluationData(root); rows=[];bins=[];screens=[]
 for f in ('random','scaffold','leader_cluster'):
  for s in map(int,args.seeds.split(',')):
   r,b,sc=one(root,data,f,s);rows+=r;bins+=b;screens+=sc;print(f'completed {f} seed={s}',flush=True)
 out=root/'results'/'phase3';out.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(out/'local_calibration_summary.csv',index=False);pd.DataFrame(bins).to_csv(out/'local_calibration_by_tanimoto.csv',index=False);pd.DataFrame(screens).to_csv(out/'screening_utility.csv',index=False)
if __name__=='__main__':main()

