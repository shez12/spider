"""Compare training-style truncated context with full recurrent inference."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from experiments.mustard_simtoolreal.train_bc import VENDOR
from rl_games.algos_torch import model_builder


def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 device='cuda:0';old=torch.load(a.checkpoint,map_location=device,weights_only=False);d=np.load(a.dataset)
 x=torch.tensor(d['obs'],device=device);y=torch.tensor(d['actions'],device=device);n,t,_=x.shape
 mask=torch.arange(t,device=device)[None]<torch.tensor(d['valid_lengths'],device=device)[:,None]
 model=model_builder.ModelBuilder().load(dict(network=old['network'],model=dict(name='continuous_a2c_logstd'))).build(dict(actions_num=29,input_shape=(141,),num_seqs=n,value_size=1,normalize_input=True,normalize_value=True,type='extra_param',coef_ids=torch.linspace(50,0,6,device=device),coef_id_idx=140)).to(device)
 model.load_state_dict(old['model']);model.eval()
 def zero():return tuple(s.to(device).zero_() for s in model.get_default_rnn_state())
 def forward(xx,h,coef):
  b,l,_=xx.shape;inp=torch.cat([xx,torch.full((b,l,1),coef,device=device)],-1).reshape(-1,141)
  mu,_,_,h=model.a2c_network(dict(obs=model.norm_obs(inp),rnn_states=h,seq_length=l))
  return mu.reshape(b,l,29),h
 report=dict(checkpoint=str(a.checkpoint),epoch=old['bc_epochs'],trajectories=n,valid_frames=int(mask.sum()),conditions={})
 with torch.no_grad():
  for coef in (0.,50.):
   full=[];truncated=[];h=zero()
   for start in range(0,t,16):
    end=min(start+16,t);pred,h=forward(x[:,start:end],h,coef);full.append(pred)
    z=zero()
    if start:_,z=forward(x[:,max(0,start-16):start],z,coef)
    pred,_=forward(x[:,start:end],z,coef);truncated.append(pred)
   full=torch.cat(full,1);truncated=torch.cat(truncated,1)
   def metrics(pred):
    error=pred-y;v=error[mask]
    delta=.0025*((pred.clamp(-1,1)-y)[:,:,:7]*mask[:,:,None]).sum(1)
    return dict(balanced_mse=float(.5*(v[:,:7].square().mean()+v[:,7:].square().mean())),arm_action_rmse=float(v[:,:7].square().mean().sqrt()),hand_action_rmse=float(v[:,7:].square().mean().sqrt()),nominal_arm_drift_max_rad=float(delta.abs().max()),nominal_arm_drift_median_max_joint_rad=float(delta.abs().max(-1).values.median()))
   report['conditions'][str(coef)]=dict(full_history=metrics(full),training_context=metrics(truncated),prediction_difference_rmse=float((full-truncated)[mask].square().mean().sqrt()))
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
