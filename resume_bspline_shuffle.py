"""Resume legacy zero-initialized joint B-spline runs with exact epoch shuffles.

The archived runner saved model/Adam but not CUDA RNG. This narrow adapter
replays its sole RNG consumer (one randperm per completed epoch) before the
first resumed epoch. It refuses a changed runner, nonzero/random model init,
a non-joint model, or unexpected RNG consumption before the first shuffle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch


def digest(tensor):
    return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()


class EpochShuffleReplay:
    def __init__(self, original, *, seed, views, completed_epochs, on_ready=None):
        self.original=original;self.seed=seed;self.views=views
        self.completed_epochs=completed_epochs;self.on_ready=on_ready;self.calls=0
        self.audit=None

    def __call__(self,n,*args,**kwargs):
        if self.calls:
            self.calls+=1
            return self.original(n,*args,**kwargs)
        if n!=self.views or args or set(kwargs)!={'device'}:
            raise ValueError('Unexpected sampler signature; refuse approximate RNG restoration')
        device=torch.device(kwargs['device'])
        reference=torch.Generator(device=device).manual_seed(self.seed)
        state=(torch.cuda.get_rng_state(device) if device.type=='cuda' else torch.get_rng_state())
        if not torch.equal(state,reference.get_state()):
            raise ValueError('Unexpected RNG consumption before the first epoch shuffle')
        for _ in range(self.completed_epochs):
            self.original(n,device=device)
            self.original(n,device=device,generator=reference)
        next_views=self.original(n,device=device)
        expected=self.original(n,device=device,generator=reference)
        state=(torch.cuda.get_rng_state(device) if device.type=='cuda' else torch.get_rng_state())
        if not torch.equal(next_views,expected) or not torch.equal(state,reference.get_state()):
            raise ValueError('Resumed view permutation or generator state failed exact replay')
        self.calls=1
        self.audit=dict(replayed_epochs=self.completed_epochs,views=n,seed=self.seed,
            first_resumed_permutation_sha256=digest(next_views),post_shuffle_rng_sha256=digest(state),
            reference_generator_permutation_and_state_identical=True)
        if self.on_ready:self.on_ready(self.audit)
        return next_views


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True)
    args=p.parse_args()
    from run_sinespin_calibration import sha256
    root=Path(__file__).resolve().parent
    exp=json.loads((args.out_dir/'experiment.json').read_text());recipe=exp['recipe']
    if ('update_scheme' in recipe or recipe.get('motion_model_config',{}).get('kind')!='cubic_bspline'
            or recipe['initialization']!='All B-spline coefficients exactly zero; circular nominal input P'):
        raise ValueError('This replay adapter only supports the audited zero-initialized joint B-spline')
    for name,expected in exp['source_sha256'].items():
        if sha256(root/name)!=expected:raise ValueError(f'Changed training source: {name}')
    source=(root/'run_sinespin_calibration.py').read_text()
    if source.count('torch.randperm(')!=1 or source.count('torch.manual_seed(')!=1:
        raise ValueError('The runner RNG protocol changed')
    checkpoint=torch.load(args.out_dir/'checkpoint.pt',map_location='cpu',weights_only=False)
    start=checkpoint['epoch']
    if not 0<start<recipe['epochs']:raise ValueError('Checkpoint is not a partial run')
    initial=torch.load(args.out_dir/'initial_model.pt',map_location='cpu',weights_only=True)
    if torch.count_nonzero(initial['raw_coefficients']).item()!=0:raise ValueError('Nonzero initialization')
    views=len(exp['train_views'])
    event_path=args.out_dir/f'resume_epoch{start:04d}_gpu{args.gpu}.json'
    if event_path.exists():raise FileExistsError(event_path)
    before_path=args.out_dir/f'experiment_before_resume_epoch{start:04d}.json'
    before_path.write_text(json.dumps(exp,indent=2)+'\n')
    event=dict(from_epoch=start,previous_gpu=exp['gpu'],resumed_gpu=args.gpu,
        checkpoint_sha256=sha256(args.out_dir/'checkpoint.pt'),
        previous_experiment_sha256=sha256(before_path),adapter_source_sha256=sha256(Path(__file__)),
        extra_initialization_or_GT_fit=False,status='starting')
    def ready(audit):
        event['sampler']=audit;event['status']='running'
        event_path.write_text(json.dumps(event,indent=2)+'\n')
        print('[resume sampler]',json.dumps(audit),flush=True)
    original=torch.randperm
    replay=EpochShuffleReplay(original,seed=recipe['seed'],views=views,completed_epochs=start,on_ready=ready)
    cmd=['run_sinespin_calibration.py','train','--resume','--gpu',str(args.gpu),
         '--out-dir',str(args.out_dir),'--input-dir',str(args.input_dir),
         '--epochs',str(recipe['epochs']),'--seed',str(recipe['seed']),
         '--batch-size',str(recipe['batch_size']),'--lr',str(recipe['lr']),
         '--view-step',str(recipe['view_step']),'--motion-model','bspline','--initialization','zero-head',
         '--spline-control-points',str(recipe['motion_model_config']['control_points']),
         '--loss','signed_lncc','--lncc-kernel-size',str(recipe['loss_config']['kernel_size']),
         '--ts-max-mm',str(recipe['bounds']['ts_max_mm']),'--tp-max-mm',str(recipe['bounds']['tp_max_mm']),
         '--rot-max-deg',str(recipe['bounds']['rot_max_deg']),
         '--loss-roi-json',str(args.out_dir/'loss_roi.json')]
    if any(recipe['regularization'][name+'_weight'] for name in ('intrinsic','translation','rotation')):
        raise ValueError('This adapter is scoped to the unregularized comparison')
    # Initialize imports that may inspect torch operators before installing the sampler.
    import monai
    from torch import _dynamo
    old_argv=sys.argv
    try:
        sys.argv=cmd;torch.randperm=replay
        import run_sinespin_calibration
        run_sinespin_calibration.main()
        assert replay.calls==recipe['epochs']-start
        event['status']='complete';event['resumed_epoch_shuffles']=replay.calls
    finally:
        torch.randperm=original;sys.argv=old_argv
        event_path.write_text(json.dumps(event,indent=2)+'\n')


if __name__=='__main__':main()
