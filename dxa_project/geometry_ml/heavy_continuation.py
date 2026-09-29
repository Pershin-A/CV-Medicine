"""Finish the small run, then run the large pipeline with artifact-stratified studies."""
import argparse,json,shutil,subprocess,sys,time,psutil
from . import retrain_reviewed as r

def read(path):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError,json.JSONDecodeError):return {}

def stop_tree(pid):
    try:parent=psutil.Process(pid)
    except psutil.NoSuchProcess:return
    processes=parent.children(recursive=True)+[parent]
    for p in processes:
        try:p.terminate()
        except psutil.NoSuchProcess:pass
    psutil.wait_procs(processes,timeout=15)

def wait_and_run(pid):
    progress=r.OUT/'heavy_continuation.json'
    r.save(progress,{'stage':'waiting_for_light','target_hours':8,'limit_hours':12,'legacy_pid':pid})
    while True:
        status=read(r.OUT/'status.json');stage=status.get('stage','')
        light_complete=all((r.OUT/p).exists() for p in ('light/evaluation/report.json','light/synthetic_evaluation/report.json','selection.json'))
        if light_complete and (stage in ('light_complete','heavy_real_timing','complete') or stage.startswith('heavy_training')):
            try:
                parent=psutil.Process(pid)
                if 'dxa_project.geometry_ml.retrain_reviewed' not in parent.cmdline():raise ValueError('Legacy PID no longer identifies the expected process')
                stop_tree(pid)
            except psutil.NoSuchProcess:pass
            break
        if stage=='failed':
            r.save(progress,{'stage':'blocked_by_previous_failure','error':status.get('error')});return
        time.sleep(1)
    r.save(progress,{'stage':'starting_stratified_large_run','target_hours':8,'limit_hours':12})
    r.event('heavy_transition',target_hours=8,limit_hours=12)
    with (r.OUT/'heavy_report_watch.log').open('a',encoding='utf-8') as log:
        subprocess.Popen([sys.executable,'-m','dxa_project.geometry_ml.report_reviewed','--watch'],cwd=r.ROOT,stdout=log,stderr=subprocess.STDOUT)
    with (r.OUT/'heavy_12h.log').open('a',encoding='utf-8') as log:
        child=subprocess.Popen([sys.executable,'-m',__package__+'.heavy_continuation','--run'],cwd=r.ROOT,stdout=log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+12*3600
        while child.poll() is None:
            if time.monotonic()>deadline:
                stop_tree(child.pid)
                r.event('failed',error='Heavy stage exceeded 12 hours; saved checkpoints retained')
                r.save(progress,{'stage':'elapsed_limit_reached','limit_hours':12});return
            time.sleep(10)
        r.save(progress,{'stage':'complete' if child.returncode==0 else 'failed','exit_code':child.returncode,'target_hours':8,'limit_hours':12})

def run():
    from .stratified_protocol import make_stratified_protocol
    device=r.setup();records=r.load_records(r.ROOT);snapshot=read(r.OUT/'data_snapshot.json')
    if r.fingerprint(records)!=snapshot:raise ValueError('Data revision changed; regenerate before training')
    folder=r.OUT/'heavy_stratified';folder.mkdir(parents=True,exist_ok=True);protocol_path=folder/'protocol.json'
    if not protocol_path.exists():
        prepared=r.OUT.parent/'next_training/protocol.json'
        if prepared.exists():shutil.copy2(prepared,protocol_path)
        else:make_stratified_protocol(records,protocol_path)
    a,b=r.split_records(records,0);p=r.make_protocol(r.ROOT,records,a,b,protocol_path);parts=p['partition_by_path']
    aug=r.load_augmented_records(r.ROOT,r.AUG,records)
    train=[x for x in records if parts[x.relative_path]=='train']+[x for x in aug if parts[x.source_id]=='train']
    valid=[x for x in records if parts[x.relative_path]=='validation'];hipvalid=[x for x in valid if x.region!='SPINE']
    synthetic=r.validation_augments(aug,parts);selected=read(r.OUT/'selection.json')['selected']
    r.save(folder/'training_data.json',{'train_images':len(train),'validation_originals':len(valid),'protocol_summary':p['summary'],
           'point_synthetic_validation':len(synthetic),'selection_scope':'configuration inherited from earlier light comparison; holdout is not unseen external data'})
    timing_path=r.OUT/'heavy_timing.json'
    if timing_path.exists() and not read(timing_path).get('stratified_protocol'):
        shutil.copy2(timing_path,r.OUT/'heavy_timing_legacy.json');timing_path.unlink()
    timing=r.heavy_timing(train,valid,selected,device);timing['stratified_protocol']=str(protocol_path);r.save(timing_path,timing)
    if not timing['allowed_full_training']:
        r.event('complete',heavy_full_training=False,heavy_estimate_hours=timing['estimate_hours'],reason='estimate_exceeds_12_hours');return
    r.fit_modules(train,valid,folder,device,'heavy')
    if not (folder/'points_report.json').exists():
        r.clear_gpu();boxes=r.proposed_boxes(hipvalid+synthetic,folder,device);r.clear_gpu()
        report=r.train_variant({**r.VARIANTS[selected['variant']],'architecture':'heavy','batch_size':2},
               [x for x in train if x.region!='SPINE'],hipvalid,folder,device,24,patience=7,
               validation_evaluator=r.Validation(hipvalid,synthetic,boxes));r.save(folder/'points_report.json',report)
    from .landmark_geometry import fit_geometry_ranges
    r.save(folder/'landmark_geometry_bounds.json',fit_geometry_ranges([x for x in records if parts[x.relative_path]=='train' and x.region!='SPINE']))
    r.evaluate_pipeline(folder,protocol_path)
    if r.fingerprint(records)!=snapshot:raise ValueError('Data changed during large-model training')
    r.event('complete',heavy_full_training=True,heavy_estimate_hours=timing['estimate_hours'],heavy_output=str(folder),target_hours=8,limit_hours=12)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--legacy-pid',type=int);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    try:
        if args.run:run()
        else:
            if args.legacy_pid is None:parser.error('--legacy-pid required')
            wait_and_run(args.legacy_pid)
    except Exception as e:r.event('failed',error=repr(e));raise
