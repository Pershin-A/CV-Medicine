"""Finish the improved pipeline after the dedicated point experiments."""
from pathlib import Path
import json,subprocess,sys,time,shutil

def main():
    root=Path(__file__).resolve().parents[2];out=root/'dxa_project/outputs/improvements_v2'
    baseline=root/'dxa_project/outputs/geometry_ml_augmented_5epochs'
    while not (out/'trained/coordinate256/report.json').exists():time.sleep(5)
    def command(name,args):
        print(json.dumps({'stage':name,'status':'started'}),flush=True)
        with (out/f'{name}.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-u','-m',*map(str,args)],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(json.dumps({'stage':name,'status':'finished'}),flush=True)
    for task in ('spine','hip','artifact'):
        folder=out/'modules'/task
        command('train_'+task,['dxa_project.geometry_ml.train','--task',task,'--epochs','20','--size','256','--batch-size','8',
              '--encoder-lr','0.00001' if task=='artifact' else '0.00002',
              '--head-lr','0.00003' if task=='artifact' else '0.0001','--scheduler','cosine','--patience','5',
              '--loader-workers','2','--balance-sources','--selection-protocol',out/'protocol.json',
              '--augmented-root',root/'dxa_project/outputs/augmented_15000_final','--output',folder])
    variants=('shared256','coordinate256')
    from .select_points import select
    selected=select(root,out);chosen=selected['variant']
    pipeline=out/'pipeline';pipeline.mkdir(exist_ok=True)
    shutil.copy2(baseline/'router.pt',pipeline/'router.pt')
    for task in ('spine','hip','artifact'):shutil.copy2(out/'modules'/task/f'{task}.pt',pipeline/f'{task}.pt')
    shutil.copy2(selected['path'],pipeline/'hip_points.pt')
    (out/'selection.json').write_text(json.dumps({'point_variant':chosen,'criterion':'inner validation mean point error only',
          'router':'unchanged original router checkpoint','other_modules':'fresh ImageNet initialization, new training partition'},indent=2),encoding='utf-8')
    from .data import load_records
    from .landmark_geometry import fit_geometry_ranges
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    train=[r for r in load_records(root) if r.region!='SPINE' and protocol['partition_by_path'][r.relative_path]=='train']
    (pipeline/'landmark_geometry_bounds.json').write_text(json.dumps(fit_geometry_ranges(train),indent=2),encoding='utf-8')
    command('point_test',['dxa_project.geometry_ml.experiments','--phase','test','--variants',','.join(variants),'--output',out])
    command('full_evaluation',['dxa_project.geometry_ml.evaluate','--checkpoints',pipeline,'--output',pipeline/'evaluation',
              '--selection-protocol',out/'protocol.json'])
    command('synthetic_evaluation',['dxa_project.geometry_ml.evaluate_augmented','--augmented-root',root/'dxa_project/outputs/augmented_15000_final',
              '--checkpoints',pipeline,'--output',pipeline/'synthetic_evaluation'])
    (out/'complete.json').write_text(json.dumps({'complete':True,'pipeline':str(pipeline)}),encoding='utf-8')
    print(json.dumps({'complete':True,'selected_points':chosen}),flush=True)

if __name__=='__main__':main()
